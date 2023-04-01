import argparse
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from transformers.models.lilt.modeling_lilt import LiltForPretraining
from transformers import HarrysConfig, HarrysInvConfig
from pl_lilt_datamodule import LiltPretrainingDataModule
import numpy as np
import pandas as pd
import pickle
import os

lang_adapter_config = HarrysInvConfig(text_output_adapter=True, layout_output_adapter=False)
layout_adapter_config = HarrysConfig(text_output_adapter=False, layout_output_adapter=True)
task_adapter_config = HarrysConfig(text_output_adapter=True, layout_output_adapter=False)
torch.set_float32_matmul_precision('medium')

class LiltPretrainingModule(LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = LiltForPretraining.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base")#("nielsr/lilt-xlm-roberta-base")
        self.model.add_adapter("layout", layout_adapter_config)
        self.model.add_adapter("language", lang_adapter_config)
        self.model.add_adapter("task", task_adapter_config)
        self.model.train_adapter(["layout", "task", "language"])
        self.model.freeze_model(True)

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.lang_weights = {}
        self.task_weights = {}
        self.layout_weights = {}

    def forward(self, batch):

        output = self.model(**batch)
        return output

    def common_step(self, batch,stage=None):
        outputs = self(batch)
        loss_dict = outputs.loss
        # self.log(loss_dict)
        loss = torch.sum(torch.stack(list(loss_dict.values())))
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch,'train')

        if batch_idx % self.args.log_weight_freq == 0:
            self.get_weight_summaries(batch_idx)

        # self.log('train_loss', loss)#, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch,'val')

        # self.log('val_loss', loss)#, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        
        weights = {'lang_weights': self.lang_weights,
                   'task_weights': self.task_weights,
                   'layout_weights': self.layout_weights}

        with open(f'{self.args.output_dir}/adapter_weights/step_{self.global_step}.pkl', 'wb') as f:
            pickle.dump(weights, f)

        self.lang_weights = {}
        self.task_weights = {}
        self.layout_weights = {}


    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch)

        self.log('test_loss', loss)#, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        if self.args.exclude_bn_bias:
            params = self.exclude_from_wt_decay(
                self.named_parameters(), weight_decay=self.args.weight_decay)
        else:
            params = self.parameters()

        optimizer = torch.optim.AdamW(params,
                                      lr=self.args.lr,
                                      betas=(0.9, 0.999),
                                      eps=1e-8,
                                      weight_decay=self.args.weight_decay)

        total_steps = self.num_training_steps

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                total_steps=total_steps,
                pct_start=self.args.sched_warmup_pct,
                max_lr=self.args.lr,
                anneal_strategy='linear',
                final_div_factor=4.0
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(
            limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        train_iters_per_epoch = batches // effective_accum

        return train_iters_per_epoch * self.trainer.max_epochs
    
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--sched_warmup_pct', type=float, default=0.1)
        parser.add_argument('--exclude_bn_bias', type=bool, default=True)
        return parser

    def get_weight_summaries(self,batch_idx):
        self.lang_weights[batch_idx] = self.get_weights('language')
        self.task_weights[batch_idx] = self.get_weights('task')
        self.layout_weights[batch_idx] = self.get_weights('layout')
        self.log('lang_weights', self.lang_weights[batch_idx].mean(axis=0)[0]) # [0] to get the mean of weights; [1] to get the std of weights
        self.log('task_weights', self.task_weights[batch_idx].mean(axis=0)[0]) # [2] to get the mean of biases; [3] to get the std of biases
        self.log('layout_weights', self.layout_weights[batch_idx].mean(axis=0)[0])
    
    def get_weights(self,adapter_name):
        if adapter_name in ['task','language']:
            module_name = 'text_output_adapter'
        else:
            module_name = 'layout_output_adapter'
        all_param = {}
        for layer,module in self.model.get_adapter(adapter_name).items():
            if layer==-1: # Skip Language invertible layer
                continue
            all_param[layer]={}
            all_weights_mean = []
            all_weights_std = []
            all_biases_mean = []
            all_biases_std = []

            for name,param in module[module_name].named_parameters():
                if 'weight' in name:
                    all_weights_mean.append(param.mean().item())
                    all_weights_std.append(param.std().item())
                elif 'bias' in name:
                    all_biases_mean.append(param.mean().item())
                    all_biases_std.append(param.std().item())                             
            all_param[layer]['weight_mean'] = np.mean(all_weights_mean)
            all_param[layer]['weight_std'] = np.mean(all_weights_std)
            all_param[layer]['bias_mean'] = np.mean(all_biases_mean)
            all_param[layer]['bias_std'] = np.mean(all_biases_std)

        return pd.DataFrame(all_param).transpose()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser = LiltPretrainingModule.add_model_specific_args(parser)
    parser = LiltPretrainingDataModule.add_model_specific_args(parser)

    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--val_check_interval", type=int, default=2000)
    parser.add_argument("--limit_val_batches", type=int, default=10)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="/home/hj36wegi/scratch/data/lilt/pretraining/")
    parser.add_argument("--log_weight_freq", type=int, default=100)

    tmp_args = "".split()
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir+'/adapter_weights', exist_ok=True)
    # init model
    model = LiltPretrainingModule(args)

    # init data
    dm = LiltPretrainingDataModule(args)

    # sanity check
    # dm.prepare_data()
    # batch = next(iter(dm.train_dataloader()))
    # out = model(batch)

    # init callbacks
    logger = TensorBoardLogger(save_dir=args.output_dir,
                               version='lilt_pretraining',
                               name='lightning_tb_logs')
    
    checkpoint = ModelCheckpoint(dirpath=args.output_dir+'/checkpoints',
                                 every_n_train_steps=100,
                                 save_top_k=-1)

    # init trainer
    trainer = Trainer(devices=args.devices, 
                        max_epochs=args.max_epochs, 
                        max_steps=args.max_steps, 
                        accelerator=args.accelerator,
                        precision=args.precision,
                        accumulate_grad_batches=args.accumulate_grad_batches,
                        logger=logger,
                        callbacks=[checkpoint],
                        val_check_interval=args.val_check_interval,
                        limit_val_batches=args.limit_val_batches,
                        detect_anomaly=False,
                        strategy=args.strategy,
                      )
    # train
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()