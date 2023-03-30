import argparse
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from transformers.models.lilt.modeling_lilt import LiltForPretraining
from transformers import HarrysConfig, HarrysInvConfig
from pl_lilt_datamodule import LiltPretrainingDataModule

lang_adapter_config = HarrysInvConfig(text_output_adapter=True, layout_output_adapter=False)
layout_adapter_config = HarrysConfig(text_output_adapter=False, layout_output_adapter=True)
task_adapter_config = HarrysConfig(text_output_adapter=True, layout_output_adapter=False)

class LiltPretrainingModule(LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = LiltForPretraining.from_pretrained("nielsr/lilt-xlm-roberta-base")
        self.model.add_adapter("layout", layout_adapter_config)
        self.model.add_adapter("language", lang_adapter_config)
        self.model.add_adapter("task", task_adapter_config)
        self.model.train_adapter(["layout", "task", "language"])
        self.model.freeze_model(False)

    def forward(self, batch):

        output = self.model(**batch)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)
    
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=0.01) 
        return parser

def parse_args():
    parser = argparse.ArgumentParser()
    parser = LiltPretrainingModule.add_model_specific_args(parser)
    parser = LiltPretrainingDataModule.add_model_specific_args(parser)

    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # init model
    model = LiltPretrainingModule(args)
    # init data
    dm = LiltPretrainingDataModule(args)
    dm.prepare_data()
    batch = next(iter(dm.train_dataloader()))

    out = model(batch)

    # init trainer
    trainer = Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20)
    # train
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()