import argparse
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from transformers.models.lilt.modeling_lilt import LiltForPretraining
from pl_lilt_datamodule import LiLTDataModule

class LiLTModel(LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = LiltForPretraining.from_pretrained("nielsr/lilt-xlm-roberta-base")

    def forward(self, input_ids, bbox, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):

        output = self.model(input_ids=input_ids, 
                            bbox=bbox, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids, 
                            position_ids=position_ids, 
                            head_mask=head_mask, 
                            labels=labels)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
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
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        return parser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # init data
    dm = LiLTDataModule(args)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    # init model
    model = LiLTModel(args)
    out = model(**batch)

    # init trainer
    trainer = Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20)
    # train
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()