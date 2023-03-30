import argparse
from pytorch_lightning import LightningModule, Trainer
# from lightning.pytorch.cli import LightningCLI
import torch
from transformers import AutoModelForTokenClassification
from transformers import HarrysConfig, HarrysInvConfig
import evaluate
from pl_lilt_datamodule import LiltFineTuningDataModule

lang_adapter_config = HarrysInvConfig(text_output_adapter=True, layout_output_adapter=False)
layout_adapter_config = HarrysConfig(text_output_adapter=False, layout_output_adapter=True)
task_adapter_config = HarrysConfig(text_output_adapter=True, layout_output_adapter=False)

class LiltFineTuningModule(LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        # self.save_hyperparameters()
        self.args = args
        self.label_list = self.args.label_list
        self.id2label = {id:label for id, label in enumerate(self.label_list)}
        self.label2id = {label:id for id, label in enumerate(self.label_list)}
        self.model = AutoModelForTokenClassification.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base", 
                                                                     id2label=self.id2label, 
                                                                     label2id=self.label2id)
        self.model.add_adapter("layout", layout_adapter_config)
        self.model.add_adapter("task", task_adapter_config)
        self.model.add_adapter("language", lang_adapter_config)
        
        self.train_metric = evaluate.load("seqeval")
        self.eval_metric = evaluate.load("seqeval")
        self.return_entity_level_metrics = False

    def get_labels(self, predictions, references):
        # Transform predictions and references tensors to numpy arrays
        if self.device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def forward(self, batch): #input_ids, attention_mask, bbox, labels):
        output = self.model(**batch) #input_ids, attention_mask, bbox)
        return output

    def get_loss(self, batch, batch_idx, stage=None):
        outputs = self(batch) #self(**batch)

        predictions = outputs.logits.argmax(-1)
        true_predictions, true_labels = self.get_labels(predictions, batch["labels"])

        if stage == "train":
            self.train_metric.add_batch(references=true_labels, predictions=true_predictions)
        else:
            self.eval_metric.add_batch(references=true_labels, predictions=true_predictions)

        loss = outputs.loss
        return loss
    
    def training_step(self, batch, batch_idx):

        loss = self.get_loss(batch, batch_idx, stage="train")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    # def on_train_epoch_end(self, outputs) -> None:

    #     train_metric = self.train_metric.compute()

    #     train_results = {"overall_accuracy": train_metric["overall_accuracy"],
    #                      "overall_f1": train_metric["overall_f1"]
    #                      }
    #     self.log_dict(train_results, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        
        val_loss = self.get_loss(batch, batch_idx, stage="eval")

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
    
    # def on_validation_epoch_end(self, outputs) -> None:
            
    #     eval_metric = self.eval_metric.compute()

    #     val_results = {"overall_accuracy": eval_metric["overall_accuracy"],
    #                    "overall_f1": eval_metric["overall_f1"]
    #                    }
    #     self.log_dict(val_results, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        test_loss = outputs.loss
        self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return test_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
    
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-5)

        return parser
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser = LiltFineTuningDataModule.add_model_specific_args(parser)
    parser = LiltFineTuningModule.add_model_specific_args(parser)
    args = parser.parse_args()
    return args

def main(args):
    dm = LiltFineTuningDataModule(args)
    args.label_list = dm.args.label_list
    model = LiltFineTuningModule(args)
    # compiled_model = torch.compile(model)
    # loader = dm.train_dataloader()
    # batch = next(iter(loader))
    # out = model.training_step(batch,0)

    trainer = Trainer(devices=1, max_epochs=3, precision=16,accelerator="auto")
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    args = parse_args()
    main(args)