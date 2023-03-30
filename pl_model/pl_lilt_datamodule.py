from lightning import LightningDataModule
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, random_split
from transformers import LayoutXLMProcessor, AutoTokenizer
from data_collator import DataCollatorForLiltPretraining
import argparse
from typing import Optional
import torch

def get_dataset():
    dataset = load_dataset('rvl_cdip')
    valid_indices = [i for i in range(len(dataset['test'])) if i!=33669]
    dataset['test']=dataset['test'].select(valid_indices)
    return dataset

class LiltPretrainingDataModule(LightningDataModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        dataset = load_from_disk(f'{self.args.data_dir}/rvl_cdip_processed')
        self.dataset = dataset.remove_columns(['image','category','words'])
        self.processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base", seq_len = self.args.seq_len)
        # self.dataset.set_transform(self.processor)
        self.collator = DataCollatorForLiltPretraining(self.processor.tokenizer)
    
    def prepare_data(self):
        self.labels = self.dataset['train'].features['label'].names
        self.id2label = {id: label for id, label in enumerate(self.labels)}
        self.label2id = {k: v for v, k in enumerate(self.labels)}

        if self.args.category != 'all':
            print(f'Filtering dataset for category : {self.args.category}\n')
            self.dataset = self.dataset.filter(lambda e: e['label']==self.label2id[self.args.category], batched=False,num_proc=32)
        self.dataset = self.dataset.remove_columns(['label'])
        self.dataset.set_format('torch')

    def setup(self, stage: Optional[str] = None):
        pass
    
    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collator)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collator)
    
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="/work/scratch/hj36wegi/data")
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--seq_len', type=int, default=512)
        parser.add_argument('--category', type=str, default='all',choices=['all','form','letter','email','handwritten','advertisement','scientific report','scientific publication','specification','file folder','news article','budget','invoice','presentation','questionnaire','resume','memo'])
        return parser

class LiltFineTuningDataModule(LightningDataModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        # self.save_hyperparameters()
        self.args = args
        self.dataset = load_dataset(f"nielsr/{self.args.task}-layoutlmv3")
        self.tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base")
        self.dataset.set_transform(self.prepare_examples)
        self.args.label_list = self.dataset["train"].features['ner_tags'].feature.names

    def prepare_examples(self, batch):
        encoding = self.tokenizer(batch["tokens"],
                                boxes=batch["bboxes"],
                                word_labels=batch["ner_tags"],
                                padding="max_length",
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
        
        return encoding
    
    def setup(self, stage=None):

        train_length = int(len(self.dataset["train"])*0.8)
        validation_length = len(self.dataset["train"]) - train_length

        self.train_dataset, self.eval_dataset = random_split(self.dataset["train"], [train_length, validation_length])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="/work/scratch/hj36wegi/data")
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--task', type=str, default='funsd',choices=['funsd', 'cord'])
        parser.add_argument('--label_list', type=list, default=[])
        return parser
    
def main():

    DataModule = LiltPretrainingDataModule
    parser = argparse.ArgumentParser()
    parser = DataModule.add_model_specific_args(parser)
    args = parser.parse_args('')
    
    dm = DataModule(args)
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    for k,v in batch.items():
        print(k, v.shape)

if __name__ == "__main__":
    main()