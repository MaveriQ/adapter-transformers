from pytorch_lightning import LightningDataModule
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import LayoutXLMProcessor, DataCollatorForLiltPretraining
import argparse
from typing import Optional
import torch

def get_dataset():
    dataset = load_dataset('rvl_cdip')
    valid_indices = [i for i in range(len(dataset['test'])) if i!=33669]
    dataset['test']=dataset['test'].select(valid_indices)
    return dataset

class LiLTDataModule(LightningDataModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.dataset = load_from_disk('rvl_cdip_fixed') # get_dataset() #load_dataset('rvl_cdip')#,split=['train','validation'])
        self.processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
        # self.dataset.set_transform(self.processor)
        self.collator = DataCollatorForLiltPretraining(self.processor.tokenizer)

    def preprocess_data(self,examples):
        # take a batch of images
        images = [img.convert("RGB") for img in examples['image']]

        encoded_inputs = self.processor(images, padding="max_length", 
                                        truncation=True, 
                                        max_length=512,
                                        return_tensors='pt',
                                        return_special_tokens_mask=True,)
        return encoded_inputs
    
    def prepare_data(self):
        self.labels = self.dataset['train'].features['label'].names
        self.id2label = {id: label for id, label in enumerate(self.labels)}
        self.label2id = {k: v for v, k in enumerate(self.labels)}

        print(f'Filtering dataset for category : {self.args.category}\n')
        self.dataset = self.dataset.filter(lambda e: e['label']==self.label2id[self.args.category], batched=False,num_proc=6)
        print('Extracting features from dataset\n')
        self.dataset = self.dataset.map(self.preprocess_data, batched=True,remove_columns=['image'],num_proc=6)

    def setup(self, stage: Optional[str] = None):
        pass
    
    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collator)

    # def test_dataloader(self):
    #     return DataLoader(self.dataset['test'], batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collator)
    
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--category', type=str, default='form',choices=['form','letter','email','handwritten','advertisement','scientific report','scientific publication','specification','file folder','news article','budget','invoice','presentation','questionnaire','resume','memo'])
        return parser
    
def main():
    parser = argparse.ArgumentParser()
    parser = LiLTDataModule.add_model_specific_args(parser)
    args = parser.parse_args('')
    
    dm = LiLTDataModule(args)
    dm.prepare_data()
    # dm.setup()
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    for k,v in batch.items():
        print(k, v.shape)

    def preprocess_data(examples):
        # take a batch of images
        images = [img.convert("RGB") for img in examples['image']]

        encoded_inputs = processor(images, padding="max_length", truncation=True, max_length=512,return_tensors='pt',return_special_tokens_mask=True)

        encoded_inputs.pop("image")
        return encoded_inputs
    
    dataset = load_dataset('rvl_cdip')#,split='test')
    processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
    dataset.set_transform(preprocess_data)
    data_collator = DataCollatorForLiltPretraining(tokenizer=processor.tokenizer, mlm=True, mlm_probability=0.15)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    batch = next(iter(loader))

    for k,v in batch.items():
        print(k, v.shape)

if __name__ == "__main__":
    main()