from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Optional, Any, Mapping, Tuple, Union
import torch
import numpy as np



@dataclass
class DataCollatorForLiltPretraining(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    seq_len : int = 512
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

        self.pad_tokens = {'input_ids':self.tokenizer.pad_token_id,
                            'attention_mask':0,
                            'special_tokens_mask':1,
                            'bbox':0}

    def pad(self,examples,key,pad_token_id):

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        if key=='bbox':
            output_shape = (self.bs,self.seq_len,4)
        else:
            output_shape = (self.bs,self.seq_len)

        output = torch.full(output_shape,pad_token_id)
        
        for i,example in enumerate(examples):
            ex_length = min(self.seq_len,example.shape[1]) # Truncate if greater than seq_len.
            if self.tokenizer.padding_side == "right":
                output[i, : ex_length] = example[:,:ex_length] # For Truncation
            else:
                output[i, -ex_length :] = example[:,:ex_length]
        return output

    def collate_fn(self,examples):

        # Convert list of encodings to dict of ragged tensors
        batch = {k:[ex[k] for ex in examples] for k in examples[0].keys()}
        collated = {}
        
        for key in batch.keys():
            if key not in self.pad_tokens.keys():
                continue
            collated[key] = self.pad(batch[key],key,self.pad_tokens[key])
            
        return collated
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        self.bs = len(examples)

        batch = {k:torch.stack([ex[k] for ex in examples],dim=0) for k in examples[0].keys()}

        batch = self.process_bounding_boxes(batch)
        # If special token mask has been preprocessed, pop it from the dict.
        self.special_tokens_mask = batch.pop("special_tokens_mask", None)
        assert self.special_tokens_mask is not None, "special_tokens_mask is not set"

        if self.mlm:
            batch["input_ids"], batch["input_labels"] = self.torch_mask_tokens(batch,input_type = 'input_ids')
            batch["bbox"], batch["bbox_labels"] = self.torch_mask_tokens(batch,input_type = 'bbox')
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        return batch

    def torch_mask_tokens(self, batch: Any, input_type = 'input_ids') -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        attention_mask = batch["attention_mask"]
        if input_type == 'input_ids':
            inputs = batch['input_ids']
            mask_shape = inputs.shape
            labels = inputs.clone()
        elif input_type == 'bbox':
            inputs = batch['bbox']
            mask_shape = inputs.shape[:2]
            labels = torch.zeros(mask_shape,dtype=torch.long)
        

        masked_indices,indices_replaced,indices_random, misaligned_indices, aligned_indices = self.get_masks(mask_shape,self.special_tokens_mask)

        if input_type == 'input_ids':
            labels[~masked_indices] = -100
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            random_words = torch.randint(len(self.tokenizer), mask_shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

        elif input_type == 'bbox': # using .unsqueeze(-1).expand_as() to broadcast the mask to the shape of the label tensor
            alignment_indices = (aligned_indices | misaligned_indices)
            batch['bbox_loc'][~alignment_indices.unsqueeze(-1).expand_as(batch['bbox_loc'])] = -100
            labels[~alignment_indices] = -100
            labels[aligned_indices] = 1
            labels[misaligned_indices] = 0

            inputs[indices_replaced.unsqueeze(-1).expand_as(inputs)] = 999 # 999 is arbitrarily picked as the index of the [MASK] token in the bbox vocabulary

            all_boxes_in_batch = inputs[attention_mask.bool()]
            num_elements = indices_random.sum() # number of elements to sample
            random_boxes = np.random.choice(all_boxes_in_batch.shape[0],num_elements.item())
            inputs[indices_random] = all_boxes_in_batch[random_boxes]

        return inputs, labels

    def get_box_number(self,x,y):
        loc = int(y*10 + x) # Get the box number from the x and y coordinates of 10x10 grid.
        if loc>99: # If the box number is greater than 99, it is out of bounds. Use -100 to skip precicting this box.
            loc = -100
        return loc
    
    def process_bounding_boxes(self,batch):

        bs,seq,_ = batch['bbox'].shape
        all_boxes = batch['bbox'].view(-1,4)
        bbox_loc = []

        for box in all_boxes:
            tl_x,tl_y,br_x,br_y = box.numpy()
            ce_x = (tl_x + br_x)/2 # Find the center of the box
            ce_y = (tl_y + br_y)/2

            tl_x,tl_y,br_x,br_y,ce_x,ce_y = int(tl_x/100),int(tl_y/100),int(br_x/100),int(br_y/100),int(ce_x/100),int(ce_y/100)
            tl = self.get_box_number(tl_x,tl_y)
            br = self.get_box_number(br_x,br_y)
            ce = self.get_box_number(ce_x,ce_y)

            bbox_loc.append((tl,br,ce))

        batch['bbox_loc'] = torch.LongTensor(bbox_loc).view(bs,seq,-1) # (bs,seq,4)
        return batch

    def get_masks(self,mask_shape,special_tokens_mask):

        probability_matrix = torch.full(mask_shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(mask_shape, 0.8)).bool() & masked_indices

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(mask_shape, 0.5)).bool() & masked_indices & ~indices_replaced

        # The random indices are misaligned
        misaligned_indices = indices_random

        # All masked indices that are not replaced by masks or randomly are aligned
        aligned_indices = masked_indices & ~indices_replaced & ~indices_random

        return masked_indices,indices_replaced,indices_random, misaligned_indices, aligned_indices

def process_example(example):
    
    images = [img.convert('RGB') for img in example['image']]
    
    features = img_processor(images)
    encoding = tokenizer(
        text=features["words"],
        boxes=features["boxes"],
        return_special_tokens_mask=True,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    )
    # encoding.pop('image')
    # encoding['image']=example['image']
    # encoding['category']=torch.LongTensor(example['label'])
    # encoding['words']=features["words"]
    return encoding

if __name__ == '__main__':
    from datasets import load_from_disk, load_dataset
    from transformers import LayoutLMv2ImageProcessor, LayoutXLMTokenizerFast

    # dataset = load_from_disk('/work/scratch/hj36wegi/data/rvl_cdip_processed')
    rvl_cdip = load_dataset('rvl_cdip')
    img_processor = LayoutLMv2ImageProcessor(do_resize=False,
                                         do_rescale=False,
                                         do_normalize=False,
                                         ocr_lang='eng',
                                         )
    tokenizer = LayoutXLMTokenizerFast.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base")

    rvl_cdip.set_transform(process_example)

    bs = 4
    examples = [rvl_cdip['train'][i] for i in range(bs)]
    collate_fn = DataCollatorForLiltPretraining(tokenizer=tokenizer,seq_len=512)
    batch = collate_fn(examples)
    print(batch)