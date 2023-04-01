from datasets import load_dataset
from transformers import LayoutLMv2ImageProcessor, LayoutXLMTokenizerFast
from transformers.models.lilt.modeling_lilt import LiltForPretraining
import torch
from data_collator import DataCollatorForLiltPretraining
from tqdm import tqdm

rvl_cdip = load_dataset('rvl_cdip',split='train',keep_in_memory=True)

img_processor = LayoutLMv2ImageProcessor(do_resize=False,
                                         do_rescale=False,
                                         do_normalize=False,
                                         ocr_lang='eng',
                                         )
tokenizer = LayoutXLMTokenizerFast.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base")

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
        truncation=True,
    )   
    return encoding

rvl_cdip.set_transform(process_example)

collator = DataCollatorForLiltPretraining(tokenizer=tokenizer,seq_len=512)
loader = torch.utils.data.DataLoader(rvl_cdip,batch_size=8,num_workers=4,shuffle=False,collate_fn=collator)
model = LiltForPretraining.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base").cuda()

for idx,batch in tqdm(enumerate(loader),total=len(loader)):
    # batch.pop('bbox_loc')
    # batch.pop('bbox_labels')
    batch_gpu = {k:v.cuda() for k,v in batch.items()}
    val = batch_gpu['bbox_loc'].max()
    if val>99:
        print(idx,val,batch['bbox_loc'].max())
        break
    with torch.no_grad():
        out = model(**batch_gpu)