from datasets import load_dataset
from transformers import LayoutLMv2ImageProcessor, LayoutXLMTokenizerFast, LayoutXLMProcessor
import os
import datasets
os.environ['TESSDATA_PREFIX'] = "/home/hj36wegi/scratch/tessdata"
datasets.config.IN_MEMORY_MAX_SIZE = 98000000000
split = 'train'
rvl_cdip = load_dataset('rvl_cdip',split=split,keep_in_memory=True)

if split=='test':
    valid_indices = [i for i in range(len(rvl_cdip)) if i!=33669]
    rvl_cdip=rvl_cdip.select(valid_indices)

img_processor = LayoutLMv2ImageProcessor(do_resize=False,
                                         do_rescale=False,
                                         do_normalize=False,
                                         ocr_lang='eng',
                                         )
tokenizer = LayoutXLMTokenizerFast.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base")

# processor = LayoutXLMProcessor(img_processor,tokenizer)

def process_example(example):
    
    features = img_processor(example['image'].convert('RGB'))
    encoding = tokenizer(
        text=features["words"],
        boxes=features["boxes"],
        return_special_tokens_mask=True
    )
    # encoding = processor(images=[img],return_special_tokens_mask=True)

    encoding['image']=example['image']
    encoding['category']=example['label']
    encoding['words']=features["words"]
    return encoding

def get_ocr(example,index):
    img = example['image'].convert('RGB')
    ocr = img_processor(img)
    ocr.pop('pixel_values')
    ocr['id'] = index
    return ocr

# ocred = rvl_cdip.map(get_ocr,with_indices=True,num_proc=24)
# print(ocred)
# ocred.save_to_disk('/home/hj36wegi/scratch/rvl_cdip_ocr')

processed = rvl_cdip.map(process_example,with_indices=False,num_proc=24)

print(processed)

processed.save_to_disk(f'/home/hj36wegi/scratch/rvl_cdip_{split}_processed')


