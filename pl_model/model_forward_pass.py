from transformers import LiltForPretraining, HarrysConfig, HarrysInvConfig
from pl_lilt_datamodule import LiltPretrainingDataModule
import argparse
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()
parser = LiltPretrainingDataModule.add_model_specific_args(parser)
args = parser.parse_args()

dm = LiltPretrainingDataModule(args)
dm.prepare_data()
trainer = dm.train_dataloader()

lang_adapter_config = HarrysInvConfig(text_output_adapter=True, layout_output_adapter=False)
layout_adapter_config = HarrysConfig(text_output_adapter=False, layout_output_adapter=True)
task_adapter_config = HarrysConfig(text_output_adapter=True, layout_output_adapter=False)

model = LiltForPretraining.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base")#.from_pretrained("nielsr/lilt-xlm-roberta-base")
model.add_adapter("layout", layout_adapter_config)
model.add_adapter("language", lang_adapter_config)
model.add_adapter("task", task_adapter_config)
model.train_adapter(["layout","task", "language"])
model.freeze_model(True)
model = model.cuda()

for batch_idx,batch in tqdm(enumerate(trainer),total=len(trainer)):
    if batch_idx in []:
        print(f"skipping..{batch_idx}")
        continue
    batch = {k:v.cuda() for k,v in batch.items()}
    out = model(**batch)