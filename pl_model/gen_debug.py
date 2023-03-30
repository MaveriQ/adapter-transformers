import argparse
from pl_lilt_datamodule import LiltPretrainingDataModule
from transformers import LiltForPretraining, HarrysConfig, HarrysInvConfig

parser = argparse.ArgumentParser()
parser = LiltPretrainingDataModule.add_model_specific_args(parser)
args = parser.parse_args()

dm = LiltPretrainingDataModule(args)
dm.prepare_data()

# dataset = dm.dataset['train']
# collator = DataCollatorForLiltPretraining(tokenizer=dm.processor.tokenizer, mlm=True, mlm_probability=0.15)
# loader = DataLoader(dataset, batch_size=4, collate_fn=collator)
# batch = next(iter(loader))
# for k,v in batch.items():
#     print(k, v.shape)

loader = dm.train_dataloader()
batch = next(iter(loader))

lang_adapter_config = HarrysInvConfig(text_output_adapter=True, layout_output_adapter=False)
layout_adapter_config = HarrysConfig(text_output_adapter=False, layout_output_adapter=True)
task_adapter_config = HarrysConfig(text_output_adapter=True, layout_output_adapter=False)

model = LiltForPretraining.from_pretrained("nielsr/lilt-xlm-roberta-base")
model.add_adapter("layout", layout_adapter_config)
model.add_adapter("language", lang_adapter_config)
model.add_adapter("task", task_adapter_config)

model.train_adapter(["layout","task", "language"])
# model.freeze_model(False)

out = model(**batch)
print(out.loss.shape)