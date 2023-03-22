from datasets import load_dataset
from transformers.adapters import AdapterTrainer, HarrysConfig
from transformers import DataCollatorForLiltPretraining, TrainingArguments, AutoTokenizer, AutoAdapterModel, LiltForPretraining
from torch.utils.data import DataLoader

adapter_config = HarrysConfig()
model = LiltForPretraining.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base")

model.add_adapter("ner", config=adapter_config)
model.train_adapter("ner")

dataset = load_dataset("nielsr/funsd-layoutlmv3",split='train')
tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base")
collator = DataCollatorForLiltPretraining(tokenizer)

def prepare_examples(batch):
    encoding = tokenizer(batch["tokens"],
                        boxes=batch["bboxes"],
                        word_labels=batch["ner_tags"],
                        padding="max_length",
                        max_length=128,
                        truncation=True,
                        return_tensors="pt",
                        return_special_tokens_mask=True)

    encoding.pop("labels")
    return encoding

dataset.set_transform(prepare_examples)
loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

batch = next(iter(loader))
print(batch.keys())
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

out = model(**batch)
print(out.keys())


training_args = TrainingArguments(
    output_dir="./examples",
    do_train=True,
    learning_rate=0.01,
    max_steps=1000,
    no_cuda=True,
    per_device_train_batch_size=2,
    remove_unused_columns=False,
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    # tokenizer=tokenizer,
)

trainer.train()