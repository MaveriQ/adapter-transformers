from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import evaluate
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from lightning import LightningDataModule, LightningModule

task = "funsd" # "cord"
dataset = load_dataset(f"nielsr/{task}-layoutlmv3")
label_list = dataset["train"].features['ner_tags'].feature.names
id2label = {id:label for id, label in enumerate(label_list)}
label2id = {label:id for id, label in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base")

def prepare_examples(batch):
    encoding = tokenizer(batch["tokens"],
                            boxes=batch["bboxes"],
                            word_labels=batch["ner_tags"],
                            padding="max_length",
                            max_length=512,
                            truncation=True,
                            return_tensors="pt")
    
    return encoding

dataset.set_transform(prepare_examples)

model = AutoModelForTokenClassification.from_pretrained("SCUT-DLVCLab/lilt-infoxlm-base", id2label=id2label, label2id=label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = evaluate.load("seqeval")

return_entity_level_metrics = False

def get_labels(predictions, references):
    # Transform predictions and references tensors to numpy arrays
    if device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels

train_dataloader = DataLoader(dataset["train"], batch_size=2, shuffle=True)
test_dataloader = DataLoader(dataset["test"], batch_size=2, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

model.to(device)

for epoch in range(50):
    print("Epoch:", epoch+1)
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # move batch to device
        batch = {k:v.to(device) for k,v in batch.items()}
        outputs = model(**batch)

        predictions = outputs.logits.argmax(-1)
        true_predictions, true_labels = get_labels(predictions, batch["labels"])
        metric.add_batch(references=true_labels, predictions=true_predictions)

        loss = outputs.loss

        if idx % 100 == 0:
            print("Loss:", loss.item())
            results = metric.compute()
            print("Overall f1:", results["overall_f1"])
            print("Overall precision:", results["overall_f1"])
            print("Overall recall:", results["overall_recall"])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

eval_metric = evaluate.load("seqeval")

for idx, batch in enumerate(tqdm(test_dataloader)):
    # move batch to device
    batch = {k:v.to(device) for k,v in batch.items()}
    with torch.no_grad():
      outputs = model(**batch)

    predictions = outputs.logits.argmax(-1)
    true_predictions, true_labels = get_labels(predictions, batch["labels"])
    eval_metric.add_batch(references=true_labels, predictions=true_predictions)