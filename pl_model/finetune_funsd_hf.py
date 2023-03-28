from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import evaluate
import torch
import numpy as np
from transformers import TrainingArguments, Trainer

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

# Taken from the token-classification example
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
args = TrainingArguments(output_dir="lilt-roberta-en-base-finetuned-funsd",
                         overwrite_output_dir=True,
                         remove_unused_columns=False,
                         warmup_steps=0.1,
                         max_steps=2000,
                         evaluation_strategy="steps",
                         eval_steps=100,
                         push_to_hub=False)

trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

trainer.train()

metrics = trainer.evaluate()
print(metrics)