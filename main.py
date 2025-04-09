import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import evaluate

MODEL_NAME = "Salesforce/codet5-small"

def load_jsonl_dataset(path):
    with open(path, "r") as f:
        lines = [json.loads(line) for line in f]
    return Dataset.from_list(lines)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    input_enc = tokenizer(
        example["input"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    target_enc = tokenizer(
        example["label"],
        padding="max_length",
        truncation=True,
        max_length=64
    )
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": target_enc["input_ids"]
    }

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu = evaluate.load("sacrebleu").compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )["score"]
    exact = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds)

    return {
        "bleu": bleu,
        "exact_match": exact
    }

if __name__ == "__main__":
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    train_data = load_jsonl_dataset("processed_data/train.json").map(
        tokenize, remove_columns=["input", "label"]
    )
    valid_data = load_jsonl_dataset("processed_data/valid.json").map(
        tokenize, remove_columns=["input", "label"]
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir="./models/codet5-if",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5,
        logging_dir="./outputs/logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model("./models/codet5-if")
    tokenizer.save_pretrained("./models/codet5-if")
