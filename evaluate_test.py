import json
import csv
import torch
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from collections import Counter
import re

# Fallback CodeBLEU (token overlap based)
def calc_codebleu(ref: str, pred: str) -> float:
    def tokenize(x): return re.findall(r"\w+|[^\s\w]", x)
    r, p = tokenize(ref), tokenize(pred)
    if not r: return 0.0
    match = sum((Counter(r) & Counter(p)).values())
    return (match / len(r)) * 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + tokenizer
model = T5ForConditionalGeneration.from_pretrained("./models/codet5-if").to(device)
tokenizer = AutoTokenizer.from_pretrained("./models/codet5-if")

# Load test dataset
with open("processed_data/test.json", "r") as f:
    data = [json.loads(line) for line in f]
dataset = Dataset.from_list(data)

# Generate predictions
def generate(example):
    inputs = tokenizer(
        example["input"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    ).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=64)

    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"predicted": pred}

dataset = dataset.map(generate)

# Evaluate predictions
sacrebleu = evaluate.load("sacrebleu")
rows = []

for ex in dataset:
    input_text = ex["input"]
    expected = ex["label"]
    predicted = ex["predicted"]

    match = str(predicted.strip() == expected.strip()).lower()
    bleu = sacrebleu.compute(predictions=[predicted], references=[[expected]])["score"]
    codebleu = calc_codebleu(expected, predicted)

    rows.append([
        input_text,
        match,
        expected,
        predicted,
        round(codebleu, 2),
        round(bleu, 2)
    ])

# Save CSV
with open("testset-results.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Input function with masked if condition",
        "Prediction correct (true/false)",
        "Expected if condition",
        "Predicted if condition",
        "CodeBLEU prediction score (0-100)",
        "BLEU-4 prediction score (0-100)"
    ])
    writer.writerows(rows)

print("testset-results.csv generated.")
