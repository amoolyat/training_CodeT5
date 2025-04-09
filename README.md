# Fine-Tuning CodeT5 for Predicting `if` Conditions in Python

This project fine-tunes the `Salesforce/codet5-small` model to predict missing `if` conditions in Python functions. The model is trained on a dataset of masked Python functions and learns to generate accurate and context-aware conditional expressions.

---

## Task Overview

Given a Python function with a masked `if` condition, the model's objective is to reconstruct the missing logical condition. For example:

**Input (masked):**
```python
def check_positive(num):
    <mask>: return "Positive"
    else: return "Non-Positive"
```

**Predicted Output:**  
`if num > 0`

---

## How It Works

- The dataset was preprocessed to mask real `if` conditions inside function bodies.
- Each sample includes:
  - A `cleaned_method` with the `<mask>` token
  - The original `target_block` (i.e., the masked `if` condition)

The model was fine-tuned on this data using supervised learning to minimize the loss between predicted and ground truth conditions.

---

## Files Included

- `main.py` — Fine-tunes the model using the Hugging Face `Trainer` API.
- `evaluate_test.py` — Runs inference on the test set and computes BLEU & CodeBLEU.
- `testset-results.csv` — Output file with all test predictions and metrics.
- `processed_data/` — Contains the `train.json`, `valid.json`, and `test.json` datasets.
- `models/` — Directory containing the fine-tuned CodeT5 model weights.
- `CodeBLEU.ipynb` — Official notebook for running Microsoft's CodeBLEU script.

---

## Evaluation Metrics

Each model prediction was evaluated using:

- **Exact Match** — Checks if the predicted `if` condition matches the reference exactly.
- **BLEU-4** — Measures n-gram overlap between the predicted and true conditions.
- **CodeBLEU** — A structural and semantic-aware evaluation metric tailored to code generation.

Evaluation was run on a 5,000-sample test set using these metrics and saved in `testset-results.csv`.

---

## How to Reproduce

1. Install dependencies:

```bash
!pip install transformers datasets accelerate evaluate sentencepiece sacrebleu codebleu
```

2. Run training:

```bash
python main.py
```

3. Run evaluation:

```bash
python evaluate_test.py
```

---

