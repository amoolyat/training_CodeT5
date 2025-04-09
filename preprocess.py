import pandas as pd
import os
import json

def mask_condition(code: str, condition: str) -> str:
    # Replace the condition with a <mask> token
    masked = code.replace(f"if {condition}", "if <mask>")
    return " ".join(masked.strip().split())

def preprocess_and_save(input_csv, output_path):
    df = pd.read_csv(input_csv)
    processed = []

    for _, row in df.iterrows():
        input_code = mask_condition(row["cleaned_method"], row["target_block"])
        label = row["target_block"]
        processed.append({"input": input_code, "label": label})

    with open(output_path, "w") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    os.makedirs("processed_data", exist_ok=True)

    preprocess_and_save("data/ft_train.csv", "processed_data/train.json")
    preprocess_and_save("data/ft_valid.csv", "processed_data/valid.json")
    preprocess_and_save("data/ft_test.csv", "processed_data/test.json")
