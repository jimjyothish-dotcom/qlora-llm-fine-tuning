"""
evaluate.py
-----------
Step 4 of the pipeline: evaluates the fine-tuned model on the held-out test
set and reports:
    - Per-field accuracy / F1 (intent, urgency, department, sentiment)
    - JSON validity rate (did the model output parseable JSON?)
    - Confusion matrix for `intent` (saved as a PNG for the README)

Usage:
    python evaluate.py \
        --adapter_dir ../training/outputs/qwen2.5-1.5b-intent-qlora \
        --test_file ../data/test.jsonl
"""

import argparse
import json
import re

import matplotlib.pyplot as plt
import torch
from peft import PeftModel
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FIELDS = ["intent", "urgency", "department", "sentiment"]


def load_model(adapter_dir, base_model=BASE_MODEL):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # fp16, not bf16 - faster on Kaggle T4 GPUs
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def build_prompt(query):
    return (
        "### Instruction:\n"
        "Classify the customer query into intent, urgency, department, and sentiment. "
        "Respond with a single valid JSON object only.\n\n"
        f"### Query:\n{query}\n\n"
        "### Classification:\n"
    )


def extract_json(text):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


@torch.no_grad()
def generate_prediction(model, tokenizer, query, max_new_tokens=80):
    prompt = build_prompt(query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--test_file", type=str, default="../data/test.jsonl")
    parser.add_argument("--confusion_matrix_out", type=str, default="confusion_matrix.png")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    print("Loading fine-tuned model...")
    model, tokenizer = load_model(args.adapter_dir, args.base_model)

    records = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.max_examples:
        records = records[: args.max_examples]

    y_true = {field: [] for field in FIELDS}
    y_pred = {field: [] for field in FIELDS}
    valid_json_count = 0

    for record in tqdm(records, desc="Evaluating"):
        raw_output = generate_prediction(model, tokenizer, record["query"])
        parsed = extract_json(raw_output)

        if parsed is not None:
            valid_json_count += 1

        for field in FIELDS:
            y_true[field].append(record[field])
            pred_val = parsed.get(field, "INVALID") if parsed else "INVALID"
            y_pred[field].append(pred_val)

    print("\n=== Results ===")
    json_validity_rate = valid_json_count / len(records)
    print(f"JSON validity rate: {json_validity_rate:.2%}")

    for field in FIELDS:
        f1 = f1_score(y_true[field], y_pred[field], average="macro", zero_division=0)
        print(f"{field:12s} macro-F1: {f1:.3f}")

    # Confusion matrix for the main `intent` field
    labels = sorted(set(y_true["intent"]) | set(y_pred["intent"]))
    cm = confusion_matrix(y_true["intent"], y_pred["intent"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(args.confusion_matrix_out, dpi=150)
    print(f"Confusion matrix saved to {args.confusion_matrix_out}")


if __name__ == "__main__":
    main()
