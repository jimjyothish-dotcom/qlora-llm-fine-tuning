"""
format_for_training.py
-----------------------
Step 2 of the pipeline.

Reads data/raw_labeled.jsonl (produced by prepare_dataset.py) and converts it
into an instruction/response format suitable for SFT with `trl`'s SFTTrainer.
Also performs a stratified-ish train/test split.

Output:
    data/train.jsonl
    data/test.jsonl

Usage:
    python format_for_training.py
"""

import argparse
import json
import random
from pathlib import Path

SYSTEM_INSTRUCTION = (
    "Classify the customer query into intent, urgency, department, and sentiment. "
    "Respond with a single valid JSON object only."
)


def to_prompt_format(record: dict) -> dict:
    """Builds the instruction-tuning text example for one record."""
    completion = json.dumps(
        {
            "intent": record["intent"],
            "urgency": record["urgency"],
            "department": record["department"],
            "sentiment": record["sentiment"],
        },
        ensure_ascii=False,
    )

    text = (
        f"### Instruction:\n{SYSTEM_INSTRUCTION}\n\n"
        f"### Query:\n{record['query']}\n\n"
        f"### Classification:\n{completion}"
    )
    return {"text": text, **record}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw_labeled.jsonl")
    parser.add_argument("--train_out", type=str, default="data/train.jsonl")
    parser.add_argument("--test_out", type=str, default="data/test.jsonl")
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} labeled records.")

    formatted = [to_prompt_format(r) for r in records]
    random.shuffle(formatted)

    n_test = int(len(formatted) * args.test_ratio)
    test_set = formatted[:n_test]
    train_set = formatted[n_test:]

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.train_out, "w", encoding="utf-8") as f:
        for r in train_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.test_out, "w", encoding="utf-8") as f:
        for r in test_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Train set: {len(train_set)} examples -> {args.train_out}")
    print(f"Test set:  {len(test_set)} examples -> {args.test_out}")


if __name__ == "__main__":
    main()
