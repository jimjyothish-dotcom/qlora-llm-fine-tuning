"""
prepare_dataset.py
------------------
Step 1 of the pipeline.

Loads the public Bitext customer-support dataset (already contains an `intent`
field) and enriches each example with synthetic `urgency` and `sentiment`
labels using GPT-4o-mini (or any OpenAI-compatible endpoint).

Output: data/raw_labeled.jsonl
    Each line: {"query": ..., "intent": ..., "urgency": ..., "sentiment": ..., "department": ...}

Usage:
    export OPENAI_API_KEY=sk-...
    python prepare_dataset.py --n_samples 2500

If you don't have an OpenAI key handy, pass --skip_synthetic to fall back to
simple heuristic labeling (keyword-based) so the pipeline still runs end to end.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Map Bitext's fine-grained intents -> a smaller department taxonomy.
# Extend this mapping if you use a different base dataset.
DEPARTMENT_MAP = {
    "cancel_order": "orders",
    "change_order": "orders",
    "place_order": "orders",
    "track_order": "orders",
    "delivery_period": "orders",
    "get_refund": "billing",
    "check_refund_policy": "billing",
    "payment_issue": "billing",
    "check_invoice": "billing",
    "check_payment_methods": "billing",
    "create_account": "account",
    "delete_account": "account",
    "edit_account": "account",
    "switch_account": "account",
    "registration_problems": "account",
    "recover_password": "account",
    "newsletter_subscription": "marketing",
    "complaint": "support",
    "contact_customer_service": "support",
    "contact_human_agent": "support",
    "review": "support",
    "change_shipping_address": "shipping",
    "set_up_shipping_address": "shipping",
}

URGENCY_KEYWORDS_HIGH = [
    "urgent", "immediately", "asap", "right now", "furious", "angry",
    "scam", "fraud", "charged twice", "not working", "broken", "still waiting",
]
SENTIMENT_NEGATIVE_KEYWORDS = [
    "angry", "frustrated", "disappointed", "terrible", "worst", "unacceptable",
    "never", "annoyed", "furious", "ridiculous",
]
SENTIMENT_POSITIVE_KEYWORDS = [
    "thank", "great", "awesome", "appreciate", "love", "happy", "excellent",
]


def heuristic_label(query: str):
    """Cheap fallback labeling when no LLM API key is available."""
    q = query.lower()
    urgency = "high" if any(k in q for k in URGENCY_KEYWORDS_HIGH) else "low"
    if urgency != "high":
        urgency = "medium" if "?" in query or len(query.split()) > 25 else "low"

    if any(k in q for k in SENTIMENT_NEGATIVE_KEYWORDS):
        sentiment = "negative"
    elif any(k in q for k in SENTIMENT_POSITIVE_KEYWORDS):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return urgency, sentiment


def llm_label(query: str, client, model="gpt-4o-mini", retries=3):
    """Use an LLM to assign urgency + sentiment labels. Returns (urgency, sentiment)."""
    prompt = f"""Classify the following customer support query.

Query: "{query}"

Return ONLY a JSON object with two keys:
- "urgency": one of ["low", "medium", "high"]
- "sentiment": one of ["positive", "neutral", "negative"]

JSON:"""

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50,
            )
            text = resp.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            return parsed.get("urgency", "medium"), parsed.get("sentiment", "neutral")
        except Exception as e:
            if attempt == retries - 1:
                print(f"[warn] LLM labeling failed for query, falling back to heuristic: {e}")
                return heuristic_label(query)
            time.sleep(1.5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_synthetic", action="store_true",
                         help="Use heuristic labeling instead of calling an LLM API.")
    parser.add_argument("--output", type=str, default="data/raw_labeled.jsonl")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading Bitext customer support dataset from HuggingFace...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )

    # Shuffle and subsample
    ds = ds.shuffle(seed=args.seed)
    n = min(args.n_samples, len(ds))
    ds = ds.select(range(n))
    print(f"Using {n} examples.")

    client = None
    if not args.skip_synthetic:
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("[warn] OPENAI_API_KEY not set, falling back to heuristic labeling.")
                args.skip_synthetic = True
            else:
                client = OpenAI(api_key=api_key)
        except ImportError:
            print("[warn] openai package not installed, falling back to heuristic labeling.")
            args.skip_synthetic = True

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="Labeling"):
            query = row["instruction"]
            intent = row["intent"]
            department = DEPARTMENT_MAP.get(intent, "general")

            if args.skip_synthetic:
                urgency, sentiment = heuristic_label(query)
            else:
                urgency, sentiment = llm_label(query, client, model=args.openai_model)

            record = {
                "query": query,
                "intent": intent,
                "urgency": urgency,
                "department": department,
                "sentiment": sentiment,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Labeled dataset written to {out_path}")


if __name__ == "__main__":
    main()
