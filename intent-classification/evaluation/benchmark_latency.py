"""
benchmark_latency.py
---------------------
Compares inference latency and estimated cost per 1,000 queries between:
    1. The fine-tuned QLoRA model (self-hosted, local generation)
    2. A GPT-4 / GPT-4o API call doing the same classification task

This produces the "ROI story" numbers used in the README/resume bullet.

Usage:
    export OPENAI_API_KEY=sk-...
    python benchmark_latency.py \
        --adapter_dir ../training/outputs/qwen2.5-1.5b-intent-qlora \
        --test_file ../data/test.jsonl \
        --n_samples 50
"""

import argparse
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Approximate public pricing, per 1M tokens (update as needed — these change often)
GPT4_INPUT_COST_PER_1M = 5.00
GPT4_OUTPUT_COST_PER_1M = 15.00


def build_prompt(query):
    return (
        "### Instruction:\n"
        "Classify the customer query into intent, urgency, department, and sentiment. "
        "Respond with a single valid JSON object only.\n\n"
        f"### Query:\n{query}\n\n"
        "### Classification:\n"
    )


def benchmark_finetuned(model, tokenizer, queries, max_new_tokens=60):
    latencies = []
    for q in queries:
        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        latencies.append(time.perf_counter() - start)
    return latencies


def benchmark_gpt4(queries, model="gpt-4o", max_new_tokens=60):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    latencies = []
    for q in queries:
        prompt = build_prompt(q)
        start = time.perf_counter()
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0,
        )
        latencies.append(time.perf_counter() - start)
    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--test_file", type=str, default="../data/test.jsonl")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--skip_gpt4", action="store_true",
                         help="Skip the live GPT-4 API calls (e.g. no API key available).")
    parser.add_argument("--gpt4_model", type=str, default="gpt-4o")
    args = parser.parse_args()

    with open(args.test_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    queries = [r["query"] for r in records[: args.n_samples]]

    print(f"Loading fine-tuned model from {args.adapter_dir}...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)  # fp16 - faster on T4
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    print(f"Benchmarking fine-tuned model on {len(queries)} queries...")
    ft_latencies = benchmark_finetuned(model, tokenizer, queries)
    ft_avg = sum(ft_latencies) / len(ft_latencies)
    print(f"Fine-tuned avg latency: {ft_avg*1000:.0f} ms  (self-hosted, ~$0 marginal cost)")

    if not args.skip_gpt4:
        print(f"Benchmarking {args.gpt4_model} on {len(queries)} queries...")
        gpt4_latencies = benchmark_gpt4(queries, model=args.gpt4_model)
        gpt4_avg = sum(gpt4_latencies) / len(gpt4_latencies)
        speedup = gpt4_avg / ft_avg
        print(f"{args.gpt4_model} avg latency: {gpt4_avg*1000:.0f} ms")
        print(f"Speedup: {speedup:.1f}x faster with fine-tuned model")

        # Rough cost estimate for 1K queries (assumes ~120 input + 60 output tokens/query)
        cost_per_1k = (
            (120 * 1000 / 1_000_000) * GPT4_INPUT_COST_PER_1M
            + (60 * 1000 / 1_000_000) * GPT4_OUTPUT_COST_PER_1M
        )
        print(f"Estimated {args.gpt4_model} cost per 1K queries: ${cost_per_1k:.2f}")
        print("Fine-tuned self-hosted cost per 1K queries: ~$0 (excluding GPU/infra amortization)")
    else:
        print("Skipping GPT-4 benchmark (--skip_gpt4). Use README table for reference numbers.")


if __name__ == "__main__":
    main()
