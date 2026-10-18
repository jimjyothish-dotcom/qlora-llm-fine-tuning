"""
train_qlora.py — QLoRA fine-tuning of Qwen2.5-1.5B-Instruct for multi-label
customer support intent classification.

FIXED VERSION (tested working on Kaggle T4 x2, July 2026):
  - Does NOT use `trl` (avoids multiple trl/peft/bitsandbytes compatibility bugs
    seen in trl==1.9.2: _patch_chunked_ce_lm_head crash, num_valid_tokens crash).
    Uses plain transformers.Trainer instead, which is stable and does the same job.
  - Uses fp16 instead of bf16 (T4 GPUs are Turing architecture and do not have
    native bf16 support - bf16 runs 5-10x SLOWER on T4 due to software emulation).

Usage:
    python train_qlora.py \
        --train_file data/train.jsonl \
        --eval_file data/test.jsonl \
        --output_dir outputs/qwen2.5-1.5b-intent-qlora \
        --epochs 3

Requires: transformers, peft, bitsandbytes, accelerate, datasets
(trl is NOT required for this script)
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--eval_file", type=str, default="data/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen2.5-1.5b-intent-qlora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading base model: {args.model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # fp16, NOT bf16 - T4 GPUs (Kaggle) don't have native bf16 support.
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading datasets...")
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "test": args.eval_file},
    )

    def tokenize_fn(example):
        out = tokenizer(
            example["text"] + tokenizer.eos_token,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize_fn, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final adapter + tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
