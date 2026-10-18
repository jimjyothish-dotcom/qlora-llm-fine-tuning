# Project Plan: Fine-Tuned Multi-Intent Classifier for Customer Support Automation

## Why this project is real / in-demand

Every company running a customer support chatbot (banking, e-commerce, SaaS, telecom) has a core
problem: routing an incoming query to the right department/action — "refund", "technical issue",
"billing", "cancellation," etc. Using a large model like GPT-4 for this simple classification task is
overkill and expensive. Companies use small fine-tuned models for fast, cheap, low-latency routing —
this pattern shows up in the AI features of platforms like Zendesk, Intercom, and Freshdesk.

**Real advantage:** for classification tasks, a small fine-tuned model (1.5B) often matches a large
model's (GPT-4) accuracy, but at 10-50x lower cost and higher speed — a clean, measurable ROI story.

## 1. Task Definition

**Input:** Customer query (raw text)

**Output:** Structured multi-label classification

```json
{
  "intent": "refund_request",
  "urgency": "high",
  "department": "billing",
  "sentiment": "negative"
}
```

## 2. Tech Stack

| Component | Choice |
|---|---|
| Platform | Kaggle Notebook (T4 x2 GPU) |
| Base model | Qwen2.5-1.5B-Instruct |
| Technique | QLoRA |
| Libraries | `transformers`, `peft`, `trl`, `bitsandbytes` |
| Tracking | Weights & Biases |
| Dataset | Bitext Customer Support dataset (public) + synthetic labels |

## 3. Dataset

- Base: HuggingFace `bitext/Bitext-customer-support-llm-chatbot-training-dataset` — already has intent
  categories and real-world support queries.
- Enhanced with GPT-4o-mini-generated urgency + sentiment labels (synthetic), since the base dataset
  doesn't include those fields.
- Size: 2,000–3,000 examples.

### Format

```
### Instruction:
Classify the customer query into intent, urgency, department, and sentiment.

### Query:
I was charged twice for my subscription this month and need this fixed urgently

### Classification:
{"intent": "billing_error", "urgency": "high", "department": "billing", "sentiment": "negative"}
```

## 4. Evaluation Metrics

- Intent accuracy (multi-class macro F1)
- JSON validity rate
- Confusion matrix (base vs fine-tuned) — visually strong for the README
- Inference speed comparison: fine-tuned 1.5B model vs GPT-4 API latency

| Metric | Base model | Fine-tuned |
|---|---|---|
| Intent F1-score | ~55-65% | ~88-92% |
| Avg latency | GPT-4: ~1.2s | Fine-tuned: ~150-250ms |
| Cost per 1K queries | GPT-4: ~$3-5 | Fine-tuned: ~$0 (self-hosted) |

## 5. Resume Bullet

> Fine-tuned Qwen2.5-1.5B via QLoRA for multi-label customer support intent classification, achieving
> 90%+ F1-score while reducing inference latency by ~5x and API costs to near-zero compared to
> GPT-4-based routing.

## Future directions in this repo

- Resume-to-JSON structured extraction (HR-tech)
- Query rewriting for search/RAG pipelines
- Legal document summarization
- Code review comment generation
