"""
Phase 4: Evaluation — Base vs Fine-Tuned vs RAG+Fine-Tuned
============================================================

Goal: run all three versions of your pipeline on the PubMedQA labeled
evaluation set and compute Exact Match, F1, and ROUGE-L for each. These
numbers are what go directly into your README and resume bullets.

    pip install rouge-score
"""

import json
import os
import pickle
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROCESSED_DIR = "data/processed"
INDEX_DIR = "data/index"
ADAPTER_DIR = "models/biogpt-lora-biomedqa"
BASE_MODEL = "microsoft/biogpt"

# Start small to make sure everything runs end-to-end, then raise this
# for the final numbers you report (e.g. 200-500).
N_EVAL = 50

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# 1. Load evaluation data (PubMedQA labeled set from Phase 1)
# ---------------------------------------------------------------------
with open(os.path.join(PROCESSED_DIR, "eval.json")) as f:
    eval_data = json.load(f)[:N_EVAL]
print(f"Evaluating on {len(eval_data)} examples")


# ---------------------------------------------------------------------
# 2. Load tokenizer + base model, then attach the LoRA adapter
# ---------------------------------------------------------------------
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.to(device).eval()

print("Attaching LoRA adapter from Phase 3...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.to(device).eval()
# `model` now has LoRA active by default. We use model.disable_adapter()
# below to temporarily get "base BioGPT" behavior from the same object —
# this avoids loading the model into memory twice.


# ---------------------------------------------------------------------
# 3. Load Phase 2 retrieval index for the RAG pipeline
# ---------------------------------------------------------------------
print("Loading Phase 2 retrieval index...")
with open(os.path.join(INDEX_DIR, "corpus.json")) as f:
    corpus = json.load(f)
faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
with open(os.path.join(INDEX_DIR, "bm25.pkl"), "rb") as f:
    bm25 = pickle.load(f)
embedder = SentenceTransformer("neuml/pubmedbert-base-embeddings", device=device)


def hybrid_search(query, top_k=2, alpha=0.5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    dense_scores, dense_idx = faiss_index.search(q_emb, len(corpus))
    dense_scores, dense_idx = dense_scores[0], dense_idx[0]
    d_range = dense_scores.max() - dense_scores.min() + 1e-9
    dense_norm = (dense_scores - dense_scores.min()) / d_range
    dense_map = dict(zip(dense_idx, dense_norm))

    bm25_scores = bm25.get_scores(query.lower().split())
    b_range = bm25_scores.max() - bm25_scores.min() + 1e-9
    bm25_norm = (bm25_scores - bm25_scores.min()) / b_range

    combined = [(i, alpha * dense_map.get(i, 0) + (1 - alpha) * bm25_norm[i]) for i in range(len(corpus))]
    combined.sort(key=lambda x: x[1], reverse=True)
    return [corpus[i]["text"] for i, _ in combined[:top_k]]


# ---------------------------------------------------------------------
# 4. Generation helper
# ---------------------------------------------------------------------
def generate(prompt, use_adapter=True, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        if use_adapter:
            output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id)
        else:
            with model.disable_adapter():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def extract_verdict(text):
    """Pull a yes/no/maybe verdict out of generated text for Exact Match scoring."""
    lowered = text.lower()
    for word in ["yes", "no", "maybe"]:
        if word in lowered[:30]:
            return word
    return lowered.split()[0] if lowered.split() else ""


def lcs_length(a, b):
    """Length of the longest common subsequence between two token lists."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_fmeasure(pred, gold):
    """ROUGE-L F-measure based on longest common subsequence — no external deps."""
    pred_tokens, gold_tokens = pred.lower().split(), gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def f1_score_text(pred, gold):
    """Simple token-overlap F1 (SQuAD-style), useful alongside ROUGE-L."""
    pred_tokens, gold_tokens = pred.lower().split(), gold.lower().split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0
    return 2 * precision * recall / (precision + recall + 1e-9)


# ---------------------------------------------------------------------
# 5. Run all three pipelines on each example and score
# ---------------------------------------------------------------------
results = {name: {"em": [], "f1": [], "rougeL": []} for name in ["base", "finetuned", "rag_finetuned"]}

print("Running evaluation...")
for i, ex in enumerate(eval_data):
    question = ex["question"]
    gold_verdict = ex["answer"].lower().strip()
    gold_long = ex.get("long_answer", "").strip()

    prompt = f"Context: {ex['context']}\nQuestion: {question}\nAnswer:"
    retrieved = hybrid_search(question, top_k=2)
    rag_prompt = f"Context: {' '.join(retrieved)}\nQuestion: {question}\nAnswer:"

    predictions = {
        "base": generate(prompt, use_adapter=False),
        "finetuned": generate(prompt, use_adapter=True),
        "rag_finetuned": generate(rag_prompt, use_adapter=True),
    }

    for name, pred in predictions.items():
        verdict = extract_verdict(pred)
        results[name]["em"].append(1.0 if verdict == gold_verdict else 0.0)
        if gold_long:
            results[name]["f1"].append(f1_score_text(pred, gold_long))
            results[name]["rougeL"].append(rouge_l_fmeasure(pred, gold_long))
        else:
            results[name]["f1"].append(0.0)
            results[name]["rougeL"].append(0.0)

    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(eval_data)} done")


# ---------------------------------------------------------------------
# 6. Report results
# ---------------------------------------------------------------------
print("=" * 60)
print(f"{'Pipeline':<16} {'Exact Match':>12} {'F1':>8} {'ROUGE-L':>8}")
labels = {"base": "Base BioGPT", "finetuned": "Fine-tuned", "rag_finetuned": "RAG+Fine-tuned"}
summary = {}
for name, label in labels.items():
    em, f1, rl = (np.mean(results[name][m]) * 100 for m in ["em", "f1", "rougeL"])
    summary[name] = {"exact_match": em, "f1": f1, "rougeL": rl}
    print(f"{label:<16} {em:>11.1f}% {f1:>7.1f}% {rl:>7.1f}%")

with open("eval_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved results to eval_results.json — use these numbers in your README/resume.")
print("Next: build the Gradio demo app and polish the GitHub repo (Phase 5).")
