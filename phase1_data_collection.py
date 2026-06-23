"""
Phase 1: Data Collection — Biomedical LLM Q&A Project
=======================================================

Goal: download PubMedQA and MedQA (USMLE), explore their structure,
and convert both into one unified JSON schema that Phase 2 (RAG) and
Phase 3 (LoRA fine-tuning) can both consume directly.

Run this in Google Colab, Kaggle, or any environment with internet
access. No GPU needed for this step.

    pip install datasets pandas
"""

import json
import os
from datasets import load_dataset

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 1. PubMedQA
# ---------------------------------------------------------------------
# pqa_labeled   -> 1k expert-annotated examples (best for evaluation)
# pqa_artificial -> 211k auto-generated examples (best for training)
print("=" * 60)
print("Loading PubMedQA (labeled)...")
pubmedqa_labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
print(pubmedqa_labeled)
print("\nExample record:")
print(json.dumps(pubmedqa_labeled["train"][0], indent=2, default=str)[:800])

print("\nLoading PubMedQA (artificial, larger training set)...")
pubmedqa_artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
print(pubmedqa_artificial)


# ---------------------------------------------------------------------
# 2. MedQA (USMLE-style, 4-option multiple choice)
# ---------------------------------------------------------------------
print("=" * 60)
print("Loading MedQA (USMLE 4-option)...")
medqa = load_dataset("GBaker/MedQA-USMLE-4-options")
print(medqa)
print("\nExample record:")
print(json.dumps(medqa["train"][0], indent=2, default=str)[:800])


# ---------------------------------------------------------------------
# 3. BioASQ — manual download required
# ---------------------------------------------------------------------
# BioASQ (Task B) is not redistributable via the datasets library.
# To get it:
#   1. Register for free at http://participants-area.bioasq.org/
#   2. Download the latest "Task B" training set (JSON format)
#   3. Save it to: data/raw/bioasq_task_b.json
#
# This step can be done in parallel while you start building with
# PubMedQA + MedQA — BioASQ adds factoid/list/yes-no questions with
# multiple gold PubMed snippets per question, which is great for
# evaluating your retriever later in Phase 2.
print("=" * 60)
bioasq_path = os.path.join(RAW_DIR, "bioasq_task_b.json")
if os.path.exists(bioasq_path):
    with open(bioasq_path) as f:
        bioasq_raw = json.load(f)
    print(f"Found BioASQ file with {len(bioasq_raw.get('questions', []))} questions.")
else:
    print("BioASQ file not found — skipping for now.")
    print("See comments above for how to download it.")
    bioasq_raw = None


# ---------------------------------------------------------------------
# 4. Unify everything into one schema
# ---------------------------------------------------------------------
# Common schema used across the whole project:
# {
#   "id": str,
#   "source": "pubmedqa" | "medqa" | "bioasq",
#   "question": str,
#   "context": str,           # supporting text/abstract(s), if any
#   "answer": str,            # short or final answer
#   "long_answer": str,       # explanation / reasoning, if available
#   "choices": dict | None,   # multiple-choice options, if any
# }

def convert_pubmedqa(split, source_name="pubmedqa"):
    records = []
    for ex in split:
        ctx = ex.get("context", {})
        contexts = ctx.get("contexts", []) if isinstance(ctx, dict) else [str(ctx)]
        records.append({
            "id": f"pubmedqa_{ex.get('pubid', len(records))}",
            "source": source_name,
            "question": ex["question"],
            "context": " ".join(contexts),
            "answer": ex.get("final_decision", ""),
            "long_answer": ex.get("long_answer", ""),
            "choices": None,
        })
    return records


def convert_medqa(split):
    records = []
    for i, ex in enumerate(split):
        records.append({
            "id": f"medqa_{i}",
            "source": "medqa",
            "question": ex["question"],
            "context": "",
            "answer": ex.get("answer", ""),
            "long_answer": "",
            "choices": ex.get("options", None),
        })
    return records


def convert_bioasq(raw):
    records = []
    if raw is None:
        return records
    for i, ex in enumerate(raw.get("questions", [])):
        snippets = [s.get("text", "") for s in ex.get("snippets", [])]
        records.append({
            "id": f"bioasq_{ex.get('id', i)}",
            "source": "bioasq",
            "question": ex["body"],
            "context": " ".join(snippets),
            "answer": ex.get("exact_answer", ""),
            "long_answer": ex.get("ideal_answer", [""])[0] if ex.get("ideal_answer") else "",
            "choices": None,
        })
    return records


print("=" * 60)
print("Converting datasets to unified schema...")

unified_train = []
unified_train += convert_pubmedqa(pubmedqa_artificial["train"])
unified_train += convert_medqa(medqa["train"])
unified_train += convert_bioasq(bioasq_raw)

unified_eval = convert_pubmedqa(pubmedqa_labeled["train"])  # pqa_labeled has no separate test split here

print(f"Unified training records: {len(unified_train)}")
print(f"Unified evaluation records: {len(unified_eval)}")

with open(os.path.join(PROCESSED_DIR, "train.json"), "w") as f:
    json.dump(unified_train, f, indent=2)

with open(os.path.join(PROCESSED_DIR, "eval.json"), "w") as f:
    json.dump(unified_eval, f, indent=2)

print(f"\nSaved unified data to {PROCESSED_DIR}/train.json and {PROCESSED_DIR}/eval.json")
print("Phase 1 complete. Next: build the PubMed abstract corpus + vector store (Phase 2).")
