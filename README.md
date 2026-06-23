# 🧬 Biomedical LLM Q&A — RAG + LoRA Fine-Tuned BioGPT

<div align="center">

**A production-style biomedical question answering system that retrieves real PubMed evidence and generates cited, grounded answers - built from scratch in 5 phases.**

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-FF6B6B?style=for-the-badge)](https://huggingface.co/spaces/beingniki/biomedical-qa)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/beingniki/biomedical-llm-qa-project)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

</div>

---

## The Problem This Solves

Biomedical professionals as clinicians, researchers, drug developers, ask complex questions that require evidence from thousands of research papers. General LLMs like GPT-4 answer these confidently but **hallucinate up to 30% of the time** in biomedical contexts: citing papers that don't exist, misquoting trial statistics, generating outdated treatment guidelines.

**This system solves that.** Before generating any answer, it retrieves real, current PubMed abstracts and grounds every response in actual published evidence with sources shown alongside the answer so users can verify every claim.

> *This is the same architectural pattern used in production at Microsoft (BioGPT), Google (Med-PaLM 2), and Elsevier (Clinical Key AI) and is directly aligned with FDA's 2025–2026 guidance on evidence-grounded AI in drug development.*

---

## What It Does

```
You ask:   "Do statins reduce the risk of cardiovascular events?"

System:    1. Searches 5,000+ indexed PubMed abstracts (FAISS + BM25)
           2. Retrieves the 3 most relevant papers with relevance scores
           3. Fine-tuned BioGPT reads the evidence and generates an answer
           
You get:   A direct answer + the actual abstracts it used to reach it
```

**Try it live →** [huggingface.co/spaces/beingniki/biomedical-qa](https://huggingface.co/spaces/beingniki/biomedical-qa)

---

## Results — Ablation Study on PubMedQA

> *Three pipelines evaluated head-to-head on the same 50 expert-annotated questions. This comparison — called an ablation study — shows the isolated contribution of each component.*

| Pipeline | Exact Match | F1 Score | ROUGE-L | What this tells us |
|---|---|---|---|---|
| 🔴 Base BioGPT (no changes) | ~45% | ~25% | ~20% | Baseline — raw model performance |
| 🟡 LoRA Fine-Tuned BioGPT | ~55% | ~35% | ~28% | +10pp from task-specific fine-tuning |
| 🟢 **RAG + Fine-Tuned (full system)** | **~65%** | **~44%** | **~38%** | **Best — retrieval + fine-tuning combined** |

*RAG + Fine-Tuning outperforms the base model by ~20 percentage points on Exact Match consistent with published BioASQ 2025 benchmark results showing hybrid retrieval systems outperform standalone LLMs on biomedical QA.*

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUESTION                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   HYBRID RETRIEVER    │
                    │                       │
                    │  ┌─────────────────┐  │
                    │  │ FAISS Index     │  │  ← Semantic / meaning-based search
                    │  │ (PubMedBERT     │  │    "finds conceptually similar papers"
                    │  │  embeddings)    │  │
                    │  └────────┬────────┘  │
                    │           │ combined  │
                    │  ┌────────▼────────┐  │
                    │  │ BM25 Index      │  │  ← Keyword-based search
                    │  │ (exact terms)   │  │    "finds exact drug/gene names"
                    │  └─────────────────┘  │
                    └───────────┬───────────┘
                                │ Top-3 PubMed abstracts
                    ┌───────────▼───────────┐
                    │  LoRA FINE-TUNED      │
                    │  BioGPT               │  ← microsoft/biogpt + LoRA adapter
                    │  (RAG prompt)         │    trained on PubMedQA + MedQA
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  ANSWER + CITATIONS   │  ← Grounded, verifiable response
                    └───────────────────────┘
```

---

##  Technical Stack

| Layer | Tool | Why This Choice |
|---|---|---|
| **Datasets** | HuggingFace Datasets | PubMedQA (211k Q&A pairs), MedQA (12.7k USMLE questions) |
| **Biomedical Embeddings** | `neuml/pubmedbert-base-embeddings` | Pre-trained on PubMed — outperforms general BERT on biomedical retrieval |
| **Dense Search** | FAISS `IndexFlatIP` | Industry-standard vector search; millisecond lookup across 5k+ passages |
| **Keyword Search** | BM25 (`rank_bm25`) | Catches exact gene/drug names that semantic search misses |
| **Base LLM** | `microsoft/biogpt` | Pre-trained on 15M PubMed abstracts - biomedical domain knowledge built-in |
| **Fine-Tuning** | PEFT + LoRA (r=4) | Trains only ~1% of parameters; saves a 10MB adapter instead of a 1.5GB model |
| **Training Hardware** | Apple MPS (M-series GPU) | PyTorch MPS backend — no cloud GPU needed for this scale |
| **Demo UI** | Gradio + HuggingFace Spaces | Live public URL; zero-install for reviewers |

---

## Key Technical Decisions — The "Why" Behind Each Choice

**Why RAG instead of just fine-tuning?**
Fine-tuning bakes knowledge into model weights at training time it goes stale and can't cite sources. RAG dynamically retrieves current evidence at inference time, making the system auditable and updatable without retraining. Critical for FDA-compliant clinical AI.

**Why LoRA instead of full fine-tuning?**
Full fine-tuning BioGPT (~347M params) requires 16GB+ GPU memory and hours of compute. LoRA adds tiny low-rank matrices (rank=4) to just the attention layers (`q_proj`, `v_proj`), training ~1% of parameters with equivalent task performance. The saved adapter is 10MB vs 1.5GB practical for deployment.

**Why hybrid BM25 + FAISS instead of just one?**
Biomedical text has exact terminology (gene symbols, drug names, p-values) that semantic embeddings blur together. BM25 catches these exact matches; FAISS catches conceptual similarity. Combined, they outperform either alone on BioASQ benchmarks.

**Why PubMedBERT for embeddings instead of general sentence-BERT?**
Domain-specific pre-training matters. PubMedBERT was trained entirely on PubMed and produces embeddings that cluster biomedical concepts more accurately than models trained on general web text.

---

## Run It Yourself

```bash
# Clone and set up
git clone https://github.com/beingniki/biomedical-llm-qa-project
cd biomedical-llm-qa-project
python -m venv .venv && source .venv/bin/activate

# Install all dependencies
pip install datasets pandas sentence-transformers faiss-cpu rank_bm25 \
            transformers peft accelerate sacremoses gradio

# Run all 5 phases in order
python phase1_data_collection.py    # ~5 min  — downloads + unifies datasets
python phase2_rag_pipeline.py       # ~10 min — builds FAISS + BM25 index
python phase3_lora_finetune.py      # ~30 min — LoRA fine-tunes BioGPT
python phase4_evaluation.py         # ~10 min — ablation evaluation
python phase5_demo_app.py           # launch  — open http://localhost:7860
```

---

## Repository Structure

```
biomedical-llm-qa-project/
│
├── phase1_data_collection.py    # Dataset download, cleaning & unification
├── phase2_rag_pipeline.py       # PubMedBERT embeddings, FAISS + BM25 indexing
├── phase3_lora_finetune.py      # LoRA fine-tuning with PEFT
├── phase4_evaluation.py         # Ablation: base vs fine-tuned vs RAG+fine-tuned
├── phase5_demo_app.py           # Gradio demo app
│
├── data/
│   ├── processed/               # train.json, eval.json  (Phase 1 output)
│   └── index/                   # faiss.index, bm25.pkl, corpus.json  (Phase 2 output)
│
├── models/
│   └── biogpt-lora-biomedqa/    # LoRA adapter weights  (Phase 3 output)
│
├── eval_results.json            # Evaluation scores across all 3 pipelines
└── README.md
```

---

## Datasets

| Dataset | Size | Format | Used For |
|---|---|---|---|
| [PubMedQA](https://pubmedqa.github.io/) | 211k artificial + 1k labeled | Yes/No/Maybe + abstract | Training + evaluation |
| [MedQA (USMLE)](https://github.com/jind11/MedQA) | 12.7k questions | 4-option multiple choice | Training diversity |
| [BioASQ](http://bioasq.org/) | Expert-curated | Factoid/list + gold snippets | Retriever evaluation |

---

## Industry Context

This project was built in 2026, directly aligned with the current state of biomedical AI:

- **FDA Draft Guidance (Jan 2025)** on AI/ML in drug development emphasises evidence-grounded, auditable AI exactly what RAG enables
- **FDA-EMA Joint Principles (Jan 2026)** on AI in medicines regulation reinforce the need for systems that can cite and verify their sources
- **Pharma adoption** — Moderna, AstraZeneca, and Roche are actively deploying RAG-based internal literature review systems for clinical trial design and regulatory submissions

---

## About the Author

**Nikita Patil**
MSc Bioinformatics AI/ML | Bioinformatics AI/ML Intern @ AskBio (Bayer AG subsidiary, Edinburgh)

Working at the intersection of bioinformatics, AI/ML, and gene therapy. This project was built in parallel with an industry internship focused on AAV capsid library modelling — demonstrating the ability to deliver both applied research and independent portfolio work simultaneously.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/nikita-patil-275a2024b/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-FF6B6B?style=flat)](https://huggingface.co/beingniki)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat&logo=gmail)](mailto:nikitapatil.work.uk@gmail.com)

---

<div align="center">

*Built with 🧬 curiosity, ☕ chai, and a lot of debugging*

**⭐ Star this repo if it helped you understand RAG + LoRA fine-tuning**

</div>
