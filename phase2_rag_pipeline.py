"""
Phase 2: RAG Pipeline — Vector Store + Hybrid Retriever
=========================================================

Goal: turn the PubMed abstracts collected in Phase 1 into a searchable
knowledge base. We build two indexes — a dense (semantic) one and a
keyword (BM25) one — and combine them into a hybrid retriever.

    pip install sentence-transformers faiss-cpu rank_bm25
"""

import json
import os
import pickle
import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# A laptop CPU can take many hours to embed 200k+ passages. For a
# portfolio demo, a few thousand well-chosen passages is plenty to
# show a working RAG pipeline. Raise this once you move to a GPU
# (Colab/Kaggle) for the full-scale version.
MAX_CORPUS_SIZE = 5000

PROCESSED_DIR = "data/processed"
INDEX_DIR = "data/index"
os.makedirs(INDEX_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 1. Build the corpus from Phase 1 outputs
# ---------------------------------------------------------------------
print("Loading Phase 1 data...")
with open(os.path.join(PROCESSED_DIR, "train.json")) as f:
    train_data = json.load(f)
with open(os.path.join(PROCESSED_DIR, "eval.json")) as f:
    eval_data = json.load(f)

all_records = train_data + eval_data

# Many records share the same abstract — dedupe so we don't index it twice
corpus = []
seen = set()
for r in all_records:
    ctx = r.get("context", "").strip()
    if ctx and ctx not in seen:
        seen.add(ctx)
        corpus.append({"id": r["id"], "text": ctx, "source": r["source"]})

print(f"Found {len(corpus)} unique passages")

if len(corpus) > MAX_CORPUS_SIZE:
    random.seed(42)
    corpus = random.sample(corpus, MAX_CORPUS_SIZE)
    print(f"Sampled down to {len(corpus)} passages for this run "
          f"(adjust MAX_CORPUS_SIZE at the top of the script)")

with open(os.path.join(INDEX_DIR, "corpus.json"), "w") as f:
    json.dump(corpus, f, indent=2)


# ---------------------------------------------------------------------
# 2. Dense embeddings with a biomedical sentence encoder
# ---------------------------------------------------------------------
print("Loading biomedical embedding model...")
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")
embedder = SentenceTransformer("neuml/pubmedbert-base-embeddings", device=device)

texts = [c["text"] for c in corpus]
print("Encoding corpus...")
embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
faiss.normalize_L2(embeddings)  # so inner product = cosine similarity


# ---------------------------------------------------------------------
# 3. Build the dense (FAISS) index
# ---------------------------------------------------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
print(f"Saved FAISS index with {index.ntotal} vectors (dim={dim})")


# ---------------------------------------------------------------------
# 4. Build the keyword (BM25) index
# ---------------------------------------------------------------------
print("Building BM25 index...")
tokenized_corpus = [text.lower().split() for text in texts]
bm25 = BM25Okapi(tokenized_corpus)
with open(os.path.join(INDEX_DIR, "bm25.pkl"), "wb") as f:
    pickle.dump(bm25, f)


# ---------------------------------------------------------------------
# 5. Hybrid retriever — combines dense + keyword scores
# ---------------------------------------------------------------------
def hybrid_search(query, top_k=5, alpha=0.5):
    """alpha = 1.0 -> pure semantic search, alpha = 0.0 -> pure keyword search"""
    # Dense similarity scores
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    dense_scores, dense_idx = index.search(q_emb, len(corpus))
    dense_scores, dense_idx = dense_scores[0], dense_idx[0]
    d_range = dense_scores.max() - dense_scores.min() + 1e-9
    dense_norm = (dense_scores - dense_scores.min()) / d_range
    dense_map = dict(zip(dense_idx, dense_norm))

    # Keyword (BM25) scores
    bm25_scores = bm25.get_scores(query.lower().split())
    b_range = bm25_scores.max() - bm25_scores.min() + 1e-9
    bm25_norm = (bm25_scores - bm25_scores.min()) / b_range

    # Weighted combination
    combined = [
        (i, alpha * dense_map.get(i, 0) + (1 - alpha) * bm25_norm[i])
        for i in range(len(corpus))
    ]
    combined.sort(key=lambda x: x[1], reverse=True)

    return [
        {"text": corpus[i]["text"], "score": float(score), "source": corpus[i]["source"]}
        for i, score in combined[:top_k]
    ]


# ---------------------------------------------------------------------
# 6. Quick test
# ---------------------------------------------------------------------
print("=" * 60)
test_question = eval_data[0]["question"]
print(f"Test question: {test_question}\n")

for i, r in enumerate(hybrid_search(test_question, top_k=3), 1):
    print(f"[{i}] score={r['score']:.3f}  source={r['source']}")
    print(r["text"][:300] + "...\n")

print("Phase 2 indexing complete. Saved to data/index/.")
print("Next: wire hybrid_search() into a prompt for your LLM (RAG generation step).")
