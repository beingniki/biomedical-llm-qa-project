"""
Phase 5: Gradio Demo App — Biomedical LLM Q&A
===============================================

Goal: wrap the full RAG + fine-tuned pipeline into a clean browser UI
that anyone can use without touching code. Deploy it free on HuggingFace
Spaces so you have a live public URL for your resume.

    pip install gradio
"""

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import json
import pickle
import torch
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------------------
# Paths — all relative to the project folder
# ---------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR    = os.path.join(BASE_DIR, "data/index")
ADAPTER_DIR  = os.path.join(BASE_DIR, "models/biogpt-lora-biomedqa")
BASE_MODEL   = "microsoft/biogpt"

# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
if torch.backends.mps.is_available():
    try:
        torch.zeros(1).to("mps")
        device = "mps"
    except Exception:
        device = "cpu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------------------
# 1. Load retrieval index (Phase 2)
# ---------------------------------------------------------------------
print("Loading retrieval index...")
with open(os.path.join(INDEX_DIR, "corpus.json")) as f:
    corpus = json.load(f)
faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
with open(os.path.join(INDEX_DIR, "bm25.pkl"), "rb") as f:
    bm25 = pickle.load(f)
embedder = SentenceTransformer("neuml/pubmedbert-base-embeddings", device=device)

# ---------------------------------------------------------------------
# 2. Load fine-tuned model (Phase 3)
# ---------------------------------------------------------------------
print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
base_model.config.pad_token_id = tokenizer.pad_token_id
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.to(device).eval()
print("Model ready.")

# ---------------------------------------------------------------------
# 3. Retrieval function
# ---------------------------------------------------------------------
def hybrid_search(query, top_k=3, alpha=0.5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    dense_scores, dense_idx = faiss_index.search(q_emb, len(corpus))
    dense_scores, dense_idx = dense_scores[0], dense_idx[0]
    d_range = dense_scores.max() - dense_scores.min() + 1e-9
    dense_norm = (dense_scores - dense_scores.min()) / d_range
    dense_map = dict(zip(dense_idx.tolist(), dense_norm.tolist()))

    bm25_scores = bm25.get_scores(query.lower().split())
    b_range = bm25_scores.max() - bm25_scores.min() + 1e-9
    bm25_norm = (bm25_scores - bm25_scores.min()) / b_range

    combined = [
        (i, alpha * dense_map.get(i, 0) + (1 - alpha) * float(bm25_norm[i]))
        for i in range(len(corpus))
    ]
    combined.sort(key=lambda x: x[1], reverse=True)
    return [(corpus[i]["text"], round(score, 3)) for i, score in combined[:top_k]]

# ---------------------------------------------------------------------
# 4. Answer generation function
# ---------------------------------------------------------------------
def generate_answer(question, retrieved_texts):
    context = " ".join(t for t, _ in retrieved_texts)[:600]
    prompt = f"Context: {context} Q: {question} A:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

# ---------------------------------------------------------------------
# 5. Main pipeline — called by Gradio on every submission
# ---------------------------------------------------------------------
def run_pipeline(question):
    if not question.strip():
        return "Please enter a question.", "", "", ""

    retrieved = hybrid_search(question, top_k=3)
    answer    = generate_answer(question, retrieved)

    sources = ""
    for i, (text, score) in enumerate(retrieved, 1):
        sources += f"[{i}] Relevance score: {score}\n{text[:400]}...\n\n"

    return answer, sources

# ---------------------------------------------------------------------
# 6. Gradio UI
# ---------------------------------------------------------------------
EXAMPLES = [
    "Do statins reduce the risk of cardiovascular events?",
    "Is metformin effective for type 2 diabetes treatment?",
    "Does vitamin D deficiency increase cancer risk?",
    "Are beta-blockers effective after myocardial infarction?",
    "Does aspirin reduce the risk of colorectal cancer?",
]

with gr.Blocks(title="Biomedical LLM Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧬 Biomedical Literature Q&A
    **RAG + LoRA fine-tuned BioGPT** | Answers grounded in real PubMed abstracts
    
    Ask a biomedical yes/no question and the system will:
    1. Retrieve the most relevant PubMed abstracts from the index
    2. Generate an evidence-based answer using a fine-tuned BioGPT model
    """)

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Your biomedical question",
                placeholder="e.g. Do statins reduce cardiovascular risk?",
                lines=2,
            )
            submit_btn = gr.Button("Get Answer", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=question_box)

        with gr.Column(scale=3):
            answer_box = gr.Textbox(label="Generated Answer", lines=4, interactive=False)
            sources_box = gr.Textbox(label="Retrieved PubMed Sources", lines=12, interactive=False)

    gr.Markdown("""
    ---
    **Model:** microsoft/biogpt fine-tuned with LoRA (PEFT, r=4) on PubMedQA + MedQA  
    **Retriever:** Hybrid BM25 + FAISS semantic search over PubMed abstracts  
    **Built by:** Nikita Patil | Bioinformatics & AI/ML Portfolio Project
    """)

    submit_btn.click(
        fn=run_pipeline,
        inputs=question_box,
        outputs=[answer_box, sources_box],
    )

# ---------------------------------------------------------------------
# 7. Launch
# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=False)   # set share=True to get a temporary public URL
