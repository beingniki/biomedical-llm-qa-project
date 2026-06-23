"""
Phase 3: LoRA Fine-Tuning — BioGPT on Biomedical Q&A
======================================================

Goal: fine-tune BioGPT with LoRA (a parameter-efficient method) on the
unified Q&A data from Phase 1, so the model learns to generate direct
answers to biomedical questions instead of just retrieving text.

    pip install transformers peft accelerate
"""

# Fix MPS memory watermark bug on macOS — must be set before importing torch
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "models/biogpt-lora-biomedqa"
MODEL_NAME = "microsoft/biogpt"

# Kept at 1000 for a stable, fast first run on a MacBook.
# Raise to 3000+ once you confirm it completes successfully.
MAX_TRAIN_EXAMPLES = 1000


# ---------------------------------------------------------------------
# 1. Load Phase 1 data and format as prompt -> answer text
# ---------------------------------------------------------------------
print("Loading Phase 1 data...")
with open(os.path.join(PROCESSED_DIR, "train.json")) as f:
    train_data = json.load(f)


def format_example(r):
    """Turn a unified record into one short training string."""
    question = r["question"]
    if r["source"] == "medqa" and r.get("choices"):
        options = " | ".join(f"{k}: {v}" for k, v in r["choices"].items())
        return f"Q: {question} Options: {options} A: {r['answer']}"
    answer = r.get("long_answer") or r.get("answer", "")
    # Truncate context to 200 chars to keep sequences short on CPU
    context = r.get("context", "")[:200]
    return f"Context: {context} Q: {question} A: {answer}"


texts = [format_example(r) for r in train_data if r.get("answer") or r.get("long_answer")]
texts = texts[:MAX_TRAIN_EXAMPLES]
print(f"Using {len(texts)} training examples")


# ---------------------------------------------------------------------
# 2. Tokenizer + dataset
# ---------------------------------------------------------------------
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(batch):
    # max_length=256 (down from 512) halves memory use and speeds up training
    return tokenizer(batch["text"], truncation=True, max_length=256, padding="max_length")


dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])


# ---------------------------------------------------------------------
# 3. Device selection — safe MPS, falls back to CPU if MPS is unstable
# ---------------------------------------------------------------------
if torch.backends.mps.is_available():
    try:
        # Quick sanity check — if MPS can allocate a small tensor, use it
        _ = torch.zeros(1).to("mps")
        device = "mps"
    except Exception:
        device = "cpu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# 4. Load base model + apply LoRA
# ---------------------------------------------------------------------
print(f"Loading base model {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # float32 is safest on CPU/MPS
)
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

lora_config = LoraConfig(
    r=4,                   # rank 4 (down from 8) — fewer params, faster, more stable
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ---------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,   # small batch = less memory pressure
    gradient_accumulation_steps=4,  # effective batch size = 2 * 4 = 8
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    use_cpu=(device == "cpu"),       # tells Trainer not to try moving to GPU
    fp16=False,                      # keep off — MPS and CPU don't support fp16 well
    bf16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting fine-tuning — estimated 20-40 min on MacBook Air...")
trainer.train()


# ---------------------------------------------------------------------
# 6. Save the LoRA adapter
# ---------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nSaved LoRA adapter to {OUTPUT_DIR}")


# ---------------------------------------------------------------------
# 7. Quick generation test
# ---------------------------------------------------------------------
print("=" * 60)
sample = train_data[0]
context = sample.get("context", "")[:200]
prompt = f"Context: {context} Q: {sample['question']} A:"

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

print("Question:", sample["question"])
print("Generated answer:")
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
print("\nPhase 3 complete. Run phase4_evaluation.py next.")