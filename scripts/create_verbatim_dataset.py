import json
import random
from datasets import load_dataset

# -----------------------------
# SETTINGS
# -----------------------------
MIN_WORDS = 10
MAX_WORDS = 30
MAX_CHARS = 180
SAMPLE_SIZE = 200
SEED = 99   

random.seed(SEED)

# -----------------------------
# LOAD MMLU
# -----------------------------
dataset = load_dataset("cais/mmlu", "all", split="test")

print("Total MMLU questions:", len(dataset))

# -----------------------------
# FILTER
# -----------------------------
def valid_length(question):
    words = len(question.split())
    chars = len(question)
    return (
        MIN_WORDS <= words <= MAX_WORDS and
        chars <= MAX_CHARS
    )

filtered = [
    {
        "id": i,
        "subject": ex["subject"],
        "question": ex["question"],
        "choices": ex["choices"],
        "answer": ex["answer"]
    }
    for i, ex in enumerate(dataset)
    if valid_length(ex["question"])
]

print("After filtering:", len(filtered))

if len(filtered) < SAMPLE_SIZE:
    raise ValueError("Not enough valid questions.")

# -----------------------------
# SAMPLE DIFFERENT 200
# -----------------------------
random.shuffle(filtered)
verbatim_data = filtered[:SAMPLE_SIZE]

# -----------------------------
# SAVE
# -----------------------------
with open("data/processed/verbatim_data.json", "w") as f:
    json.dump(verbatim_data, f, indent=2)

print("verbatim_data.json created with", len(verbatim_data), "questions.")