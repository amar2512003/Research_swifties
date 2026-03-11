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
SEED = 70

random.seed(SEED)

# -----------------------------
# LOAD MMLU (all subjects, test split)
# -----------------------------
dataset = load_dataset("cais/mmlu", "all", split="test")

print("Total MMLU questions:", len(dataset))

# -----------------------------
# FILTER BY LENGTH
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
# SAMPLE 200
# -----------------------------
random.shuffle(filtered)
clean_data = filtered[:SAMPLE_SIZE]

# -----------------------------
# SAVE
# -----------------------------
with open("data/raw/clean_data.json", "w") as f:
    json.dump(clean_data, f, indent=2)

print("clean_data.json created with", len(clean_data), "questions.")