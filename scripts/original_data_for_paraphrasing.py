from datasets import load_dataset
import json
import random
import re
from pathlib import Path
from collections import defaultdict

# -------------------
# CONFIG
# -------------------
SEED = 42
TARGET_SAMPLES = 200
OUTPUT_PATH = "data/paraphrases/original_data.json"

random.seed(SEED)

MIN_WORD_COUNT = 10
MAX_WORD_COUNT = 30
MAX_QUESTION_LENGTH = 180

# -------------------
# FILTER FUNCTION
# -------------------
def is_pure_mcq(example):
    question = example["question"].strip()
    word_count = len(question.split())

    # 0️⃣ Enforce minimum word limit
    if word_count < MIN_WORD_COUNT:
        return False

    # 1️⃣ Remove fill-in-the-blank
    if "____" in question or "blank" in question.lower():
        return False

    # 2️⃣ Remove heavy numeric problems
    math_patterns = [
        r"\d+\s*[\+\-\*\/]\s*\d+",
        r"\bsolve\b",
        r"\bcalculate\b",
        r"\bfind the value\b",
        r"\bwhat is the value\b",
        r"\bcompute\b"
    ]
    for pattern in math_patterns:
        if re.search(pattern, question.lower()):
            return False

    # 3️⃣ Remove very long questions
    if len(question) > MAX_QUESTION_LENGTH:
        return False

    if word_count > MAX_WORD_COUNT:
        return False

    # 4️⃣ Ensure proper MCQ
    if not example.get("choices") or len(example["choices"]) != 4:
        return False

    return True


# -------------------
# LOAD DATASET
# -------------------
dataset = load_dataset("cais/mmlu", "all", split="test")

# -------------------
# GROUP BY SUBJECT
# -------------------
subject_questions = defaultdict(list)

for idx, ex in enumerate(dataset):
    if is_pure_mcq(ex):
        subject = ex["subject"]
        item = {
            "id": f"mmlu_{idx}",
            "subject": subject,
            "question": ex["question"],
            "choices": ex["choices"],
            "answer": int(ex["answer"])
        }
        subject_questions[subject].append(item)

all_subjects = list(subject_questions.keys())
random.shuffle(all_subjects)

print(f"Total subjects after filtering: {len(all_subjects)}")

selected = []

# -------------------
# TAKE 1 PER SUBJECT
# -------------------
for subject in all_subjects:
    if len(selected) >= TARGET_SAMPLES:
        break
    if subject_questions[subject]:
        selected.append(random.choice(subject_questions[subject]))

# -------------------
# IF LESS THAN TARGET, FILL REMAINING
# -------------------
if len(selected) < TARGET_SAMPLES:
    print("Not enough subjects with valid questions. Filling remaining randomly...")

    remaining_pool = []
    for questions in subject_questions.values():
        remaining_pool.extend(questions)

    selected_ids = set(q["id"] for q in selected)
    remaining_pool = [q for q in remaining_pool if q["id"] not in selected_ids]

    needed = TARGET_SAMPLES - len(selected)
    selected.extend(random.sample(remaining_pool, needed))

# -------------------
# SAVE FILE
# -------------------
Path("data/raw").mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(selected)} questions.")