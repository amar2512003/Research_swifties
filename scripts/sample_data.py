import json

# -------------------------
# LOAD FILES
# -------------------------
with open("data/paraphrases/original_data.json",encoding="utf-8") as f:
    original_data = json.load(f)

with open("data/verbatim/verbatim_data.json",encoding="utf-8") as f:
    verbatim_data = json.load(f)

with open("data/clean/clean_data.json",encoding="utf-8") as f:
    clean_data = json.load(f)

# -------------------------
# ADD SPLIT TYPE LABELS
# -------------------------
def add_split_type(data, label):
    new_data = []
    for ex in data:
        new_data.append({
            "subject": ex["subject"],
            "question": ex["question"],
            "choices": ex["choices"],
            "answer": ex["answer"],
            "split_type": label
        })
    return new_data

original_labeled = add_split_type(original_data, "paraphrase")
clean_labeled = add_split_type(clean_data, "clean")
verbatim_labeled = add_split_type(verbatim_data, "verbatim")

# -------------------------
# COMBINE ALL
# -------------------------
sampledata = original_labeled + clean_labeled + verbatim_labeled

# -------------------------
# SAVE
# -------------------------
with open("data/sample/sample_data.json", "w") as f:
    json.dump(sampledata, f, indent=4)

print("sample_data.json created successfully.")
print("Total questions:", len(sampledata))