
# MMLU Contamination Detection with LoRA Fine-Tuning

## Overview
This project investigates **training data contamination in Large Language Models (LLMs)** using the **MMLU benchmark**.

The goal is to determine whether a model performs better on benchmark questions because it has **memorized them during training** rather than truly understanding the task.

We fine-tune **Microsoft Phi-2** using **LoRA (Low-Rank Adaptation)** and evaluate the model on three dataset splits:

- Verbatim – original benchmark questions
- Paraphrased – reworded versions of the same questions
- Clean – unseen questions not used during training

The difference between performance on verbatim and clean datasets is used to measure the **contamination gap**.

---

## Project Structure

mmlu_contamination_project/

│

├── data/
│   ├── sampled/
│   │   └── mmlu_sampled_600.json
│   │
│   ├── paraphrased/
│   │   └── mmlu_paraphrased_final.json
│   │
│   └── final_splits/
│       ├── mmlu_verbatim.json
│       └── mmlu_clean.json
│
├── models/
│   ├── model_clean/
│   ├── model_paraphrased/
│   └── model_verbatim/
│
├── notebooks/
│   └── member1_Finetune.ipynb
│
├── results/
│
├── scripts/
│
└── README.md

---

## Methodology

### Dataset Preparation
- Sampled MMLU benchmark questions
- Generated paraphrased variants
- Created three evaluation splits

### Model
Base Model:
microsoft/phi-2

Fine-tuning Method:
LoRA (Low-Rank Adaptation)

Libraries used:
- transformers
- peft
- datasets
- accelerate

---

## Evaluation

Metric used:

Contamination Gap = Verbatim Accuracy - Clean Accuracy

Interpretation:

High gap → possible memorization  
Low / zero gap → no clear contamination

---

## Results

Verbatim Accuracy: 0.217  
Paraphrased Accuracy: 0.217  
Clean Accuracy: 0.217  

Contamination Gap: **0.0**

This suggests **no detectable benchmark contamination in the current setup**.

---

## Requirements

Install dependencies:

pip install transformers peft datasets accelerate torch pandas matplotlib

---

## How to Run

Open the notebook:

notebooks/member1_Finetune.ipynb

Run all cells sequentially to:
1. Prepare dataset
2. Fine-tune model
3. Evaluate contamination gap

---

## Author

Amar Sinha  


---

## License

Educational / research use.
