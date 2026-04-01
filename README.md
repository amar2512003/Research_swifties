# MMLU Contamination Detection in LLMs

## Overview

This project studies **benchmark contamination in Large Language Models (LLMs)** using the MMLU dataset.

We test whether a model performs well because it **understands the task** or has **memorized training data**.

We use:

- Controlled fine-tuning (Member 1)
- Prefix completion detection (Member 2)

---

## Dataset

We use 600 MMLU examples split into:

| Type        | Size | Description                     |
| ----------- | ---- | ------------------------------- |
| Verbatim    | 200  | Exact benchmark questions       |
| Paraphrased | 200  | Same meaning, different wording |
| Clean       | 200  | Unseen questions                |

---

## Models

Base model: microsoft/phi-2

Fine-tuned using LoRA:

| Model   | Training Data |
| ------- | ------------- |
| Model_V | Verbatim      |
| Model_P | Paraphrased   |

---

## Method 1 — Accuracy (Member 1)

CODE EXECUTION ORDER:

1. RUN PYTHON FILES UNDER INITIAL_SCRIPTS TO GENERATE THE VERBATIM, CLEAN, ORIGINAL DATA (200 QS EACH) AND COMBINE TO CREATE SAMPLE DATA(600 QS)
2. RUN PARAPHRASING.IPYNB ON ORIGINAL_DATA.CSV AND CREATE PARAPHRASED DATA FILE.
3. RUN MODEL-FINETUNING.IPYNB TO TRAIN 2 BASE MODELS, ONE ON VERBATIM DATA AND ONE ON PARAPHRASED DATA.
4. MEMBER 2 CODES CAN BE RUN IN ANY ORDER I THINK
5. IGNORE MMLU_CONTAMINATION_PROJECT FOLDER

Metric:
Contamination Gap = Verbatim Accuracy − Clean Accuracy

### Results

| Dataset     | Accuracy |
| ----------- | -------- |
| Verbatim    | 0.217    |
| Paraphrased | 0.217    |
| Clean       | 0.217    |

**Gap = 0.0 → No detection**

---

## Method 2 — Prefix Completion (Member 2)

### Idea

If a model has memorized data, it can **reconstruct the rest of a question from a prefix**.

### Steps

- Use 25%, 50%, 75% prefixes
- Generate completion
- Compare with ground truth using:
  - ROUGE-L
  - Edit distance

---

## Results

| Condition           | AUC   | TPR   | FPR   |
| ------------------- | ----- | ----- | ----- |
| Model_V Verbatim    | 0.707 | 0.695 | 0.365 |
| Model_V Paraphrased | 0.525 | 0.475 | 0.375 |
| Model_P Verbatim    | 0.464 | 0.030 | 0.005 |
| Model_P Paraphrased | 0.801 | 0.725 | 0.205 |

---

## Key Findings

- Prefix completion works for **verbatim contamination**
- Performance drops for **paraphrased contamination**
- Complete failure in: Model_P → Verbatim (TPR = 0.03)

### Insight

> Detection depends heavily on **surface-form (wording)**, not just meaning

---

## Conclusion

- Accuracy alone cannot detect contamination
- Prefix completion reveals **hidden memorization patterns**
- Detection methods are **not robust to paraphrasing**

---

## License

Research / educational use
