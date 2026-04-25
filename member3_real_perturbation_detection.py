#!/usr/bin/env python3
"""
Member 3: Perturbation Sensitivity Detection + Ensemble

This script implements the Member 3 pipeline for the contamination project:
1. Load Model_V / Model_P and the three dataset splits.
2. Generate four perturbation types per example.
3. Score contamination using perturbation sensitivity.
4. Evaluate each contamination condition against the clean split.
5. Build a held-out ensemble with Member 2's prefix-completion scores.
6. Save metrics, confidence intervals, significance tests, and failure cases.

Notes
-----
- The evaluation design matches the paper setup: each contaminated split is
  compared against the clean split for the same model.
- Ensemble weights are selected on a held-out 50% validation split and reported
  on the remaining 50% test split.
- If Phi-2 is not available locally, pass --mock to exercise the pipeline with
  synthetic perturbation scores only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_FILES = {
    "verbatim": ROOT / "data" / "verbatim" / "verbatim_data.json",
    "paraphrased": ROOT / "data" / "paraphrases" / "original_data.json",
    "clean": ROOT / "data" / "clean" / "clean_data.json",
}
MODEL_DIRS = {
    "model_v": ROOT / "models" / "Model_V",
    "model_p": ROOT / "models" / "Model_P",
}
PREFIX_FILES = {
    "model_v_verbatim": ROOT / "prefix_scores" / "prefix_scores_model_v_verbatim.csv",
    "model_v_paraphrased": ROOT / "prefix_scores" / "prefix_scores_model_v_paraphrase.csv",
    "model_v_clean": ROOT / "prefix_scores" / "prefix_scores_model_v_clean.csv",
    "model_p_verbatim": ROOT / "prefix_scores" / "prefix_scores_model_p_verbatim.csv",
    "model_p_paraphrased": ROOT / "prefix_scores" / "prefix_scores_model_p_paraphrase.csv",
    "model_p_clean": ROOT / "prefix_scores" / "prefix_scores_model_p_clean.csv",
}
RESULTS_DIR = ROOT / "member3_results"
RESULTS_DIR.mkdir(exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    low = np.min(arr)
    high = np.max(arr)
    if np.isclose(high, low):
        return np.zeros_like(arr)
    return (arr - low) / (high - low)


def percentile_ci(values: Sequence[float], low: float = 2.5, high: float = 97.5) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"low": 0.0, "high": 0.0}
    return {"low": float(np.percentile(arr, low)), "high": float(np.percentile(arr, high))}


def bootstrap_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
) -> Dict[str, Dict[str, float]]:
    aucs: List[float] = []
    tprs: List[float] = []
    fprs: List[float] = []

    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = scores[idx]

        if len(np.unique(y_b)) < 2:
            continue

        aucs.append(float(roc_auc_score(y_b, s_b)))

        preds = (s_b >= threshold).astype(int)
        pos = y_b == 1
        neg = y_b == 0
        tprs.append(float(preds[pos].mean()) if np.any(pos) else 0.0)
        fprs.append(float(preds[neg].mean()) if np.any(neg) else 0.0)

    return {
        "auc_ci": percentile_ci(aucs),
        "tpr_ci": percentile_ci(tprs),
        "fpr_ci": percentile_ci(fprs),
    }


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return (
        float(roc_auc_score(y_true, scores)),
        float(thresholds[idx]),
        float(tpr[idx]),
        float(fpr[idx]),
    )


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    preds = (scores >= threshold).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    tpr = float(preds[pos].mean()) if np.any(pos) else 0.0
    fpr = float(preds[neg].mean()) if np.any(neg) else 0.0
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0
    return auc, tpr, fpr


def best_threshold_under_fpr_budget(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
) -> Tuple[float, float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = float(roc_auc_score(y_true, scores))

    valid_idx = [idx for idx, current_fpr in enumerate(fpr) if current_fpr <= target_fpr + 1e-12]
    if valid_idx:
        best_idx = max(valid_idx, key=lambda idx: (tpr[idx], -fpr[idx]))
        return auc, float(thresholds[best_idx]), float(tpr[best_idx]), float(fpr[best_idx])

    best_idx = int(np.argmin(fpr))
    return auc, float(thresholds[best_idx]), float(tpr[best_idx]), float(fpr[best_idx])


@dataclass
class ExampleResult:
    example_id: str
    subject: str
    question: str
    original_prediction: Optional[str]
    original_correct: int
    original_correct_prob: float
    perturbed_correct_avg: float
    perturbed_correct_prob_avg: float
    sensitivity: float
    perturbation_details: List[Dict[str, object]]


class PerturbationGenerator:
    def __init__(self, seed: int = SEED) -> None:
        self.rng = random.Random(seed)
        self.name_replacements = {
            "John": "Michael",
            "Mary": "Sarah",
            "Alice": "Emma",
            "Bob": "David",
            "James": "Robert",
            "William": "Thomas",
            "Elizabeth": "Jennifer",
        }
        self.word_replacements = {
            "what": "which",
            "which": "what",
            "best": "most accurately",
            "following": "below",
            "describes": "characterizes",
            "concept": "idea",
            "example": "instance",
            "reason": "cause",
            "effect": "result",
        }

    def swap_names(self, question: str) -> str:
        words = []
        for token in question.split():
            bare = re.sub(r"[^A-Za-z]", "", token)
            replacement = self.name_replacements.get(bare)
            if replacement:
                words.append(token.replace(bare, replacement))
            else:
                words.append(token)
        return " ".join(words)

    def change_numbers(self, question: str) -> str:
        def modify(match: re.Match[str]) -> str:
            value = int(match.group())
            if value <= 5:
                return str(value + 1)
            if value <= 20:
                return str(value + 3)
            return str(value + max(1, int(round(value * 0.1))))

        return re.sub(r"\b\d+\b", modify, question, count=1)

    def minor_paraphrase(self, question: str) -> str:
        tokens = []
        for token in question.split():
            bare = re.sub(r"[^A-Za-z]", "", token)
            replacement = self.word_replacements.get(bare.lower())
            if replacement:
                if bare[:1].isupper():
                    replacement = replacement.capitalize()
                tokens.append(token.replace(bare, replacement))
            else:
                tokens.append(token)
        return " ".join(tokens)

    def shuffle_options(self, choices: Sequence[str], answer_idx: int) -> Tuple[List[str], int]:
        order = list(range(len(choices)))
        self.rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        new_answer_idx = order.index(answer_idx)
        return shuffled, new_answer_idx

    def generate_all(self, example: Dict[str, object]) -> List[Dict[str, object]]:
        question = str(example["question"])
        choices = list(example["choices"])
        answer_idx = int(example["answer"])

        shuffled_choices, shuffled_answer = self.shuffle_options(choices, answer_idx)
        return [
            {
                "type": "name_swap",
                "question": self.swap_names(question),
                "choices": choices,
                "answer": answer_idx,
            },
            {
                "type": "number_change",
                "question": self.change_numbers(question),
                "choices": choices,
                "answer": answer_idx,
            },
            {
                "type": "option_shuffle",
                "question": question,
                "choices": shuffled_choices,
                "answer": shuffled_answer,
            },
            {
                "type": "minor_paraphrase",
                "question": self.minor_paraphrase(question),
                "choices": choices,
                "answer": answer_idx,
            },
        ]


class ModelInference:
    def __init__(self, model, tokenizer, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.answer_labels = ["A", "B", "C", "D"]
        self.answer_token_ids = {
            label: self._resolve_answer_token_id(label) for label in self.answer_labels
        }

    def format_prompt(self, question: str, choices: Sequence[str]) -> str:
        lines = [f"Question: {question}", ""]
        for i, choice in enumerate(choices):
            lines.append(f"{chr(65 + i)}) {choice}")
        lines.extend(["", "Answer:"])
        return "\n".join(lines)

    def _resolve_answer_token_id(self, label: str) -> int:
        for candidate in [label, f" {label}"]:
            token_ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if len(token_ids) == 1:
                return int(token_ids[0])
        raise ValueError(f"Could not find single-token encoding for answer label {label}")

    def predict_with_probs(self, question: str, choices: Sequence[str]) -> Tuple[str, Dict[str, float]]:
        prompt = self.format_prompt(question, choices) + " "
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        next_token_logits = outputs.logits[0, -1]
        candidate_logits = torch.stack(
            [next_token_logits[self.answer_token_ids[label]] for label in self.answer_labels]
        )
        candidate_probs = torch.softmax(candidate_logits, dim=0).detach().cpu().numpy()
        prob_map = {
            label: float(candidate_probs[idx]) for idx, label in enumerate(self.answer_labels)
        }
        prediction = max(prob_map, key=prob_map.get)
        return prediction, prob_map


class PerturbationSensitivityDetector:
    def __init__(self, inference: ModelInference) -> None:
        self.inference = inference

    @staticmethod
    def is_correct(prediction: Optional[str], answer_idx: int) -> int:
        if prediction is None:
            return 0
        return int(prediction == chr(65 + answer_idx))

    def score_example(self, example: Dict[str, object], perturbations: List[Dict[str, object]]) -> ExampleResult:
        question = str(example["question"])
        choices = list(example["choices"])
        answer_idx = int(example["answer"])
        correct_label = chr(65 + answer_idx)

        original_prediction, original_probs = self.inference.predict_with_probs(question, choices)
        original_correct = self.is_correct(original_prediction, answer_idx)
        original_correct_prob = float(original_probs[correct_label])

        details: List[Dict[str, object]] = []
        pert_correct: List[int] = []
        pert_correct_probs: List[float] = []
        effective_perturbations = [
            perturbation
            for perturbation in perturbations
            if str(perturbation["question"]) != question or perturbation["choices"] != choices
        ]

        for perturbation in effective_perturbations:
            pred, prob_map = self.inference.predict_with_probs(
                str(perturbation["question"]),
                perturbation["choices"],
            )
            correct = self.is_correct(pred, int(perturbation["answer"]))
            correct_label_pert = chr(65 + int(perturbation["answer"]))
            pert_correct.append(correct)
            pert_correct_probs.append(float(prob_map[correct_label_pert]))
            details.append(
                {
                    "type": perturbation["type"],
                    "prediction": pred,
                    "correct": correct,
                    "correct_answer_prob": float(prob_map[correct_label_pert]),
                    "question_changed": str(perturbation["question"]) != question,
                }
            )

        pert_avg = float(np.mean(pert_correct)) if pert_correct else 0.0
        pert_prob_avg = float(np.mean(pert_correct_probs)) if pert_correct_probs else original_correct_prob

        # Use confidence drop rather than raw binary accuracy drop to reduce ties and noise.
        sensitivity = float(original_correct_prob - pert_prob_avg)
        return ExampleResult(
            example_id=str(example.get("id", "unknown")),
            subject=str(example.get("subject", "unknown")),
            question=question,
            original_prediction=original_prediction,
            original_correct=original_correct,
            original_correct_prob=original_correct_prob,
            perturbed_correct_avg=pert_avg,
            perturbed_correct_prob_avg=pert_prob_avg,
            sensitivity=sensitivity,
            perturbation_details=details,
        )


def load_json(path: Path) -> List[Dict[str, object]]:
    with path.open() as handle:
        return json.load(handle)


def load_datasets() -> Dict[str, List[Dict[str, object]]]:
    datasets = {name: load_json(path) for name, path in DATA_FILES.items()}
    print("Loaded datasets:")
    for name, data in datasets.items():
        print(f"  - {name}: {len(data)} examples")
    return datasets


def load_prefix_scores() -> Dict[str, np.ndarray]:
    scores = {}
    print("Loaded Member 2 prefix scores:")
    for key, path in PREFIX_FILES.items():
        df = pd.read_csv(path)
        scores[key] = df["score"].to_numpy(dtype=float)
        print(f"  - {key}: {len(scores[key])} scores")
    return scores


def load_model_pair(base_model_name: str, device: torch.device):
    print(f"Loading Phi-2 base model from: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.eos_token_id

    models = {}
    for model_key, adapter_dir in MODEL_DIRS.items():
        print(f"  - loading adapter: {adapter_dir}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            config=config,
            trust_remote_code=True,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        base_model.config.pad_token_id = tokenizer.eos_token_id
        base_model.eval()
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model = model.to(device)
        model.eval()
        models[model_key] = model
    return tokenizer, models


def generate_mock_scores(split_name: str, size: int, rng: np.random.Generator) -> np.ndarray:
    if split_name == "verbatim":
        return rng.beta(8, 3, size=size)
    if split_name == "paraphrased":
        return rng.beta(5, 4, size=size)
    return rng.beta(2, 8, size=size)


def compute_sensitivity_scores(
    model_key: str,
    model,
    tokenizer,
    device: torch.device,
    datasets: Dict[str, List[Dict[str, object]]],
    sample_size: Optional[int],
    mock: bool,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[ExampleResult]]]:
    perturbation_gen = PerturbationGenerator(seed=SEED)
    scores: Dict[str, np.ndarray] = {}
    details: Dict[str, List[ExampleResult]] = {}

    for split_name, data in datasets.items():
        active_data = data[:sample_size] if sample_size else data
        print(f"Scoring {model_key} on {split_name} ({len(active_data)} examples)")

        if mock:
            scores[split_name] = generate_mock_scores(split_name, len(active_data), rng)
            details[split_name] = []
            continue

        detector = PerturbationSensitivityDetector(ModelInference(model, tokenizer, device))
        split_results: List[ExampleResult] = []
        for example in active_data:
            perturbations = perturbation_gen.generate_all(example)
            split_results.append(detector.score_example(example, perturbations))

        details[split_name] = split_results
        scores[split_name] = np.asarray([row.sensitivity for row in split_results], dtype=float)

        out_path = RESULTS_DIR / f"{model_key}_{split_name}_perturbation_details.json"
        with out_path.open("w") as handle:
            json.dump([asdict(row) for row in split_results], handle, indent=2)

    return scores, details


def make_stratified_split(labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_cut = len(pos_idx) // 2
    neg_cut = len(neg_idx) // 2

    val_idx = np.concatenate([pos_idx[:pos_cut], neg_idx[:neg_cut]])
    test_idx = np.concatenate([pos_idx[pos_cut:], neg_idx[neg_cut:]])
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return val_idx, test_idx


def optimize_ensemble(
    prefix_scores: np.ndarray,
    perturb_scores: np.ndarray,
    labels: np.ndarray,
    seed: int,
    target_fpr: float,
) -> Dict[str, object]:
    val_idx, test_idx = make_stratified_split(labels, seed)
    p_norm = normalize(prefix_scores)
    s_norm = normalize(perturb_scores)

    best = {
        "w_prefix": 0.5,
        "w_perturb": 0.5,
        "validation_auc": -1.0,
        "validation_tpr": -1.0,
        "validation_fpr": 1.0,
        "threshold": 0.5,
    }
    for w_prefix in np.arange(0.0, 1.01, 0.05):
        w_perturb = 1.0 - w_prefix
        val_scores = w_prefix * p_norm[val_idx] + w_perturb * s_norm[val_idx]
        auc, threshold, val_tpr, val_fpr = best_threshold_under_fpr_budget(
            labels[val_idx],
            val_scores,
            target_fpr=target_fpr,
        )
        candidate = (val_tpr, -val_fpr, auc)
        incumbent = (best["validation_tpr"], -best["validation_fpr"], best["validation_auc"])
        if candidate > incumbent:
            best = {
                "w_prefix": float(round(w_prefix, 2)),
                "w_perturb": float(round(w_perturb, 2)),
                "validation_auc": auc,
                "validation_tpr": val_tpr,
                "validation_fpr": val_fpr,
                "threshold": threshold,
            }

    test_scores = best["w_prefix"] * p_norm[test_idx] + best["w_perturb"] * s_norm[test_idx]
    test_auc, test_tpr, test_fpr = metrics_at_threshold(
        labels[test_idx],
        test_scores,
        best["threshold"],
    )
    return {
        "validation_idx": val_idx,
        "test_idx": test_idx,
        "test_scores": test_scores,
        "test_labels": labels[test_idx],
        "weights": {"prefix": best["w_prefix"], "perturbation": best["w_perturb"]},
        "validation_auc": best["validation_auc"],
        "validation_tpr": best["validation_tpr"],
        "validation_fpr": best["validation_fpr"],
        "test_auc": test_auc,
        "test_tpr": test_tpr,
        "test_fpr": test_fpr,
        "threshold": best["threshold"],
        "target_fpr_budget": target_fpr,
    }


def build_failure_cases(
    examples: List[Dict[str, object]],
    prefix_scores: np.ndarray,
    perturb_scores: np.ndarray,
    prefix_threshold: float,
    perturb_threshold: float,
    limit: int = 10,
) -> Dict[str, List[Dict[str, object]]]:
    prefix_flags = prefix_scores >= prefix_threshold
    perturb_flags = perturb_scores >= perturb_threshold

    prefix_miss_pert_hit = []
    prefix_hit_pert_miss = []
    both_miss = []

    for idx, example in enumerate(examples):
        row = {
            "example_id": example.get("id"),
            "subject": example.get("subject"),
            "question": example.get("question"),
            "prefix_score": float(prefix_scores[idx]),
            "perturbation_score": float(perturb_scores[idx]),
        }
        if (not prefix_flags[idx]) and perturb_flags[idx]:
            prefix_miss_pert_hit.append(row)
        elif prefix_flags[idx] and (not perturb_flags[idx]):
            prefix_hit_pert_miss.append(row)
        elif (not prefix_flags[idx]) and (not perturb_flags[idx]):
            both_miss.append(row)

    return {
        "prefix_miss_perturbation_hit": prefix_miss_pert_hit[:limit],
        "prefix_hit_perturbation_miss": prefix_hit_pert_miss[:limit],
        "both_miss": both_miss[:limit],
    }


def evaluate_condition(
    model_key: str,
    contamination_split: str,
    datasets: Dict[str, List[Dict[str, object]]],
    prefix_store: Dict[str, np.ndarray],
    perturb_store: Dict[str, Dict[str, np.ndarray]],
    sample_size: Optional[int],
    rng: np.random.Generator,
) -> Dict[str, object]:
    positive_examples = datasets[contamination_split][:sample_size] if sample_size else datasets[contamination_split]
    clean_examples = datasets["clean"][:sample_size] if sample_size else datasets["clean"]

    prefix_pos = prefix_store[f"{model_key}_{contamination_split}"][: len(positive_examples)]
    prefix_neg = prefix_store[f"{model_key}_clean"][: len(clean_examples)]
    pert_pos = perturb_store[model_key][contamination_split][: len(positive_examples)]
    pert_neg = perturb_store[model_key]["clean"][: len(clean_examples)]

    prefix_all = np.concatenate([prefix_pos, prefix_neg])
    pert_all = np.concatenate([pert_pos, pert_neg])
    labels = np.concatenate([np.ones(len(prefix_pos)), np.zeros(len(prefix_neg))]).astype(int)
    val_idx, test_idx = make_stratified_split(labels, SEED)

    prefix_val_auc, prefix_threshold, prefix_val_tpr, prefix_val_fpr = youden_threshold(
        labels[val_idx],
        prefix_all[val_idx],
    )
    pert_val_auc, pert_threshold, pert_val_tpr, pert_val_fpr = youden_threshold(
        labels[val_idx],
        pert_all[val_idx],
    )

    prefix_auc, prefix_tpr, prefix_fpr = metrics_at_threshold(
        labels[test_idx],
        prefix_all[test_idx],
        prefix_threshold,
    )
    pert_auc, pert_tpr, pert_fpr = metrics_at_threshold(
        labels[test_idx],
        pert_all[test_idx],
        pert_threshold,
    )

    target_fpr_budget = prefix_val_fpr if prefix_val_tpr >= pert_val_tpr else pert_val_fpr
    ensemble = optimize_ensemble(
        prefix_all,
        pert_all,
        labels,
        seed=SEED,
        target_fpr=target_fpr_budget,
    )

    prefix_ci = bootstrap_metric(labels[test_idx], prefix_all[test_idx], prefix_threshold, rng)
    pert_ci = bootstrap_metric(labels[test_idx], pert_all[test_idx], pert_threshold, rng)
    ensemble_ci = bootstrap_metric(
        ensemble["test_labels"],
        ensemble["test_scores"],
        ensemble["threshold"],
        rng,
    )

    test_pos_mask = ensemble["test_labels"] == 1
    prefix_test_scores = prefix_all[test_idx]
    pert_test_scores = pert_all[test_idx]
    prefix_test_flags = (prefix_test_scores >= prefix_threshold).astype(int)
    pert_test_flags = (pert_test_scores >= pert_threshold).astype(int)
    ensemble_test_flags = (ensemble["test_scores"] >= ensemble["threshold"]).astype(int)

    positive_prefix = prefix_test_flags[test_pos_mask]
    positive_pert = pert_test_flags[test_pos_mask]
    positive_ensemble = ensemble_test_flags[test_pos_mask]
    best_individual = positive_prefix.copy()
    if positive_pert.mean() > positive_prefix.mean():
        best_individual = positive_pert

    ttest = ttest_rel(positive_ensemble, best_individual) if len(best_individual) > 1 else None
    t_statistic = 0.0
    p_value = 1.0
    if ttest is not None:
        if np.isfinite(ttest.statistic):
            t_statistic = float(ttest.statistic)
        if np.isfinite(ttest.pvalue):
            p_value = float(ttest.pvalue)

    positive_indices = np.where(labels == 1)[0]
    failure_cases = build_failure_cases(
        positive_examples,
        prefix_pos,
        pert_pos,
        prefix_threshold,
        pert_threshold,
    )

    return {
        "condition": f"{model_key}_{contamination_split}",
        "model": model_key,
        "contamination_type": contamination_split,
        "prefix": {
            "auc": prefix_auc,
            "threshold": prefix_threshold,
            "tpr": prefix_tpr,
            "fpr": prefix_fpr,
            "validation_auc": prefix_val_auc,
            "validation_tpr": prefix_val_tpr,
            "validation_fpr": prefix_val_fpr,
            **prefix_ci,
        },
        "perturbation": {
            "auc": pert_auc,
            "threshold": pert_threshold,
            "tpr": pert_tpr,
            "fpr": pert_fpr,
            "validation_auc": pert_val_auc,
            "validation_tpr": pert_val_tpr,
            "validation_fpr": pert_val_fpr,
            **pert_ci,
        },
        "ensemble": {
            "weights": ensemble["weights"],
            "validation_auc": ensemble["validation_auc"],
            "validation_tpr": ensemble["validation_tpr"],
            "validation_fpr": ensemble["validation_fpr"],
            "auc": ensemble["test_auc"],
            "threshold": ensemble["threshold"],
            "tpr": ensemble["test_tpr"],
            "fpr": ensemble["test_fpr"],
            "target_fpr_budget": ensemble["target_fpr_budget"],
            **ensemble_ci,
        },
        "delta_tpr": float(positive_ensemble.mean() - max(positive_prefix.mean(), positive_pert.mean())),
        "paired_t_test": {
            "t_statistic": t_statistic,
            "p_value": p_value,
        },
        "split_sizes": {
            "positive_total": int(len(positive_indices)),
            "clean_total": int(len(clean_examples)),
            "validation_total": int(len(val_idx)),
            "test_total": int(len(test_idx)),
        },
        "failure_cases": failure_cases,
    }


def summarise(results: List[Dict[str, object]], method: str, contamination_type: str) -> Dict[str, float]:
    rows = [r[method] for r in results if r["contamination_type"] == contamination_type]
    return {
        "avg_tpr": float(np.mean([row["tpr"] for row in rows])) if rows else 0.0,
        "avg_auc": float(np.mean([row["auc"] for row in rows])) if rows else 0.0,
    }


def build_comparison_table(results: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "condition": result["condition"],
                "prefix_tpr": result["prefix"]["tpr"],
                "prefix_fpr": result["prefix"]["fpr"],
                "prefix_auc": result["prefix"]["auc"],
                "perturbation_tpr": result["perturbation"]["tpr"],
                "perturbation_fpr": result["perturbation"]["fpr"],
                "perturbation_auc": result["perturbation"]["auc"],
                "ensemble_tpr": result["ensemble"]["tpr"],
                "ensemble_fpr": result["ensemble"]["fpr"],
                "ensemble_auc": result["ensemble"]["auc"],
                "ensemble_prefix_weight": result["ensemble"]["weights"]["prefix"],
                "ensemble_perturbation_weight": result["ensemble"]["weights"]["perturbation"],
                "delta_tpr": result["delta_tpr"],
                "paired_t_test_p_value": result["paired_t_test"]["p_value"],
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Member 3 perturbation sensitivity workflow")
    parser.add_argument("--base-model", default=os.environ.get("PHI2_BASE_MODEL", "microsoft/phi-2"))
    parser.add_argument("--sample-size", type=int, default=None, help="Optional cap per split for fast testing")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mock", action="store_true", help="Use synthetic perturbation scores if the model is unavailable")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Reserved for future extension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("Member 3: Perturbation Sensitivity Detection + Ensemble")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Mock mode: {args.mock}")

    datasets = load_datasets()
    prefix_scores = load_prefix_scores()

    if args.mock:
        tokenizer = None
        models = {"model_v": None, "model_p": None}
    else:
        try:
            tokenizer, models = load_model_pair(args.base_model, device)
        except Exception as exc:
            raise RuntimeError(
                "Could not load Phi-2 locally. Use --mock for a dry run or set PHI2_BASE_MODEL "
                "to a local Phi-2 path."
            ) from exc

    perturb_scores: Dict[str, Dict[str, np.ndarray]] = {}
    perturb_details: Dict[str, Dict[str, List[ExampleResult]]] = {}
    for model_key in ["model_v", "model_p"]:
        model_scores, model_details = compute_sensitivity_scores(
            model_key=model_key,
            model=models[model_key],
            tokenizer=tokenizer,
            device=device,
            datasets=datasets,
            sample_size=args.sample_size,
            mock=args.mock,
            rng=rng,
        )
        perturb_scores[model_key] = model_scores
        perturb_details[model_key] = model_details

    condition_results = []
    primary_conditions = [
        ("model_v", "verbatim"),
        ("model_p", "paraphrased"),
    ]
    for model_key, contamination_split in primary_conditions:
        condition_results.append(
            evaluate_condition(
                model_key=model_key,
                contamination_split=contamination_split,
                datasets=datasets,
                prefix_store=prefix_scores,
                perturb_store=perturb_scores,
                sample_size=args.sample_size,
                rng=rng,
            )
        )

    comparison_df = build_comparison_table(condition_results)
    comparison_path = RESULTS_DIR / "member3_real_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    aggregates = {
        "prefix": {
            "verbatim": summarise(condition_results, "prefix", "verbatim"),
            "paraphrased": summarise(condition_results, "prefix", "paraphrased"),
        },
        "perturbation": {
            "verbatim": summarise(condition_results, "perturbation", "verbatim"),
            "paraphrased": summarise(condition_results, "perturbation", "paraphrased"),
        },
        "ensemble": {
            "verbatim": summarise(condition_results, "ensemble", "verbatim"),
            "paraphrased": summarise(condition_results, "ensemble", "paraphrased"),
        },
    }

    final_results = {
        "version": "Member 3 pipeline with held-out ensemble evaluation",
        "mock_mode": args.mock,
        "sample_size": args.sample_size,
        "conditions": condition_results,
        "aggregates": aggregates,
        "degradation_analysis": {
            "prefix_degradation": aggregates["prefix"]["verbatim"]["avg_tpr"] - aggregates["prefix"]["paraphrased"]["avg_tpr"],
            "perturbation_degradation": aggregates["perturbation"]["verbatim"]["avg_tpr"] - aggregates["perturbation"]["paraphrased"]["avg_tpr"],
            "ensemble_degradation": aggregates["ensemble"]["verbatim"]["avg_tpr"] - aggregates["ensemble"]["paraphrased"]["avg_tpr"],
        },
    }

    json_path = RESULTS_DIR / "member3_real_results.json"
    with json_path.open("w") as handle:
        json.dump(final_results, handle, indent=2)

    print("\nComparison table:")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved results to: {json_path}")
    print(f"Saved comparison table to: {comparison_path}")


if __name__ == "__main__":
    main()
