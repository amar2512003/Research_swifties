# Member 3 Summary

## Scope

This note summarizes the Member 3 work on:

- perturbation sensitivity detection
- ensemble detection with Member 2 prefix scores
- what changed across versions
- which result version is the strongest
- what the current results mean for the paper

## Versions We Ran

### 1. Early baseline

This was the first working Member 3 pipeline, but it had important methodological issues:

- paraphrased contamination was evaluated on the paraphrased text instead of the original wording
- perturbation sensitivity used a coarse binary accuracy-drop score
- several perturbations were often ineffective or noisy
- ensemble comparison was not fully fair across methods

Because of those issues, the early results were unstable and not reliable enough to use as the final table.

### 2. Corrected real pipeline

This version fixed the major issues above:

- paraphrased evaluation uses the original wording
- perturbation sensitivity uses correct-answer probability drop
- no-op perturbations are skipped
- only the valid experiment conditions are reported:
  - `model_v_verbatim`
  - `model_p_paraphrased`

This made the perturbation method behave much more plausibly.

### 3. Improved batch / v4

This version is the strongest final variant so far.

It added:

- safer perturbations
- weighted perturbation aggregation
- a TPR-oriented ensemble target
- fairer thresholding under a fixed false-positive budget

This is the best version to use as the primary result table.

## Best Current Results

These are the latest v4 improved-batch results reported from Colab.

| Condition | Prefix TPR | Prefix FPR | Prefix AUC | Perturb TPR | Perturb FPR | Perturb AUC | Ensemble TPR | Ensemble FPR | Ensemble AUC | Ensemble Weights | Delta TPR | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `model_v_verbatim` | 0.42 | 0.14 | 0.7259 | 0.49 | 0.35 | 0.6750 | 0.64 | 0.33 | 0.7653 | prefix 0.55 / perturb 0.45 | +0.15 | 0.0034 |
| `model_p_paraphrased` | 0.68 | 0.14 | 0.8104 | 0.35 | 0.25 | 0.6058 | 0.68 | 0.15 | 0.8139 | prefix 0.95 / perturb 0.05 | 0.00 | 1.0000 |

Aggregate degradation values:

- prefix degradation: `-0.26`
- perturbation degradation: `+0.14`
- ensemble degradation: `-0.04`

## What These Results Mean

### Main takeaway

The best-supported conclusion is:

- perturbation sensitivity is useful for verbatim contamination
- perturbation sensitivity is much weaker for paraphrased contamination
- prefix completion remains the strongest practical detector for paraphrased contamination in this project
- the ensemble helps in some cases, but not uniformly

### Strongest ensemble result

The most important positive finding is:

- for `model_v_verbatim`, the ensemble improved TPR from the best individual method baseline by `+0.15`
- this improvement was statistically significant with `p = 0.0034`

This is the clearest ensemble win in the current experiments.

### Paraphrased contamination result

For `model_p_paraphrased`:

- prefix completion is still the strongest detector
- perturbation sensitivity underperforms substantially
- the ensemble mostly falls back to prefix behavior

This means the ensemble does not currently provide a meaningful TPR gain on paraphrased contamination.

### Interpretation of degradation

The perturbation detector now shows the expected direction:

- `perturbation_degradation = +0.14`

That means perturbation sensitivity performs better on verbatim contamination than on paraphrased contamination, which is a believable and publishable finding.

However:

- `prefix_degradation = -0.26`

In this setup, prefix completion is actually stronger on paraphrased contamination than verbatim contamination. This is opposite to the original expectation, so it should be written as an empirical finding rather than forced into the hypothesis.

## Recommended Paper Framing

Use the v4 improved-batch results as the main Member 3 result table.

Recommended interpretation:

1. Perturbation sensitivity was not as robust to paraphrased contamination as expected.
2. Prefix completion remained the strongest detector for paraphrased contamination in our controlled setup.
3. The ensemble improved detection for verbatim contamination, but the gain did not transfer to paraphrased contamination.
4. This supports the broader paper theme that contamination detectors fail in different ways depending on contamination type.

## How To Use Multiple Versions

Do not mix rows from different versions into one main table.

Best practice:

- use v4 improved-batch as the primary result table
- mention the corrected real pipeline as an ablation/debugging step
- explain that earlier unstable results came from implementation and evaluation issues that were later fixed

Suggested structure:

- main results table: v4 improved-batch
- short ablation paragraph:
  - early baseline was unstable
  - corrected pipeline fixed evaluation alignment and score design
  - improved batch produced the strongest ensemble gain

## Recommended Final Position

If the goal is a strong and honest paper story, stop tuning here and write the results section from these numbers.

Reason:

- there is now one clear ensemble success case
- the paraphrased failure case is still informative
- this matches the project title and research question well

## Files To Cite

- main script: `member3_improved_batch.py`
- previous corrected script: `member3_real_perturbation_detection.py`
- improved outputs:
  - `member3_results/member3_improved_results.json`
  - `member3_results/member3_improved_comparison.csv`
