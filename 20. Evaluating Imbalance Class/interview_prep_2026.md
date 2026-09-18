# Evaluating Imbalanced Classification — Senior ML Engineer Interview Prep (2026)

> Pairs with folder 21 (handling techniques) — this doc is specifically about *measuring* model quality correctly when classes are imbalanced, which is where most junior-level mistakes happen.

## TL;DR Refresher

- **Accuracy is actively misleading** on imbalanced data — a classifier that always predicts the majority class gets high accuracy while being useless (the classic 99%-accuracy-on-1%-fraud trap).
- Precision, Recall, F1, and the **Precision-Recall (PR) curve / PR-AUC** are generally more informative than ROC-AUC on heavily imbalanced data, because ROC-AUC's false-positive-rate term is diluted by the large number of true negatives, making it look artificially good even for a mediocre minority-class detector.
- Confusion matrix, per-class metrics (macro/micro/weighted F1), and cost-sensitive/business-aligned metrics (e.g., expected cost given false-positive/false-negative business costs) round out a proper imbalanced-evaluation toolkit.

## What's New / State of the Art (2025–2026)

- **PR-AUC over ROC-AUC as the default reporting metric for imbalanced problems is now firmly standard practice** in 2025–2026 industry and research writeups — this shift (well-argued since Davis & Goadrich's original 2006 paper but only fully mainstream in the last several years) is treated as baseline knowledge; failing to name this trade-off unprompted is a notable gap at senior level.
- **Calibration evaluation alongside discrimination metrics** has become standard for imbalanced classifiers deployed as risk scores (fraud, credit, healthcare triage) — reporting only AUC/F1 without also checking calibration (reliability diagrams, Brier score, calibration slope/intercept) is increasingly flagged as incomplete evaluation in 2025–2026 production ML review practices (see the Evaluation Metrics doc for the general calibration discussion).
- **Business-cost-weighted metrics over generic F1** — 2025–2026 senior/staff-level practice increasingly pushes past generic F1 (which implicitly assumes false positives and false negatives are equally costly) toward an explicit expected-cost or `Fβ` metric where `β` is derived from the actual business cost ratio (e.g., cost of missing fraud vs. cost of a false fraud alert) — being able to derive and justify the right `β` or cost matrix for a given business problem is a concrete 2025–2026 senior-interview differentiator.
- **Threshold-independent evaluation during model development, threshold-tuned evaluation before deployment** — a two-stage evaluation discipline (evaluate ranking quality with PR-AUC during model selection, then separately tune and validate a specific decision threshold against business constraints before shipping) is the modern standard workflow, replacing older single-threshold-only evaluation habits.

## Senior-Level Interview Questions

**Q1. Why is accuracy a poor metric for imbalanced classification, and what should you report instead?**
Strong answer: With a 99:1 class ratio, a trivial always-majority-class classifier achieves 99% accuracy while having zero recall on the minority class — accuracy simply doesn't penalize this failure mode. Report precision, recall, F1 (or `Fβ` matched to business costs) per class, plus PR-AUC for threshold-independent ranking quality, and always inspect the full confusion matrix rather than a single scalar.

**Q2. Explain precisely why ROC-AUC can look deceptively good on imbalanced data while PR-AUC reveals the true weakness.**
Strong answer: ROC-AUC plots true positive rate vs. false positive rate (`FP / (FP+TN)`); when negatives vastly outnumber positives, a moderate absolute number of false positives is a tiny fraction of the huge negative pool, keeping FPR low and ROC-AUC looking strong even if precision (`TP / (TP+FP)`, which compares false positives against the *much smaller* pool of positive predictions) is actually poor. PR-AUC (precision vs. recall) doesn't involve true negatives at all, so it directly reflects how well the model concentrates its positive predictions — a far more honest signal when the positive class is rare.

**Q3. Derive the relationship between precision, recall, and `Fβ`, and explain how you'd choose `β` for a fraud-detection system where missing fraud costs 10x more than a false alarm.**
Strong answer: `Fβ = (1+β²)·(precision·recall) / (β²·precision + recall)` — `β>1` weights recall more heavily (penalizes missed positives more), `β<1` weights precision more heavily. If missing fraud (a false negative) costs 10x a false alarm (false positive), you want recall weighted roughly proportionally more — a common approach is setting `β` such that the metric's implicit precision/recall trade-off reflects that cost ratio (e.g., `β=√10≈3.16` under one common derivation linking `Fβ` to a cost ratio, though in practice many senior practitioners instead directly build an explicit expected-cost function — `Cost = FN_cost·FN_count + FP_cost·FP_count` — and optimize/report that instead of reverse-engineering a `β`, since it's more transparent and directly tied to the business justification).

**Q4. Your imbalanced classifier has excellent PR-AUC in offline evaluation, but the operations team says the deployed model "cries wolf too often" at the chosen threshold. What's going on and how do you fix it?**
Strong answer: PR-AUC is threshold-independent — it measures ranking quality across all thresholds, but says nothing about whether the *specific* threshold chosen for deployment is well-calibrated to the operational tolerance for false positives. This is an evaluation-methodology gap: good ranking (PR-AUC) doesn't guarantee a good operating point was selected. Fix by explicitly tuning the decision threshold against a precision (or business-cost) target using a validation set (e.g., "find the threshold that achieves ≥90% precision, report the resulting recall"), rather than defaulting to 0.5, and validate that chosen threshold's real-world false-positive rate matches operational capacity before shipping.

**Q5. How do you evaluate an imbalanced *multiclass* classification problem (e.g., 20 product categories, wildly different frequencies)?**
Strong answer: Report macro-averaged F1 (unweighted average across classes — treats rare classes as equally important, revealing if the model is ignoring them) *alongside* weighted-averaged F1 (weighted by class frequency — reflects overall/aggregate performance) rather than either alone, since they can diverge sharply and tell different stories; inspect the full per-class confusion matrix (not just an aggregate scalar) to identify which specific rare classes are being systematically confused with the majority classes, and consider per-class PR curves for the classes that matter most to the business even if they're numerically rare.

## Common Pitfalls & Follow-Up Probes

- Reporting only ROC-AUC on a severely imbalanced problem without also reporting PR-AUC — a very common and very findable gap.
- Using a default 0.5 classification threshold without justification on an imbalanced problem — the threshold should be explicitly tuned against a validation-set target.
- Computing metrics after resampling the evaluation set (SMOTE, undersampling) — evaluation must always happen on the original, untouched class distribution to reflect real-world performance; resampling is a training-time technique only (see folder 21).

## Key Resources

- Davis & Goadrich (2006), *"The Relationship Between Precision-Recall and ROC Curves"* — the foundational argument for preferring PR curves under class imbalance.
- [Handling Imbalanced Classification: What Works Better Than SMOTE (2026)](https://www.analyticsvidhya.com/blog/2026/07/class-imbalance-ml/)
- scikit-learn model-evaluation documentation (`precision_recall_curve`, `average_precision_score`, `fbeta_score`).
