# Feature Selection — Senior ML Engineer Interview Prep (2026)

> Complements folder 25 (creating features) — this is about removing/ranking them. Also connects tightly to folder 27 (dimensionality reduction), which senior interviewers expect you to distinguish clearly.

## TL;DR Refresher

- **Filter methods**: rank features by a statistic independent of any model (correlation with target, mutual information, chi-squared test, variance threshold) — fast, model-agnostic, but ignore feature interactions and downstream-model-specific behavior.
- **Wrapper methods**: use a model's actual performance to evaluate feature subsets (Recursive Feature Elimination, forward/backward selection) — captures interactions and model-specific effects, but computationally expensive (requires retraining per subset evaluated).
- **Embedded methods**: feature selection happens as a byproduct of model training itself (Lasso's L1-induced sparsity, tree-based feature importance/gain) — a good practical middle ground between filter speed and wrapper accuracy.
- Feature selection ≠ dimensionality reduction: selection keeps a subset of the *original* features (interpretable, no transformation); reduction (PCA, UMAP) creates *new* transformed features that are combinations of the originals (often uninterpretable individually) — this distinction is a common and important interview clarification.

## What's New / State of the Art (2025–2026)

- **SHAP-based feature selection has become a standard, more trustworthy alternative to raw model-native importance** (e.g., Gini/MDI importance from trees, as flagged as biased in the Random Forest and Decision Tree docs) — using aggregated SHAP values to rank and prune features is now common 2025–2026 production practice, since it's more robust to feature-cardinality bias and correlation effects than naive impurity-based importance.
- **Feature selection as a leakage/robustness safeguard, not just a dimensionality-reduction tool** — 2025–2026 practice increasingly frames feature selection partly as a defense against overfitting to spurious/leaky features (the "too-good-to-be-true" feature flagged in the Feature Engineering doc) — a disciplined feature-selection/review process is now considered part of a broader model-governance and leakage-prevention pipeline, not purely a performance-optimization step.
- **Automated feature selection inside AutoML/AutoGluon-style pipelines** and inside gradient-boosting workflows (e.g., using built-in feature importance plus permutation importance as an automatic pruning step before final model training) continues to reduce how much manual wrapper-method search senior engineers do by hand, shifting the interview focus from "how do you run RFE" toward "how do you validate and trust an automated selection's output."
- **Stability selection and selection-under-resampling** (checking whether a selected feature subset stays consistent across bootstrap resamples or CV folds) has become a more emphasized 2025–2026 practice for building trust in a feature-selection result before shipping it, given growing scrutiny of ML reproducibility in production.

## Senior-Level Interview Questions

**Q1. Compare filter, wrapper, and embedded feature selection — give a concrete scenario where each is the clearly right choice.**
Strong answer: **Filter** — you have 100,000 candidate features (e.g., genomic data) and need a fast first-pass prune before any model training is even feasible; use mutual information or correlation thresholding to cut down to a manageable set. **Wrapper** — you have ~50 candidate features, model training is cheap, and you specifically need the subset that maximizes a particular model's validation performance (feature interactions matter and you can afford to retrain many times); use RFE with cross-validation. **Embedded** — you're already training a Lasso or gradient-boosting model for the primary task and want feature selection "for free" as part of that same training run rather than a separate pipeline stage; use the model's native sparsity/importance output directly.

**Q2. Why is Gini/MDI feature importance from a Random Forest or single tree potentially misleading for feature selection, and what would you use instead?**
Strong answer: As covered in the Random Forest doc, MDI importance is computed on training data and is biased toward high-cardinality and continuous features (more possible split points inflate their apparent importance), and doesn't properly account for correlated features "splitting credit" between them. For feature selection specifically, this bias can cause you to systematically over-select noisy high-cardinality features and under-select genuinely important low-cardinality ones. Prefer permutation importance (measures the actual performance drop when a feature's values are shuffled, computed on held-out data) or aggregated SHAP values (game-theoretically fair attribution, accounts for interactions) for a more trustworthy selection signal.

**Q3. Explain Recursive Feature Elimination (RFE) step by step, and its main computational weakness.**
Strong answer: RFE trains a model on the full feature set, ranks features by some importance measure (model coefficients, feature importance), removes the least important feature(s), and repeats — retraining the model on the shrinking feature set at each step — until reaching a target feature count or performance criterion. Weakness: each step requires a full model retrain, so for `p` features removed one at a time, that's `O(p)` full training runs — prohibitively expensive for both large `p` and expensive-to-train models (deep learning, large gradient-boosting ensembles); mitigate by removing features in larger batches per step (RFE with a step size >1) or switching to a cheaper embedded method for the initial coarse pruning before a final wrapper-method refinement on a much smaller candidate set.

**Q4. You suspect your gradient-boosting model's feature-selection result (top 20 features by importance) isn't stable — different random seeds give noticeably different rankings. How do you investigate and what does instability actually tell you?**
Strong answer: Run **stability selection** — repeat the feature-importance ranking across multiple bootstrap resamples or CV folds (and/or multiple random seeds), and look at how consistently each feature appears in the top-k across runs; report a stability score (e.g., selection frequency) rather than trusting a single run's ranking. Instability specifically suggests either (a) many features carry redundant/correlated signal, so the model arbitrarily favors one of several similarly-informative features each run (not necessarily a problem for prediction accuracy, but a problem if you need a *specific, reproducible* feature list for interpretability or a downstream simplified model), or (b) genuinely low signal-to-noise in the dataset, where "important" features are barely distinguishable from noise — the fix and interpretation differ, so this diagnostic step matters before you act on an unstable ranking.

**Q5. A stakeholder asks you to build a simplified 5-feature model for regulatory explainability from an original 200-feature gradient-boosting model, without significantly sacrificing accuracy. Walk through your approach.**
Strong answer: Use SHAP-based importance ranking on the full 200-feature model to identify genuinely high-impact features (not raw MDI importance, per Q2); shortlist candidates accounting for correlation (don't pick 5 mutually redundant top features — use a correlation-aware selection or greedy forward-selection wrapper evaluating actual validation performance of small combinations); train a new model restricted to the selected 5 features and quantify the accuracy gap against the full model explicitly (report both, honestly, to the stakeholder — a 5-feature model's simplicity/explainability is a deliberate trade-off, not a free upgrade); validate the reduced model's stability (per Q4) before committing to it as the regulator-facing model.

## Common Pitfalls & Follow-Up Probes

- Conflating feature selection with dimensionality reduction (PCA/UMAP) when a stakeholder specifically needs interpretable, original features — this distinction is a frequent and important interview clarification.
- Selecting features using the full dataset (including validation/test) before splitting — the same leakage pattern flagged for scalers (folder 18) and SMOTE (folder 21) applies here: fit selection on training data only.
- Trusting a single run's feature-importance ranking without checking stability across seeds/folds, especially with many correlated candidate features.

## Key Resources

- Guyon & Elisseeff (2003), *"An Introduction to Variable and Feature Selection"* — the classic survey, still broadly applicable.
- Meinshausen & Bühlmann (2010), *"Stability Selection"* — the stability-selection paper.
- Lundberg & Lee, *"A Unified Approach to Interpreting Model Predictions"* (SHAP) — see also the XGBoost doc for TreeSHAP specifics.
