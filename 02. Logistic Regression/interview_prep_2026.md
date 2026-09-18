# Logistic Regression — Senior ML Engineer Interview Prep (2026)

> Companion to the logistic regression notebook. Focuses on 2025–2026 developments and how senior interviews probe this topic beyond the sigmoid/log-loss derivation.

## TL;DR Refresher

- Models `P(y=1|x) = σ(wᵀx + b) = 1/(1+e^{-(wᵀx+b)})`, fit by maximizing log-likelihood (equivalently minimizing binary cross-entropy), typically via (L-)BFGS, Newton-Raphson/IRLS, or gradient descent.
- Decision boundary is linear in the log-odds; regularization (L1/L2/Elastic Net) controls overfitting exactly as in linear regression.
- Multiclass via one-vs-rest or softmax (multinomial logistic regression).

## What's New / State of the Art (2025–2026)

- **Logistic regression as the calibration gold standard.** As industry has gotten far more serious about *calibrated* probabilities (risk scoring, ad auctions, healthcare triage), logistic regression's naturally well-calibrated output (versus tree ensembles or neural nets, which often need Platt scaling / isotonic regression on top) is now explicitly cited as a reason to keep it in production stacks even when raw AUC is lower than a boosted model.
- **Foundation-model-assisted feature construction feeding a simple logistic head.** A recurring 2025–2026 industry pattern: use an LLM or pretrained embedding model to generate rich features/embeddings, then fit an interpretable, auditable logistic regression on top for the final decision layer — common in fraud, credit, and content-moderation systems where the *decision* must be explainable even if upstream features are learned. See discussion of "foundation models feeding interpretable heads" ([PubMed 2026 study on LR vs. foundation models for forecasting](https://pubmed.ncbi.nlm.nih.gov/41245920/)).
- **Regulatory pressure (EU AI Act, evolving US state AI regulation)** has pushed high-stakes classification pipelines (lending, insurance, hiring) back toward inherently interpretable models like logistic regression, or toward "glass-box" wrappers (GAMs, EBMs) — a senior candidate should be able to discuss this trade-off explicitly.
- **Recalibration in production** — drift causes calibration decay faster than discrimination (AUC) decay; 2025–2026 MLOps best practice is to monitor calibration slope/intercept continuously, not just AUC ([calibration slope/intercept methodology](https://metricgate.com/docs/calibration-slope-intercept-logistic/)).

## Senior-Level Interview Questions

**Q1. Derive the gradient of the log-loss with respect to weights, and explain why logistic regression training is a convex problem.**
Strong answer: `∂L/∂w = Xᵀ(σ(Xw) - y)` (same clean form as linear regression's gradient, up to the sigmoid). The negative log-likelihood of a Bernoulli model is convex in `w` because it's a composition of a convex loss with a linear function, so gradient descent/Newton's method converges to a global optimum.

**Q2. Your model has AUC 0.85 but stakeholders complain the predicted probabilities "don't mean anything." What's happening and how do you fix it?**
Strong answer: AUC measures ranking/discrimination, not calibration. Diagnose with a reliability diagram / calibration curve and Brier score decomposition; fix with Platt scaling or isotonic regression on a held-out calibration set, or refit with a properly specified link function and interaction terms. This is a classic "seniority tell" question — juniors conflate AUC with correctness of the probabilities.

**Q3. How do you handle severe class imbalance (e.g., 0.1% positive rate) in logistic regression specifically?**
Strong answer: Class weights (`class_weight='balanced'` or manual weighting that reflects business cost asymmetry) adjust the loss; note that naive oversampling (SMOTE) can distort the log-odds intercept — if you resample, you must correct the intercept afterward (offset correction) to recover calibrated probabilities on the true population, or use `class_weight` instead, which doesn't have this bias.

**Q4. Explain odds ratio interpretation of a coefficient, and how you'd explain a coefficient of 0.7 to a non-technical stakeholder.**
Strong answer: `e^{β_j}` is the multiplicative change in odds for a one-unit increase in `x_j`, holding others fixed. `β_j = 0.7` → `e^0.7 ≈ 2.01`, i.e., odds roughly double per unit increase — this needs to be phrased in odds, not raw probability, since the probability effect is non-linear (depends on baseline).

**Q5. Newton-Raphson / IRLS vs. SGD for fitting logistic regression — when would you choose each?**
Strong answer: IRLS (iteratively reweighted least squares) converges in very few iterations and gives exact solutions for small/medium `n, p` (used by `statsmodels`), but each iteration is `O(np²+p³)`. SGD/mini-batch (as in `scikit-learn`'s `SGDClassifier` or deep learning frameworks) scales to millions of rows and streaming data but needs learning-rate tuning and more iterations.

**Q6. Multicollinearity in logistic regression — does it bias coefficients or predictions?**
Strong answer: It doesn't bias predictions/decision boundary but inflates coefficient variance, making individual coefficients unstable/uninterpretable (and possibly the wrong sign) — critical if the business use case needs coefficient interpretation (e.g., credit adverse-action reasons), even if pure classification accuracy is unaffected. Mitigate with Ridge/Elastic Net or VIF-based feature pruning.

**Q7. How would you design a system where logistic regression is the "explainable decision layer" on top of a learned embedding model?**
Strong answer: Freeze/pretrain the embedding model (or an LLM-derived feature extractor), then fit logistic regression on the embeddings/derived features as the final, auditable layer — gives you monotonic, coefficient-level explanations for regulators while still capturing non-linear signal upstream. Discuss the trade-off: you lose end-to-end differentiability/joint optimization for compliance and debuggability.

## Common Pitfalls & Follow-Up Probes

- Confusing "linear decision boundary in feature space" with "no non-linear relationships representable" — remind interviewer you can add interaction/polynomial terms.
- Not knowing that perfect separation causes coefficients to diverge to infinity (needs regularization or Firth's penalized likelihood).
- Forgetting the log-loss is the *proper scoring rule* justification for using it over accuracy during training.

## Key Resources

- [Devinterview.io — Logistic Regression Interview Questions (2026)](https://github.com/Devinterview-io/logistic-regression-interview-questions)
- [Calibration slope/intercept from logistic recalibration](https://metricgate.com/docs/calibration-slope-intercept-logistic/)
- [From Logistic Regression to Foundation Models — PubMed 2026](https://pubmed.ncbi.nlm.nih.gov/41245920/)
