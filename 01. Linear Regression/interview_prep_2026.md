# Linear Regression — Senior ML Engineer Interview Prep (2026)

> Companion to `linear_regression.ipynb`. This doc assumes the core theory (OLS, normal equation, gradient descent, multivariable regression) and focuses on what a **senior** candidate is expected to add in 2026: production judgment, statistical rigor, and awareness of where linear models still win against deep/foundation models.

## TL;DR Refresher

- Linear regression minimizes squared residuals: closed-form via the normal equation `β = (XᵀX)⁻¹Xᵀy`, or iteratively via (stochastic/mini-batch) gradient descent when `X` is large or `XᵀX` is ill-conditioned.
- Assumes linearity in parameters, homoscedastic and independent errors, and (for inference, not prediction) normally distributed residuals.
- Regularized variants: Ridge (L2, shrinks coefficients, handles multicollinearity), Lasso (L1, induces sparsity/feature selection), Elastic Net (mix of both).

## What's New / State of the Art (2025–2026)

- **Linear models as the "boring baseline that keeps winning."** With the industry's heavy 2025–2026 investment in LLMs and tabular foundation models, senior interviewers increasingly probe whether you reach for a well-regularized linear model *first* — for latency-critical, interpretable, or low-data settings — before escalating to gradient boosting or neural approaches. Being able to justify "why not deep learning" is now a real signal of seniority.
- **Double/debiased machine learning & causal regression** have become mainstream in industry (pricing, marketing-mix modeling, experimentation teams) — using regularized linear models as nuisance estimators inside causal-inference pipelines (e.g., `EconML`, `DoWhy`) rather than only for plain prediction.
- **Conformal prediction** is now a standard way to wrap linear (and other) regressors with distribution-free prediction intervals in production, replacing hand-wavy "±1.96·σ" assumptions — expect this to come up when you're asked about uncertainty quantification.
- **Elastic Net / coordinate descent at scale** — libraries like `scikit-learn`, `glum`, and GPU-accelerated solvers (`cuML`) make regularized linear regression practical on tens of millions of rows, which keeps it competitive as a first production model even at "big data" scale.
- Interview prep resources for 2026 increasingly frame linear regression questions around **bias-variance trade-offs under regularization** and **feature-leakage/data-drift** rather than pure derivation of the normal equation ([Devinterview.io 2026 bank](https://github.com/Devinterview-io/linear-regression-interview-questions), [InterviewPal 2026 guide](https://www.interviewpal.com/blog/25-machine-learning-interview-questions-for-2026-and-how-senior-candidates-actually-answer-them)).

## Senior-Level Interview Questions

**Q1. Derive the normal equation and explain when you would *not* use it.**
What they're testing: whether you understand the linear algebra and its computational limits.
Strong answer: Minimize `J(β) = ||y - Xβ||²`; setting `∂J/∂β = 0` gives `XᵀXβ = Xᵀy` → `β = (XᵀX)⁻¹Xᵀy`. Avoid it when `XᵀX` is singular/ill-conditioned (multicollinearity), when `p` (features) is large relative to `n` (inverting an `O(p³)` matrix is expensive), or when data arrives in a stream — use gradient descent, QR/SVD decomposition, or Ridge instead.

**Q2. Why does Ridge regression help with multicollinearity, and what happens to coefficients as λ → ∞?**
Strong answer: Ridge adds `λ‖β‖²` to the loss, which adds `λI` to `XᵀX` before inversion, making it invertible/well-conditioned even when columns are correlated. As `λ → ∞`, all coefficients shrink to 0 (underfitting); as `λ → 0`, it converges to OLS. Ridge trades bias for reduced variance — useful whenever `Var(β̂)` is inflated by correlated predictors.

**Q3. Lasso vs Ridge vs Elastic Net — when would you pick each in a real feature set with 5,000 correlated features?**
Strong answer: Lasso does implicit feature selection but is unstable under high correlation (arbitrarily picks one of a correlated group and can be non-unique). Ridge keeps all features and is stable but doesn't sparsify. Elastic Net (`α·L1 + (1-α)·L2`) is usually the practical senior-level answer here — it groups and shrinks correlated features together while still sparsifying, which is exactly the pathology described.

**Q4. Your model has great R² on training and validation but degrades within 2 weeks in production. Walk through your diagnosis.**
Strong answer: This is a data/feature-drift question wearing a linear-regression costume. Check: (a) feature distribution drift (PSI/KL divergence per feature), (b) label drift or delayed-label bias, (c) leakage of a feature only available at train time (e.g., a feature computed using future information), (d) whether the linear relationship itself is non-stationary (seasonality, regime change) — propose monitoring with drift detectors and periodic retraining/online updating.

**Q5. How do you get calibrated prediction intervals for a linear regression deployed in a business-critical pricing system?**
Strong answer: Classical intervals assume Gaussian, homoscedastic residuals — often false. Use **split conformal prediction**: hold out a calibration set, compute residual quantiles, and wrap predictions with distribution-free intervals that guarantee coverage regardless of the true error distribution. This is now a standard senior-level expectation, not a research nicety.

**Q6. When is gradient descent preferable to the closed-form solution, and how do you choose a learning rate at scale?**
Strong answer: Prefer GD/SGD when `n` or `p` is large (closed form is `O(np² + p³)`), when doing online/streaming learning, or when adding non-quadratic regularization. In practice, use adaptive optimizers (Adam) or a learning-rate schedule/line search rather than a fixed rate, and monitor loss curves for divergence vs. plateauing.

**Q7. A stakeholder asks "why did the model predict $120k for this house?" — how do you answer with a linear model vs. a black-box model?**
Strong answer: This is the interpretability angle senior candidates are expected to raise proactively: linear regression gives an additive, exact decomposition (`coefficient × feature value` per term), which is directly explainable — a genuine advantage over needing SHAP/LIME approximations for tree ensembles or neural nets. Flag this as a legitimate reason to prefer linear models in regulated domains (credit, insurance, healthcare).

## Production / System-Design Angle

- **Feature-store consistency:** the biggest real-world failure mode for linear models isn't the math, it's train/serve skew — a feature computed differently online vs. offline silently breaks coefficients' validity. Be ready to describe how you'd guard against this (shared feature-transformation code, feature store with point-in-time correctness).
- **Retraining cadence:** because linear models are cheap to retrain, senior candidates should propose continuous/online retraining (e.g., recursive least squares, SGD warm-starts) instead of static periodic batch retraining when the underlying relationship drifts.

## Common Pitfalls & Follow-Up Probes

- Forgetting to standardize/scale features before regularization (Ridge/Lasso penalties are scale-dependent).
- Treating R² as sufficient — an interviewer may push on adjusted R², residual plots, and heteroscedasticity (Breusch–Pagan) as follow-ups.
- Ignoring that OLS coefficients are BLUE (Gauss–Markov) only under specific assumptions — be ready to state the assumptions precisely if asked.
- Not distinguishing prediction intervals from confidence intervals.

## Key Resources

- [Devinterview.io — Linear Regression Interview Questions (2026)](https://github.com/Devinterview-io/linear-regression-interview-questions)
- [InterviewPal — 25 ML Interview Questions for 2026: How Senior Candidates Answer](https://www.interviewpal.com/blog/25-machine-learning-interview-questions-for-2026-and-how-senior-candidates-actually-answer-them)
- [InterviewBit — Linear Regression Interview Questions (2025)](https://www.interviewbit.com/linear-regression-interview-questions/)
- Conformal prediction: *"A Gentle Introduction to Conformal Prediction"* (Angelopoulos & Bates) — standard senior-level reference for distribution-free uncertainty.
