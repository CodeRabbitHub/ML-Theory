# Evaluation Metrics — Senior ML Engineer Interview Prep (2026)

> Broader than folder 20 (which is imbalance-specific) — this doc covers the general evaluation toolkit plus the 2025–2026 shift toward continuous production monitoring as an extension of "evaluation."

## TL;DR Refresher

- **Regression**: MSE/RMSE (penalizes large errors, same units as target for RMSE), MAE (robust, interpretable), R²/adjusted R² (variance explained), MAPE (percentage error — unstable near zero targets).
- **Classification**: accuracy, precision/recall/F1, ROC-AUC, PR-AUC (see folder 20 for imbalance nuance), log-loss/Brier score (probabilistic quality), confusion matrix.
- **Calibration** (distinct from discrimination): reliability diagrams, calibration slope/intercept, Expected Calibration Error (ECE) — measures whether predicted probabilities match observed frequencies, independent of ranking ability.

## What's New / State of the Art (2025–2026)

- **Evaluation has expanded from a one-time offline step into continuous production monitoring** as the dominant 2025–2026 framing — MLOps practice now treats "evaluation" as an ongoing pipeline (tracking live prediction distributions, calibration decay, and drift) rather than a single train/test-split number reported once before deployment ([ML Model Monitoring: Metrics, Drift Tests, Playbooks](https://www.domo.com/glossary/ml-model-monitoring), [Model Drift in Production 2026 Guide](https://alldaystech.com/guides/artificial-intelligence/model-drift-detection-monitoring-response)).
- **Data drift vs. model/concept drift as a distinguished pair of failure modes** is now standard vocabulary — *data drift* (input feature distributions shift, e.g., via PSI/KL-divergence/KS-test monitoring) vs. *concept drift* (the relationship between features and target itself changes, only detectable once labels arrive) — being able to distinguish these and name appropriate detection methods for each is now baseline senior knowledge ([Model Drift vs Data Drift, 2026](https://futureagi.com/blog/model-vs-data-drift-how-to-identify-and-handle-it/)).
- **Delayed-label monitoring strategies** — a lot of 2025–2026 production-evaluation discussion focuses specifically on the practical problem that true labels often arrive late (e.g., fraud confirmed weeks later, loan default confirmed months later) — proxy metrics (prediction-distribution drift, model confidence distribution shifts) are used as early-warning signals *before* delayed ground truth arrives to confirm actual performance degradation.
- **Calibration monitoring alongside discrimination monitoring** — since calibration tends to decay faster than discrimination (AUC) under drift, 2025–2026 best practice explicitly tracks calibration metrics (not just AUC/accuracy) as part of ongoing production evaluation, directly connecting to the Logistic Regression and Loss Functions docs' calibration discussions.

## Senior-Level Interview Questions

**Q1. Explain the difference between a model's discrimination (ranking ability) and calibration (probability correctness), with an example where a model has great discrimination but terrible calibration.**
Strong answer: Discrimination measures whether the model correctly *ranks* positives above negatives (AUC-ROC/PR-AUC don't care about the actual probability values, only their relative order); calibration measures whether a predicted probability of, say, 0.7 actually corresponds to a ~70% real-world positive rate among all cases predicted at that probability. Example: a model that outputs 0.9 for every true positive and 0.6 for every true negative has perfect discrimination (AUC=1, since positives always rank above negatives) but could be badly miscalibrated if the true positive rate given a 0.6 prediction is actually 5%, not 60% — a monotonic transformation of a model's raw scores preserves discrimination but can completely destroy calibration, which is exactly why Platt scaling/isotonic regression (monotonic recalibration) can fix calibration without touching AUC.

**Q2. Distinguish data drift from concept drift, and describe a detection strategy for each in a production system where labels arrive 30 days late.**
Strong answer: Data drift is a shift in `P(X)` (input feature distributions) — detectable *immediately* on live traffic without needing labels, via distribution-comparison tests (Population Stability Index, Kolmogorov-Smirnov test, or KL-divergence between a reference window and current window, per feature). Concept drift is a shift in `P(y|X)` (the true relationship between features and target) — this fundamentally requires actual labels to detect directly, so with a 30-day label lag you'd rely on proxy signals in the interim (e.g., the model's prediction-confidence distribution shifting, or a sudden change in the *rate* of positive predictions without a corresponding data-drift signal, which can hint at concept drift before confirmed labels arrive), then confirm with real performance metrics once labels catch up, and trigger retraining/investigation accordingly.

**Q3. Derive Expected Calibration Error (ECE) and explain its main practical weakness.**
Strong answer: ECE bins predictions into `M` bins by predicted probability (e.g., 10 bins of width 0.1), and computes `ECE = Σₘ (|Bₘ|/n)·|acc(Bₘ) - conf(Bₘ)|` — the weighted average absolute gap between each bin's average predicted confidence and its actual observed accuracy/positive rate. Its main weakness: it's sensitive to binning choice (number and width of bins) — a model can appear well-calibrated with coarse binning while hiding significant miscalibration within a bin (e.g., a bin with average confidence 0.7 could contain a mix of predictions that are wildly over- and under-confident, canceling out in the bin average) — adaptive binning (equal-count rather than equal-width bins) or continuous alternatives (kernel-based calibration error estimators) partially address this.

**Q4. You're asked to choose a single "north star" evaluation metric for a churn-prediction model that will drive a retention-offer budget allocation. Walk through your reasoning.**
Strong answer: Resist "just use AUC" as the answer — start from the business decision the metric needs to inform: if the model ranks customers for a limited retention-offer budget (e.g., top 5,000 customers get an offer), the relevant metric is closer to **precision at k** (or lift/gain at the specific budget-constrained cutoff) — how many of the top-k ranked customers are actual churners — rather than a threshold-independent aggregate like AUC, which doesn't reflect performance at the *specific* operating point the business will actually use. Report AUC/PR-AUC during model development for general ranking-quality comparison across model versions, but validate and report precision@k (matched to the real budget) as the metric that actually determines whether the model creates business value.

**Q5. Design a monitoring dashboard for a deployed classification model, specifying what you'd track and at what cadence.**
Strong answer: Real-time/daily: prediction-distribution statistics (mean predicted probability, % positive predictions) and input feature drift (PSI per key feature) — available immediately, no label lag. Weekly/as-labels-arrive: realized precision/recall/F1 and calibration metrics (reliability diagram, ECE) against ground truth as it becomes available, compared against the training-time baseline with statistical significance/control-chart-style alerting rather than eyeballing single-point changes. Periodic (monthly/quarterly): full model re-evaluation including subgroup/fairness slicing (performance broken out by relevant segments) and a decision on whether drift has crossed a retraining threshold — tie alerting thresholds to business-impact estimates (e.g., "alert if estimated cost impact of drift exceeds $X"), not just statistical significance alone, so the dashboard drives action rather than just observation.

## Common Pitfalls & Follow-Up Probes

- Reporting a single point-in-time metric as if it's a permanent property of the model, without a monitoring/re-evaluation plan for production drift.
- Conflating "high AUC" with "good, deployable model" without checking calibration, threshold-specific performance, or business-cost alignment.
- Not distinguishing data drift (detectable immediately, unsupervised) from concept drift (requires labels) when designing a monitoring strategy — a very common gap under interview pressure.

## Key Resources

- [Model Drift vs Data Drift in 2026: Detection & Mitigation Guide](https://futureagi.com/blog/model-vs-data-drift-how-to-identify-and-handle-it/)
- Guo et al. (2017), *"On Calibration of Modern Neural Networks"* — the paper that popularized ECE and modern calibration analysis.
- [ML Model Monitoring: Metrics, Drift Tests, Playbooks](https://www.domo.com/glossary/ml-model-monitoring)
