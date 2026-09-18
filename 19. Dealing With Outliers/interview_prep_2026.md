# Dealing With Outliers — Senior ML Engineer Interview Prep (2026)

> Focus: outlier/anomaly detection as its own discipline in 2025–2026, not just a preprocessing footnote before model training.

## TL;DR Refresher

- Statistical/univariate methods: z-score threshold (`|z|>3`), IQR method (`Q1-1.5·IQR`, `Q3+1.5·IQR`) — simple, fast, but only catch single-feature outliers and assume roughly unimodal/symmetric distributions.
- Multivariate methods: Mahalanobis distance (accounts for feature correlation — see folder 18), Isolation Forest (isolates anomalies via random recursive partitioning — anomalies require fewer splits to isolate), Local Outlier Factor (LOF — density-based, flags points with much lower local density than their neighbors), One-Class SVM (folder 04).
- Treatment options once detected: remove, cap/winsorize, transform (log/Box-Cox to reduce skew's influence), impute, or — increasingly the 2025–2026-preferred default — **keep and flag**, letting a downstream robust model or explicit anomaly-score feature handle them rather than deleting potentially informative data.

## What's New / State of the Art (2025–2026)

- **Isolation Forest remains the dominant general-purpose production anomaly detector in 2025–2026**, and continues to see active research extending it: **Deep Isolation Forest** (representation learning combined with isolation-based scoring for better performance on complex, high-dimensional, non-linear data) and **Online/streaming Isolation Forest** (updating the isolation-tree ensemble incrementally as new data arrives, rather than requiring full retraining) are both concrete 2025–2026-relevant extensions worth naming when asked "how would you improve on vanilla Isolation Forest?" ([Online Isolation Forest, arXiv 2505.09593](https://arxiv.org/abs/2505.09593), [Deep Isolation Forest, IEEE](https://ieeexplore.ieee.org/document/10108034/)).
- **The "delete outliers" default is increasingly discouraged in 2025–2026 practice** in favor of explicit anomaly-aware modeling — deleting outliers silently assumes they're all noise/errors, but in fraud, health, and industrial-monitoring domains the outliers are frequently the entire point of the model; senior candidates are expected to ask "is this outlier noise or signal?" before defaulting to removal.
- **Federated and edge anomaly detection** — Isolation Forest's lightweight, tree-based nature makes it a popular choice for on-device/edge anomaly detection (e.g., IoT sensor monitoring) in 2025–2026 federated-learning architectures, where you can't centralize raw data but still need local anomaly flags ([Federated Learning-Based Anomaly Detection with Isolation Forest in IoT-Edge, ACM 2025](https://dl.acm.org/doi/10.1145/3702995)).
- **Outlier detection as a first-class MLOps monitoring signal**, not just a training-time preprocessing step — production systems increasingly run anomaly detection continuously on live feature distributions as part of data-quality/drift monitoring (overlapping with the Evaluation Metrics doc's drift-detection discussion), catching upstream pipeline bugs before they corrupt model predictions.

## Senior-Level Interview Questions

**Q1. Explain how Isolation Forest detects anomalies, and why it doesn't need a distance or density metric the way LOF or DBSCAN do.**
Strong answer: Isolation Forest builds an ensemble of random trees, each recursively splitting the data on a randomly chosen feature and random split value; the core insight is that anomalies — being few and different — get **isolated into their own leaf in fewer splits** than normal points, which require many splits to separate from the dense "normal" region surrounding them. The anomaly score is derived from the average path length (number of splits) to isolate each point across all trees, normalized against the expected path length for a given sample size — no distance computation or density estimation is needed, which is precisely why it scales well (roughly `O(n log n)`) to high-dimensional, large datasets where distance-based methods (LOF, Mahalanobis) become expensive or less reliable.

**Q2. Compare Isolation Forest, LOF, and One-Class SVM — when would you choose each?**
Strong answer: **Isolation Forest** — best general-purpose default: fast, scales well, handles high dimensionality reasonably, doesn't need a distance metric. **LOF** — better when anomalies are only anomalous *relative to a local neighborhood* (varying-density data, similar to why DBSCAN beats a single global threshold) — e.g., a point that's normal in a sparse region but would be anomalous in a dense one; more expensive (`O(n²)`-ish without indexing) and less scalable. **One-Class SVM** — best when you have a reasonably well-behaved, roughly convex "normal" region and limited data, and want a margin-based boundary rather than a score; more sensitive to kernel/hyperparameter tuning and less scalable to large `n` than Isolation Forest.

**Q3. A junior teammate proposes removing all points with `|z-score| > 3` from the training set before training a fraud model. What's your feedback?**
Strong answer: Push back directly — in a fraud-detection context, extreme/outlier feature values (unusual transaction amounts, atypical timing patterns) are very often exactly the *signal* the model needs to learn, not noise to discard; blanket z-score removal risks deleting the minority-class (fraud) examples disproportionately, since fraud often manifests as statistical outliers by nature — this would actively harm the model's ability to detect the thing it's built to detect. Recommend instead: investigate whether flagged points are actual anomalies vs. genuine rare-but-valid patterns, keep them in training (fraud detection is class-imbalance territory — see folders 20–21), and consider outlier scores as an additional *feature* rather than a filter.

**Q4. Design a real-time anomaly-monitoring system for a live feature pipeline feeding a production model, using Isolation Forest.**
Strong answer: Train Isolation Forest on a representative historical window of "known-good" feature distributions; score incoming feature vectors in real time (Isolation Forest scoring is fast — no retraining needed per prediction); alert when the anomaly-score distribution over a rolling window shifts significantly (e.g., using online/incremental Isolation Forest updates or periodic retraining on recent windows to adapt to legitimate distribution drift without becoming stale); pair with per-feature statistical drift checks (PSI) as a complementary, more interpretable signal, since Isolation Forest's aggregate anomaly score doesn't by itself explain *which* feature(s) drove the flag — SHAP-style attribution on the isolation score, or per-feature drift metrics, fill that gap.

**Q5. When is Mahalanobis distance a better outlier-detection choice than Isolation Forest, and when is it worse?**
Strong answer: Better when the "normal" data is genuinely well-approximated by a multivariate Gaussian (or you've transformed it to be so) and you specifically want to detect multivariate correlation-breaking anomalies (see the concrete height/weight example in the Distances & Feature Scaling doc) with an interpretable, closed-form distance. Worse when the true data distribution is strongly multi-modal or non-Gaussian (Mahalanobis distance assumes a single elliptical "normal" region defined by one global mean/covariance, so it fails on data with multiple legitimate clusters of normal behavior) or when `p` (features) is large relative to `n` (covariance matrix `Σ` becomes poorly estimated/singular) — Isolation Forest makes no such distributional assumption and handles these cases more gracefully.

## Common Pitfalls & Follow-Up Probes

- Defaulting to "just remove outliers" without first asking whether they represent noise or the actual signal of interest (fraud, disease, failure events).
- Computing z-score/IQR outlier detection independently per feature when the true anomaly is only visible in feature *combinations* — a cue to bring up Mahalanobis distance or Isolation Forest.
- Not knowing Isolation Forest's anomaly score requires a chosen contamination rate/threshold to convert to a binary label — the threshold choice itself is a business decision, not a purely statistical one.

## Key Resources

- Liu, Ting & Zhou (2008), *"Isolation Forest"* (ICDM) — the original paper.
- [Online Isolation Forest (2025)](https://arxiv.org/abs/2505.09593)
- [Deep Isolation Forest for Anomaly Detection (IEEE)](https://ieeexplore.ieee.org/document/10108034/)
- Breunig et al. (2000), *"LOF: Identifying Density-Based Local Outliers"* — the LOF paper.
