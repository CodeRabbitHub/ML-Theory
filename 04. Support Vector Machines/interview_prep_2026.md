# Support Vector Machines — Senior ML Engineer Interview Prep (2026)

> Focuses on where SVMs genuinely still fit in a 2026 ML stack, and the depth senior interviewers expect on margins, kernels, and duals.

## TL;DR Refresher

- Finds the maximum-margin hyperplane separating classes: `min ½‖w‖² + C·Σξᵢ` subject to `yᵢ(wᵀxᵢ+b) ≥ 1-ξᵢ`, `ξᵢ ≥ 0` (soft margin).
- The dual formulation expresses the solution purely in terms of dot products, enabling the **kernel trick** (RBF, polynomial, sigmoid) to implicitly map to high-/infinite-dimensional feature spaces without computing the mapping explicitly.
- Only support vectors (points on/inside the margin) determine the boundary — a sparse, memory-efficient decision rule at inference time.

## What's New / State of the Art (2025–2026)

- **SVMs have receded from "default model" status but remain the interview-favorite for testing optimization/duality intuition**, precisely because gradient boosting and tabular foundation models (TabPFN, see the GBM/XGBoost docs) now dominate tabular benchmarks and deep learning dominates unstructured data. Senior interviews use SVM questions less to test "would you use this in production" and more to test whether you deeply understand convex optimization, margins, and the bias-variance trade-off via `C` and kernel bandwidth `γ` — this shift in framing is itself worth naming out loud if asked "would you use an SVM today?"
- **Where SVMs still win in 2025–2026 production systems:** small-to-medium, high-dimensional, low-sample-count problems — genomics/bioinformatics (gene expression classification), some cybersecurity/malware-detection pipelines, and one-class SVM for anomaly/novelty detection where labeled anomalies are scarce. Medical/health-data literature continues to publish SVM applications for exactly this small-n, high-p regime ([Cureus 2025 review of SVM in public-health mass-data analysis](https://www.cureus.com/articles/325571-support-vector-machines-a-literature-review-on-their-application-in-analyzing-mass-data-for-public-health.pdf)).
- **Kernel methods research has partly merged with deep learning theory** — the Neural Tangent Kernel (NTK) framework shows infinitely-wide neural networks behave like kernel machines, and this connection is an increasingly common "bridge" question at senior/staff level to test whether you see SVM/kernel theory and deep learning as related, not separate, paradigms.
- **GPU-accelerated and approximate SVM solvers** (e.g., ThunderSVM, cuML's SVM) make SVMs usable on larger datasets than the classical `O(n²)`–`O(n³)` QP solvers allowed, but they're still rarely chosen over linear models + good features or gradient boosting for tabular data at real "big data" scale.

## Senior-Level Interview Questions

**Q1. Derive why maximizing the margin is equivalent to minimizing `‖w‖²`, and explain the role of support vectors.**
Strong answer: Margin width is `2/‖w‖`, so maximizing margin ⟺ minimizing `‖w‖` (or `½‖w‖²` for a smooth, convex QP). At the optimum, only points with non-zero Lagrange multipliers `αᵢ` (the support vectors — on or inside the margin) affect `w = Σαᵢyᵢxᵢ`; all other points could be removed without changing the boundary.

**Q2. Explain the kernel trick and why it lets you work in infinite-dimensional feature spaces without ever computing the mapping.**
Strong answer: The dual objective and decision function depend on data only through inner products `xᵢᵀxⱼ`. A kernel `K(xᵢ,xⱼ) = φ(xᵢ)ᵀφ(xⱼ)` computes that inner product directly in the (possibly infinite-dimensional) transformed space in closed form — e.g., the RBF kernel `exp(-γ‖xᵢ-xⱼ‖²)` corresponds to an infinite-dimensional feature map — without ever materializing `φ(x)`. Must satisfy Mercer's condition (positive semi-definite kernel matrix) to be valid.

**Q3. How do `C` and `γ` (RBF bandwidth) interact, and how would you tune them?**
Strong answer: `C` controls the margin/misclassification trade-off (small `C` → wider margin, more tolerance for violations, more bias; large `C` → tries to classify all training points correctly, more variance/overfitting risk). `γ` controls how far the influence of a single training example reaches (small `γ` → smoother, more bias; large `γ` → tighter fit around points, more variance). They interact multiplicatively — always grid/random/Bayesian-search them jointly (e.g., a log-scale grid), never independently, since the optimal `C` shifts with `γ`.

**Q4. Why is SVM training `O(n²)`–`O(n³)` in practice, and how does that shape your production decision?**
Strong answer: The dual QP involves an `n×n` kernel (Gram) matrix; standard SMO-style solvers scale roughly quadratically to cubically with `n`. This is the main reason SVMs are avoided for very large `n` in production — you'd reach for a linear SVM (`liblinear`, which scales near-linearly) if the data is linearly separable-ish, or switch to gradient boosting/neural nets, rather than push a kernel SVM to millions of rows.

**Q5. Compare a linear SVM to logistic regression — when would each be preferred?**
Strong answer: Both fit linear decision boundaries; SVM optimizes hinge loss (margin-maximizing, sparse support-vector solution, no native probability output) while logistic regression optimizes log-loss (probabilistic, calibrated). Prefer SVM when you only need the decision boundary and want maximum-margin robustness with fewer assumptions; prefer logistic regression when you need well-calibrated probabilities or interpretable odds ratios (as in the Logistic Regression doc).

**Q6. What is one-class SVM and where would you use it today?**
Strong answer: Learns a boundary enclosing "normal" data by finding a maximum-margin hyperplane separating the data from the origin in feature space (or a minimum-enclosing hypersphere, SVDD) — used for novelty/anomaly detection when you have plentiful normal examples but few or no labeled anomalies, e.g., fraud/intrusion detection cold-start before enough labeled fraud exists to train a supervised classifier.

## Common Pitfalls & Follow-Up Probes

- Forgetting SVMs require feature scaling (kernel/margin computations are scale-sensitive) — a very common "gotcha" follow-up.
- Confusing soft-margin slack variables with regularization direction (larger `C` = *less* regularization, opposite of Ridge's `λ`).
- Not knowing SVMs don't natively output probabilities — Platt scaling (fitting a sigmoid on the decision function) is needed, and it's a post-hoc approximation, not a first-class output.

## Key Resources

- [Cureus (2025) — SVM literature review, public health mass-data analysis](https://www.cureus.com/articles/325571-support-vector-machines-a-literature-review-on-their-application-in-analyzing-mass-data-for-public-health.pdf)
- [FutureAGI — Support Vector Machines Glossary (2026)](https://futureagi.com/glossary/support-vector-machines/)
- Jacot et al., *"Neural Tangent Kernel"* — for the kernel/deep-learning bridge question at staff level.
