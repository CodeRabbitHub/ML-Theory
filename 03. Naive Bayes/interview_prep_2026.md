# Naive Bayes — Senior ML Engineer Interview Prep (2026)

> Companion to `naive_bayes.ipynb`. Focuses on why this "old, simple" algorithm still shows up in senior interviews in the LLM era, and what's changed.

## TL;DR Refresher

- Applies Bayes' theorem with the (naive) conditional-independence assumption: `P(y|x₁..xₙ) ∝ P(y)·Π P(xᵢ|y)`.
- Variants: Gaussian NB (continuous features), Multinomial NB (counts, e.g., word frequencies), Bernoulli NB (binary features).
- Extremely fast to train (closed-form counting/MLE), works well with high-dimensional sparse data (text), and is a strong baseline for text classification.

## What's New / State of the Art (2025–2026)

- **"LLMs or Naive Bayes? Old Gems or New Ways" (PEARC 2026 / arXiv 2609.13185)** — a widely discussed 2025–2026 paper directly comparing LLM-based classification against Naive Bayes on text classification tasks, finding NB remains extremely competitive on cost, latency, and in-domain accuracy for well-defined classification tasks, while LLMs win on zero-shot generalization to unseen label schemas. This is now a live interview talking point: *"when do you not need an LLM?"* ([arXiv:2609.13185](https://arxiv.org/abs/2609.13185)).
- **NB as a fast pre-filter in LLM pipelines.** A common 2025–2026 production pattern: use a cheap NB (or logistic regression) classifier to triage/route requests (e.g., "does this need the expensive LLM call at all?") before invoking a large model — cost-optimization architecture that senior candidates are expected to know about.
- **Calibration and use as a generative baseline for semi-supervised learning** (EM-based NB with unlabeled data) remains relevant in low-label-budget settings, which are increasingly common where labeling is done by LLMs and needs a cheap sanity-check model.
- NB continues to be the textbook example for **spam filtering, sentiment triage, and streaming/online classification** because its parameters (class priors, conditional likelihoods) update in O(1) per new example — no retraining pass needed, a genuine advantage over gradient-based models for real-time systems.

## Senior-Level Interview Questions

**Q1. The "naive" independence assumption is almost always false in real data. Why does Naive Bayes still work well in practice?**
Strong answer: NB only needs to get the *argmax* class right, not accurate probability estimates — even with correlated features, the assumption's errors often cancel out in a way that preserves the correct ranking of classes (Domingos & Pazzani's classic result). It's a classification-optimal, not probability-optimal, classifier in many regimes.

**Q2. Compare Gaussian, Multinomial, and Bernoulli NB — how do you choose, and what happens if you use the wrong one?**
Strong answer: Multinomial NB models word/feature *counts* (good for bag-of-words with frequency signal); Bernoulli NB models feature *presence/absence* (good for short texts where repeated words carry less signal); Gaussian NB assumes continuous features are normally distributed per class. Using Gaussian NB on skewed count data, or Bernoulli on frequency-heavy data, systematically miscalibrates likelihoods and hurts accuracy — always match the likelihood family to the actual feature-generating process.

**Q3. How do you handle the zero-frequency problem, and what's the effect of different smoothing values?**
Strong answer: Laplace/Lidstone smoothing adds `α` (typically 1, or tuned) to every count so unseen feature-class combinations don't zero out the whole product. Larger `α` → more uniform/regularized likelihoods (more bias, less variance); `α` should be tuned via cross-validation, especially with small class-conditional sample sizes.

**Q4. Design a system where Naive Bayes is used to cost-optimize an LLM-based classification pipeline.**
Strong answer: Train a lightweight NB (or logistic regression) on historical labeled data as a fast first-pass classifier; route high-confidence predictions directly, and only forward low-confidence/ambiguous cases to the expensive LLM. Monitor NB/LLM agreement rate over time to detect drift and decide when to retrain NB or adjust the confidence threshold — a real 2025–2026 cost-engineering pattern for text classification at scale.

**Q5. Why is Naive Bayes well-suited to online/streaming learning, and how would you implement incremental updates?**
Strong answer: All you need to update are class priors and per-feature-per-class counts (or running mean/variance for Gaussian NB) — no gradient computation or full retraining pass is required, so each new labeled example can update the model in O(features) time. This makes it attractive for real-time spam/abuse filters that must adapt within seconds of new labels arriving.

**Q6. When would you *not* use Naive Bayes, and what would you replace it with?**
Strong answer: When feature interactions are strong and predictive (NB can't model them), when you need calibrated probabilities out of the box (logistic regression is generally better calibrated), or when you have enough data/compute that a discriminative model (logistic regression, gradient boosting, or a fine-tuned transformer) clearly outperforms it — NB is a baseline/production-efficiency tool, not usually the final answer for complex tasks.

## Common Pitfalls & Follow-Up Probes

- Forgetting to work in log-space (`log P(y) + Σ log P(xᵢ|y)`) to avoid numerical underflow with many features.
- Not knowing NB is a generative model (models `P(x,y)`) vs. logistic regression's discriminative approach (models `P(y|x)` directly) — a classic paired interview question (Ng & Jordan's NB vs. LR asymptotic analysis).
- Ignoring feature independence violations that *do* matter (e.g., highly duplicated/redundant features double-counting evidence).

## Key Resources

- [LLMs or Naive Bayes? Old Gems or New Ways — arXiv 2609.13185 (2026)](https://arxiv.org/abs/2609.13185)
- [FutureAGI — Naive Bayes Models Glossary (2026)](https://futureagi.com/glossary/naive-bayes-models/)
- Ng & Jordan, *"On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes"* — still the canonical theory reference.
