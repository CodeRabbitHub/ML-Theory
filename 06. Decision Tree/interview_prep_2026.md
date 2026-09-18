# Decision Trees — Senior ML Engineer Interview Prep (2026)

> Trees are the atomic unit behind Random Forests, GBM, XGBoost, LightGBM and CatBoost (folders 07–13) — senior interviewers use single-tree questions to test whether you *actually* understand splitting criteria and overfitting mechanics before trusting your answers about the ensembles built on top.

## TL;DR Refresher

- Recursively partitions feature space by choosing, at each node, the (feature, threshold) split that most improves a purity criterion: Gini impurity or entropy/information gain for classification, variance reduction (MSE) for regression.
- Fully grown trees have zero training bias but very high variance (they memorize training data) — controlled via pruning (cost-complexity/CCP), `max_depth`, `min_samples_leaf`, `min_samples_split`.
- Naturally handles mixed feature types, non-linear relationships and interactions, requires no feature scaling, but is unstable (small data changes → very different tree).

## What's New / State of the Art (2025–2026)

- **Single decision trees are almost never deployed alone in 2026** — they're the interview vehicle for testing ensemble intuition (bagging/boosting) rather than a production choice, except where extreme interpretability/auditability is a hard requirement (e.g., a single shallow tree as a regulator-facing decision rule, or a "surrogate tree" distilled from a black-box model for explainability).
- **Surrogate/distillation trees for explaining black-box models** are a live 2025–2026 production pattern: fit a shallow decision tree to approximate a complex model's (e.g., XGBoost or a neural net's) predictions, trading some fidelity for a globally interpretable, presentable decision rule — increasingly required in regulated ML pipelines.
- **Oblique/multivariate trees and learned-split trees** (splits on linear combinations of features rather than single-feature axis-aligned splits) have seen renewed research interest as a middle ground between plain trees and neural approaches, though axis-aligned CART-style trees remain completely dominant in industry practice.
- Modern interview banks in 2026 increasingly pair decision-tree questions directly with the ensemble methods built on them, expecting you to explain *why* an ensemble of high-variance, low-bias trees (bagging) or low-variance, high-bias shallow trees (boosting) is a deliberate design choice, not an accident ([Devinterview.io Decision Tree Q bank 2026](https://github.com/Devinterview-io/decision-tree-interview-questions)).

## Senior-Level Interview Questions

**Q1. Derive/explain Gini impurity vs. entropy — do they usually produce meaningfully different trees?**
Strong answer: Gini `= 1 - Σp_i²`, entropy `= -Σp_i log₂ p_i`; both are concave impurity measures that are 0 for pure nodes and maximized at uniform class distribution. In practice they produce very similar trees — Gini is preferred in CART implementations (e.g., scikit-learn's default) mainly because it avoids the log computation and is marginally faster, not because it's meaningfully more accurate.

**Q2. Why does an unpruned decision tree have (near) zero training error but poor generalization, and what are your levers to fix it?**
Strong answer: A fully grown tree keeps splitting until leaves are pure (or `min_samples_leaf=1`), effectively memorizing training data — each leaf can end up representing a single training point, giving zero bias but very high variance. Levers: pre-pruning (`max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`), post-pruning (cost-complexity pruning via `ccp_alpha`, selected by cross-validated validation-set performance), or move to an ensemble (bagging to reduce variance, boosting to control bias directly with shallow trees).

**Q3. How does a decision tree handle a mix of continuous and categorical features, and what's the complexity of finding the best split for a continuous feature?**
Strong answer: For continuous features, CART sorts values and considers thresholds between consecutive distinct values — an `O(n log n)` sort followed by an `O(n)` scan per feature per node. For categorical features with high cardinality, native handling (as in CatBoost, discussed in folder 13) uses target statistics instead of naive one-hot/exhaustive-subset search, which is `O(2^{k-1})` and intractable for large `k`.

**Q4. Explain feature importance from a decision tree (Gini/MDI importance) and why it can be misleading.**
Strong answer: Mean decrease in impurity (MDI) sums the impurity reduction each feature contributes across all splits, weighted by the number of samples reaching that node. It's biased toward high-cardinality features (more possible split points → more chances to appear to reduce impurity) and computed on training data, so it can overstate importance for features correlated with noise. Prefer permutation importance or SHAP values for more reliable, model-agnostic importance in production reporting.

**Q5. You're asked to build a single interpretable decision rule to explain a production XGBoost model's decisions to a regulator. How do you do it?**
Strong answer: Train a shallow (depth 3–5) "surrogate" decision tree to predict the XGBoost model's *outputs* (not the original labels), using the original features; evaluate its fidelity (agreement rate) with the black-box model on held-out data, and clearly disclose that it's an approximation. Discuss the fidelity/interpretability trade-off explicitly — this is the standard way trees remain relevant for explainability at senior level even when they're not the production model.

**Q6. Why are decision trees so sensitive to small changes in training data, and how does this instability actually become a *feature* rather than a bug at scale?**
Strong answer: High-variance instability comes from the greedy, hierarchical nature of splits — an early split change cascades to every downstream split. This instability is exactly what bagging (Random Forest) exploits: averaging many high-variance, low-correlation trees trained on bootstrap samples cancels out the instability, converting a weakness of a single tree into the source of a strong ensemble's low-variance predictions.

## Common Pitfalls & Follow-Up Probes

- Confusing information gain (used to pick the split) with feature importance (aggregated post-hoc metric).
- Not knowing that trees can overfit even with regularization if `min_samples_leaf` is left at its default (1) on small/noisy data.
- Forgetting that decision trees can't extrapolate beyond the range of training data for regression (predictions are bounded by leaf averages).

## Key Resources

- [Devinterview.io — Decision Tree Interview Questions (2026)](https://github.com/Devinterview-io/decision-tree-interview-questions)
- [Medium — Decision Trees & Random Forests: The Complete Interview Guide (2026)](https://medium.com/@nehasingh890.ns/everything-you-need-to-know-about-decision-trees-random-forests-the-complete-interview-guide-2280b0ec5668)
- Breiman et al., *Classification and Regression Trees* (CART) — still the canonical source for split-criteria theory.
