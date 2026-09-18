# Senior ML Engineer Interview Prep — 2026 Edition

This is a companion layer on top of the theory notebooks already in this repo. Every topic folder now has an **`interview_prep_2026.md`** alongside its notebook, covering:

- **What's New / State of the Art (2025–2026)** — how the field has moved since the original theory notebook was written, with sourced references.
- **Senior-Level Interview Questions** — 5–7 questions per topic with full model answers, pitched at the depth expected of a senior/staff ML engineer (derivations, trade-off reasoning, production judgment), not entry-level definitions.
- **Production / System-Design Angle** — where relevant, how the topic shows up in real infrastructure decisions.
- **Common Pitfalls & Follow-Up Probes** — the traps interviewers use to test whether your understanding is genuinely deep or memorized.
- **Key Resources** — papers and 2025–2026 sources worth reading further.

None of this replaces the original notebooks — read those first for the core theory, then use these docs to update your mental model to 2026 and rehearse how a senior candidate would actually be expected to answer.

## Suggested Study Order

**1. Classical Supervised Learning** — the foundation every later topic builds on.
1. [Linear Regression](<01. Linear Regression/interview_prep_2026.md>)
2. [Logistic Regression](<02. Logistic Regression/interview_prep_2026.md>)
3. [Naive Bayes](<03. Naive Bayes/interview_prep_2026.md>)
4. [Support Vector Machines](<04. Support Vector Machines/interview_prep_2026.md>)
5. [K-Nearest Neighbors](<05. K Nearest Neighbour/interview_prep_2026.md>) — now essentially "approximate nearest neighbor / vector search theory" in 2026.

**2. Trees & Ensembles** — where most 2026 tabular-ML interviews spend the most time.
6. [Decision Trees](<06. Decision Tree/interview_prep_2026.md>)
7. [Random Forest](<07. Random Forest/interview_prep_2026.md>)
8. [Bagging vs. Boosting](<08. Bagging And Boosting/interview_prep_2026.md>) — read this before AdaBoost/GBM as the conceptual bridge.
9. [AdaBoost](<09. AdaBoost/interview_prep_2026.md>)
10. [Gradient Boosting Machine](<10. Gradient Boosting Machine/interview_prep_2026.md>) — the "derive it from scratch" topic.
11. [XGBoost](<11. XGBoost/interview_prep_2026.md>)
12. [LightGBM](<12. LightGBM/interview_prep_2026.md>)
13. [CatBoost](<13. CatBoost/interview_prep_2026.md>)

**3. Unsupervised Learning / Clustering** — now framed heavily around embeddings.
14. [Clustering Methods (overview)](<14. Clustering Methods/interview_prep_2026.md>) — read first for the algorithm-selection framework.
15. [KMeans](<15. KMeans/interview_prep_2026.md>)
16. [DBSCAN](<16. DBSCAN/interview_prep_2026.md>) — includes the HDBSCAN upgrade path.
17. [Mean Shift](<17. Mean Shift/interview_prep_2026.md>)

**4. Data & Modeling Fundamentals** — the topics that cut across every model above.
18. [Distances & Feature Scaling](<18. Distances & Feature Scaling/interview_prep_2026.md>)
19. [Dealing With Outliers](<19. Dealing With Outliers/interview_prep_2026.md>)
20. [Evaluating Imbalanced Classes](<20. Evaluating Imbalance Class/interview_prep_2026.md>)
21. [Handling Imbalanced Datasets](<21. Handling Imbalance Dataset/interview_prep_2026.md>)
22. [Hyperparameter Tuning](<22. Hyperparameter Tuning/interview_prep_2026.md>)
23. [Loss Functions](<23. Loss Functions/interview_prep_2026.md>)
24. [Evaluation Metrics](<24. Evaluation Metrics/interview_prep_2026.md>)
25. [Feature Engineering](<25. Feature Engineering/interview_prep_2026.md>) — covers LLM-assisted feature engineering, a genuinely new 2025–2026 topic.
26. [Feature Selection](<26. Feature Selection/interview_prep_2026.md>)
27. [Reducing Dimensionality](<27. Reducing Dimensionality/interview_prep_2026.md>)

## The Big 2025–2026 Themes That Cut Across This Repo

A few threads recur across many of the individual docs — worth internalizing as talking points that show you understand the *current* landscape, not just 2020-era ML theory:

- **Tabular foundation models (TabPFN and successors)** are now a real third option alongside Random Forest and gradient boosting for small-to-medium tabular data, with zero tuning required — see the Random Forest, GBM, and XGBoost docs for the current nuance on when they compete and when classic ensembles still win.
- **Everything KNN/clustering-related now runs through embeddings.** HNSW-based approximate nearest neighbor search powers every RAG/vector-database system; HDBSCAN has become the default upgrade over both KMeans and plain DBSCAN for embedding-space clustering.
- **LLMs are now participants in the ML pipeline, not just the model being built** — LLM-assisted/automated feature engineering (LLM-FE and similar) and cheap classifiers (Naive Bayes, logistic regression) as cost-optimizing pre-filters in front of expensive LLM calls are both concrete, current patterns.
- **Calibration and drift monitoring have become first-class, continuous concerns**, not one-time offline checks — this shows up in the Logistic Regression, XGBoost, Evaluation Metrics, and Evaluating Imbalanced Classes docs.
- **The "beyond SMOTE" shift**: class-weighting and focal loss are now generally preferred as the first response to class imbalance, with resampling techniques as a secondary tool rather than the default.
- **Explainability (SHAP/TreeSHAP) is treated as mandatory infrastructure**, not an optional add-on, for any tree-ensemble model in a regulated or customer-facing production context.

## How to Use This for Interview Prep

1. Read the original notebook for a topic to refresh the core math.
2. Read that topic's `interview_prep_2026.md`, focusing on the "What's New" section to update your framing.
3. Cover the model answers, then try to reconstruct each answer from memory out loud — the derivations (gain formulas, gradient derivations, bias-variance arguments) are exactly what separates a senior answer from a memorized definition.
4. Use the "Common Pitfalls" sections as a checklist of things interviewers specifically probe for.
