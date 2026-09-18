# Clustering Methods — Senior ML Engineer Interview Prep (2026)

> Overview doc for folders 15–17 (KMeans, DBSCAN, Mean Shift). Use this to frame the "which clustering algorithm and why" decision-tree before drilling into any one algorithm's internals.

## TL;DR Refresher

- Clustering is unsupervised partitioning of data into groups such that within-group similarity is high and between-group similarity is low, with no ground-truth labels to optimize against directly.
- Major families: **centroid-based** (KMeans — assumes convex, roughly equal-sized, spherical clusters), **density-based** (DBSCAN/HDBSCAN — finds arbitrary-shaped clusters, naturally handles noise/outliers, no need to specify `k`), **hierarchical** (agglomerative/divisive — produces a dendrogram, no fixed `k` required upfront), and **mode-seeking** (Mean Shift — finds density modes via gradient ascent, no `k` needed, but sensitive to bandwidth).
- Evaluation without labels: internal metrics (silhouette score, Davies-Bouldin, Calinski-Harabasz) when no ground truth exists; external metrics (ARI, NMI, homogeneity/completeness) when a reference labeling is available for validation.

## What's New / State of the Art (2025–2026)

- **HDBSCAN has become the de facto "upgrade path" from both KMeans and DBSCAN** in 2025–2026 production and research usage — it removes DBSCAN's single global `eps` limitation (handles clusters of varying density) while keeping DBSCAN's core advantages (arbitrary shapes, automatic noise detection, no need to pre-specify `k`), and is now the default recommendation in most modern clustering-comparison writeups ([HDBSCAN vs DBSCAN, 2025](https://blog.dailydoseofds.com/p/hdbscan-vs-dbscan), [HDBSCAN: the supercharged version of DBSCAN](https://www.dailydoseofds.com/hdbscan-the-supercharged-version-of-dbscan-an-algorithmic-deep-dive/)).
- **Clustering on learned embeddings, not raw features, is now the dominant real-world pattern** — with embeddings from foundation models widely available (text, image, tabular), 2025–2026 clustering pipelines typically embed first (often followed by a dimensionality-reduction step — see folder 27) and cluster in that space, which changes practical considerations (curse of dimensionality management, choice of distance metric — cosine vs. Euclidean) versus classic "cluster the raw tabular features" examples.
- **Clustering for LLM-era use cases** — topic discovery over document embeddings, deduplication of near-identical training data at scale, user/behavioral segmentation on top of learned representations, and anomaly/outlier discovery in monitoring pipelines are the concrete 2025–2026 production contexts where clustering algorithm choice is actively debated in interviews, more than classic "customer segmentation on 5 numeric features" toy examples.
- **Scalability**: mini-batch KMeans, and GPU-accelerated clustering (`cuML`) remain the practical answer for clustering at the scale modern embedding pipelines produce (millions of high-dimensional vectors) — a senior candidate should know these exist and roughly why (avoiding full-dataset passes per iteration).

## Senior-Level Interview Questions

**Q1. Give a decision framework: given a new clustering problem, how do you choose between KMeans, DBSCAN/HDBSCAN, and hierarchical clustering?**
Strong answer: Ask three questions — (1) Do you know/can you estimate `k` in advance, and are clusters expected to be roughly convex/spherical and similar size? → KMeans is fast and a reasonable default. (2) Are clusters likely arbitrary-shaped, of varying density, with meaningful noise/outlier points that shouldn't be forced into a cluster? → DBSCAN/HDBSCAN. (3) Do you need a multi-resolution view (a dendrogram to cut at different granularities) or a small dataset where `O(n²)`/`O(n³)` cost is acceptable? → Hierarchical clustering. In 2025–2026 practice, HDBSCAN has absorbed much of DBSCAN's use case while removing its main weakness (single global density threshold), so it's often the practical first choice over plain DBSCAN.

**Q2. Without ground-truth labels, how do you evaluate whether a clustering result is "good," and what are the failure modes of each metric?**
Strong answer: Silhouette score (per-point ratio of intra- vs. nearest-inter-cluster distance, averaged) is the most common internal metric but assumes convex clusters and penalizes DBSCAN-style arbitrary shapes unfairly; Davies-Bouldin and Calinski-Harabasz have similar convexity assumptions. If any partial ground truth or proxy labels exist (even a small labeled sample), external metrics (Adjusted Rand Index, Normalized Mutual Information) are more trustworthy. In practice, combine quantitative metrics with qualitative inspection (visualize via UMAP/t-SNE, manually review sample points per cluster) — no single metric is fully trustworthy for validating unsupervised results.

**Q3. Your clustering pipeline embeds documents with a transformer model, reduces dimensionality, then clusters. Walk through the key design decisions at each stage.**
Strong answer: Embedding stage — choose a model whose training objective matches your notion of similarity (e.g., a sentence-embedding model trained with contrastive/cosine objectives, so cosine distance is the "native" metric downstream, as discussed in the KNN doc). Dimensionality-reduction stage — UMAP is generally preferred over PCA before density-based clustering because it better preserves local neighborhood structure that density-based methods rely on (though it distorts global distances, which matters if you need genuinely meaningful inter-cluster distances, not just separation). Clustering stage — HDBSCAN is a strong default on the reduced embedding space since embedding clusters are rarely uniform density or convex.

**Q4. A stakeholder wants exactly 8 customer segments for a marketing campaign, but your data doesn't naturally partition into 8 clean clusters. How do you handle this tension?**
Strong answer: Be explicit that this is a business-constraint-vs-data-structure conflict, not purely a technical problem: KMeans with `k=8` will produce 8 clusters regardless of whether the data actually supports 8 distinct groups (it will split a genuinely uniform region or merge genuinely distinct sub-populations to satisfy `k`). Recommend running an unconstrained method (HDBSCAN, or KMeans across a range of `k` with the elbow/silhouette method) first to characterize the data's *actual* structure, present that finding to the stakeholder, and then decide together whether to force `k=8` (accepting some artificial splits/merges) or negotiate the segment count based on what the data supports.

**Q5. Explain why clustering evaluation metrics like silhouette score can mislead you when comparing KMeans against DBSCAN/HDBSCAN results on the same dataset.**
Strong answer: Silhouette score implicitly rewards convex, well-separated, roughly spherical clusters (it's essentially measuring how KMeans-friendly the result is) — so comparing a KMeans result against a DBSCAN result using silhouette score is comparing them on a metric structurally biased toward KMeans's assumptions, and DBSCAN's noise points (label -1) further complicate the metric's interpretation. A fairer comparison uses density-based evaluation metrics (e.g., DBCV — Density-Based Clustering Validation) alongside domain-specific/business validation, not a single geometry-biased internal metric.

## Common Pitfalls & Follow-Up Probes

- Treating "clustering" as a single algorithm rather than a family with fundamentally different assumptions about cluster shape/density — always name the assumption a chosen algorithm makes.
- Forgetting that clustering results are highly sensitive to feature scaling and distance-metric choice (see folder 18) — this is nearly always a fair follow-up probe.
- Not distinguishing clustering (unsupervised grouping) from classification (supervised) when asked to justify why you didn't just "train a classifier" — a genuinely common confusion to preempt.

## Key Resources

- [HDBSCAN vs. DBSCAN (2025)](https://blog.dailydoseofds.com/p/hdbscan-vs-dbscan)
- [Advanced Customer Segmentation: HDBSCAN, DBSCAN, and K-Means Compared](https://pub.towardsai.net/advanced-customer-segmentation-a-comprehensive-comparison-of-hdbscan-dbscan-and-k-means-ce3bcaa7f1a2)
- Moulavi et al. (2014), *"Density-Based Clustering Validation"* (DBCV) — the metric to cite for fair density-based-clustering evaluation.
