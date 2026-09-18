# KMeans — Senior ML Engineer Interview Prep (2026)

> See folder 14 for the broader "which clustering algorithm" framing. This doc drills into KMeans mechanics, its known failure modes, and its 2025–2026 role in embedding-heavy pipelines.

## TL;DR Refresher

- Iterative algorithm (Lloyd's algorithm): (1) initialize `k` centroids, (2) assign each point to its nearest centroid, (3) recompute centroids as the mean of assigned points, (4) repeat until convergence — minimizes within-cluster sum of squares (WCSS/inertia): `Σₖ Σ_{x∈Cₖ} ‖x - μₖ‖²`.
- Guaranteed to converge (WCSS monotonically decreases each iteration) but only to a **local** optimum — sensitive to initialization, hence `k-means++` seeding (probabilistically spreads initial centroids apart) is the standard default over random initialization.
- Assumes convex, roughly-equal-variance, roughly-equal-sized clusters — fundamentally a Voronoi-partition method, so it struggles with non-convex shapes, varying cluster density/size, and is sensitive to outliers (means are outlier-sensitive, unlike medians).

## What's New / State of the Art (2025–2026)

- **Mini-batch KMeans remains the standard way to scale to millions of high-dimensional embedding vectors** — instead of using the full dataset per iteration, it updates centroids using small random batches, trading a small amount of solution quality for a large reduction in compute/memory, which is exactly the trade-off needed for 2025–2026-scale embedding clustering pipelines.
- **KMeans as an embedding-space quantizer, not just a segmentation tool** — a distinctly 2025–2026 usage pattern: KMeans (or its vector-quantization cousins) is used inside vector-database indexing (IVF indexes bucket vectors by nearest KMeans centroid before ANN search — see the KNN doc) and inside model-compression pipelines (weight/activation quantization) — a senior candidate who only frames KMeans as "customer segmentation" is missing a large, current chunk of where it's actually used in production ML infrastructure today.
- **`k-means++` and elbow-method selection remain standard**, but 2025–2026 practice increasingly pairs KMeans with silhouette-based or gap-statistic-based `k` selection rather than eyeballing an elbow plot alone, especially in automated/pipeline settings where no human is reviewing every clustering run.
- HDBSCAN (folder 14/16) has captured much of KMeans's traditional "exploratory segmentation" use case where `k` isn't known in advance, narrowing KMeans's role toward cases where speed, scale, or a genuinely known/fixed `k` (e.g., quantization into a fixed codebook size) matter more than shape flexibility.

## Senior-Level Interview Questions

**Q1. Prove that Lloyd's algorithm (KMeans) is guaranteed to converge, and explain why it isn't guaranteed to find the global optimum.**
Strong answer: Both steps of each iteration can only decrease (or keep equal) the WCSS objective — reassigning each point to its nearest centroid can't increase its contribution to WCSS (by definition of "nearest"), and recomputing each centroid as the cluster mean is the exact minimizer of within-cluster squared distance for a fixed assignment. Since WCSS is bounded below by 0 and monotonically non-increasing over a finite number of possible assignments, the algorithm must converge (assignments stop changing) in finite steps. It's not guaranteed global-optimal because WCSS is non-convex in the joint (assignment, centroid) space — different initializations converge to different local optima, which is exactly why `k-means++` and multiple random restarts (`n_init`) are standard practice.

**Q2. Explain `k-means++` initialization and why it improves on random initialization.**
Strong answer: Instead of picking `k` initial centroids uniformly at random (which can accidentally place multiple centroids close together, wasting cluster capacity), `k-means++` picks the first centroid uniformly at random, then each subsequent centroid with probability proportional to its squared distance from the nearest already-chosen centroid — this biases initialization toward spreading centroids apart across the data's actual spread, provably giving an `O(log k)`-competitive expected approximation to the optimal WCSS, and empirically converges faster and to better local optima than random init.

**Q3. Walk through 2–3 methods for choosing `k`, and their respective weaknesses.**
Strong answer: **Elbow method** — plot WCSS vs. `k`, look for the point of diminishing returns; subjective, the "elbow" is often ambiguous. **Silhouette score** — average per-point silhouette across `k` values, pick the max; more principled but assumes convex clusters (same bias noted in the Clustering Methods doc) and is `O(n²)` to compute naively. **Gap statistic** — compares WCSS against WCSS under a null reference distribution (uniform random data), picks `k` where the gap is maximized; more statistically grounded but computationally heavier (needs multiple reference-dataset clusterings). In practice, combine at least two of these plus domain judgment — no single method is fully reliable alone.

**Q4. Why does KMeans fail on non-convex clusters (e.g., two concentric rings), and how would you detect this failure in a real pipeline before it causes a downstream problem?**
Strong answer: KMeans partitions space into Voronoi cells around centroids — any cluster shape that isn't well-approximated by "closest centroid" (like concentric rings, elongated/curved shapes) gets sliced incorrectly, because the algorithm has no concept of connectivity or density, only distance-to-centroid. Detect this by visualizing clusters (via PCA/UMAP projection) rather than trusting WCSS/silhouette alone, since both metrics can look "acceptable" even on a badly-mismatched clustering if you don't visually inspect cluster shapes — a good practical habit to state explicitly in an interview.

**Q5. You're using KMeans to build an IVF index for approximate nearest-neighbor search over 10M embedding vectors. What's KMeans's actual role here, and what would go wrong if you picked a bad `k`?**
Strong answer: KMeans partitions the vector space into `k` Voronoi cells (each with a centroid, the "inverted list" bucket); at query time, you search only the nearest few centroids' buckets instead of the whole dataset — this is IVF's core speed trick, and it's a direct, concrete production use of KMeans distinct from "clustering as the end deliverable." Too small a `k` → each bucket is too large, search is still slow (not enough pruning); too large a `k` → each bucket may miss true nearest neighbors that fall just across a Voronoi boundary in a neighboring cell (recall drops), and index build/maintenance cost grows — `k` here is a latency/recall trade-off knob, tuned empirically against a labeled recall benchmark, not chosen via silhouette score.

## Common Pitfalls & Follow-Up Probes

- Forgetting KMeans requires feature scaling (unscaled features with different magnitudes distort the Euclidean-distance-based assignment step) — a near-universal follow-up.
- Using KMeans on categorical data without acknowledging the mismatch (Euclidean mean is meaningless for categories) — K-modes/K-prototypes are the correct alternatives, worth naming.
- Not knowing that KMeans' objective (minimize WCSS) is NP-hard in general, and Lloyd's algorithm is a heuristic, not an exact solver — some candidates present it as if it's provably optimal.

## Key Resources

- Arthur & Vassilvitskii (2007), *"k-means++: The Advantages of Careful Seeding"* — the seeding paper.
- [Comparing the State-of-the-Art Clustering Algorithms (2025)](https://medium.com/@sina.nazeri/comparing-the-state-of-the-art-clustering-algorithms-1e65a08157a1)
- Jégou, Douze & Schmid, *"Product Quantization for Nearest Neighbor Search"* — background for KMeans's role in IVF/ANN indexing.
