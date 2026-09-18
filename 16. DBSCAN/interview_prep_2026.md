# DBSCAN — Senior ML Engineer Interview Prep (2026)

> See folder 14 for the family overview. This doc covers DBSCAN's mechanics in depth and its evolution into HDBSCAN, which is now the practical 2025–2026 default for most of DBSCAN's use cases.

## TL;DR Refresher

- Density-based clustering: a point is a **core point** if at least `min_samples` points lie within radius `eps` of it; core points within `eps` of each other are directly density-reachable and get chained into the same cluster; **border points** are within `eps` of a core point but aren't core themselves; everything else is **noise** (label `-1`).
- No need to specify the number of clusters `k` upfront; naturally identifies arbitrary-shaped clusters and flags outliers/noise as a first-class output, not an afterthought.
- Core weakness: uses a single **global** `eps`, so it struggles when clusters have meaningfully different densities — a `eps` that correctly separates a dense cluster may merge or completely miss a sparser one.

## What's New / State of the Art (2025–2026)

- **HDBSCAN is now the standard recommendation over plain DBSCAN in most 2025–2026 writeups and production pipelines** — it builds a hierarchy of DBSCAN solutions across all `eps` values, then extracts the most stable clusters across that hierarchy, removing the need to hand-pick a single global `eps` and handling variable-density clusters that break plain DBSCAN ([HDBSCAN: The Supercharged Version of DBSCAN, 2025](https://www.dailydoseofds.com/hdbscan-the-supercharged-version-of-dbscan-an-algorithmic-deep-dive/)). A senior candidate should treat "DBSCAN vs. HDBSCAN" as a near-default follow-up to any DBSCAN question in 2026.
- **DBSCAN/HDBSCAN as the standard tool for outlier/anomaly flagging on embeddings** — because noise points are a first-class output (not requiring a separate anomaly-detection model), density-based clustering is commonly used in 2025–2026 pipelines to flag anomalous log entries, unusual user behavior, or out-of-distribution embeddings, alongside or instead of dedicated methods like Isolation Forest (folder 19).
- **Scalability improvements**: approximate/GPU-accelerated DBSCAN implementations (`cuML`) and tree-based neighbor queries (KD-tree/Ball-tree for low-dim, or approximate methods for high-dim embeddings — see the KNN doc's HNSW discussion) are what make DBSCAN/HDBSCAN practical beyond the classical `O(n²)` naive neighbor-query cost on large 2025–2026-scale embedding datasets.
- **HDBSCAN's `min_cluster_size` and cluster-stability-based extraction** (using the Excess of Mass or "leaf" cluster-selection methods) is a more nuanced hyperparameter story than DBSCAN's `eps`/`min_samples`, and is increasingly expected knowledge for anyone claiming DBSCAN-family expertise in a 2026 interview.

## Senior-Level Interview Questions

**Q1. Walk through DBSCAN's core/border/noise point classification and cluster-formation process step by step.**
Strong answer: For each unvisited point, count neighbors within `eps`; if `≥ min_samples` (including itself), mark it **core** and start/expand a cluster by recursively including all points density-reachable from it (directly density-reachable = within `eps` of a core point; the recursion chains through core points, "border-hopping" to reach far-apart-but-connected regions). Points within `eps` of a core point but not themselves core are **border points**, assigned to that cluster but not used to expand it further. Points that are neither core nor border after processing are labeled **noise**. Complexity is `O(n log n)` with a spatial index (KD-tree/Ball-tree) for the neighbor queries, or `O(n²)` naively.

**Q2. Explain precisely why a single global `eps` breaks down on data with clusters of different densities, with a concrete example.**
Strong answer: Consider two true clusters — one tight/dense (points within, say, 0.1 units of each other) and one sparse (points 1.0 units apart) — plus background noise between them. A small `eps` (tuned to correctly resolve the dense cluster) will see the sparse cluster's points as isolated noise (not enough neighbors within that radius); a large `eps` (tuned to correctly resolve the sparse cluster) will likely merge the dense cluster with nearby noise or other clusters, since the radius is too generous everywhere. There is no single `eps` that's simultaneously correct for both densities — this is precisely the failure mode HDBSCAN is designed to fix by not committing to one global density threshold.

**Q3. Explain, at a conceptual level, how HDBSCAN builds a cluster hierarchy and extracts a single flat clustering from it.**
Strong answer: HDBSCAN conceptually runs DBSCAN across *all* possible `eps` values simultaneously by building a minimum spanning tree over a "mutual reachability distance" (a density-aware distance metric that inflates distances in sparse regions), producing a dendrogram of nested clusters that appear/split/disappear as the implicit `eps` threshold varies. It then computes each candidate cluster's **stability** (how long/persistently it exists across that range of thresholds) and extracts the set of clusters that maximizes total stability (via the Excess-of-Mass algorithm) — effectively picking, per-region, the "right" local density threshold rather than a single global one, and directly giving a `min_cluster_size` parameter (far more intuitive than tuning `eps`).

**Q4. Your DBSCAN clustering on 100k user-behavior embedding vectors labels 40% of points as noise — is this a bug, and how do you investigate?**
Strong answer: Not necessarily a bug — DBSCAN's noise designation is often the *useful* output (flagging genuinely anomalous/rare behavior patterns), especially on real-world embeddings where a meaningful fraction of users don't fit clean archetypes. Investigate by: (a) sampling and manually reviewing some "noise" points — are they truly outliers, or is `eps`/`min_samples` simply mis-tuned for this data's density? (b) checking whether the embedding space itself has the curse-of-dimensionality issue (distances concentrating, making everything look "far" from everything else) — consider a dimensionality-reduction step first (UMAP, per folder 27); (c) trying HDBSCAN, which is less sensitive to a single mistuned global threshold and would reveal whether the high noise rate persists with a more robust method.

**Q5. Compare DBSCAN and Isolation Forest as anomaly-detection tools — when would you use one over the other, or both together?**
Strong answer: DBSCAN/HDBSCAN's noise points are a byproduct of density-based clustering — good when anomalies are genuinely "not near any dense group," and you also want the *normal* structure clustered into interpretable groups as a side benefit. Isolation Forest (folder 19) is a purpose-built anomaly detector using isolation-based scoring, generally faster and more scalable to high dimensions, and gives a continuous anomaly score (not just a binary noise/not-noise label) — better when you need a ranked severity score or need to set a tunable anomaly-rate threshold independent of clustering structure. In practice, combining both (ensemble of anomaly signals) is a common, defensible production pattern for high-stakes fraud/monitoring systems.

## Common Pitfalls & Follow-Up Probes

- Choosing `eps` via a fixed rule of thumb without validating against a k-distance plot (sorted distance to the `min_samples`-th nearest neighbor — the standard `eps`-selection heuristic).
- Not knowing DBSCAN's runtime/quality depends heavily on distance-metric choice and dimensionality — high-dimensional embeddings often need a spatial index that degrades toward brute force (see the curse-of-dimensionality discussion in the KNN doc).
- Treating "noise" (label -1) as always meaning "bad data" rather than potentially the most interesting output of the whole analysis.

## Key Resources

- Ester et al. (1996), *"A Density-Based Algorithm for Discovering Clusters"* — the original DBSCAN paper.
- Campello, Moulavi & Sander (2013), *"Density-Based Clustering Based on Hierarchical Density Estimates"* — the HDBSCAN paper.
- [HDBSCAN: The Supercharged Version of DBSCAN — An Algorithmic Deep Dive (2025)](https://www.dailydoseofds.com/hdbscan-the-supercharged-version-of-dbscan-an-algorithmic-deep-dive/)
