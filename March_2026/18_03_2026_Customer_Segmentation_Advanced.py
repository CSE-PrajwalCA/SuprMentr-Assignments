"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ADVANCED K-MEANS WITH SILHOUETTE ANALYSIS                      ║
║                             March 18, 2026                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

ADVANCED CLUSTERING TECHNIQUES:
- Silhouette Score: Measure of cluster quality and separation
- Range: -1 (bad) to 1 (excellent), where 0 is poor quality
- Tests stability across multiple random initializations
- Combines both cohesion (within-cluster compactness) and separation

CLUSTERING STABILITY:
- K-Means is sensitive to initialization
- Different random starts can lead to different final clusters
- Testing multiple seeds shows how stable our clusters are
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: SYNTHETIC DATASET GENERATION (Same as FILE 15)
# ═══════════════════════════════════════════════════════════════════════════

print("="*80)
print("STEP 1: GENERATING SYNTHETIC MALL CUSTOMER DATASET")
print("="*80)

np.random.seed(42)

n_samples = 200

annual_income = np.random.uniform(15, 137, n_samples)
spending_score = np.random.uniform(1, 100, n_samples)
age = np.random.uniform(18, 70, n_samples)

X = np.column_stack((annual_income, spending_score, age))

print(f"\nDataset Shape: {X.shape}")
print(f"Features: [Annual Income ($1000s), Spending Score (1-100), Age (years)]")
print(f"Total samples: {n_samples} mall customers")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: SILHOUETTE SCORE EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 2: SILHOUETTE SCORE ANALYSIS")
print("="*80)

silhouette_explanation = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                    SILHOUETTE SCORE: DETAILED GUIDE                     ║
╚═══════════════════════════════════════════════════════════════════════════╝

WHAT IS SILHOUETTE SCORE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The Silhouette Score measures how well each sample fits into its assigned cluster.
It combines two aspects:
  1. COHESION: How close the point is to other points in SAME cluster
  2. SEPARATION: How far the point is from closest points in OTHER clusters

MATHEMATICAL FORMULA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each data point i:

a(i) = Average distance from point i to all other points in SAME cluster
       a(i) = (1/nᵢ) × Σ ||xᵢ - xⱼ||  (for all j in same cluster)
       → LOW a(i) = tight cluster (good!)
       → HIGH a(i) = spread out cluster (bad)

b(i) = Minimum average distance from point i to points in OTHER clusters
       b(i) = min_j { (1/nⱼ) × Σ ||xᵢ - xₖ|| }  (for all k in cluster j≠i)
       → HIGH b(i) = far from other clusters (good!)
       → LOW b(i) = close to other clusters (bad, overlap!)

Silhouette Coefficient of point i:
    s(i) = (b(i) - a(i)) / max(a(i), b(i))

INTERPRETATION:
    s(i) = 1:  Point is well-separated from other clusters (excellent!)
    s(i) = 0:  Point is on cluster boundary (ambiguous)
    s(i) = -1: Point is closer to other clusters than its own (bad assignment!)

OVERALL SILHOUETTE SCORE:
    Score = (1/n) × Σ s(i)  (average silhouette of all points)

RANGES:
    Score ≈ 1.0:   Dense, well-separated clusters (EXCELLENT)
    Score ≈ 0.5:   Reasonable cluster structure (GOOD)
    Score ≈ 0.0:   Overlapping clusters, poor separation (WEAK)
    Score < 0.0:   Many misclassified points (BAD)

INTERPRETATION BY K:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

k=2: Usually high score (two well-separated groups)
k=3: Moderate score (data naturally clusters into ~3 groups)
k=4: May decrease (forcing unnecessary splits)
k=5+: Likely lower (over-segmentation, artificial clusters)

The PEAK in silhouette score suggests OPTIMAL k!

SILHOUETTE VS ELBOW METHOD:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ELBOW METHOD:
    + Simple, intuitive
    + Uses WCSS (within-cluster sum of squares)
    - Requires visual interpretation
    - Elbow not always clear

SILHOUETTE SCORE:
    + Statistical measure (0-1 range, comparable)
    + Considers both cohesion and separation
    + Clear maximum (optimal k)
    - Computationally expensive (O(n²))
    - Less interpretable for beginners

BEST PRACTICE: Use BOTH methods complementary!
"""

print(silhouette_explanation)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: COMPUTE SILHOUETTE SCORES FOR K=2 TO K=6
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 3: COMPUTING SILHOUETTE SCORES FOR DIFFERENT K VALUES")
print("="*80)

silhouette_scores = {}

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate silhouette score
    sil_score = silhouette_score(X, cluster_labels)
    silhouette_scores[k] = sil_score
    
    print(f"k={k} | Silhouette Score: {sil_score:.4f}")

# Find optimal k
optimal_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\n✓ OPTIMAL k = {optimal_k} (Silhouette Score: {silhouette_scores[optimal_k]:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: STABILITY ANALYSIS - TEST MULTIPLE RANDOM SEEDS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 4: CLUSTERING STABILITY ACROSS DIFFERENT RANDOM SEEDS")
print("="*80)

print("\nTesting stability: How consistent are clusters with different initializations?")
print("(Same data, different random starts should give similar clusters)\n")

seeds = [0, 1, 2, 3, 4]
results_per_seed = {}

for k_test in [3]:  # Focus on k=3 for detailed analysis
    print(f"\n{'='*75}")
    print(f"K = {k_test}")
    print(f"{'='*75}")
    
    for seed in seeds:
        kmeans = KMeans(n_clusters=k_test, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X, labels)
        
        results_per_seed[seed] = {
            'labels': labels,
            'inertia': inertia,
            'sil_score': sil_score,
            'centers': kmeans.cluster_centers_
        }
        
        print(f"Seed {seed}: Inertia={inertia:.2f}, Silhouette={sil_score:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: FINAL K-MEANS WITH K=3 AND SILHOUETTE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 5: DETAILED SILHOUETTE ANALYSIS FOR K=3")
print("="*80)

final_k = 3
final_kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X)

# Calculate silhouette scores per sample
sil_samples = silhouette_samples(X, cluster_labels)

# Overall silhouette score
overall_sil = silhouette_score(X, cluster_labels)

print(f"\nOverall Silhouette Score (k={final_k}): {overall_sil:.4f}")
print("\nSilhouette Score Per Cluster:")
print("-" * 60)

for i in range(final_k):
    cluster_sil = sil_samples[cluster_labels == i]
    avg_sil = cluster_sil.mean()
    
    print(f"Cluster {i}:")
    print(f"  • Average Silhouette: {avg_sil:.4f}")
    print(f"  • Min: {cluster_sil.min():.4f}, Max: {cluster_sil.max():.4f}")
    print(f"  • Std Dev: {cluster_sil.std():.4f}")
    print(f"  • Sample Count: {(cluster_labels == i).sum()}")
    
    # Quality assessment
    if avg_sil > 0.7:
        quality = "EXCELLENT (very tight, well-separated)"
    elif avg_sil > 0.5:
        quality = "GOOD (reasonable cohesion)"
    elif avg_sil > 0.25:
        quality = "FAIR (some overlap with other clusters)"
    else:
        quality = "POOR (significant overlap, ambiguous assignment)"
    
    print(f"  • Quality: {quality}")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: DETAILED SILHOUETTE SAMPLES STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 6: SILHOUETTE METRICS DETAILED BREAKDOWN")
print("="*80)

print(f"\nTotal Samples: {len(sil_samples)}")
print(f"Samples with s(i) > 0.5:  {(sil_samples > 0.5).sum()} ({(sil_samples > 0.5).sum()/len(sil_samples)*100:.1f}%)")
print(f"Samples with 0 < s(i) <= 0.5: {((sil_samples > 0) & (sil_samples <= 0.5)).sum()} ({((sil_samples > 0) & (sil_samples <= 0.5)).sum()/len(sil_samples)*100:.1f}%)")
print(f"Samples with s(i) <= 0:   {(sil_samples <= 0).sum()} ({(sil_samples <= 0).sum()/len(sil_samples)*100:.1f}%)")

print("\nInterpretation:")
print("  • High % of s(i) > 0.5: Clusters are well-defined and separated")
print("  • High % of s(i) <= 0: Points misclassified, clusters overlap significantly")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: VISUALIZATION - SCATTER PLOT WITH CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 7: VISUALIZING CLUSTERS (INCOME VS SPENDING SCORE)")
print("="*80)

# Use income and spending score for 2D visualization
plt.figure(figsize=(12, 5))

# Plot 1: Cluster scatter plot
plt.subplot(1, 2, 1)
colors = ['red', 'blue', 'green']
for i in range(final_k):
    cluster_points = X[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               c=colors[i], label=f'Cluster {i}', alpha=0.6, s=50)

# Plot cluster centers
centers_2d = final_kmeans.cluster_centers_[:, :2]
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
           c='black', marker='X', s=300, edgecolors='white', linewidth=2,
           label='Centers')

plt.xlabel('Annual Income ($1000s)', fontsize=11, fontweight='bold')
plt.ylabel('Spending Score (1-100)', fontsize=11, fontweight='bold')
plt.title('K-Means Clustering (k=3)\nIncome vs Spending', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Silhouette score distribution
plt.subplot(1, 2, 2)
y_lower = 10
colors_sil = ['red', 'blue', 'green']

for i in range(final_k):
    cluster_sil = sil_samples[cluster_labels == i]
    cluster_sil.sort()
    
    size_cluster = cluster_sil.shape[0]
    y_upper = y_lower + size_cluster
    
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_sil,
                      facecolor=colors_sil[i], edgecolor=colors_sil[i], alpha=0.7,
                      label=f'Cluster {i}')
    
    y_lower = y_upper + 10

plt.axvline(x=overall_sil, color='red', linestyle='--', linewidth=2,
           label=f'Average: {overall_sil:.3f}')

plt.xlabel('Silhouette Coefficient', fontsize=11, fontweight='bold')
plt.ylabel('Cluster Label', fontsize=11, fontweight='bold')
plt.title('Silhouette Plot (k=3)', fontsize=12, fontweight='bold')
plt.legend(loc='best')
plt.tight_layout()

plt.savefig('silhouette_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Scatter and silhouette plots saved as 'silhouette_analysis.png'")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: CLUSTERING STABILITY CONCLUSIONS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 8: CLUSTERING STABILITY CONCLUSIONS")
print("="*80)

inertias = [results_per_seed[i]['inertia'] for i in seeds]
sil_scores = [results_per_seed[i]['sil_score'] for i in seeds]

print(f"\nInertia Values Across Seeds:")
for seed, inertia in zip(seeds, inertias):
    print(f"  Seed {seed}: {inertia:.2f}")

inertia_variance = np.var(inertias)
inertia_mean = np.mean(inertias)
inertia_cv = (np.std(inertias) / inertia_mean) * 100

print(f"\nInertia Statistics:")
print(f"  • Mean: {inertia_mean:.2f}")
print(f"  • Std Dev: {np.std(inertias):.2f}")
print(f"  • Coefficient of Variation: {inertia_cv:.2f}%")

print(f"\nSilhouette Scores Across Seeds:")
for seed, sil in zip(seeds, sil_scores):
    print(f"  Seed {seed}: {sil:.4f}")

sil_variance = np.var(sil_scores)
sil_mean = np.mean(sil_scores)
sil_cv = (np.std(sil_scores) / sil_mean) * 100

print(f"\nSilhouette Score Statistics:")
print(f"  • Mean: {sil_mean:.4f}")
print(f"  • Std Dev: {np.std(sil_scores):.4f}")
print(f"  • Coefficient of Variation: {sil_cv:.2f}%")

print("\nStability Assessment:")
if inertia_cv < 5:
    print("  ✓ STABLE: Clustering results are consistent across seeds")
    print("    → Different initializations converge to similar solutions")
    print("    → This is a good sign of robust clustering!")
elif inertia_cv < 15:
    print("  ~ MODERATE: Some variation but within acceptable range")
    print("    → Generally consistent, occasional variations")
else:
    print("  ✗ UNSTABLE: Highly sensitive to initialization")
    print("    → Different seeds produce very different results")
    print("    → May need to increase n_init or reconsider k")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: COMPREHENSIVE EXPLANATION OF SILHOUETTE FORMULA
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*80)
print("MATHEMATICAL DEEP DIVE: SILHOUETTE FORMULA IN DETAIL")
print("="*80)

print("""
SILHOUETTE COEFFICIENT DETAILED BREAKDOWN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Calculate a(i) - Cohesion (Within-Cluster Distance)
────────────────────────────────────────────────────────────

For each point xᵢ and cluster C_j containing xᵢ:

    a(i) = (1/|C_j| - 1) × Σ ||xᵢ - xₖ||   (for all xₖ in C_j, k ≠ i)

Interpretation:
    • a(i) measures how similar xᵢ is to other points in ITS cluster
    • LOW a(i) → Well-integrated into its cluster (good)
    • HIGH a(i) → Far from cluster members (bad fit)
    • Range: [0, ∞]

Example Calculation:
    If cluster has 5 points: [A, B, C, D, E]
    To compute a(B):
        a(B) = (distance B to A + distance B to C + distance B to D + distance B to E) / 4
        → Average distance to cluster-mates


Step 2: Calculate b(i) - Separation (Between-Cluster Distance)
────────────────────────────────────────────────────────────────

For each point xᵢ and cluster C_k that doesn't contain xᵢ:

    d(i, k) = (1/|C_k|) × Σ ||xᵢ - xⱼ||   (for all xⱼ in C_k)

Then:
    b(i) = min_k { d(i, k) }    (distance to CLOSEST other cluster)

Interpretation:
    • b(i) measures distance to nearest cluster that doesn't contain xᵢ
    • HIGH b(i) → Far from other clusters (good separation)
    • LOW b(i) → Close to another cluster (poor separation)
    • Range: [0, ∞]

Example Calculation:
    For point B in Cluster 1, compute distance to other clusters:
        d(B, Cluster2) = avg distance from B to all Cluster2 points
        d(B, Cluster3) = avg distance from B to all Cluster3 points
    b(B) = min(d(B, Cluster2), d(B, Cluster3), ...)
        → Distance to NEAREST cluster


Step 3: Calculate s(i) - Silhouette Coefficient
────────────────────────────────────────────────

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

This formula combines cohesion and separation:

    Why max(a(i), b(i))?
        • Normalizes to range [-1, 1]
        • If a(i) >> b(i): Can't achieve high silhouette (cluster overlap)
        • If b(i) >> a(i): Can achieve high silhouette (good assignment)

Interpretation:
    s(i) = 1:    b(i) >> a(i) → Very far from other clusters, close to own
    s(i) = 0:    b(i) ≈ a(i) → Equidistant from own cluster and nearest other
    s(i) = -1:   a(i) >> b(i) → Closer to other cluster than own


Step 4: Calculate Overall Score
────────────────────────────────

    Avg Silhouette = (1/n) × Σ s(i)    (average of all samples)

Typical Interpretation:
    0.71 - 1.00: Strong structure
    0.51 - 0.70: Reasonable structure
    0.26 - 0.50: Weak structure
    < 0.25:      No substantial structure


COMPUTATIONAL COMPLEXITY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For n samples and k clusters:

    WCSS (Elbow): O(n × k × d × iterations) 
    SILHOUETTE:   O(n² × d)  ← More expensive! All pairwise distances

Why expensive?
    • Must compute distance from every point to EVERY other point
    • For n=200: 200×199/2 = 19,900 distance calculations
    • For n=10,000: 50 million distance calculations!

Therefore: Silhouette expensive for large datasets, but very informative.


ADVANTAGES OF SILHOUETTE ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Single number [-1, 1] is comparable across datasets
✓ Considers both within-cluster cohesion AND between-cluster separation
✓ Identifies points that are ambiguously assigned
✓ Clear maximum suggests optimal k
✓ Mathematically principled (based on distances)

✗ LIMITATIONS:
✗ Computationally expensive (O(n²))
✗ May favor convex clusters (Euclidean metric)
✗ Assumes similar-sized clusters
✗ Cannot be optimized directly (not used in training)
""")

print("\n" + "="*80)
print("ADVANCED CLUSTERING ANALYSIS COMPLETE")
print("="*80)
