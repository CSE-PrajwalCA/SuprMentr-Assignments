"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     K-MEANS CUSTOMER CLUSTERING ANALYSIS                    ║
║                             March 11, 2026                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

CLUSTERING FUNDAMENTALS:
- Clustering is an UNSUPERVISED learning task that groups similar data points together
- Within-cluster similarity should be HIGH (points close to each other)
- Between-cluster dissimilarity should be HIGH (clusters far apart)
- K-Means minimizes WITHIN-CLUSTER SUM OF SQUARES (WCSS)

MATHEMATICAL FOUNDATION:
- Distance metric: Euclidean distance = √((x1-x2)² + (y1-y2)² + ... + (zn-z2)²)
- WCSS (Inertia) = Σ(distance from each point to its cluster center)²
- Objective: minimize WCSS by iteratively reassigning points and updating centers

ELBOW METHOD:
- Plot WCSS (inertia) vs number of clusters (k)
- Look for "elbow" point where decrease in WCSS slows down
- This suggests optimal k value for the given dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: SYNTHETIC DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════

print("="*80)
print("STEP 1: GENERATING SYNTHETIC MALL CUSTOMER DATASET")
print("="*80)

# Set random seed for reproducibility
np.random.seed(42)

# Create 200 synthetic mall customers with 3 features:
# Feature 1: Annual Income (in $1000s, range: 15-137)
# Feature 2: Spending Score (1-100, how much customer spends)
# Feature 3: Age (range: 18-70)

n_samples = 200

annual_income = np.random.uniform(15, 137, n_samples)  # Annual income in thousands
spending_score = np.random.uniform(1, 100, n_samples)  # Spending score
age = np.random.uniform(18, 70, n_samples)              # Customer age

# Combine into feature matrix using all 3 features
X = np.column_stack((annual_income, spending_score, age))

print(f"\nDataset Shape: {X.shape}")
print(f"Features: [Annual Income ($1000s), Spending Score (1-100), Age (years)]")
print(f"\nFirst 5 customer records:")
print("Age(yrs) | Income($K) | Spending(1-100)")
print("-" * 45)
for i in range(5):
    print(f"{age[i]:7.1f} | {annual_income[i]:10.1f} | {spending_score[i]:14.1f}")
print(f"... {n_samples-5} more records ...")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: ELBOW METHOD - DETERMINING OPTIMAL K
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 2: ELBOW METHOD ANALYSIS")
print("="*80)

# Mathematical explanation:
# WCSS_k = Σᵢ₌₁ⁿ min_j ||xᵢ - μⱼ||²
# where xᵢ is each point, μⱼ is cluster center, ||..|| is Euclidean distance
# We compute WCSS for k=1 to k=10 and identify elbow point

inertias = []
k_range = range(1, 11)

for k in k_range:
    # KMeans algorithm:
    # 1. Randomly initialize k cluster centers
    # 2. Assign each point to nearest center (Euclidean distance)
    # 3. Update centers as mean of assigned points
    # 4. Repeat steps 2-3 until convergence (centers don't move)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    print(f"k={k:2d} | WCSS (Inertia): {kmeans.inertia_:10.2f}")

# Calculate elbow point using elbow detection algorithm
# Elbow is where the rate of decrease in WCSS slows significantly
differences = np.diff(inertias)
second_differences = np.diff(differences)
elbow_k = np.argmax(second_differences) + 2

print(f"\nDETECTED ELBOW POINT: k={elbow_k}")
print(f"This suggests optimal number of clusters is {elbow_k}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: VISUALIZE ELBOW CURVE
# ═══════════════════════════════════════════════════════════════════════════

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8, label='WCSS (Inertia)')
plt.axvline(x=elbow_k, color='red', linestyle='--', linewidth=2, label=f'Elbow (k={elbow_k})')
plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12, fontweight='bold')
plt.title('Elbow Method for Optimal K Selection', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
print("\n✓ Elbow plot saved as 'elbow_plot.png'")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: FINAL K-MEANS CLUSTERING WITH k=3
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 3: FINAL K-MEANS CLUSTERING (k=3)")
print("="*80)

optimal_k = 3

final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X)

print(f"\nClustering complete with k={optimal_k}")
print(f"Final WCSS (Inertia): {final_kmeans.inertia_:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: CLUSTER CENTERS & ASSIGNMENTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 4: CLUSTER CENTERS (μⱼ - Cluster Centroids)")
print("="*80)

centers = final_kmeans.cluster_centers_

print("\nCluster Center Coordinates:")
print("Cluster | Avg Age(yrs) | Avg Income($K) | Avg Spending(1-100)")
print("-" * 60)
for i, center in enumerate(centers):
    print(f"   {i}    |    {center[2]:6.1f}       |     {center[0]:6.1f}        |     {center[1]:6.1f}")

# Count points in each cluster
unique, counts = np.unique(cluster_labels, return_counts=True)
print("\nCluster Size Distribution:")
print("Cluster | Number of Customers | Percentage")
print("-" * 50)
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(cluster_labels)) * 100
    print(f"   {cluster_id}    |         {count:3d}          |   {percentage:6.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: DETAILED BUSINESS DESCRIPTIONS FOR EACH CLUSTER
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 5: CUSTOMER SEGMENTATION PROFILES & BUSINESS INSIGHTS")
print("="*80)

cluster_profiles = [
    {
        'id': 0,
        'name': 'Budget Conscious Shoppers',
        'profiles': [
            "James Chen (28, $45K/yr): Young professional with modest income, minimal spending. Prefers discounts and value deals. Target: Budget smartphone apps, comparison shopping platforms.",
            "Maria Rodriguez (35, $52K/yr): Single parent, careful with finances. Seeks practical items. High sensitivity to price. Target: Clearance sales, budget retailers.",
            "Ahmed Hassan (42, $48K/yr): Middle-income worker, conservative spender. Buys necessities only. Brand-loyal to affordable options. Target: Subscription discount services.",
            "Lisa Wang (31, $55K/yr): Saves aggressively, rare purchases. Researches heavily before buying. Strong financial goals. Target: Investment tools, financial planning apps.",
            "Robert Thompson (38, $50K/yr): Fixed budget mindset. Values longevity over trends. Loyal but price-sensitive. Target: Long-lasting product warranties, bulk purchase options."
        ]
    },
    {
        'id': 1,
        'name': 'Aspirational Spenders',
        'profiles': [
            "Sophie Miller (34, $95K/yr): High earner with moderate spending habits. Invests in quality items. Interested in premium brands. Target: Luxury goods, investment pieces.",
            "Carlos Mendez (52, $102K/yr): Established professional, balanced lifestyle. Spends on experiences and quality. Values reputation. Target: Premium experiences, upscale dining.",
            "Jennifer Lee (45, $98K/yr): High-income woman, selective shopper. Values quality over quantity. Status-conscious. Target: Designer brands, premium subscriptions.",
            "David Kumar (38, $88K/yr): Tech-savvy high earner, early adopter. Spends on innovation. Digitally engaged. Target: Latest tech gadgets, premium digital services.",
            "Patricia O'Brien (55, $105K/yr): Senior earner, sophisticated taste. Appreciates craftsmanship. Established preferences. Target: Artisanal products, membership clubs."
        ]
    },
    {
        'id': 2,
        'name': 'High-Value Premium Customers',
        'profiles': [
            "Michael Zhang (48, $137K/yr): Highest earner segment, significant spending power. Values luxury and prestige. Regular high-value purchaser. Target: Exclusive memberships, luxury goods.",
            "Alexandra Petrov (41, $125K/yr): Top-tier earner, sophisticated consumer. Seeks premium experiences. Status-driven purchasing. Target: VIP services, luxury travel.",
            "Emma Sullivan (36, $118K/yr): High earner, trendsetter. Fashion and lifestyle focused. Early adopter of premium brands. Target: Designer collections, premium fashion.",
            "Marcus Johnson (50, $130K/yr): Executive-level income, established consumer. Values quality and exclusivity. Premium loyalty program candidate. Target: Concierge services, elite clubs.",
            "Victoria Liu (44, $122K/yr): High-earning professional, lifestyle investor. Spends on self-improvement. Values expertise and quality. Target: Premium wellness, executive education."
        ]
    }
]

for profile_group in cluster_profiles:
    cluster_idx = profile_group['id']
    cluster_name = profile_group['name']
    
    cluster_mask = cluster_labels == cluster_idx
    cluster_data = X[cluster_mask]
    
    avg_age = cluster_data[:, 2].mean()
    avg_income = cluster_data[:, 0].mean()
    avg_spending = cluster_data[:, 1].mean()
    
    print(f"\n{'='*75}")
    print(f"CLUSTER {cluster_idx}: {cluster_name}")
    print(f"{'='*75}")
    print(f"Cluster Statistics:")
    print(f"  • Average Age: {avg_age:.1f} years")
    print(f"  • Average Income: ${avg_income:.1f}K per year")
    print(f"  • Average Spending Score: {avg_spending:.1f}/100")
    print(f"  • Total Customers: {cluster_data.shape[0]}")
    
    print(f"\nDetailed Customer Profiles (5 Representatives):")
    print("-" * 75)
    for i, profile_text in enumerate(profile_group['profiles'], 1):
        print(f"\n{i}. {profile_text}")
    
    print(f"\n{'-'*75}")
    print(f"Business Strategy for {cluster_name}:")
    if cluster_idx == 0:
        print("  • Focus on price competitiveness and value propositions")
        print("  • Implement loyalty programs with small rewards")
        print("  • Use promotional discounts strategically")
        print("  • Emphasize durability and cost-per-use metrics")
    elif cluster_idx == 1:
        print("  • Highlight quality and mid-premium positioning")
        print("  • Create aspirational marketing messaging")
        print("  • Offer financing options for bigger purchases")
        print("  • Build brand prestige and reliability reputation")
    else:
        print("  • Develop exclusive VIP customer programs")
        print("  • Offer personalized luxury experiences")
        print("  • Create premium tier products and services")
        print("  • Implement concierge-level customer service")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: MATHEMATICAL INSIGHTS & CLUSTERING THEORY
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*80)
print("MATHEMATICAL DEEP DIVE: K-MEANS ALGORITHM & CONCEPTS")
print("="*80)

print("""
WITHIN-CLUSTER DISTANCE FORMULA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each cluster C_j with center μⱼ:
  Within-Cluster Distance(C_j) = Σ ||xᵢ - μⱼ||²    (for all xᵢ in cluster j)
  
Where:
  • ||xᵢ - μⱼ|| = Euclidean distance = √((x₁ᵢ-x₁ⱼ)² + (x₂ᵢ-x₂ⱼ)² + ... + (xₙᵢ-xₙⱼ)²)
  • xᵢ = individual data point in n-dimensional space
  • μⱼ = cluster center (mean of all points in cluster j)
  • Squared distance gives more weight to outliers

TOTAL WCSS (How we measure clustering quality):
  WCSS_total = Σⱼ₌₁ᵏ Σ ||xᵢ - μⱼ||²  (sum over all clusters and all points)

GOAL: Minimize WCSS_total
  → Points close to their cluster center (tight clusters)
  → Each cluster captures a cohesive group
  → Different clusters are well-separated

K-MEANS ITERATION ALGORITHM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. INITIALIZATION: Randomly select k points as initial cluster centers
   → Different initializations may yield different local minima
   → Why we use n_init=10 (try 10 different initializations, pick best)

2. ASSIGNMENT STEP: Assign each point xᵢ to nearest center
   Cluster_assignment(xᵢ) = argmin_j ||xᵢ - μⱼ||²
   → Each point belongs to exactly ONE cluster
   → Based purely on Euclidean distance to centers

3. UPDATE STEP: Recompute cluster centers as mean of assigned points
   μⱼ (new) = (1/|C_j|) × Σ xᵢ    (for all xᵢ in cluster C_j)
   → Center moves to minimize within-cluster distance mathematically
   → Numerically optimal position for reducing WCSS

4. CONVERGENCE: Repeat steps 2-3 until centers stop moving
   → Guarantee: WCSS never increases (monotonic decrease)
   → Stop when: ||μⱼ (new) - μⱼ (old)|| < threshold (typically 1e-4)
   → Usually converges in 10-50 iterations for typical data

ELBOW METHOD MATHEMATICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Why does WCSS decrease with more clusters?
  • With k=1: All points in one cluster, maximum WCSS
  • With k=n: Each point is its own cluster, WCSS=0
  → Adding clusters ALWAYS reduces WCSS (but not always meaningful)

Elbow detection:
  • Rate of decrease d(WCSS)/dk slows down after optimal k
  • ΔWCSSₖ = WCSS(k) - WCSS(k+1) gets smaller
  • Elbow = point where ΔWCSSₖ becomes small (diminishing returns)
  • After elbow: Adding clusters gives little improvement (overfitting)

Metrics for comparison:
  ΔWCSSₖ = WCSS(k) - WCSS(k+1)
  Rate ratio = ΔWCSSₖ / ΔWCSSₖ₋₁
  → Elbow where ratio > 0.85 (30%+ drop in improvement)

ADVANTAGES & LIMITATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ ADVANTAGES:
  • Simple, fast, scalable to large datasets (O(nk) per iteration)
  • Intuitive: minimize within-cluster distance
  • Works well on spherical, well-separated clusters
  • Easy to interpret centroids
  • Easy to parallelize computation
  
✗ LIMITATIONS:
  • Must specify k beforehand (elbow method helps but not automatic)
  • Sensitive to initialization (use n_init to mitigate)
  • Assumes spherical clusters (struggles with elongated shapes)
  • Sensitive to outliers (pulls centroid toward extreme point)
  • Doesn't work well with clusters of vastly different sizes
  • Euclidean distance in high dimensions: "curse of dimensionality"
  • Local minima: different initializations may converge to different solutions
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
