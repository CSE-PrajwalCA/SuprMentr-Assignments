"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  DECISION TREES FOR CLASSIFICATION                          ║
║                             March 12, 2026                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE EXPLANATION: HOW DECISION TREES WORK
# ═══════════════════════════════════════════════════════════════════════════

print("="*85)
print("DECISION TREES: COMPREHENSIVE EDUCATIONAL GUIDE (600+ words)")
print("="*85)

explanation = """
╔════════════════════════════════════════════════════════════════════════════╗
║                      DECISION TREES EXPLAINED                             ║
║          A Hierarchical Approach to Classification Problems               ║
╚════════════════════════════════════════════════════════════════════════════╝

1. WHAT IS A DECISION TREE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A Decision Tree is a supervised learning model that makes predictions by recursively 
splitting the feature space into rectangular regions. It's called a "tree" because it 
resembles an inverted tree structure with:

• ROOT NODE: The starting point with all data samples
• INTERNAL NODES: Decision points that split data based on feature values
• BRANCHES: Represent the outcome of each decision (yes/no, left/right)
• LEAF NODES: Final classification labels (prediction outcomes)

Example: To classify customers as "Buy" or "Not Buy":
    Is Income > $50K?
        ├─ YES → Is Spending Score > 60?
        │          ├─ YES → Leaf: "HIGH VALUE" (Buy)
        │          └─ NO  → Leaf: "MEDIUM VALUE" (Maybe)
        └─ NO  → Leaf: "LOW VALUE" (Not Buy)

Trees are INTERPRETABLE: You can explain each decision to stakeholders. Unlike 
"black box" models like neural networks, trees show exactly how decisions are made.


2. ENTROPY: MEASURING IMPURITY IN DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before we split, we need to measure how "mixed" or "pure" our data is. ENTROPY 
quantifies this randomness/disorder in classification labels.

ENTROPY FORMULA:
    H(S) = -Σ pᵢ × log₂(pᵢ)
    
Where:
    • S = dataset
    • pᵢ = proportion of class i in dataset
    • log₂ = logarithm base 2
    • The negative sign ensures positive entropy

INTERPRETATION:
    • H = 0: Perfect purity (all same class) → No split needed
    • H = 1: Maximum disorder (50-50 split for 2 classes) → Maximum confusion
    • 0 < H < 1: Partial impurity → Need better splits

EXAMPLE: Dataset with 4 "Buy" and 6 "Not Buy" samples
    p_Buy = 4/10 = 0.4
    p_NotBuy = 6/10 = 0.6
    H = -[0.4×log₂(0.4) + 0.6×log₂(0.6)]
    H = -[0.4×(-1.322) + 0.6×(-0.737)]
    H = -[-0.529 - 0.442]
    H ≈ 0.971 (very high disorder → good candidate for splitting)


3. INFORMATION GAIN: HOW MUCH DO WE IMPROVE BY SPLITTING?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A split is only valuable if it REDUCES entropy (increases purity). We measure 
improvement using INFORMATION GAIN.

INFORMATION GAIN FORMULA:
    IG(S, A) = H(S) - Σ |Sᵥ|/|S| × H(Sᵥ)
    
Where:
    • S = original dataset (before split)
    • A = feature we're splitting on
    • Sᵥ = subset of S where feature A = value v
    • |Sᵥ|/|S| = proportion of samples in subset Sᵥ
    • H(S) = entropy before split
    • H(Sᵥ) = entropy of each subset after split

INTERPRETATION:
    • IG = 0: Split provides no improvement (pure waste)
    • IG = 1: Perfect split (complete separation of classes)
    • Higher IG = better split (more entropy reduction)

The decision tree algorithm GREEDILY selects the split with MAXIMUM information gain 
at each node. This is a greedy approach (local optimization) not guaranteed to be 
globally optimal, but works well in practice.


4. TREE SPLITS: THE DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

At each internal node, the tree creates a BINARY split based on a single feature:

LEFT BRANCH:  Feature_j ≤ Threshold  (go left if condition is true)
RIGHT BRANCH: Feature_j > Threshold  (go right if condition is false)

Example splits:
    "If feature1 > 5.2, go right; else go left"
    "If Age ≤ 35, go left (younger); else go right (older)"
    "If Income > $75K, go right (high earner); else go left (modest earner)"

The algorithm searches through ALL possible:
    • Features (which column to split on)
    • Thresholds (what value to split at)
    
And selects the combination that maximizes information gain.


5. GINI IMPURITY: ALTERNATIVE TO ENTROPY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Another measure of impurity is GINI INDEX (or GINI IMPURITY), often used instead 
of entropy because it's computationally faster (no logarithms).

GINI FORMULA:
    Gini(S) = 1 - Σ pᵢ²
    
Where:
    • pᵢ = proportion of class i
    • No logarithms needed (simpler computation)

INTERPRETATION:
    • Gini = 0: Pure (all same class)
    • Gini = 0.5: Maximum impurity (50-50 binary split)
    • Works similarly to entropy in practice

For binary classification (2 classes, p₁ and p₂ = 1 - p₁):
    Gini = 2 × p₁ × (1 - p₁)

Example: 40% class A, 60% class B:
    Gini = 1 - (0.4² + 0.6²) = 1 - (0.16 + 0.36) = 1 - 0.52 = 0.48


6. BUILDING THE TREE (RECURSIVE ALGORITHM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The tree grows recursively:

BUILD_TREE(S, depth):
    1. IF all samples in S belong to same class:
       → CREATE LEAF NODE with that class label
       → RETURN (base case: pure node, stop splitting)
    
    2. IF max_depth reached:
       → CREATE LEAF NODE with majority class
       → RETURN (base case: depth limit, prevent overfitting)
    
    3. FOR each feature:
       FOR each possible threshold:
           - Split samples into LEFT and RIGHT
           - Calculate information gain
           - Remember feature/threshold with MAXIMUM gain
    
    4. SELECT best feature and threshold (maximum information gain)
    
    5. SPLIT data: LEFT = samples where feature ≤ threshold
                   RIGHT = samples where feature > threshold
    
    6. RECURSIVELY call:
       LEFT_SUBTREE = BUILD_TREE(LEFT_samples, depth+1)
       RIGHT_SUBTREE = BUILD_TREE(RIGHT_samples, depth+1)
    
    7. RETURN internal node with split rule + subtrees


7. OVERFITTING & MAX_DEPTH PARAMETER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Decision trees are prone to OVERFITTING because they can grow very deep and 
memorize training data rather than learning generalizable patterns.

Max_depth Controls Tree Depth:
    • max_depth=1: Just one split (very weak learner)
    • max_depth=3: Reasonable tree (often good balance)
    • max_depth=4: Deeper, more specific rules
    • max_depth=None: Unlimited growth (very likely to overfit)

HIGH BIAS (underfitting): Shallow tree, misses important patterns
LOW BIAS (good):          Medium tree, balanced learning
HIGH VARIANCE (overfitting): Deep tree, fits training noise

Solution: Use cross-validation to find optimal max_depth.


8. ADVANTAGES & LIMITATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ ADVANTAGES:
  • Highly interpretable: Easy to explain decisions
  • No feature scaling needed (works with raw values)
  • Captures non-linear relationships naturally
  • Fast prediction time: O(depth) complexity
  • Works with mixed feature types (numeric + categorical)
  • Handles feature interactions implicitly
  
✗ LIMITATIONS:
  • Prone to overfitting (deep trees memorize data)
  • Greedy splitting: Local optimization, not global optimal
  • Unstable: Small data changes → completely different tree
  • Biased toward dominant features (if one class dominates)
  • Struggle with imbalanced datasets
  • Creating large trees is computationally expensive

Solution to overfitting: Ensemble methods like Random Forests combine many trees!
"""

print(explanation)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: SYNTHETIC DATASET CREATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*85)
print("STEP 1: CREATE SYNTHETIC DATASET FOR CLASSIFICATION")
print("="*85)

np.random.seed(42)

# Create 200 samples with 2 features
n_samples = 200

# Feature 1: Numerical value from 1 to 10
feature1 = np.random.uniform(1, 10, n_samples)

# Feature 2: Numerical value from 1 to 10
feature2 = np.random.uniform(1, 10, n_samples)

# Create classification rule: which class (A or B) each sample belongs to
# Class A: if feature1 + feature2 > 10
# Class B: if feature1 + feature2 ≤ 10 AND feature1 > 4
# Otherwise: randomly assign for some mix

class_labels = np.zeros(n_samples, dtype=int)  # 0 = Class A, 1 = Class B

for i in range(n_samples):
    if feature1[i] > 5.2:  # First decision rule
        if feature2[i] > 6.0:  # Second decision rule
            class_labels[i] = 1  # Class B
        else:
            class_labels[i] = 0  # Class A
    else:
        if feature2[i] > 4.5:
            class_labels[i] = 1
        else:
            class_labels[i] = 0

# Add some noise (10% random flips)
noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
class_labels[noise_idx] = 1 - class_labels[noise_idx]

# Combine features
X = np.column_stack((feature1, feature2))
y = class_labels

print(f"Dataset shape: {X.shape}")
print(f"Number of Class A (0) samples: {(y == 0).sum()}")
print(f"Number of Class B (1) samples: {(y == 1).sum()}")
print(f"\nFirst 5 samples:")
print("Feature1 | Feature2 | Class")
print("-" * 35)
for i in range(5):
    print(f"  {feature1[i]:6.2f}  |  {feature2[i]:6.2f}  |  {class_labels[i]}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: SPLIT DATA INTO TRAIN & TEST SETS
# ═══════════════════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: TRAIN DECISION TREE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*85)
print("STEP 2: TRAIN DECISION TREE WITH max_depth=4")
print("="*85)

# Train decision tree with max_depth=4 for readability
# max_depth=4 means maximum 4 levels of splitting
# Deeper trees would be more complex and prone to overfitting

# criterion='entropy' uses information gain (entropy-based)
# criterion='gini' uses gini impurity (alternate measure)

dt_classifier = DecisionTreeClassifier(
    max_depth=4,           # Maximum tree depth (controls complexity)
    criterion='entropy',   # Use entropy for information gain calculation
    random_state=42,       # For reproducibility
    min_samples_split=5    # Minimum samples to split a node
)

dt_classifier.fit(X_train, y_train)

print("✓ Decision Tree trained successfully")
print(f"Tree depth: {dt_classifier.get_depth()}")
print(f"Number of leaves: {dt_classifier.get_n_leaves()}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: VISUALIZE TREE STRUCTURE (TEXT FORMAT)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*85)
print("STEP 3: TREE STRUCTURE (TEXT EXPORT FOR HUMAN READING)")
print("="*85)

tree_rules = export_text(dt_classifier, feature_names=['feature1', 'feature2'])

print("\nDecision Tree Rules (Text Format):")
print("-" * 85)
print(tree_rules)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: DETAILED EXPLANATION OF TREE SPLITS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*85)
print("STEP 4: DETAILED EXPLANATION OF EACH SPLIT")
print("="*85)

def explain_tree_splits(tree, feature_names):
    """
    Recursively traverse tree and explain each split with interpretation.
    """
    feature = tree.feature
    threshold = tree.threshold
    left_child = tree.children_left
    right_child = tree.children_right
    value = tree.value
    
    def recurse(node, depth=0):
        indent = "  " * depth
        
        if tree.feature[node] == -2:  # Leaf node
            # Leaf node: show class and sample count
            class_counts = value[node][0]
            total_samples = int(sum(class_counts))
            class_pred = np.argmax(class_counts)
            
            print(f"{indent}→ LEAF NODE (Prediction: Class {class_pred})")
            print(f"{indent}  Class A: {int(class_counts[0])} samples")
            print(f"{indent}  Class B: {int(class_counts[1])} samples")
            print(f"{indent}  Total: {total_samples} samples")
            return
        
        # Internal node: show split rule
        feat_idx = feature[node]
        thresh = threshold[node]
        feat_name = feature_names[feat_idx]
        
        print(f"{indent}SPLIT {depth+1}: If {feat_name} > {thresh:.2f}")
        print(f"{indent}├─ LEFT (≤ {thresh:.2f}):")
        
        recurse(left_child[node], depth + 1)
        
        print(f"{indent}└─ RIGHT (> {thresh:.2f}):")
        
        recurse(right_child[node], depth + 1)
    
    recurse(0)

explain_tree_splits(dt_classifier.tree_, ['feature1', 'feature2'])

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: PREDICTIONS & PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*85)
print("STEP 5: MODEL PERFORMANCE EVALUATION")
print("="*85)

# Make predictions
y_train_pred = dt_classifier.predict(X_train)
y_test_pred = dt_classifier.predict(X_test)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_precision = precision_score(y_train, y_train_pred, zero_division=0)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)

train_recall = recall_score(y_train, y_train_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)

print("\n" + "-"*85)
print("METRIC DEFINITIONS:")
print("-"*85)

print("""
ACCURACY: Percentage of correct predictions overall
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    • TP (True Positive): Correctly predicted Class B
    • TN (True Negative): Correctly predicted Class A
    • FP (False Positive): Incorrectly predicted Class B (was actually A)
    • FN (False Negative): Incorrectly predicted Class A (was actually B)

PRECISION: Of all Class B predictions, how many were correct?
    Precision = TP / (TP + FP)
    • Focus: Reducing false alarms (Type I errors)
    • Use when: Cost of false positives is high
    • Example: Email spam detection (don't want false positives)

RECALL (or SENSITIVITY): Of all actual Class B samples, how many did we catch?
    Recall = TP / (TP + FN)
    • Focus: Reducing missed cases (Type II errors)
    • Use when: Cost of false negatives is high
    • Example: Disease detection (don't want to miss sick people)

TRADE-OFF: Precision ↑ usually means Recall ↓ (and vice versa)
    • High precision: Conservative predictions, fewer errors but more missed cases
    • High recall: Aggressive predictions, catches most cases but more false alarms
    • F1 Score = 2 × (Precision × Recall) / (Precision + Recall) balances both
""")

print("\n" + "-"*85)
print("PERFORMANCE ON TRAINING SET:")
print("-"*85)
print(f"Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Precision: {train_precision:.4f}")
print(f"Recall:    {train_recall:.4f}")

# Confusion matrix for training
cm_train = confusion_matrix(y_train, y_train_pred)
print("\nConfusion Matrix (Training):")
print(f"                 Predicted A | Predicted B")
print(f"Actually A:           {cm_train[0, 0]:3d}      |     {cm_train[0, 1]:3d}")
print(f"Actually B:           {cm_train[1, 0]:3d}      |     {cm_train[1, 1]:3d}")

print("\n" + "-"*85)
print("PERFORMANCE ON TEST SET:")
print("-"*85)
print(f"Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")

# Confusion matrix for test
cm_test = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test):")
print(f"                 Predicted A | Predicted B")
print(f"Actually A:           {cm_test[0, 0]:3d}      |     {cm_test[0, 1]:3d}")
print(f"Actually B:           {cm_test[1, 0]:3d}      |     {cm_test[1, 1]:3d}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: DEEP MATHEMATICAL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*85)
print("MATHEMATICAL DEEP DIVE: ENTROPY & GINI CALCULATIONS")
print("="*85)

print("""
ENTROPY DETAILED CALCULATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For a dataset with n_A samples of Class A and n_B samples of Class B:

Total samples: N = n_A + n_B
Proportions: p_A = n_A / N,  p_B = n_B / N

Entropy formula:
    H(S) = -p_A × log₂(p_A) - p_B × log₂(p_B)

Special cases:
    • If p_A = 1 and p_B = 0 (all Class A):
      H = -1×log₂(1) - 0×log₂(0) = -1×0 - 0 = 0 (pure)
    
    • If p_A = 0.5 and p_B = 0.5 (balanced):
      H = -0.5×log₂(0.5) - 0.5×log₂(0.5)
      H = -0.5×(-1) - 0.5×(-1)
      H = 0.5 + 0.5 = 1.0 (maximum disorder)

INFORMATION GAIN DETAILED CALCULATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When we split on feature F at threshold T:

LEFT subset:   All samples where F ≤ T
RIGHT subset:  All samples where F > T

Weighted entropy after split:
    H_after = (|LEFT|/N) × H(LEFT) + (|RIGHT|/N) × H(RIGHT)

Information Gain:
    IG = H_before - H_after

Interpretation:
    • IG = 0: Split has no value (entropy unchanged)
    • IG = 1: Perfect split (completely separated classes)
    • Higher IG = better split (more entropy reduction)

The algorithm tests ALL possible (feature, threshold) pairs and selects the 
combination with Maximum Information Gain. This is GREEDY: optimal locally, 
not necessarily globally.

GINI IMPURITY CALCULATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gini formula (measures probability of misclassification if random label assigned):
    Gini(S) = 1 - Σ (pᵢ)²
    Gini(S) = 1 - p_A² - p_B²

For balanced data (p_A = p_B = 0.5):
    Gini = 1 - 0.5² - 0.5² = 1 - 0.25 - 0.25 = 0.5 (maximum)

For pure data (p_A = 1, p_B = 0):
    Gini = 1 - 1² - 0² = 0 (minimum, perfect purity)

Relationship to Entropy:
    Gini ≈ Entropy/log₂(K) where K = number of classes
    Similar behavior but no logarithm (computationally simpler)

TREE COMPLEXITY & OVERFITTING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bias-Variance Tradeoff:

MAX_DEPTH = 1 (Stump):
    • HIGH BIAS (underfitting): Too simple, misses patterns
    • LOW VARIANCE: Stable across different datasets
    • Likely UNDERFITS (poor training accuracy)

MAX_DEPTH = 4 (Medium Tree):
    • BALANCED: Good generalization
    • GOOD: Captures important patterns without overfitting
    • RECOMMENDED: This is what we use!

MAX_DEPTH = None (Full Growth):
    • LOW BIAS: Captures all patterns
    • HIGH VARIANCE: Sensitive to training data noise
    • Likely OVERFITS: High training accuracy, poor test accuracy

Solution: Cross-validate to find optimal max_depth!
""")

print("\n" + "="*85)
print("DECISION TREE TRAINING COMPLETE")
print("="*85)
