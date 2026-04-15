"""
Word Embeddings and Semantic Similarity Analysis
==================================================
Author: AI Assignment Module
Date: 27 March 2026

This script explores word embeddings and semantic similarity between word pairs.

DISTRIBUTIONAL HYPOTHESIS (CORE CONCEPT):
=========================================
"Words that occur in similar contexts have similar meanings."

Key Idea: The meaning of a word is determined by the company it keeps.
- If two words appear alongside the same sets of words, they're semantically similar
- Example: "doctor" and "nurse" both appear near "hospital", "patient", "medicine"
- This distributional pattern suggests they're related in meaning

WORD EMBEDDINGS:
================
A word embedding is a dense vector (list of numbers) representing a word's meaning.
- Dimension: typically 50-300 values per word
- Pre-trained embeddings: Learned from billions of words (Word2Vec, GloVe, FastText)
- Semantic relationships encoded as geometrical relationships in vector space

COSINE SIMILARITY FORMULA (MATHEMATICAL):
===========================================
For vectors A and B:

                    A · B
    similarity = ─────────────────────
                 ||A|| × ||B||

Where:
- A · B = dot product = Σ(A[i] × B[i]) for all dimensions
- ||A|| = magnitude/norm = √(Σ(A[i]²))
- ||B|| = magnitude/norm = √(Σ(B[i]²))

Result Range: [0, 1] (for normalized vectors)
- 1.0 = identical direction (perfect similarity)
- 0.5 = somewhat similar
- 0.0 = orthogonal (no similarity)
- Negative values possible with unnormalized vectors

WHY COSINE SIMILARITY?
- Measures angle between vectors, not distance
- Invariant to vector magnitude (scale-independent)
- Perfect for comparing semantic relationships
- Computationally efficient (just dot products)

TYPES OF SEMANTIC RELATIONSHIPS:
=================================
1. SYNONYMY: Similar meaning (doctor ≈ physician)
2. ANTONYMY: Opposite meaning (good ↔ bad) [low similarity]
3. ANALOGY: Relational similarity (king - man ≈ queen - woman)
4. HYPERNYMY: Category relationship (dog is a hypernym of poodle)
5. HYPONYMY: Subcategory relationship (poodle is a hyponym of dog)

LIMITATIONS OF WORD EMBEDDINGS:
- Can't distinguish word senses (bank = financial institution vs river bank)
- Limited to pre-training data
- Don't understand context within a sentence
- May encode biases from training data
"""

import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd

# ============================================================================
# STEP 1: ESTABLISH WORD PAIRS AND THEIR SEMANTIC RELATIONSHIPS
# ============================================================================

word_pairs = [
    ("doctor", "nurse", "Professional Relationship: Both medical professionals"),
    ("king", "queen", "Gender Relationship: Opposite gender royalty"),
    ("cat", "dog", "Semantic Similarity: Both domesticated animals/pets"),
    ("good", "bad", "Antonym Relationship: Opposite meanings"),
    ("apple", "orange", "Hypernym Relationship: Both fruits")
]

print("="*100)
print("WORD EMBEDDINGS AND SEMANTIC SIMILARITY ANALYSIS")
print("="*100)
print()

print("WORD PAIRS UNDER ANALYSIS:")
print("-"*100)
for idx, (word1, word2, relationship) in enumerate(word_pairs, 1):
    print(f"{idx}. ({word1:10} ↔ {word2:10}) → {relationship}")

print()

# ============================================================================
# STEP 2: CREATE WORD EMBEDDINGS (RANDOM VECTORS FOR DEMONSTRATION)
# ============================================================================

# In real-world scenarios, we'd use:
# - Word2Vec: Trained on Google News corpus (300-dimensional)
# - GloVe: Stanford's Global Vectors for Word Representation
# - FastText: Facebook's subword-aware embeddings
# - BERT features: Modern transformer-based embeddings

# For this educational demonstration, we create semantic-ish vectors
# These are designed to show realistic similarity patterns

np.random.seed(42)

# Define embedding vectors that reflect semantic relationships
embeddings = {
    # Medical domain (doctor, nurse) - high similarity
    "doctor":  np.array([0.8, 0.7, 0.3, 0.2, 0.1, 0.9, 0.4, 0.6, 0.5, 0.2]),
    "nurse":   np.array([0.75, 0.72, 0.32, 0.25, 0.08, 0.88, 0.42, 0.58, 0.52, 0.18]),
    
    # Royalty domain (king, queen) - high similarity
    "king":    np.array([0.9, 0.2, 0.7, 0.85, 0.3, 0.1, 0.8, 0.4, 0.6, 0.9]),
    "queen":   np.array([0.88, 0.22, 0.72, 0.83, 0.32, 0.12, 0.82, 0.38, 0.58, 0.88]),
    
    # Pet/animal domain (cat, dog) - high similarity
    "cat":     np.array([0.2, 0.9, 0.4, 0.1, 0.7, 0.5, 0.8, 0.9, 0.3, 0.4]),
    "dog":     np.array([0.18, 0.88, 0.42, 0.12, 0.72, 0.52, 0.78, 0.88, 0.32, 0.42]),
    
    # Opposites (good, bad) - low similarity
    "good":    np.array([0.85, 0.4, 0.8, 0.7, 0.6, 0.3, 0.9, 0.2, 0.7, 0.8]),
    "bad":     np.array([0.1, 0.7, 0.15, 0.25, 0.4, 0.75, 0.2, 0.85, 0.3, 0.15]),
    
    # Fruits (apple, orange) - medium similarity
    "apple":   np.array([0.3, 0.5, 0.8, 0.4, 0.6, 0.7, 0.2, 0.5, 0.9, 0.3]),
    "orange":  np.array([0.32, 0.52, 0.82, 0.42, 0.62, 0.72, 0.22, 0.52, 0.88, 0.32])
}

print("\nEMBEDDING VECTORS (10-dimensional space):")
print("-"*100)
for word in ["doctor", "nurse", "king", "queen", "cat", "dog", "good", "bad", "apple", "orange"]:
    vector = embeddings[word]
    magnitude = np.linalg.norm(vector)
    print(f"{word:10} → {vector} [magnitude: {magnitude:.4f}]")

print()

# ============================================================================
# STEP 3: CALCULATE COSINE SIMILARITY FOR ALL WORD PAIRS
# ============================================================================

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    MATHEMATICAL IMPLEMENTATION:
    ===========================
    1. Compute dot product: A · B = Σ(A[i] × B[i])
    2. Compute magnitudes: ||A|| = √(Σ(A[i]²)), ||B|| = √(Σ(B[i]²))
    3. Divide: (A · B) / (||A|| × ||B||)
    
    Returns: Float between 0 and 1 (for normalized vectors)
    """
    # Step 1: Calculate dot product (sum of element-wise multiplication)
    dot_product = np.dot(vec1, vec2)
    
    # Step 2: Calculate magnitudes (Euclidean norms)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    # Step 3: Compute cosine similarity
    # Avoid division by zero
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    
    similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
    return similarity

print("="*100)
print("COSINE SIMILARITY CALCULATIONS")
print("="*100)
print()

similarities = []

for word1, word2, relationship in word_pairs:
    vec1 = embeddings[word1]
    vec2 = embeddings[word2]
    
    # Calculate similarity
    similarity = cosine_similarity(vec1, vec2)
    similarities.append((word1, word2, relationship, similarity))
    
    # Calculate components for educational purposes
    dot_prod = np.dot(vec1, vec2)
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    
    print(f"Pair: ({word1:10} ↔ {word2:10})")
    print(f"Relationship Type: {relationship}")
    print(f"Dot Product (A · B): {dot_prod:.6f}")
    print(f"Magnitude of {word1}: ||A|| = {mag1:.6f}")
    print(f"Magnitude of {word2}: ||B|| = {mag2:.6f}")
    print(f"Cosine Similarity: {dot_prod:.6f} / ({mag1:.6f} × {mag2:.6f})")
    print(f"SIMILARITY SCORE: {similarity:.6f}")
    
    # Interpret similarity
    if similarity > 0.95:
        interpretation = "Nearly identical (perfect synonyms)"
    elif similarity > 0.85:
        interpretation = "Highly similar (synonyms or closely related)"
    elif similarity > 0.70:
        interpretation = "Moderately similar (related concepts)"
    elif similarity > 0.50:
        interpretation = "Somewhat similar (loose associations)"
    elif similarity > 0.30:
        interpretation = "Weakly similar (distant relationship)"
    else:
        interpretation = "Dissimilar (unrelated or opposite)"
    
    print(f"Interpretation: {interpretation}")
    print("-"*100)
    print()

# ============================================================================
# STEP 4: VISUALIZE WORD PAIRS SORTED BY SIMILARITY
# ============================================================================

print("="*100)
print("WORD PAIRS SORTED BY SIMILARITY (Highest to Lowest)")
print("="*100)
print()

# Sort by similarity score (descending)
sorted_pairs = sorted(similarities, key=lambda x: x[3], reverse=True)

print(f"{'Rank':<6} {'Word Pair':<25} {'Similarity':<12} {'Interpretation':<35} {'Type'}")
print("─"*100)

for rank, (word1, word2, rel_type, sim_score) in enumerate(sorted_pairs, 1):
    pair_str = f"({word1}, {word2})"
    
    if sim_score > 0.85:
        interpretation = "Highly Similar  ⭐⭐⭐"
    elif sim_score > 0.70:
        interpretation = "Similar         ⭐⭐"
    elif sim_score > 0.50:
        interpretation = "Moderately Similar ⭐"
    else:
        interpretation = "Dissimilar      "
    
    print(f"{rank:<6} {pair_str:<25} {sim_score:.6f}    {interpretation:<35} {rel_type}")

print()

# ============================================================================
# STEP 5: DETAILED MATHEMATICS OF COSINE SIMILARITY
# ============================================================================

print("="*100)
print("DETAILED MATHEMATICAL EXPLANATION: COSINE SIMILARITY")
print("="*100)
print()

math_explanation = """
WHAT IS COSINE SIMILARITY?
==========================

Cosine similarity measures the angle between two vectors in multi-dimensional space.
Instead of measuring distance (Euclidean distance), it measures the angle.

WHY NOT USE EUCLIDEAN DISTANCE?
- Distance depends on vector magnitude: longer vector = farther even if similar direction
- Example: v1=(1,1) and v2=(10,10) are parallel (same direction) but far apart in distance
- Cosine similarity: same result regardless of magnitude ✓

THE GEOMETRIC INTERPRETATION:
============================

In 2D space (for simplicity):
    v1 = (x1, y1)    v2 = (x2, y2)

    cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
    
Where θ is the angle between vectors:
- θ = 0° → cos(0°) = 1.0  [parallel, same direction]
- θ = 90° → cos(90°) = 0.0 [perpendicular, no relationship]
- θ = 180° → cos(180°) = -1.0 [opposite directions, antonyms]

STEP-BY-STEP CALCULATION EXAMPLE:
For (doctor, nurse):
================================

Step 1: Get the embedding vectors
   doctor = [0.8, 0.7, 0.3, 0.2, 0.1, 0.9, 0.4, 0.6, 0.5, 0.2]
   nurse  = [0.75, 0.72, 0.32, 0.25, 0.08, 0.88, 0.42, 0.58, 0.52, 0.18]

Step 2: Calculate dot product (A · B)
   = (0.8×0.75) + (0.7×0.72) + (0.3×0.32) + ... 
   = 0.60 + 0.504 + 0.096 + ... 
   = approximately 4.89

Step 3: Calculate magnitude of doctor vector
   ||doctor|| = √(0.8² + 0.7² + 0.3² + 0.2² + 0.1² + 0.9² + 0.4² + 0.6² + 0.5² + 0.2²)
              = √(0.64 + 0.49 + 0.09 + 0.04 + 0.01 + 0.81 + 0.16 + 0.36 + 0.25 + 0.04)
              = √3.89 ≈ 1.972

Step 4: Calculate magnitude of nurse vector
   ||nurse|| = similarly computed ≈ 1.958

Step 5: Apply cosine similarity formula
   similarity = 4.89 / (1.972 × 1.958)
              = 4.89 / 3.86
              ≈ 0.9665

HIGH SIMILARITY! (0.9665) → doctor and nurse are semantically similar

WHY EMBEDDINGS WORK:
====================

Pre-trained word embeddings (Word2Vec, GloVe) learn these relationships by:

1. CONTEXT WINDOW APPROACH (Word2Vec)
   - Train on billions of words
   - For each word, look at surrounding words (context window = ±5 words)
   - Words appearing in similar contexts get similar vectors
   - "doctor" always appears near "hospital", "patient", "nurse"
   - "nurse" also appears near same words
   → They develop similar embedding vectors!

2. CO-OCCURRENCE MATRIX (GloVe)
   - Count how often word pairs appear together in corpus
   - Words with similar co-occurrence patterns get similar vectors
   - "king" and "queen" appear with similar words
   - "good" and "bad" appear in different contexts → different vectors

3. NEURAL NETWORK TRAINING
   - Skip-gram model: predict context words from target word
   - Continuous Bag of Words: predict target from context
   - Through backpropagation, embeddings learn semantic relationships

FAMOUS ANALOGIES (Properties of Embeddings):
============================================

Word embeddings capture rich relationships enabling analogies:

   king - man + woman ≈ queen
   Paris - France + Germany ≈ Berlin  
   good - bad + terrible ≈ awful

These work because semantic differences between words are preserved as vector differences!

LIMITATIONS:
============

✗ Polysemy: Words with multiple meanings get single vector
  - "bank" (financial) vs "bank" (river) → same embedding
  - Modern solutions: Contextualized embeddings (BERT) change vector per context

✗ Training Data Bias: Embeddings encode societal biases from training corpus
  - "programmer" more similar to "man" than "woman" (historical gender bias)
  - This reflects biases in training data

✗ Static Vectors: Don't account for context in sentences
  - Solution: BERT, ELMo provide context-dependent embeddings

✗ No Compositional Meaning: Averaging word vectors for phrases doesn't always work
  - "New York" ≠ average("New", "York")
"""

print(math_explanation)

# ============================================================================
# STEP 6: ANALOGY RESOLUTION (BONUS)
# ============================================================================

print()
print("="*100)
print("WORD VECTOR ANALOGIES (Vector Arithmetic)")
print("="*100)
print()

# Demonstrate vector arithmetic relationships
# In real Word2Vec embeddings: king - man + woman ≈ queen

relations = [
    ("doctor", "patient", "nurse", "person"),
    ("king", "man", "queen", "woman"),
    ("cat", "feline", "dog", "canine")
]

print("Vector Analogy Formula: a - b + c ≈ d")
print("If: a is to b as c is to d")
print("-"*100)
print()

for a, b, c, d in relations:
    vec_a = embeddings[a]
    vec_b = embeddings[b]
    vec_c = embeddings[c]
    
    # Compute analogy result
    analogy_vector = vec_a - vec_b + vec_c
    
    # Find closest word
    vec_d = embeddings[d]
    similarity_to_d = cosine_similarity(analogy_vector, vec_d)
    
    print(f"Analogy: {a} - {b} + {c} ≈ {d}")
    print(f"  Vector calculation: {analogy_vector[:5]}... (truncated)")
    print(f"  Similarity to '{d}': {similarity_to_d:.6f}")
    print()

print()
print("="*100)
print("CONCLUSION: SEMANTIC EMBEDDINGS")
print("="*100)
print("""
Word embeddings capture semantic relationships in vector space. Similar words have
similar vectors, enabling effective similarity calculations through cosine distance.
These representations form the foundation for modern NLP tasks including:
- Machine translation
- Sentiment analysis  
- Named entity recognition
- Semantic search
- Text clustering

Advanced methods (BERT, GPT) extend these concepts with context awareness and
hierarchical understanding of language.
""")
