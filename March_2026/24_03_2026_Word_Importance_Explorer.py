"""
TF-IDF Feature Extraction for Text Analysis
============================================
Author: AI Assignment Module
Date: 24 March 2026

This script demonstrates TF-IDF (Term Frequency-Inverse Document Frequency)
feature extraction from text documents using scikit-learn.

TF-IDF FORMULA EXPLANATION:
==========================
TF (Term Frequency) = (Number of times term t appears in document) / (Total number of terms in document)
    - Measures how frequently a word appears in a specific document
    - High TF = word is important in that document

IDF (Inverse Document Frequency) = log(Total number of documents / Number of documents containing term t)
    - Measures how rare/common a word is across all documents
    - High IDF = word is rare (appears in few documents)
    - Low IDF = word is common (appears in many documents)

TF-IDF = TF × IDF
    - Combines both metrics
    - High TF-IDF = word is frequent in document AND rare across corpus
    - Low TF-IDF = word is common across documents (less informative)

WHY TF-IDF IS USEFUL:
- Eliminates common stop words (the, is, and) that have high TF but low IDF
- Emphasizes unique/distinctive words in documents
- Used for: document similarity, search ranking, text classification
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# ============================================================================
# STEP 1: CREATE 5 SAMPLE DOCUMENTS ON DIFFERENT TOPICS
# ============================================================================

documents = {
    "Sports": """Cricket is a sport that captivates millions of fans worldwide with its rich history 
    spanning over four centuries. The Indian Premier League has revolutionized the game with its fast-paced 
    T20 format, attracting global talent and massive television audiences. Players like Virat Kohli and 
    Rohit Sharma have become household names, inspiring young athletes to pursue the sport. Cricket combines 
    strategy, athleticism, and skill as batsmen face bowlers in intense contests. Stadium crowds reach 
    deafening levels of excitement. Professional cricket teams compete in multiple formats: Test matches 
    lasting five days, One Day Internationals of fifty overs, and Twenty20 leagues. Cricket training 
    includes batting practice, bowling techniques, and fielding drills. The sport demands mental toughness, 
    focus, and teamwork.""",
    
    "Politics": """Political systems across democracies function through regular elections where citizens vote 
    for representatives. Election campaigns involve candidates debating policies on healthcare, education, 
    and economy. Political parties develop manifestos addressing voter concerns and national priorities. 
    Government institutions include legislatures, executive branches, and judicial systems operating with 
    checks and balances. Political leaders shape policies affecting millions of people daily. National 
    assemblies debate laws, pass budgets, and oversee government agencies. Political discourse involves 
    ideologies ranging from left-wing to right-wing perspectives. Citizens engage through voting, activism, 
    and public participation in democratic processes. Elections determine which party controls government 
    and implements their policies for the next term.""",
    
    "Technology": """Artificial intelligence and machine learning are transforming industries through 
    automation and pattern recognition algorithms. Neural networks process massive amounts of data to make 
    predictions and decisions. Python has become the dominant programming language for AI development due to 
    libraries like TensorFlow and PyTorch. Cloud computing platforms provide scalable infrastructure for 
    training complex models. Software development uses version control systems like Git for collaboration. 
    Cybersecurity protects systems from hackers and data breaches using encryption techniques. Databases store 
    and retrieve information efficiently using SQL queries. Mobile applications run on smartphones and tablets, 
    serving billions of users globally. Technology companies invest heavily in research and development of 
    emerging technologies like quantum computing and blockchain.""",
    
    "Entertainment": """Movies and television shows provide entertainment through storytelling and visual 
    effects that transport audiences to different worlds. Actors deliver performances that connect emotionally 
    with viewers and create memorable characters. Directors control the overall vision of films, managing budgets 
    and creative decisions. Streaming platforms like Netflix and Amazon Prime have disrupted traditional cinema 
    distribution models. Comedy shows and dramas explore human experiences and relationships in compelling narratives. 
    Music concerts and live performances bring artists close to enthusiastic fans. Cinema production involves 
    cinematography, sound design, and editing to create immersive experiences. Box office revenue measures a 
    film's commercial success and popularity. Celebrity actors become cultural icons influencing fashion and 
    lifestyle trends among audiences.""",
    
    "Science": """Biology explores the structure and function of living organisms through cellular and molecular 
    research methods. DNA contains genetic information that determines organism characteristics and traits. Evolution 
    explains how species adapt and change over millions of years through natural selection processes. Chemistry studies 
    matter and reactions between elements and compounds forming new substances. Physics describes forces, energy, and 
    motion through mathematical equations and laws. Experiments involve hypotheses, data collection, and analysis to 
    draw scientific conclusions. Microscopes reveal microscopic structures invisible to human eyes. Climate science 
    investigates atmospheric patterns and global warming effects on ecosystems. Medical research develops treatments 
    and vaccines to cure diseases and improve human health. Scientific peer review ensures research quality and 
    validity before publication in journals."""
}

# ============================================================================
# STEP 2: INITIALIZE TFIDFVECTORIZER AND FIT ON DOCUMENTS
# ============================================================================

# TfidfVectorizer automatically:
# 1. Tokenizes text into individual words
# 2. Removes common English stop words (the, is, a, etc.)
# 3. Converts to lowercase
# 4. Calculates TF-IDF scores for each word in each document
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

# Fit the vectorizer on documents and transform to TF-IDF matrix
doc_list = list(documents.values())
doc_names = list(documents.keys())
tfidf_matrix = vectorizer.fit_transform(doc_list)

print("="*80)
print("TF-IDF FEATURE EXTRACTION ANALYSIS")
print("="*80)
print()

# ============================================================================
# STEP 3: EXTRACT AND PRINT VOCABULARY
# ============================================================================

# The vocabulary is the set of unique words learned from documents
vocabulary = vectorizer.get_feature_names_out()
print(f"VOCABULARY (Total unique words: {len(vocabulary)})")
print("-" * 80)
print(f"Sample of learned words: {list(vocabulary[:20])}")
print()

# ============================================================================
# STEP 4: PRINT TF-IDF MATRIX
# ============================================================================

# Convert sparse matrix to dense array for display
tfidf_array = tfidf_matrix.toarray()

print("TF-IDF MATRIX SHAPE: (documents × features)")
print(f"Dimensions: {tfidf_array.shape[0]} documents × {tfidf_array.shape[1]} features")
print()
print("TF-IDF SCORES (First 10 words, all documents):")
print("-" * 80)

# Create a readable DataFrame for visualization
tfidf_df = pd.DataFrame(
    tfidf_array[:, :10],
    columns=vocabulary[:10],
    index=doc_names
)
print(tfidf_df.round(4))
print()

# ============================================================================
# STEP 5: IDENTIFY MOST IMPORTANT WORDS IN EACH DOCUMENT
# ============================================================================

print("TOP-5 MOST IMPORTANT WORDS PER DOCUMENT (by TF-IDF Score)")
print("="*80)

# For each document, find words with highest TF-IDF scores
top_words_per_doc = {}
for doc_idx, doc_name in enumerate(doc_names):
    # Get TF-IDF scores for this document
    scores = tfidf_array[doc_idx]
    
    # Get indices of top 5 words (highest scores)
    top_indices = np.argsort(scores)[-5:][::-1]  # Sort descending
    
    # Get word names and their scores
    top_words = [(vocabulary[idx], scores[idx]) for idx in top_indices]
    top_words_per_doc[doc_name] = top_words
    
    # Print formatted table
    print()
    print(f"📄 {doc_name.upper()}")
    print("-" * 80)
    print(f"{'Rank':<6} {'Word':<20} {'TF-IDF Score':<15} {'Significance':<20}")
    print("-" * 80)
    
    for rank, (word, score) in enumerate(top_words, 1):
        # Interpret score significance
        if score > 0.15:
            significance = "Highly Important ⭐"
        elif score > 0.10:
            significance = "Important ✓"
        else:
            significance = "Moderately Important"
        
        print(f"{rank:<6} {word:<20} {score:<15.6f} {significance:<20}")

print()
print()

# ============================================================================
# STEP 6: EXPLAIN WHY CERTAIN WORDS SCORE HIGH
# ============================================================================

print("DETAILED ANALYSIS: Why Certain Words Score High")
print("="*80)
print()

analysis_text = """
WHY DO CERTAIN WORDS HAVE HIGH TF-IDF SCORES?

1. RARE + FREQUENT IN DOCUMENT (High Score)
   Example: "cricket" in Sports document
   - Appears multiple times in Sports document (high TF)
   - Appears in NO other documents (high IDF, rare word)
   - TF-IDF = high × high = VERY HIGH ✓
   - These are the most discriminative words!

2. COMMON ACROSS DOCUMENTS (Low Score)
   Example: "system" appears in Politics AND Technology
   - Appears in document (contributes to TF)
   - Appears in multiple documents (lowers IDF, common word)
   - TF-IDF = moderate × low = LOW ✗
   - Stop words like "and", "the" would have even lower scores

3. UNIQUE TO TOPIC (Medium-High Score)
   Example: "algorithm" in Technology document
   - Unique identifier for the document's topic
   - Appears once or twice (medium TF)
   - Appears in few documents (high IDF)
   - TF-IDF = medium × high = MEDIUM-HIGH ✓

4. STOP WORDS ARE ELIMINATED
   Words like "is", "the", "a", "and" are removed by TfidfVectorizer
   - These appear in ALL documents (IDF = log(5/5) ≈ 0)
   - Automatically filtered out during preprocessing
   - Don't contribute meaningful information for classification

TF-IDF IS NOT PERFECT FOR:
- Understanding word context or meaning
- Capturing word order or sentence structure
- Recognizing synonyms (king ≠ monarch, but similar meaning)
- Modern NLP might use word embeddings (Word2Vec, BERT) instead
"""

print(analysis_text)
print()

# ============================================================================
# STEP 7: DOCUMENT SIMILARITY USING TF-IDF
# ============================================================================

print("DOCUMENT SIMILARITY ANALYSIS")
print("="*80)
print()

# Calculate cosine similarity between documents
# Similarity = dot product of normalized vectors (TF-IDF already normalized)
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tfidf_matrix)
similarity_df = pd.DataFrame(
    similarity_matrix,
    columns=doc_names,
    index=doc_names
)

print("Document Similarity Matrix (0 = completely different, 1 = identical):")
print("-" * 80)
print(similarity_df.round(4))
print()

print("INTERPRETATION:")
for i in range(len(doc_names)):
    for j in range(i+1, len(doc_names)):
        sim = similarity_matrix[i, j]
        doc1, doc2 = doc_names[i], doc_names[j]
        if sim > 0.1:
            status = "Related topics"
        else:
            status = "Distinct topics"
        print(f"{doc1:15} ↔ {doc2:15}: {sim:.4f} ({status})")

print()
print("="*80)
print("CONCLUSION:")
print("="*80)
print("""
TF-IDF successfully extracted distinguishing features from documents.
- Each document has unique high-scoring words
- Stop words (common across documents) have zero or near-zero scores
- Rare but frequent-in-document words score highest
- This technique is foundational for text classification and information retrieval
""")
