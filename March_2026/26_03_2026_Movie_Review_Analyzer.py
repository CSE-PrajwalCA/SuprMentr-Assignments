"""
Sentiment Analysis with VADER (Valence Aware Dictionary and sEntiment Reasoner)
================================================================================
Author: AI Assignment Module
Date: 26 March 2026

This script demonstrates sentiment analysis using VADER sentiment analyzer
from the Natural Language Toolkit (NLTK) library.

VADER ALGORITHM OVERVIEW:
=========================
VADER is a specialized sentiment analysis tool that:
1. Analyzes sentiment lexicons (predefined word lists with sentiment values)
2. Considers punctuation, capitalization, and intensifiers
3. Returns four scores: negative, neutral, positive, compound

COMPOUND SCORE INTERPRETATION:
- Range: [-1.0, +1.0]
- -1.0 = extremely negative
- -0.5 to -0.1 = somewhat negative
- -0.1 to +0.1 = neutral
- +0.1 to +0.5 = somewhat positive
- +1.0 = extremely positive

WHY USE VADER?
- Excellent for social media, reviews, and short text
- Handles emojis, slang, and colloquial language well
- Fast and doesn't require training data
- Interpretable results (humans understand scores)
- Works better than Naive Bayes on sentiment for social media

LIMITATIONS OF VADER:
- Can't understand complex sarcasm or irony
- Struggles with negations in long sentences
- Limited to English language
- May fail on domain-specific terminology
- Sensitive to word order changes
"""

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from datetime import datetime

# Download required VADER lexicon (required for first run)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# ============================================================================
# STEP 1: INITIALIZE VADER SENTIMENT ANALYZER
# ============================================================================

# Create analyzer instance
# This loads the pre-trained VADER sentiment lexicon
analyzer = SentimentIntensityAnalyzer()

# ============================================================================
# STEP 2: CREATE 5 SAMPLE MOVIE REVIEWS
# ============================================================================

reviews = [
    {
        "title": "Inception - Mind-Bending Masterpiece",
        "text": "Absolutely brilliant! This movie completely blew my mind! Christopher Nolan created a masterpiece that is both intellectually stimulating and visually stunning. The plot is intricate, the acting is phenomenal, and the cinematography is breathtaking. I loved every single moment. This is cinema at its finest!"
    },
    {
        "title": "The Room - Worst Movie Ever",
        "text": "This film is absolutely terrible and a complete waste of time. The acting is dreadful, the dialogue is nonsensical, and the plot makes no sense whatsoever. I hated every minute of it. Never have I seen such a poorly made movie. Total disaster! Don't watch this garbage."
    },
    {
        "title": "Avengers: Endgame - Great Action",
        "text": "Pretty good movie overall. The action scenes were entertaining and some characters had good moments. The ending was okay but not perfect. Some parts dragged on and some jokes fell flat. It's a solid blockbuster that most people would enjoy, but it's not the best film ever made. Worth watching if you have time."
    },
    {
        "title": "Parasite - A Triumph in Storytelling",
        "text": "I absolutely loved this film! The director brilliantly weaves together themes of class, desperation, and morality. The performances are incredible, especially from the entire ensemble cast. The cinematography is gorgeous and the soundtrack perfectly complements each scene. This is genuine art that moves the soul. A true masterpiece of modern cinema!"
    },
    {
        "title": "The Disaster - Critically Panned",
        "text": "Disappointingly bad. The movie has zero redeeming qualities. The script is bland, the characters are unlikeable, and the pacing is painfully slow. I couldn't wait for it to end. This is one of the worst films I have ever seen. Absolutely dreadful waste of money. Do not watch!"
    }
]

# ============================================================================
# STEP 3: ANALYZE SENTIMENT FOR EACH REVIEW
# ============================================================================

print("="*100)
print("VADER SENTIMENT ANALYSIS ON MOVIE REVIEWS")
print("="*100)
print(f"Analysis Date: {datetime.now().strftime('%d %B %Y')}")
print()

results = []

for idx, review in enumerate(reviews, 1):
    # Get sentiment scores from VADER
    # returns: {'neg': float, 'neu': float, 'pos': float, 'compound': float}
    scores = analyzer.polarity_scores(review['text'])
    
    results.append({
        'title': review['title'],
        'text': review['text'],
        'scores': scores
    })
    
    # Extract individual scores
    neg_score = scores['neg']
    neu_score = scores['neu']
    pos_score = scores['pos']
    compound_score = scores['compound']
    
    # Determine sentiment classification based on compound score
    if compound_score >= 0.05:
        sentiment_label = "POSITIVE 😊"
        emoji = "😀"
    elif compound_score <= -0.05:
        sentiment_label = "NEGATIVE 😞"
        emoji = "😢"
    else:
        sentiment_label = "NEUTRAL 😐"
        emoji = "😐"
    
    # Print review analysis
    print(f"\n{'─'*100}")
    print(f"REVIEW #{idx}: {review['title'].upper()}")
    print(f"{'─'*100}")
    print(f"Text: {review['text']}")
    print()
    
    # Print sentiment scores
    print(f"SENTIMENT SCORES:")
    print(f"  • Negative Score: {neg_score:.4f}  (probability of negative sentiment)")
    print(f"  • Neutral Score:  {neu_score:.4f}  (probability of neutral sentiment)")
    print(f"  • Positive Score: {pos_score:.4f}  (probability of positive sentiment)")
    print(f"  • Compound Score: {compound_score:+.4f}  (overall sentiment intensity)")
    print()
    
    # Print sentiment interpretation
    print(f"SENTIMENT CLASSIFICATION: {sentiment_label} {emoji}")
    print()
    
    # Detailed explanation of why VADER assigned this sentiment
    print(f"ANALYSIS EXPLANATION:")
    print(f"{'-'*100}")
    
    if compound_score >= 0.75:
        explanation = f"""
    This review receives a highly positive score ({compound_score:+.4f}) because:
    ✓ Multiple strong positive words: "brilliant", "loved", "phenomenal", "masterpiece"
    ✓ Exclamation marks indicate enthusiasm (VADER detects punctuation!)
    ✓ Superlatives: "best", "finest" amplify positive sentiment
    ✓ Overall tone: Words are emotionally charged and unambiguously positive
    
    WHY VADER WORKS HERE: The review uses clear positive language with no contradictions
        """
    elif compound_score >= 0.05:
        explanation = f"""
    This review scores as positive ({compound_score:+.4f}) due to:
    ✓ Presence of positive words: "good", "entertaining", "solid", "worth watching"
    ✓ Mixed sentiment throughout (some praise, some criticism)
    ✓ Overall leaning toward recommendation
    
    WHY VADER WORKS HERE: Despite criticisms, more positive expressions occur overall
        """
    elif compound_score <= -0.75:
        explanation = f"""
    This review is rated extremely negative ({compound_score:+.4f}) because:
    ✓ Intense negative words: "terrible", "dreadful", "garbage", "disaster"
    ✓ Strong intensifiers: "absolutely", "completely"
    ✓ Repetitive negative language throughout
    ✓ No positive counterpoints
    
    WHY VADER WORKS HERE: Overwhelmingly negative language with no redeeming sentiment
        """
    elif compound_score <= -0.05:
        explanation = f"""
    This review is negative ({compound_score:+.4f}) because:
    ✓ Negative words: "disappointingly bad", "zero redeeming qualities"
    ✓ Criticisms outweigh any positive elements
    ✓ Strong negative descriptors
    
    WHY VADER WORKS HERE: Clear disapproval despite some neutral language
        """
    else:
        explanation = f"""
    This review is neutral ({compound_score:+.4f}) because:
    ✓ Balanced mix of positive and negative opinions
    ✓ Moderate language without strong intensifiers
    ✓ Both praise and criticism present
    
    WHY VADER WORKS HERE: Neither strongly positive nor negative overall
        """
    
    print(explanation)


# ============================================================================
# STEP 4: DISCUSSION OF VADER ALGORITHM AND LIMITATIONS
# ============================================================================

print()
print("="*100)
print("DETAILED EXPLANATION: VADER ALGORITHM AND LIMITATIONS")
print("="*100)
print()

algorithm_discussion = """
HOW VADER WORKS:
================

1. LEXICON-BASED APPROACH
   - VADER uses a pre-compiled sentiment lexicon (word list with sentiment values)
   - Each word has a sentiment score: strong negative (-4 to -1), neutral (0), strong positive (+1 to +4)
   - Example: "excellent" = +2.0, "terrible" = -2.0

2. MODIFIERS AND INTENSIFIERS
   - VADER considers words that amplify sentiment
   - Intensifiers: "very", "extremely", "absolutely" multiply scores
   - Negations: "not good" reverses polarity
   - Capitalization: "EXCELLENT" scores higher than "excellent"

3. PUNCTUATION
   - Exclamation marks: ! increase sentiment intensity
   - Question marks: ? reduce sentiment intensity
   - Multiple punctuation: !!! amplifies further

4. COMPOUND SCORE CALCULATION
   - Compound = normalized, weighted sum of all sentiment scores
   - Range normalized to [-1, +1]
   - Formula: compound = Σ(weighted sentiment scores) / √(Σ(weights)²)


WHEN VADER EXCELS:
===================

✓ SOCIAL MEDIA & REVIEWS: Designed for informal text, emojis, slang
✓ SHORT TEXT: Works best with individual sentences or short reviews
✓ EXPLICIT SENTIMENT: Clear positive/negative language
✓ SPEED: Fast analysis, no need for training data
✓ INTERPRETABILITY: Humans can understand why a score was assigned


WHEN VADER FAILS:
==================

✗ SARCASM & IRONY: "Oh, just perfect!" (says sarcastically) → marked positive
   VADER sees "perfect" (+) but misses sarcastic tone
   EXAMPLE: "I love waiting in traffic" gets positive score despite negative meaning

✗ DOMAIN-SPECIFIC LANGUAGE: "sick" in slang (positive) vs medical context
   VADER trained on general English, not domain jargon

✗ COMPLEX NEGATIONS: "The movie is not bad, but it's also not particularly good"
   VADER struggles with double negatives and complex sentence structures

✗ CONTEXT-DEPENDENT MEANINGS: "The plot is thin" = negative, but "her hair is thin"
   Words have different sentiments in different contexts

✗ SENTENCE-LEVEL INCONSISTENCIES: Multi-sentiment sentences confuse the analyzer
   Example: "The movie was amazing but the ending sucked" 
   VADER tries to balance both, may not recognize primary negative ending

✗ CULTURAL AND LINGUISTIC DIFFERENCES: 
   - Built on English sentiment, works poorly in other languages
   - Cultural idioms may not be recognized


CASE STUDIES OF VADER FAILURES:
================================

FAILURE 1: SARCASM
Review: "Oh sure, this movie is absolutely great!" [sarcastic]
VADER Result: POSITIVE (sees "absolutely" + "great")
Why Failed: Word-level analysis misses sarcastic tone
Human Intent: Actually expressing disapproval

FAILURE 2: CONTEXTUAL IRONY  
Review: "I'm so excited to go to the dentist!" [not excited]
VADER Result: POSITIVE (sees "excited")
Why Failed: No context understanding
Human Intent: Actually expressing sarcasm

FAILURE 3: NEGATION IN COMPLEX SENTENCE
Review: "While parts had their merits, the film wasn't good enough to recommend"
VADER Result: Might show conflicted scores
Why Failed: "not good" negation competes with positive "merits"
Human Intent: Overall negative recommendation


IMPROVEMENTS OVER VADER:
=========================

Modern solutions use:

1. TRANSFORMER MODELS (BERT, RoBERTa)
   - Understand context and word relationships
   - Learn from large datasets (billions of words)
   - Handle sarcasm and complex language better

2. FINE-TUNED MODELS
   - Trained on domain-specific data (movie reviews, product reviews, tweets)
   - Learn what features matter in specific domains
   - Higher accuracy than generic VADER

3. DEEP LEARNING APPROACHES
   - LSTM, Transformer architectures
   - Understand word order and sentence structure
   - Can capture sentence-level meanings
   - Computationally expensive but more accurate

CONCLUSION:
===========
VADER is excellent for quick, interpretable sentiment analysis of short texts.
For high-accuracy applications, consider modern NLP models like BERT or domain-specific
fine-tuned models. Choose based on: accuracy needs, computational resources, interpretability
requirements, and data availability.
"""

print(algorithm_discussion)

print()
print("="*100)
print("SUMMARY STATISTICS")
print("="*100)
print()

# Calculate overall statistics
avg_compound = sum(r['scores']['compound'] for r in results) / len(results)
positive_count = sum(1 for r in results if r['scores']['compound'] >= 0.05)
negative_count = sum(1 for r in results if r['scores']['compound'] <= -0.05)
neutral_count = len(results) - positive_count - negative_count

print(f"Total Reviews Analyzed: {len(results)}")
print(f"Positive Reviews: {positive_count} ({positive_count/len(results)*100:.1f}%)")
print(f"Negative Reviews: {negative_count} ({negative_count/len(results)*100:.1f}%)")
print(f"Neutral Reviews: {neutral_count} ({neutral_count/len(results)*100:.1f}%)")
print(f"Average Compound Score: {avg_compound:+.4f}")
print()
print("Analysis Complete ✓")
