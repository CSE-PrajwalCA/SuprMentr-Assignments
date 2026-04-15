"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TEXT DATA QUALITY ISSUES & CHALLENGES                    ║
║                             March 20, 2026                                  ║
║                                                                              ║
║  PROBLEM IDENTIFICATION EXERCISE:                                           ║
║  This script demonstrates common data quality issues in NLP (Natural        ║
║  Language Processing). Each messy sentence is analyzed to identify          ║
║  problems and discuss impact on machine learning models.                    ║
║                                                                              ║
║  Note: NO SOLUTIONS PROVIDED - This is about understanding the problems    ║
║        and why they matter. Solutions come in the next module.             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re

print("="*85)
print("TEXT DATA QUALITY CHALLENGES IN NLP")
print("Why Raw Text is a Nightmare for Machine Learning")
print("="*85)

# Create list of 20 messy sentences with common NLP problems
messy_sentences = [
    # 1. Extra whitespace
    "  hello   world  ",
    
    # 2. Contractions (informal English)
    "I'm gonna go",
    
    # 3. Inconsistent casing (uppercase/lowercase)
    "HELLO world",
    
    # 4. Special characters and punctuation
    "hello, world!!!",
    
    # 5. Spelling errors and typos
    "helo wrld",
    
    # 6. Unicode characters and accents
    "café is located in Montréal",
    
    # 7. Mixed whitespace types (tabs, newlines)
    "hello\tworld\ntest",
    
    # 8. Standalone numbers and symbols
    "Contact: 123-456-7890 for details",
    
    # 9. HTML entities and encoding issues
    "Price is &pound;50 &amp; &lt;50% off",
    
    # 10. URLs embedded in text
    "Visit https://www.example.com/page?id=123 for more info",
    
    # 11. Email addresses
    "Email me at john.doe@example.com please",
    
    # 12. Multiple punctuation marks
    "Really??? That's incredible!!!",
    
    # 13. Abbreviated words and acronyms
    "The USA govt. passed new legislation",
    
    # 14. Repeated characters
    "Soooooo cooooool!!!",
    
    # 15. Mixed languages
    "Hello! Bonjour! 你好! Hola!",
    
    # 16. Slang and internet speak
    "lol ur awesome tbh",
    
    # 17. Possessives and apostrophe variations
    "That's John's book; it's his",
    
    # 18. Leading/trailing punctuation
    "...hello... \"world\" ...",
    
    # 19. Timestamps embedded in text
    "Meeting at 2024-03-20 14:30:00 in room 101",
    
    # 20. Professional jargon and domain-specific terms
    "The ML model's F1-score improved by 3.5% using XGBoost",
]

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: DETAILED ANALYSIS OF EACH PROBLEM
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*85)
print("DETAILED ANALYSIS OF 20 TEXT QUALITY PROBLEMS")
print("="*85)

analyses = [
    {
        'number': 1,
        'title': 'Extra & Irregular Whitespace',
        'sentence': "  hello   world  ",
        'issues': [
            "Leading whitespace (2 spaces at beginning)",
            "Irregular spacing between words (3 spaces instead of 1)",
            "Trailing whitespace (2 spaces at end)",
            "Inconsistent use of space characters",
            "May confuse tokenizers (split on whitespace) creating spurious tokens"
        ]
    },
    {
        'number': 2,
        'title': 'Contractions (Informal English)',
        'sentence': "I'm gonna go",
        'issues': [
            "\"I'm\" = contraction of \"I am\" with apostrophe",
            "\"gonna\" = informal contraction of \"going to\"",
            "Contractions are non-standard words not in formal dictionaries",
            "ML model may not recognize \"I'm\" as equivalent to \"I am\"",
            "Sentiment analysis: hard to detect without expansion (e.g., negations in \"don't\")",
            "Word embeddings: contractions get different vectors than expanded forms"
        ]
    },
    {
        'number': 3,
        'title': 'Inconsistent Casing (UPPERCASE/lowercase)',
        'sentence': "HELLO world",
        'issues': [
            "\"HELLO\" in all caps (shouting, emphasis) vs \"world\" lowercase",
            "Machine sees \"HELLO\" and \"hello\" as DIFFERENT words",
            "Doubles model vocabulary size unnecessarily (hello, Hello, HELLO, hELLO all separate)",
            "Word embeddings: untrained on rare variations (all-caps word very infrequent)",
            "Named entities: proper nouns (names, places) need caps but confuse algorithms",
            "Negation in all caps has different meaning (for emphasis/anger)"
        ]
    },
    {
        'number': 4,
        'title': 'Special Characters & Excessive Punctuation',
        'sentence': "hello, world!!!",
        'issues': [
            "Comma after \"hello\" (normal punctuation)",
            "Triple exclamation marks \"!!!\" (emphasis, emotion)",
            "Punctuation carries semantic meaning (! = excitement, ? = question)",
            "Different tokenizers handle punctuation differently (keep vs remove)",
            "Multiple punctuation confuses parsing (!!!??? has no semantic difference)",
            "Models may overweight rare punctuation patterns"
        ]
    },
    {
        'number': 5,
        'title': 'Typos & Spelling Errors',
        'sentence': "helo wrld",
        'issues': [
            "\"helo\" instead of \"hello\" (missing 'l')",
            "\"wrld\" instead of \"world\" (missing vowels)",
            "Model trained on correct spellings likely has no vector for \"helo\"",
            "Unknown word handling: either skip (lose information) or use <UNK> token",
            "Spelling errors widespread in social media/user-generated content",
            "Hard to detect without spell-checker (some \"errors\" are valid slang)",
            "Edit distance algorithms could find closest match but computationally expensive"
        ]
    },
    {
        'number': 6,
        'title': 'Unicode Characters & Diacritical Marks',
        'sentence': "café is located in Montréal",
        'issues': [
            "\"café\" has accent over 'e' (é, Unicode character U+00E9)",
            "\"Montréal\" has accent over 'e' (é)",
            "Model trained on English text may not have vectors for accented characters",
            "Different encodings (UTF-8, ASCII, Latin-1) handle accents differently",
            "Removing accents (normalization) loses information but helps standardization",
            "Non-Latin characters entirely problematic (Chinese, Arabic, emoji)",
            "Display issues: accented characters may render incorrectly on some systems"
        ]
    },
    {
        'number': 7,
        'title': 'Mixed Whitespace Types (tabs, newlines)',
        'sentence': "hello\\tworld\\ntest",
        'issues': [
            "Tab character (\\t) used instead of space",
            "Newline character (\\n) embedded in text",
            "Different whitespace types (space, tab, newline) treated differently",
            "Some tokenizers may not split on tabs/newlines correctly",
            "Sentence boundaries: newline should often mark sentence break",
            "Reading from different file formats (TSV vs CSV) creates this mess"
        ]
    },
    {
        'number': 8,
        'title': 'Standalone Numbers & Special Symbols',
        'sentence': "Contact: 123-456-7890 for details",
        'issues': [
            "Numeric sequence \"123-456-7890\" (phone number format)",
            "Numbers: should they be kept, removed, normalized, tokenized?",
            "Hyphenated numbers: is it one token or four (123, -, 456, -, 7890)?",
            "Context: phone number meaningless for sentiment/topic but important for contact extraction",
            "Different tokenizers split numbers differently",
            "Numeric embeddings: rare/large numbers OOV (out-of-vocabulary)"
        ]
    },
    {
        'number': 9,
        'title': 'HTML Entities & Encoding Issues',
        'sentence': "Price is &pound;50 &amp; &lt;50% off",
        'issues': [
            "\"&pound;\" = HTML entity for £ (pound symbol)",
            "\"&amp;\" = HTML entity for & (ampersand)",
            "\"&lt;\" = HTML entity for < (less-than)",
            "Raw HTML entities not decoded in preprocessing",
            "Text scraped from web often contains these entities",
            "Models expect decoded text (£, &, <) not entities"
        ]
    },
    {
        'number': 10,
        'title': 'URLs Embedded in Text',
        'sentence': "Visit https://www.example.com/page?id=123 for more info",
        'issues': [
            "Long URL \"https://www.example.com/page?id=123\"",
            "Contains special characters: /, :, ?, =, &",
            "Query parameters important for systems but noise for NLP",
            "Should be removed (noise) or extracted (useful for entity recognition)",
            "Domain names contain meaningful info (example.com = company name)",
            "URL length variable, making fixed-length input problematic"
        ]
    },
    {
        'number': 11,
        'title': 'Email Addresses',
        'sentence': "Email me at john.doe@example.com please",
        'issues': [
            "Email \"john.doe@example.com\"",
            "Contains special character @ and . (periods)",
            "Should be recognized as email (entity) vs treated as regular text",
            "Privacy concern: emails should be removed/redacted in many applications",
            "Tokenization: is it one token or split into components?"
        ]
    },
    {
        'number': 12,
        'title': 'Multiple/Excessive Punctuation',
        'sentence': "Really??? That's incredible!!!",
        'issues': [
            "Triple question marks \"???\" (emphasis, emotion)",
            "Triple exclamation \"!!!\" (strong emotion)",
            "Semantic meaning: \"Really?\" vs \"Really???\" both mean questioning",
            "But varying punctuation creates different tokens",
            "Internet speak: multiple punctuation normalized to single in formal writing",
            "Sentiment analysis: emotional punctuation should be detected"
        ]
    },
    {
        'number': 13,
        'title': 'Abbreviated Words & Acronyms',
        'sentence': "The USA govt. passed new legislation",
        'issues': [
            "\"USA\" = acronym (United States of America)",
            "\"govt.\" = abbreviation of \"government\" with period",
            "Period after abbreviation confuses sentence/word end detection",
            "Acronyms may not be in model's vocabulary (especially new ones)",
            "Context needed: USA could mean United States of America or World University Services",
            "Expansion difficult without domain knowledge"
        ]
    },
    {
        'number': 14,
        'title': 'Repeated Characters (Stretched Words)',
        'sentence': "Soooooo cooooool!!!",
        'issues': [
            "\"Soooooo\" = \"So\" with repeated 'o' (emphasis, emotion)",
            "\"cooooool\" = \"cool\" with repeated 'o's (enthusiastic emotion)",
            "Model sees \"Soooooo\" as different word from \"So\"",
            "Repeated characters indicate emotion/stress but lost if characters normalized",
            "Internet/social media language: very common in informal writing",
            "Sentiment analysis: these indicate strong emotion"
        ]
    },
    {
        'number': 15,
        'title': 'Mixed Languages',
        'sentence': "Hello! Bonjour! 你好! Hola!",
        'issues': [
            "\"Hello\" = English, \"Bonjour\" = French, \"你好\" = Chinese, \"Hola\" = Spanish",
            "Multiple languages in single sentence (language switching/code-switching)",
            "ML model trained on English won't have vectors for Chinese characters",
            "Different alphabets (Latin, Hanzi): require different encoding",
            "Machine translation systems need to identify language first",
            "Language detection models struggle with mixed-language text"
        ]
    },
    {
        'number': 16,
        'title': 'Internet Slang & Text Speak',
        'sentence': "lol ur awesome tbh",
        'issues': [
            "\"lol\" = laugh out loud (not in formal English dictionary)",
            "\"ur\" = informal spelling of \"your\" (texting shorthand)",
            "\"tbh\" = to be honest (internet acronym)",
            "Very common in social media but absent from formal text corpora",
            "Spell checkers flag as errors but are meaningful in context",
            "Modern language: corpora from 2010 won't have these"
        ]
    },
    {
        'number': 17,
        'title': 'Possessives & Apostrophe Variations',
        'sentence': "That's John's book; it's his",
        'issues': [
            "\"That's\" = contraction of \"That is\"",
            "\"John's\" = possessive form of John (John + apostrophe + s)",
            "\"it's\" = contraction of \"it is\"",
            "Apostrophes used for both contractions AND possessives",
            "Inconsistent handling: some systems expand all, others keep all",
            "Ambiguity: \"it's\" (it is) vs \"its\" (possessive) are commonly confused"
        ]
    },
    {
        'number': 18,
        'title': 'Leading/Trailing Punctuation & Quotes',
        'sentence': "...hello... \"world\" ...",
        'issues': [
            "Ellipsis \"...\" before word (indicates trailing off from previous text)",
            "Double quotes around \"world\" (emphasis or cited text)",
            "Ellipsis after first \"hello\"",
            "Should be removed? Should tokenize separately? Context dependent",
            "Quotation marks might indicate speaker/dialogue (important for NER)",
            "Different quote styles: \", ', `, « », depending on language/region"
        ]
    },
    {
        'number': 19,
        'title': 'Timestamps & Temporal Data',
        'sentence': "Meeting at 2024-03-20 14:30:00 in room 101",
        'issues': [
            "Timestamp \"2024-03-20 14:30:00\" (ISO format: YYYY-MM-DD HH:MM:SS)",
            "Context: timestamp meaningful for calendar/scheduling but noise for sentiment",
            "Different timestamp formats worldwide (MM/DD/YYYY vs DD/MM/YYYY vs ISO)",
            "Should be removed (noise) or parsed as structured data (entity)",
            "Room number \"101\" is numeric but categorical (room identifier)",
            "Mixed temporal and room number data needs careful parsing"
        ]
    },
    {
        'number': 20,
        'title': 'Professional Jargon & Domain-Specific Terms',
        'sentence': "The ML model's F1-score improved by 3.5% using XGBoost",
        'issues': [
            "\"ML\" = Machine Learning (jargon/acronym)",
            "\"F1-score\" = performance metric (hyphenated technical term)",
            "\"XGBoost\" = proper noun (software library name)",
            "Domain knowledge needed: these terms mean nothing to general audience",
            "Out-of-domain text: general model won't recognize these terms",
            "Tokenization: is \"F1-score\" one token or three?"
        ]
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# PRINT DETAILED ANALYSIS FOR EACH PROBLEM
# ═══════════════════════════════════════════════════════════════════════════

for analysis in analyses:
    print("\n" + "─"*85)
    print(f"PROBLEM {analysis['number']}: {analysis['title']}")
    print("─"*85)
    
    print(f"\nAKA: {analysis['sentence']!r}\n")
    
    print("Issues Identified:")
    for i, issue in enumerate(analysis['issues'], 1):
        print(f"  {i}. {issue}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: IMPACT ON MACHINE LEARNING MODELS
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*85)
print("WHY DO THESE TEXT QUALITY ISSUES MATTER FOR MACHINE LEARNING?")
print("="*85)

impact_explanation = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                    IMPACT OF UNCLEAN TEXT ON ML SYSTEMS                    ║
╚═════════════════════════════════════════════════════════════════════════════╝

1. VOCABULARY EXPLOSION & SPARSITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: Without cleaning, the model's vocabulary balloons unnecessarily.

Example: The word "hello" appears with variations:
  • hello, Hello, HELLO, hELLo (casing variations)
  • hello!, hello?, hello... (punctuation attached)
  • helo (typo)
  • "hello" (with quotes)
  
Result: Model treats 8+ variations as DIFFERENT WORDS instead of same meaning
  • Vocabulary goes from 10,000 core words to 50,000+ with noise
  • Word embeddings: each word needs separate learned vector
  • Sparse data: rare variations have insufficient training examples
  • Model performance degrades with sparse, high-dimensional features

Solution approach: Normalization to reduce vocabulary to core meaningful words


2. FEATURE DILUTION & WEAK SIGNAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: Same semantic meaning split across multiple representations.

Sentiment Analysis Example:
  Positive samples with variations:
    • "This movie is AMAZING!!!"
    • "this movie is amazing"
    • "This movie is amazing :)"
    • "This movie is amaaazing"
    
  Model sees different text, learns separate features, signal dispersed
  • 4 samples saying same thing look like 4 different examples
  • Model needs more training data to learn "amazing" = positive
  • With 100k samples but 5x variations = really only 20k unique meanings

Solution approach: Normalize to extract consistent signal from noisy data


3. OUT-OF-VOCABULARY (OOV) WORDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: Words not in training vocabulary cause model failures.

Examples:
  • Typos: \"helo\", \"wrld\" never seen during training
  • Slang: \"ur\", \"lol\" may not be in formal text corpora
  • Misspellings: Common in social media but model unprepared
  
Model handling options (all problematic):
  • Skip unknown word: Lose information (sentence becomes \"This is [UNK] is great\")
  • Replace with <UNK> token: All unknown words treated identically (loss of nuance)
  • Use character-level model: Slower, loses word-level semantics
  • Ignore: Crash on unknown words

Real-world example: Word2Vec model trained on 2010 news has no vector for:
  • \"blockchain\", \"cryptocurrency\", \"selfie\" (didn't exist in 2010)
  • Using such model on 2020 data: 1000s of OOV words = poor performance

Solution approach: Clean text and spell-check to match training vocabulary


4. PARSING & TOKENIZATION ERRORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: Unclean text produces wrong word boundaries and sentence boundaries.

Examples:
  Input: \"hello!!!world\"
  • Simple split on spaces: [\"hello!!!world\"] – ONE token (wrong!)
  • Ideal tokenization: [\"hello\", \"!\", \"!\", \"!\", \"world\"] or [\"hello\", \"world\"]
  
  Input: \"Mr. Smith went home.\"
  • Sentence boundary: period after \"Mr\" is NOT end of sentence
  • But simple rule \"split on \".\" causes problems
  • Need domain knowledge: \"Mr.\", \"Dr.\", \"Inc.\" are special
  
  Input: \"It's a USA govt. issue\"
  • Apostrophe: is part of contraction (keep) or punctuation (separate)?
  • Period: after abbreviation (don't split) not sentence end
  • Different tokenizer choices affect downstream processing

Result: Wrong tokenization = wrong features = wrong model predictions

Solution approach: Standardize text format before tokenization


5. SEMANTIC MEANING LOST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: Emotionally-charged text produces wrong interpretation.

Sentiment Analysis Problem:
  • \"sooooo boring\" vs \"boring\" – same word but intensifier lost
  • \"REALLY GREAT!!!\" vs \"really great\" – emphasis/emotion lost
  • \"it's not bad\" – double negative, sentiment inverted
  
Spam Detection Problem:
  • \"FREE!!!\" vs \"free\" – aggressive punctuation signals spam
  • Multiple punctuation \"??? !!! !!!\" is spam characteristic
  
Entity Recognition Problem:
  • \"John's book\" – need to recognize \"John's\" as person entity
  • Simply remove apostrophe: \"Johns book\" loses structure

Solution approach: Careful preprocessing that preserves semantic markers


6. ENCODING & REPRESENTATION ISSUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: Non-ASCII characters cause encoding problems.

Technical problems:
  • Python 2 vs Python 3 string handling differences
  • Database encoding (UTF-8 vs Latin-1 vs ASCII)
  • File reading: correct encoding must be specified
  
Example:
  • \"café\" in UTF-8 encoding: correct display
  • \"café\" read with ASCII encoding: displays as garbage
  • Model confusion: never seen correct \"café\" form
  
Multi-language problem:
  • \"hello 你好 مرحبا\" – different scripts completely
  • Tokenization rules different per language
  • Word embeddings: need separate for each language
  • Single model can't handle all scripts equally well

Solution approach: Force UTF-8 encoding, handle multi-language separately


7. TRAINING/TEST MISMATCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: Different preprocessing on training vs production data.

Scenario:
  • Model trained on clean text: \"The movie was great\"
  • Production input: \"The moovie was gr8!!!\"
  
If training preprocessing removes punctuation but production doesn't:
  • Training: \"the movie was great\"
  • Production: \"the moovie was gr8\"
  
Features don't match what model learned → Wrong predictions

Database of sentiment scores:
  • Training data: professional movie reviews (clean, well-written)
  • Production: Twitter reviews (messy, abbreviations, emojis, typos)
  • Data distribution COMPLETELY DIFFERENT
  • Model trained on clean data fails on messy real-world data

Solution approach: Ensure identical preprocessing pipeline for train and production


8. CURSE OF DIMENSIONALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: Expanded vocabulary makes feature space huge, requiring exponentially 
more training data.

Math:
  • Clean vocabulary: 10,000 words → 10,000-dimensional feature space
  • Unclean vocabulary: 50,000 words → 50,000-dimensional feature space
  • With N training samples and D dimensions:
    Data sufficiency: Need roughly N >> D (more samples than dimensions)
    
  • With 100,000 training samples:
    • Clean (D=10K): 100K > 10K ✓ (sufficient)
    • Unclean (D=50K): 100K > 50K ✓ (borderline)
    • Rare spelling: 50K is mostly zeros (sparsity curse)
  
Result: High variance, poor generalization, overfitting

Solution approach: Dimensionality reduction through aggressive preprocessing


9. COMPUTATIONAL EFFICIENCY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: Large vocabulary and noise makes training slow and memory-intensive.

Complexity:
  • Vocabulary size directly impacts:
    Training time: O(N × V × D) where N=samples, V=vocab, D=dimensions
    Memory: Word embeddings matrix V × D in memory always
    
  • Example: Word2Vec with 500-dimensional embeddings:
    • Clean vocab (10K): 10K × 500 = 5 million floats = 20 MB
    • Unclean vocab (50K): 50K × 500 = 25 million floats = 100 MB
    • 5x larger model = 5x more training time!
  
  • Production latency:
    If model lookup time depends on vocabulary size
    Clean: microseconds per word
    Unclean: milliseconds (10x slower)
  
Large-scale example: Processing 1 billion documents:
  • Preprocessing overhead: 2-3 hours (one-time)
  • Model training: hours to days depending on optimization
  • Cleanup text: converts days of training to hours!

Solution approach: Text preprocessing usually 80% of development time, huge ROI


10. BIAS & FAIRNESS ISSUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: Unclean text can hide or introduce bias in preprocessing.

Toxicity Detection Example:
  Raw text includes: \"User123\", \"@Twitter\", URLs, emoji
  • Usernames might be correlated with bot accounts
  • If preprocessing removes accounts, random sampling changes
  • Removes information about identity-based toxicity
  
  Inappropriate removal of \"usernames\" masks harassment patterns:
    • Targeted harassment: specific users mentioned
    • If preprocessing removes, model can't detect targeting
  
Sentiment Analysis Bias:
  • Cleaned text: \"This is great\"
  • Original: \"This is great for guys\" (gender-specific context)
  • Removing context → model learns gender-neutral sentiment
  • Applied to \"%This is great for ladies\" - wrongly predicts positive
  
Fairness principle: Preprocessing can inadvertently:
  • Hide discriminatory patterns (mask real bias)
  • Create artificial disparities (unequal treatment of groups)
  • Remove information important for fair assessment

Solution approach: Thoughtful preprocessing respecting fairness considerations


SUMMARY: Why Text Preprocessing Matters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Without proper preprocessing:
✗ Bloated vocabulary → sparse features → need more data → more computation
✗ Inconsistent representations → weak signal → poor model performance
✗ Tokenization errors → wrong input features → systematic errors
✗ Encoding issues → crashes, garbage output → system failures
✗ Train/test mismatch → works in lab, fails in production
✗ Semantic loss → wrong interpretation of intent/emotion → wrong predictions
✗ Unfair treatment → systematic bias against certain groups
✗ Slow inference → 10x slower predictions → can't serve users in real-time

With proper preprocessing:
✓ Consistent representation → strong signal → better learning
✓ Reduced vocabulary → denser features → less data needed
✓ Correct tokenization → right input features → consistent behavior
✓ Handled encoding → robust to text variations
✓ Train/test alignment → reliable production system
✓ Preserved semantics → accurate intent/emotion detection
✓ Fair treatment → equitable across groups
✓ Fast inference → sub-millisecond predictions → scales to millions

BOTTOM LINE: 80% of NLP project time is preprocessing/cleaning. It's not glamorous
but it's CRITICAL for production systems that work reliably on real-world messy data.
"""

print(impact_explanation)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: ADDITIONAL CATEGORIES OF PROBLEMS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*85)
print("ADDITIONAL PROBLEM CATEGORIES NOT COVERED IN DETAIL")
print("="*85)

additional_problems = """
BEYOND THE 20 EXAMPLES - Other Common Issues:

LINGUISTIC PROBLEMS:
─────────────────────
• Negation Handling: \"not good\" vs \"good\" – negation flips meaning
• Sarcasm: \"Oh, GREAT!\" when frustrated – opposite meaning
• Idioms: \"raining cats and dogs\" – literal translation meaningless
• Word Sense Disambiguation: \"bank\" (river vs financial institution)
• Anaphora: \"John met Mary. He gave her the book.\" – resolve \"he\", \"her\"
• Ellipsis: \"I like apples and you do too\" – implicit \"like apples\"

DOCUMENT-LEVEL PROBLEMS:
────────────────────────
• Leading/trailing boilerplate: Website headers, footers, navigation
• Duplicates: Same content appears multiple times (copy-paste)
• Near-duplicates: Slightly different versions of same content
• Language mixing: Different language sections in same document
• Template artifacts: \"[FILL IN]\" placeholders, numbered lists marker artifacts
• Page breaks: Multiple documents concatenated without separator

ENCODING PROBLEMS:
──────────────────
• Byte Order Mark (BOM): Invisible UTF-8 marker fails tokenization
• Mojibake: Encoding errors causing garbage display
• Character normalization: \"é\" can be represented two ways (different bytes)
• Combining characters: Accents separate from base character
• Right-to-left text: Arabic, Hebrew need special handling

METADATA & STRUCTURAL PROBLEMS:
───────────────────────────────
• Missing metadata: No creation date, author, source
• Inconsistent metadata: Type/format varies document-to-document
• Semi-structured mixtures: Tables, lists formatted as text
• Markup confusion: HTML tags mixed with plain text
• Version control artifacts: Git merge markers, diff markers

REAL-WORLD DATASET PROBLEMS:
─────────────────────────────
• Imbalanced classes: 5% positive, 95% negative samples
• Noisy labels: Human annotators made mistakes in labeling
• Distribution shift: Training data from 2015, applying to 2025 data
• Missing completely at random (MCAR): Random holes in data
• Missing not at random (MNAR): Systematic missing patterns
  Example: Offensive comments deleted by moderation (missing != random)
• Temporal drift: Language evolves, models must adapt

ACTION ITEMS FOR NEXT MODULE:
─────────────────────────────
Module 21 (\"Build a Text Cleaner\") will provide SOLUTIONS:
  → Creating clean_text() function
  → Practical preprocessing pipeline
  → Handling each problem type systematically
  → Before/after comparisons
  → Edge case testing
"""

print(additional_problems)

print("\n" + "="*85)
print("CONCLUSION: IDENTIFYING PROBLEMS IS THE FIRST STEP")
print("="*85)

conclusion = """
This module focused on PROBLEM IDENTIFICATION:
Understanding WHY data quality matters and WHAT can go wrong.

You've learned:
✓ Common text data quality issues (20+ examples)
✓ Why each issue matters for ML models
✓ Impact on model performance, fairness, and production reliability
✓ How different problems compound (cascading failures)

You now understand that TEXT PREPROCESSING is not a boring necessity:
It's the FOUNDATION of reliable NLP systems.

Professional practitioners spend 70-80% of project time on data quality:
  • Understanding the data (EDA)
  • Identifying problems (this module)
  • Building cleaning pipelines (next module)
  • Validating solutions
  • Monitoring in production

The next module will teach the SOLUTIONS and give you a powerful toolkit
to handle real-world messy text data professionally.
"""

print(conclusion)

print("\n" + "="*85)
print("TEXT CHALLENGES ANALYSIS COMPLETE")
print("="*85)
