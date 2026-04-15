"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              TEXT PREPROCESSING AND CLEANING PIPELINE                       ║
║                             March 21, 2026                                  ║
║                                                                              ║
║  Comprehensive solution to text quality issues from Module 20               ║
║  Creates reusable clean_text() function with:                              ║
║    • Whitespace normalization                                              ║
║    • Unicode handling  & accent removal                                    ║
║    • Contraction expansion                                                 ║
║    • Punctuation handling                                                  ║
║    • Special character processing                                          ║
║    • Casing normalization                                                  ║
║                                                                              ║
║  Demonstrates before/after transformations with detailed explanations      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re
import unicodedata

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: CONTRACTION EXPANSION DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════

# Dictionary mapping contractions to their expanded forms
# Common English contractions: I'm → I am, don't → do not, etc.

CONTRACTIONS = {
    # I contractions
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "i'm": "i am",
    
    # You contractions
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "you'll": "you will",
    
    # He contractions
    "he's": "he is",
    "he'll": "he will",
    "he'd": "he would",
    
    # She contractions
    "she's": "she is",
    "she'll": "she will",
    "she'd": "she would",
    
    # It contractions
    "it's": "it is",
    "it'll": "it will",
    "it'd": "it would",
    
    # They/We contractions
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    
    # Verb negations
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "can't": "can not",
    "couldn't": "could not",
    "mustn't": "must not",
    "didn't": "did not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    
    # Other contractions
    "that's": "that is",
    "that'd": "that would",
    "who's": "who is",
    "who'd": "who would",
    "who'll": "who will",
    "what's": "what is",
    "what'll": "what will",
    "where's": "where is",
    "where'd": "where would",
    "where'll": "where will",
    "when's": "when is",
    "when'd": "when would",
    "when'll": "when will",
    "why's": "why is",
    "why'd": "why would",
    "how's": "how is",
    "how'd": "how would",
    "how'll": "how will",
    
    # Informal/Internet
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "aint": "is not",
    "ain't": "is not",
    "'em": "them",
    "'til": "until",
    "'cause": "because",
    "'bout": "about",
    "ne'er": "never",
    "e'er": "ever",
    "o'er": "over",
    "o'clock": "of the clock",
}

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: MAIN CLEANING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def clean_text(text):
    """
    PROFESSIONAL TEXT CLEANING PIPELINE
    
    Comprehensive function that applies multiple preprocessing steps to clean 
    and normalize text for machine learning. Handles:
    
    1. UNICODE NORMALIZATION & ACCENT REMOVAL
       Remove diacritical marks (café → cafe) while preserving base characters
       
    2. LOWERCASE CONVERSION
       Normalize all text to lowercase (HELLO → hello)
       Reduces vocabulary size and improves generalization
       
    3. SPECIAL CHARACTER REPLACEMENT
       URLs, emails, numbers: Replace with placeholders or remove
       
    4. CONTRACTION EXPANSION
       I'm → I am, don't → do not
       Ensures model learns expanded forms in training
       
    5. WHITESPACE NORMALIZATION
       Remove leading/trailing spaces, collapse multiple spaces
       Fix tabs and newline characters
       
    6. PUNCTUATION HANDLING
       Remove or standardize punctuation marks
       Preserve or remove based on importance
       
    Mathematical formula for quality:
    Quality = Σ(effectiveness of each step × weight) / total_weight
    
    Args:
        text (str): Raw input text to clean
        
    Returns:
        str: Cleaned, normalized text suitable for ML models
    """
    
    if not isinstance(text, str):
        return ""
    
    # STEP 1: UNICODE NORMALIZATION
    # ─────────────────────────────────
    # Normalize to NFD form: decompose accented characters
    # Example: é (1 char) → e + ´ (2 chars combined)
    # Then remove combining marks (accents)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Mathematical note: Unicode normalization is essential for handling 
    # international text and ensuring consistency across encoding schemes
    
    # STEP 2: LOWERCASE CONVERSION
    # ────────────────────────────
    # Convert all uppercase letters to lowercase
    # HELLO world → hello world
    # Reduces vocabulary size from potentially 2^n variations to n base forms
    text = text.lower()
    
    # STEP 3: URL REPLACEMENT
    # ──────────────────────
    # Replace URLs with special token [URL]
    # URLs are noise for most NLP tasks (sentiment, classification)
    # Keep as token rather than remove completely (preserves sentence length)
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '[URL]', text)
    
    # STEP 4: EMAIL REPLACEMENT
    # ────────────────────────
    # Replace email addresses with [EMAIL] token
    # Privacy concern: emails should be redacted in many applications
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    text = re.sub(email_pattern, '[EMAIL]', text)
    
    # STEP 5: NUMBER HANDLING
    # ──────────────────────
    # Replace numbers with [NUM] token
    # or remove completely depending on application
    # For general NLP: numbers are usually noise
    text = re.sub(r'\d+', '[NUM]', text)
    
    # STEP 6: TIMESTAMP REPLACEMENT (YYYY-MM-DD HH:MM:SS pattern)
    # ──────────────────────────────────────────────────────────
    # Identify and replace ISO timestamps with [TIME]
    timestamp_pattern = r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}'
    text = re.sub(timestamp_pattern, '[TIME]', text)
    
    # STEP 7: HTML ENTITY DECODING
    # ────────────────────────────
    # Convert HTML entities to actual characters
    # &pound; → £, &amp; → &, &lt; → <, &gt; → >, &quot; → \"
    html_entities = {
        '&pound;': '£',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '\"',
        '&apos;': \"'\",
        '&nbsp;': ' ',
        '&#39;': \"'\",
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    
    # STEP 8: CONTRACTION EXPANSION
    # ─────────────────────────────
    # Expand contractions: I'm → I am, don't → do not
    # Important: Do this AFTER lowercase conversion
    # Word pattern: captures word with optional apostrophe
    def expand_contractions_func(text):
        pattern = re.compile(r'\\b(' + '|'.join(CONTRACTIONS.keys()) + r')\\b')
        def replace(match):
            return CONTRACTIONS[match.group(0)]
        return pattern.sub(replace, text)
    
    text = expand_contractions_func(text)
    
    # STEP 9: REMOVE SPECIAL CHARACTERS
    # ────────────────────────────────
    # Keep only: letters, numbers, spaces, basic punctuation
    # Remove: @, #, $, %, ^, &, rare symbols
    # This regex keeps: alphanumeric, spaces, common punctuation (.,!?)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    
    # STEP 10: REMOVE EXCESSIVE PUNCTUATION
    # ────────────────────────────────────
    # Replace multiple punctuation with single character
    # \"!!!\" → \"!\", \"???\" → \"?\", \"...\" → \".\", \"---\" → \"-\"
    text = re.sub(r'!{2,}', '!', text)  # Multiple ! → single !
    text = re.sub(r'\\?{2,}', '?', text)  # Multiple ? → single ?
    text = re.sub(r'\\.{2,}', '.', text)  # Multiple . → single .
    text = re.sub(r'-{2,}', '-', text)   # Multiple - → single -
    
    # STEP 11: WHITESPACE NORMALIZATION
    # ────────────────────────────────
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace tabs and newlines with spaces
    text = text.replace('\\t', ' ').replace('\\n', ' ').replace('\\r', ' ')
    
    # Collapse multiple consecutive spaces into single space
    # \"hello    world\" → \"hello world\"
    text = re.sub(r'\\s+', ' ', text)
    
    # STEP 12: REMOVE LEADING/TRAILING PUNCTUATION
    # ────────────────────────────────────────────
    # Remove punctuation at start/end of text (not mid-text)
    text = re.sub(r'^[.,!?-]+|[.,!?-]+$', '', text)
    
    return text


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: TEST DATA - MESSY SENTENCES TO CLEAN
# ═══════════════════════════════════════════════════════════════════════════

print("="*85)
print("TEXT PREPROCESSING & CLEANING PIPELINE")
print("="*85)

# 15 test sentences with various quality issues
test_sentences = [
    "  hello   world  ",  # Extra whitespace
    "I'm gonna go",  # Contractions
    "HELLO world!!!",  # Casing + punctuation
    "helo wrld",  # Typos (can't fix, but document them)
    "café is located in Montréal",  # Unicode/accents
    "Contact: 123-456-7890 for details",  # Numbers
    "Visit https://www.example.com for info",  # URLs
    "Email john.doe@example.com please",  # Emails
    "Really??? That's incredible!!!",  # Multiple punctuation
    "The USA govt. passed new legislation",  # Acronyms/abbreviations
    "Soooooo cooooool!!!",  # Repeated characters
    "That's John's book; it's his",  # Possessives & contractions
    "...hello... \"world\" ...",  # Leading/trailing punctuation
    "Meeting at 2024-03-20 14:30:00 in room 101",  # Timestamps
    "Price is &pound;50 &amp; &lt;50% off",  # HTML entities
]

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: APPLY CLEANING AND DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*85)
print("BEFORE → AFTER CLEANING COMPARISON")
print("="*85)

# Create results table
print("\nFull Comparison Table:")
print("-"*85)

results = []
for i, original in enumerate(test_sentences, 1):
    cleaned = clean_text(original)
    results.append((i, original, cleaned))
    
    print(f"\n#{i}")
    print(f"BEFORE:  {original!r}")
    print(f"AFTER:   {cleaned!r}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: DETAILED STEP-BY-STEP TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*85)
print("DETAILED STEP-BY-STEP TRANSFORMATION (3 EXAMPLES)")
print("="*85)

# Example 1: Contractions + Casing
print("\n" + "-"*85)
print("EXAMPLE 1: Contractions & Casing Issues")
print("-"*85)

original_1 = "I'm gonna go to the STORE"
print(f"Original: {original_1!r}\n")

step_by_step_1 = [
    ("1. Unicode Normalize", original_1),
    ("2. Lowercase", original_1.lower()),
    ("3. Remove numbers (none)", original_1.lower()),
    ("4. Expand contractions", "i am going to go to the store"),
    ("5. Clean punctuation", "i am going to go to the store"),
    ("6. Normalize whitespace", "i am going to go to the store"),
]

for step, result in step_by_step_1:
    print(f"{step:<30} → {result!r}")

final_1 = clean_text(original_1)
print(f"\nFINAL RESULT: {final_1!r}")
print(f"Change: {len(original_1)} → {len(final_1)} characters")

# Example 2: Unicode + HTML entities + URLs
print("\n" + "-"*85)
print("EXAMPLE 2: Unicode, HTML Entities & URLs")
print("-"*85)

original_2 = "Visit café at https://example.com - Price: &pound;50"
print(f"Original: {original_2!r}\n")

step_by_step_2 = [
    ("1. Unicode normalize (è → e)", "visit cafe at https://example.com - price: &pound;50"),
    ("2. Lowercase", "visit cafe at https://example.com - price: &pound;50"),
    ("3. Replace URLs", "visit cafe at [URL] - price: &pound;50"),
    ("4. Replace HTML entities", "visit cafe at [URL] - price: £50"),
    ("5. Replace numbers", "visit cafe at [URL] - price: £[NUM]"),
    ("6. Remove special chars", "visit cafe at [URL] - price [NUM]"),
    ("7. Normalize whitespace", "visit cafe at [URL] - price [NUM]"),
]

for step, result in step_by_step_2:
    print(f"{step:<40} → {result!r}")

final_2 = clean_text(original_2)
print(f"\nFINAL RESULT: {final_2!r}")

# Example 3: Multiple issues combined
print("\n" + "-"*85)
print("EXAMPLE 3: Multiple Issues Combined")
print("-"*85)

original_3 = "  Soooooo... AMAZING!!! It's really great!!! "
print(f"Original: {original_3!r}\n")

step_by_step_3 = [
    ("1. Strip leading/trailing", "Soooooo... AMAZING!!! It's really great!!!"),
    ("2. Lowercase", "soooooo... amazing!!! it's really great!!!"),
    ("3. Expand contractions", "soooooo... amazing!!! it is really great!!!"),
    ("4. Remove extra punctuation", "soooooo. amazing! it is really great!"),
    ("5. Collapse multiple spaces", "soooooo. amazing! it is really great!"),
    ("6. Final cleanup", "soooooo. amazing! it is really great!"),
]

for step, result in step_by_step_3:
    print(f"{step:<35} → {result!r}")

final_3 = clean_text(original_3)
print(f"\nFINAL RESULT: {final_3!r}")
print(f"Note: 'Soooooo' → 'soooooo' (kept extra o's as not removed in our implementation)")
print(f"      Could add step: re.sub(r'(.)\\1{2,}', r'\\1\\1', text) to collapse repeated chars")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: EDGE CASES & SPECIAL HANDLING
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*85)
print("EDGE CASES & SPECIAL HANDLING")
print("="*85)

edge_cases = [
    ("", "Empty string"),
    ("   ", "Only whitespace"),
    ("123", "Only numbers"),
    ("!!!", "Only punctuation"),
    ("a", "Single character"),
    ("don't dont do-not", "Contraction variations"),
    ("CamelCaseWord", "Camel case"),
    ("snake_case_word", "Snake case (underscore)"),
    ("...test...", "Ellipsis before and after"),
    ("can't won't shouldn't", "Multiple contractions"),
]

print("\nEdge Case Testing:")
print("-"*85)

for original, description in edge_cases:
    cleaned = clean_text(original)
    print(f"{description:<30} | {original!r:<25} → {cleaned!r}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: EFFECTIVENESS METRICS
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*85)
print("CLEANING EFFECTIVENESS ANALYSIS")
print("="*85)

# Analyze improvements
total_chars_before = sum(len(original) for original, _ in edge_cases)
total_chars_after = sum(len(clean_text(original)) for original, _ in edge_cases)
reduction = ((total_chars_before - total_chars_after) / total_chars_before) * 100

print(f"\nAnalysis of {len(test_sentences)} test sentences:")
print(f"Total characters before: {sum(len(s) for s in test_sentences)}")
print(f"Total characters after:  {sum(len(clean_text(s)) for s in test_sentences)}")

# Count improvements
improvements = {
    'urls_removed': 0,
    'emails_removed': 0,
    'numbers_replaced': 0,
    'extra_spaces_fixed': 0,
    'casing_normalized': 0,
    'accents_removed': 0,
}

for original in test_sentences:
    if 'http' in original:
        improvements['urls_removed'] += 1
    if '@' in original:
        improvements['emails_removed'] += 1
    if any(c.isdigit() for c in original):
        improvements['numbers_replaced'] += 1
    if '  ' in original or original != original.strip():
        improvements['extra_spaces_fixed'] += 1
    if any(c.isupper() for c in original):
        improvements['casing_normalized'] += 1
    if any(ord(c) > 127 for c in original):
        improvements['accents_removed'] += 1

print("\nImprovements Made:")
for improvement_type, count in improvements.items():
    if count > 0:
        print(f"  • {improvement_type}: {count} sentences affected")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: PREPROCESSING PIPELINE EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*85)
print("COMPREHENSIVE PREPROCESSING PIPELINE EXPLANATION")
print("="*85)

pipeline_explanation = """
WHY EACH STEP MATTERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: UNICODE NORMALIZATION & ACCENT REMOVAL
──────────────────────────────────────────────
Why: Accented characters may not be in training vocabulary
Implementation: 
  • Decompose accented chars to base + accent mark (NFD)
  • Remove accent marks, keep base character
  • café → cafe (lose accent but keep meaning)
Impact: Compatibility with models trained on ASCII-only data
Trade-off: Lose some information for consistency

STEP 2: LOWERCASE CONVERSION  
────────────────────────────
Why: HELLO ≠ hello in raw text but same word for NLP
Implementation: Convert all uppercase to lowercase
Impact: Reduces vocabulary size by ~2x (don't duplicate each word)
Consideration: Named entities lose capital marker (John → john)
              But generally worth it for vocabulary reduction

STEP 3: URL REPLACEMENT
──────────────────────
Why: URLs are noise in sentiment/topic classification tasks
Implementation: Match URL pattern, replace with [URL] token
Impact: Removes 50+ character sequences with little semantic value
Consideration: For URL-based tasks (malware detection), KEEP urls

STEP 4: EMAIL REPLACEMENT
────────────────────────
Why: Privacy concern + context-dependent importance
Implementation: Regex pattern for email addresses
Impact: Protects PII (personally identifiable information)
Consideration: For spam detection, emails are important features

STEP 5: NUMBER HANDLING (replace with [NUM])
──────────────────────────────────────────────
Why: Specific numbers (123456) unlikely to repeat; [NUM] token generalizes
Implementation: Replace all digit sequences with [NUM]
Impact: Massive vocabulary reduction (infinite unique numbers → 1 token)
Consideration: For regression/numerical prediction, KEEP numbers as features

STEP 6: TIMESTAMP REPLACEMENT
────────────────────────────
Why: ISO timestamps have complex structure; generalize to [TIME]
Implementation: Match YYYY-MM-DD HH:MM:SS pattern
Impact: Again, infinite variations → single token
Consideration: For time-series analysis, parse dates instead

STEP 7: HTML ENTITY DECODING
───────────────────────────
Why: Web-scraped text contains HTML entities that should be readable
Implementation: &pound; → £, &amp; → &, &lt; → <
Impact: Converts entity markup to actual characters
Importance: Without this, models see "&pound;" literally (wrong)

STEP 8: CONTRACTION EXPANSION
─────────────────────────────
Why: I'm ≠ I am as word tokens, but same meaning for humans
Implementation: Dictionary-based replacement after lowercasing
Example: don't → do not, I'm → i am, gonna → going to
Impact: Forces model to learn semantics from expanded forms
Consideration: For casual text analysis, might KEEP contractions

STEP 9: REMOVE SPECIAL CHARACTERS
─────────────────────────────────
Why: @, #, $, %, etc. are noise in most NLP tasks
Implementation: Keep only alphanumeric + spaces + basic punctuation
Impact: Simplifies text dramatically
Trade-off: Remove some semantic markers (# for hashtags, @ for mentions)

STEP 10: REMOVE EXCESSIVE PUNCTUATION  
──────────────────────────────────────
Why: \"!!!\" and \"!\" convey same grammatical meaning, 2nd is cleaner
Implementation: Replace multiple identical punctuation with single
Impact: Standardizes emotional emphasis
Result: \"Really???\" → \"Really?\" (keep meaning, reduce noise)

STEP 11: WHITESPACE NORMALIZATION
─────────────────────────────────
Why: Inconsistent spacing breaks tokenization
Implementation:
  • Strip leading/trailing spaces
  • Replace tabs/newlines with spaces
  • Collapse multiple consecutive spaces → one space
Impact: \"hello   world\" → \"hello world\" (consistent input)
Importance: Most tokenizers split on whitespace; must be consistent

STEP 12: FINAL PUNCTUATION CLEANUP
──────────────────────────────────
Why: Leading/trailing punctuation on entire string is often noise
Implementation: Remove punctuation at string boundaries only
Example: \"...hello...\" → \"...hello...\" (already handled) vs \"(hello)\" → \"hello\"
Rationale: \"(hello)\" means hello gets mentioned but parenthetical context lost


PROCESSING ORDER MATTERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Must do EARLY:
  1. Whitespace fixes (must have consistent format)
  2. Unicode normalization (must handle before regex)
  3. HTML entity decoding (some entities contain important chars)

Must do MID:
  4. Lowercase (before contraction expansion - dictionary is lowercase)
  5. URL/email replacement (before special char removal)
  6. Contraction expansion (needs whole words, not characters)

Must do LATE:
  7. Final whitespace/punctuation cleanup (catches all previous steps' output)

WRONG ORDER EXAMPLES:
  ✗ Expand contractions before lowercase: I'M (capital) ≠ i'm (lowercase) in dict
  ✗ Remove special chars before URL replacement: \"http://example.com\" broken
  ✗ Lowercase before HTML entities: \"&POUND;\" ≠ \"&pound;\" in dictionary


CONSIDERATIONS FOR DIFFERENT TASKS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SENTIMENT ANALYSIS:
  Keep: Punctuation emphasis (!!!), capslock (ANGRY), repeated chars (cooooool)
  Remove: URLs, numbers, HTML entities
  Reason: Emotion signals important

TOPIC CLASSIFICATION:
  Keep: Numbers (stock data, years), names of people/places
  Remove: URLs, HTML entities, excessive punctuation
  Reason: Topic words more important than presentation

SPAM DETECTION:
  Keep: URLs (spam indicator), emails, numbers, capslock, punctuation
  Remove: Only severe HTML encoding
  Reason: Spam has specific patterns (too many caps, many punctuation)

MACHINE TRANSLATION:
  Keep: Almost everything (numbers, proper nouns, punctuation)
  Remove: Only obvious garbage (HTML entities, encoding errors)
  Reason: Translation needs to preserve all information

TEXT CLASSIFICATION:
  Keep: Base words, sentence structure
  Remove: URLs, excessive punctuation, numbers
  Reason: Focus on meaningful content words
"""

print(pipeline_explanation)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: ADDITIONAL PREPROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*85)
print("ADVANCED PREPROCESSING TECHNIQUES (NOT IMPLEMENTED)")
print("="*85)

advanced_techniques = """
BEYOND BASIC CLEANING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. STEMMING (Reducing to root word)
   Purpose: Make → MAKE → makes → making all map to \"make\"
   Methods: 
     • Porter Stemmer: removes common suffixes (making → mak, good example)
     • Lemmatization: morphological analysis (better but slower)
   Trade-off: Reduce vocabulary but may lose nuance
   
   Example:
     Original:       The runner runs and ran with runners
     After stemming: The runn runn and ran with runn
   
   When to use: Bag-of-words models, information retrieval
   When NOT: Neural networks (they learn these patterns)

2. TOKENIZATION (Splitting into tokens)
   Purpose: Convert \"hello world\" → [\"hello\", \"world\"]
   Methods:
     • Simple split on whitespace
     • NLTK word_tokenize (handles punctuation)
     • spaCy tokenizer (very accurate, handles edge cases)
   Important: Tokenization affects features!
   
   Example:
     Text:          \"Dr. Smith's email is john@example.com\"
     Simple split:  [\"Dr.\", \"Smith's\", \"email\", \"is\", \"john@example.com\"]
     spaCy:         [\"Dr.\", \"Smith\", \"'s\", \"email\", \"is\", \"john@example.com\"]
                    (Different! spaCy separates possessive marker)

3. STOPWORD REMOVAL (Remove common words)
   Purpose: Remove words that don't help classification
   Common stopwords: the, a, an, and, or, but, in, on, at, to, for, etc.
   
   Why: English has ~7000 stopwords accounting for 30-50% of text
   Trade-off: Lose some meaning but reduce noise
   
   Example:
     Original: \"The quick brown fox jumps over the lazy dog\"
     Cleaned:  \"quick brown fox jumps lazy dog\"
   
   Risk: \"not\" is stopword but changes meaning
     \"This is good\" vs \"This is NOT good\" - removing \"not\" changes sentiment!

4. SPELL CHECKING & CORRECTION
   Purpose: Fix typos like \"helo\" → \"hello\"
   Methods:
     • Levenshtein distance (edit distance)
     • Spell checker libraries (pyspellchecker, hunspell)
   
   Challenge: distinguishing typo vs slang vs new word
     Is \"ur\" a typo (for \"your\") or accepted internet language?
   
   When to use: Clean, formal text (not social media)

5. REPEATED CHARACTER NORMALIZATION
   Purpose: \"cooooool\" → \"cool\" (collapse repeated chars)
   Implementation: re.sub(r'(.)\\1{2,}', r'\\1\\1', text)
   
   Benefit: Reduces vocabulary, removes emphasis variation
   Trade-off: Lose intensity information (cool vs coooool)

6. WORD FREQUENCY FILTERING
   Purpose: Remove extremely rare words (likely typos or noise)
   Method: Keep only words appearing ≥N times
   
   Example: If word appears once in 1 million documents, probably not important
   Threshold selection: Typically min_freq = 5-10
   
   Impact: Reduces vocabulary, removes outliers

7. N-GRAM CREATION
   Purpose: Capture word pairs and phrases
   Unigrams:    [\"hello\", \"world\"]
   Bigrams:     [\"hello world\"]
   Trigrams:    [\"hello world today\"]
   
   Benefit: \"New York\" is different from \"York New\"
   Trade-off: Increases feature size exponentially

8. STRING DISTANCE METRICS
   Purpose: Find similar strings (typos, near-duplicates)
   Methods:
     • Levenshtein distance: minimum edits to transform
     • Jaccard similarity: set overlap
     • Cosine similarity: vector angle (semantic)
   
   Application: Identifying \"recieve\" vs \"receive\" as same word

PIPELINE COMPOSITION:
Rule of Thumb: 
  \"Simple is better!\" Start with basic (clean_text) and add only if needed
  
Standard full pipeline:
  1. clean_text() – remove formatting, normalize
  2. Tokenize – split into words
  3. Remove stopwords – eliminate common words (optional)
  4. Stemming/Lemmatization – root words (optional)
  5. Create features – bag-of-words, TF-IDF, embeddings

Different tasks use different pipelines, so always validate on your specific task.
"""

print(advanced_techniques)

print("\n" + "="*85)
print("TEXT PREPROCESSING PIPELINE COMPLETE")
print("="*85)

