"""
================================================================================
FILE: 03_04_2026_NLP_Mini_App.py
TITLE: Interactive NLP Text Analysis Application
================================================================================

An interactive Natural Language Processing application that performs text
analysis using TF-IDF feature extraction, sentence tokenization, and a
simple keyword-based chatbot. Users can input multiple lines of text and
receive analysis of important words, sentence structure, and chatbot responses
based on keyword matching with a knowledge base.

KEY FEATURES:
- Multiline text input (blank line to finish)
- TF-IDF feature extraction for important word identification
- Sentence tokenization and analysis
- Keyword-based chatbot with knowledge base
- Output saved to text file
- Heavy comments explaining NLP concepts

AUTHOR: NLP Lab
DATE: April 3, 2026
================================================================================
"""

import re
import math
from collections import defaultdict, Counter
from datetime import datetime

# ============================================================================
# SECTION 1: CORE NLP UTILITY CLASSES AND FUNCTIONS
# ============================================================================

class SimpleTokenizer:
    """
    A simple tokenizer for breaking text into sentences and words.
    
    This class handles:
    - Sentence splitting based on punctuation markers
    - Word tokenization with filtering
    - Normalization (lowercasing, punctuation removal)
    
    The tokenizer is basic but sufficient for educational purposes.
    Production systems would use NLTK or spaCy.
    """
    
    def __init__(self):
        # Sentence boundary markers - these characters often end sentences
        self.sentence_endings = r'[.!?]+'
        
        # Common patterns we want to remove
        self.punctuation = r'[^\w\s]'
    
    def tokenize_sentences(self, text):
        """
        Split text into sentences.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            list: List of sentences (strings)
            
        EXPLANATION:
        We use regex to split on sentence-ending punctuation (. ! ?).
        This is a naive approach - more sophisticated methods handle
        abbreviations, decimal numbers, and ellipsis better.
        """
        # Split on sentence endpoints, keeping track of boundaries
        sentences = re.split(self.sentence_endings, text)
        
        # Filter empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def tokenize_words(self, text):
        """
        Split text into words (tokens).
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            list: List of word tokens (lowercase)
            
        EXPLANATION:
        Word tokenization involves:
        1. Converting to lowercase (normalization)
        2. Removing punctuation
        3. Splitting on whitespace
        4. Filtering empty tokens
        
        This is a simple approach. Real tokenizers handle contractions
        ('don't' -> 'do', 'n't'), hyphenation, and special characters.
        """
        # Convert to lowercase for case-insensitive analysis
        text = text.lower()
        
        # Split on whitespace
        tokens = text.split()
        
        # Remove punctuation from each token
        cleaned_tokens = []
        for token in tokens:
            # Remove non-word characters (keeping only alphanumeric + underscore)
            cleaned = re.sub(self.punctuation, '', token)
            
            # Only keep non-empty tokens
            if cleaned:
                cleaned_tokens.append(cleaned)
        
        return cleaned_tokens


class TFIDFAnalyzer:
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) analyzer.
    
    TF-IDF is a numerical statistic that reflects how important a word is
    to a document in a collection of documents. 
    
    FORMULA:
    TF-IDF(term, doc) = TF(term, doc) × IDF(term, collection)
    
    WHERE:
    - TF(term, doc) = frequency of term in document / total words in document
      (higher when word appears more in this specific document)
    - IDF(term, collection) = log(total documents / documents containing term)
      (higher when term appears in fewer documents - more specific/unique)
    
    INTUITION:
    - Common words (the, a, is) have low IDF (appear in all documents)
    - Rare words have high IDF (appear in few documents)
    - Word importance = how frequent in this doc × how unique to this doc
    
    APPLICATION:
    In this assignment, we treat the entire input text as one "document"
    and identify "important" words = those with highest TF-IDF scores.
    """
    
    def __init__(self):
        """Initialize the TF-IDF analyzer."""
        self.tokenizer = SimpleTokenizer()
        self.word_frequencies = Counter()  # Frequency of each word
        self.document_count = 0  # Number of documents analyzed
        self.document_word_presence = defaultdict(int)  # How many docs contain each word
    
    def calculate_tf(self, words):
        """
        Calculate Term Frequency (TF) for given words.
        
        TF = frequency of word in text / total words in text
        
        EXAMPLE:
        Text: "machine learning machine learning deep learning"
        Word count: {machine: 2, learning: 3, deep: 1}, total: 6
        TF(machine) = 2/6 = 0.333
        TF(learning) = 3/6 = 0.5
        TF(deep) = 1/6 = 0.167
        
        Args:
            words (list): List of word tokens
            
        Returns:
            dict: {word: tf_score} for each unique word
        """
        # Count frequency of each word
        word_counts = Counter(words)
        
        # Total number of words
        total_words = len(words)
        
        # Calculate TF for each word
        tf_scores = {}
        for word, count in word_counts.items():
            # TF = count / total words
            tf_scores[word] = count / total_words if total_words > 0 else 0
        
        return tf_scores
    
    def calculate_idf(self, all_documents_words):
        """
        Calculate Inverse Document Frequency (IDF).
        
        IDF quantifies how unique a word is across documents.
        
        FORMULA:
        IDF(word) = log(total_documents / documents_containing_word)
        
        EXAMPLE:
        If we have 100 documents total:
        - Word "the" appears in 95 documents: IDF = log(100/95) ≈ 0.05 (low)
        - Word "unicorn" appears in 2 documents: IDF = log(100/2) ≈ 1.70 (high)
        
        We use log to prevent IDF from growing too large.
        
        Args:
            all_documents_words (list): List of word lists (each list = document)
            
        Returns:
            dict: {word: idf_score} for each unique word
        """
        # Count how many documents contain each word
        word_document_count = defaultdict(int)
        total_docs = len(all_documents_words)
        
        # For each document, count how many words it contains
        for words in all_documents_words:
            unique_words = set(words)  # Get unique words in this document
            for word in unique_words:
                word_document_count[word] += 1  # This word appears in this document
        
        # Calculate IDF for each word
        idf_scores = {}
        for word, doc_count in word_document_count.items():
            # IDF = log(total documents / documents containing word)
            # Add 1 to denominator to avoid division by zero
            idf_scores[word] = math.log(total_docs / (1 + doc_count))
        
        return idf_scores
    
    def get_important_words(self, text, num_top_words=10):
        """
        Identify most important words in text using TF-IDF.
        
        The importance of a word is determined by:
        1. How frequently it appears in THIS text (TF)
        2. How unique/rare it is in general (IDF)
        
        EXAMPLE OUTPUT:
        If text is about machine learning:
        - "machine" and "learning" would have high TF-IDF
        - "the", "a", "is" would be filtered out (low importance)
        
        Args:
            text (str): Input text to analyze
            num_top_words (int): How many top words to return
            
        Returns:
            dict: {word: tfidf_score} for top important words
        """
        # Tokenize text into words
        words = self.tokenizer.tokenize_words(text)
        
        # For this simple version, we only have one document
        # So IDF will be 0 for all words (they all appear in 1/1 documents)
        # Therefore, we use just TF as importance measure
        
        # Calculate term frequency
        tf_scores = self.calculate_tf(words)
        
        # Sort by TF score and get top words
        sorted_words = sorted(tf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N words
        top_words = dict(sorted_words[:num_top_words])
        
        return top_words


class SimpleChatbot:
    """
    A simple keyword-based chatbot that matches user input against
    a knowledge base of keywords and responds with predefined responses.
    
    This is a RULE-BASED chatbot, not a neural network chatbot.
    It works by:
    1. Extracting keywords from user input
    2. Searching knowledge base for matching keywords
    3. Returning associated responses
    4. Falling back to default response if no match found
    
    ADVANTAGES:
    - Deterministic and predictable
    - Easy to debug and maintain
    - Fast execution
    - No training required
    
    DISADVANTAGES:
    - Limited understanding of nuance
    - Requires manual knowledge base creation
    - Can't generalize to unseen inputs
    - Doesn't understand context beyond exact keywords
    
    PRODUCTION SYSTEMS:
    Modern chatbots use transformer models (like GPT-4) that can:
    - Understand context and nuance
    - Generate novel responses
    - Handle misspellings and variations
    - Engage in complex conversations
    """
    
    def __init__(self):
        """Initialize chatbot with knowledge base."""
        # Knowledge base: list of (keywords, response) tuples
        # Multiple keywords can trigger same response
        self.knowledge_base = [
            # Greeting responses
            {
                'keywords': ['hello', 'hi', 'hey', 'greetings'],
                'response': 'Hello! I\'m happy to chat with you. How can I help?'
            },
            {
                'keywords': ['how', 'are', 'you'],
                'response': 'I\'m doing great, thank you for asking! How are you doing?'
            },
            
            # Weather responses
            {
                'keywords': ['weather', 'rain', 'sunny', 'temperature'],
                'response': 'I don\'t have weather data, but I hope it\'s nice where you are!'
            },
            
            # Help responses
            {
                'keywords': ['help', 'assist', 'support'],
                'response': 'I\'m here to help! Ask me about machine learning, NLP, or data science.'
            },
            
            # Learning responses
            {
                'keywords': ['learn', 'learning', 'understand', 'explain'],
                'response': 'Learning is awesome! What topic would you like to learn about?'
            },
            
            # Data/AI responses
            {
                'keywords': ['data', 'machine', 'learning', 'ai', 'algorithm'],
                'response': 'That\'s an interesting topic! Machine learning is the future of AI. What specific aspect interests you?'
            },
            
            # Thank you responses
            {
                'keywords': ['thank', 'thanks', 'appreciate'],
                'response': 'You\'re welcome! Happy to help.'
            },
            
            # Goodbye responses
            {
                'keywords': ['bye', 'goodbye', 'farewell', 'see', 'later'],
                'response': 'Goodbye! It was nice chatting with you. Have a great day!'
            }
        ]
        
        # Fallback responses for when no keywords match
        self.fallback_responses = [
            'That\'s interesting! Can you tell me more?',
            'I\'m not sure I understood that. Could you rephrase?',
            'That\'s a good point. What else would you like to discuss?',
            'I see. Tell me more about what you mean.',
            'Interesting observation! Do you have any other thoughts?'
        ]
        
        # For rotation through fallback responses
        self.fallback_index = 0
    
    def extract_keywords(self, text):
        """
        Extract keywords from user input.
        
        This converts the input text to a set of lowercase words.
        
        EXAMPLE:
        Input: "How are you doing today?"
        Output: {'how', 'are', 'you', 'doing', 'today'}
        
        Args:
            text (str): User input text
            
        Returns:
            set: Set of lowercase words in the text
        """
        # Convert to lowercase and split into words
        words = text.lower().split()
        return set(words)
    
    def find_matching_response(self, user_input):
        """
        Find appropriate response based on user input keywords.
        
        ALGORITHM:
        1. Extract keywords from user input
        2. For each knowledge base entry:
           - Check if any of its keywords appear in user input
           - If match found, return that response
        3. If no match found, return fallback response
        
        IMPROVEMENT: Could score matches by number of keywords found
        and return response with highest score. This would handle
        inputs matching multiple categories better.
        
        Args:
            user_input (str): User's text input
            
        Returns:
            str: Chatbot response
        """
        # Extract keywords from user input
        user_keywords = self.extract_keywords(user_input)
        
        # Search knowledge base for matching keywords
        for knowledge_entry in self.knowledge_base:
            entry_keywords = set(knowledge_entry['keywords'])
            
            # Check if any keywords match
            if user_keywords & entry_keywords:  # Set intersection
                return knowledge_entry['response']
        
        # No match found - return fallback response
        response = self.fallback_responses[self.fallback_index]
        self.fallback_index = (self.fallback_index + 1) % len(self.fallback_responses)
        return response


# ============================================================================
# SECTION 2: TEXT ANALYSIS AND PROCESSING
# ============================================================================

def analyze_text(text):
    """
    Perform comprehensive text analysis.
    
    This function orchestrates the analysis pipeline:
    1. Sentence tokenization
    2. Word tokenization
    3. Important word identification
    4. Text statistics
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Analysis results with keys:
            - original_text: Input text
            - sentences: List of sentences
            - words: List of all words
            - unique_words: Set of unique words
            - important_words: Top words by TF-IDF
            - statistics: Text statistics
    """
    # Initialize analyzers
    tokenizer = SimpleTokenizer()
    tfidf = TFIDFAnalyzer()
    
    # Tokenize into sentences
    sentences = tokenizer.tokenize_sentences(text)
    
    # Tokenize into words
    words = tokenizer.tokenize_words(text)
    
    # Get unique words
    unique_words = set(words)
    
    # Get important words using TF-IDF
    important_words = tfidf.get_important_words(text, num_top_words=15)
    
    # Calculate statistics
    statistics = {
        'total_characters': len(text),
        'total_words': len(words),
        'unique_words': len(unique_words),
        'total_sentences': len(sentences),
        'average_sentence_length': len(words) / len(sentences) if sentences else 0,
        'average_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'word_frequencies': Counter(words).most_common(10)
    }
    
    return {
        'original_text': text,
        'sentences': sentences,
        'words': words,
        'unique_words': unique_words,
        'important_words': important_words,
        'statistics': statistics
    }


# ============================================================================
# SECTION 3: INPUT AND OUTPUT HANDLING
# ============================================================================

def get_multiline_input(prompt="Enter text (blank line to finish):\n"):
    """
    Get multiline text input from user.
    
    Collects multiple lines of text until user enters a blank line.
    This is useful for prose, paragraphs, or long text input.
    
    EXAMPLE:
    User enters:
        "The quick brown fox"
        "jumps over the lazy dog"
        ""  <- blank line triggers end of input
    
    Result: "The quick brown fox\njumps over the lazy dog"
    
    Args:
        prompt (str): Initial prompt to display
        
    Returns:
        str: All input lines joined with newlines
    """
    print(prompt)
    lines = []
    
    while True:
        # Get a line from user
        line = input()
        
        # Empty line signals end of input
        if line == "":
            break
        
        # Add line to list
        lines.append(line)
    
    # Join all lines with newline characters
    return "\n".join(lines)


def save_analysis_to_file(analysis_results, filename=None):
    """
    Save text analysis results to a file.
    
    Creates a formatted output file with all analysis results.
    Filename defaults to timestamp if not provided.
    
    Args:
        analysis_results (dict): Results from analyze_text()
        filename (str): Output filename (default: timestamp-based)
        
    Returns:
        str: Path to saved file
    """
    # Generate default filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nlp_analysis_{timestamp}.txt"
    
    # Build output text
    output = []
    output.append("=" * 80)
    output.append("NLP TEXT ANALYSIS REPORT")
    output.append("=" * 80)
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("")
    
    # Original text section
    output.append("-" * 80)
    output.append("ORIGINAL TEXT")
    output.append("-" * 80)
    output.append(analysis_results['original_text'])
    output.append("")
    
    # Statistics section
    output.append("-" * 80)
    output.append("TEXT STATISTICS")
    output.append("-" * 80)
    stats = analysis_results['statistics']
    output.append(f"Total characters: {stats['total_characters']}")
    output.append(f"Total words: {stats['total_words']}")
    output.append(f"Unique words: {stats['unique_words']}")
    output.append(f"Total sentences: {stats['total_sentences']}")
    output.append(f"Average sentence length: {stats['average_sentence_length']:.2f} words")
    output.append(f"Average word length: {stats['average_word_length']:.2f} characters")
    output.append("")
    
    # Sentences section
    output.append("-" * 80)
    output.append("SENTENCES (Total: {})".format(len(analysis_results['sentences'])))
    output.append("-" * 80)
    for i, sentence in enumerate(analysis_results['sentences'], 1):
        output.append(f"{i}. {sentence}")
    output.append("")
    
    # Important words section
    output.append("-" * 80)
    output.append("IMPORTANT WORDS (by TF-IDF Score)")
    output.append("-" * 80)
    for i, (word, score) in enumerate(analysis_results['important_words'].items(), 1):
        output.append(f"{i}. {word:25} (score: {score:.4f})")
    output.append("")
    
    # Most common words section
    output.append("-" * 80)
    output.append("MOST COMMON WORDS")
    output.append("-" * 80)
    for i, (word, count) in enumerate(stats['word_frequencies'], 1):
        output.append(f"{i}. {word:25} (frequency: {count})")
    output.append("")
    
    output.append("=" * 80)
    output.append("END OF ANALYSIS")
    output.append("=" * 80)
    
    # Write to file
    output_text = "\n".join(output)
    try:
        with open(filename, 'w') as f:
            f.write(output_text)
        print(f"\n✓ Analysis saved to: {filename}")
        return filename
    except IOError as e:
        print(f"\n✗ Error saving file: {e}")
        return None


# ============================================================================
# SECTION 4: INTERACTIVE APPLICATION MAIN LOOP
# ============================================================================

def run_nlp_application():
    """
    Main interactive NLP application loop.
    
    This function:
    1. Displays welcome message
    2. Gets user text input (multiline)
    3. Performs text analysis
    4. Displays analysis results
    5. Runs chatbot interaction
    6. Saves results to file
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE NLP TEXT ANALYSIS APPLICATION".center(80))
    print("=" * 80)
    print()
    print("This application analyzes your input text using NLP techniques:")
    print("• Text tokenization (sentences and words)")
    print("• Term Frequency-Inverse Document Frequency (TF-IDF) analysis")
    print("• Important word identification")
    print("• Keyword-based chatbot responses")
    print()
    
    # Get text input from user
    print("STEP 1: TEXT INPUT")
    print("-" * 80)
    user_text = get_multiline_input()
    
    # Validate input
    if not user_text.strip():
        print("\n✗ No text provided. Exiting.")
        return
    
    print(f"\n✓ Received {len(user_text)} characters of text")
    
    # Perform text analysis
    print("\nSTEP 2: TEXT ANALYSIS")
    print("-" * 80)
    print("Analyzing text...")
    analysis = analyze_text(user_text)
    
    # Display analysis results
    print("\nANALYSIS RESULTS:")
    print()
    print(f"📊 Text Statistics:")
    print(f"   • Characters: {analysis['statistics']['total_characters']}")
    print(f"   • Words: {analysis['statistics']['total_words']}")
    print(f"   • Unique words: {analysis['statistics']['unique_words']}")
    print(f"   • Sentences: {analysis['statistics']['total_sentences']}")
    print(f"   • Avg sentence length: {analysis['statistics']['average_sentence_length']:.1f} words")
    print()
    
    print(f"🔤 Top 10 Most Important Words (by frequency):")
    for i, (word, count) in enumerate(analysis['statistics']['word_frequencies'][:10], 1):
        print(f"   {i:2}. {word:20} → frequency: {count}")
    print()
    
    print(f"⭐ Top Important Words (by TF-IDF score):")
    for i, (word, score) in enumerate(analysis['important_words'].items(), 1):
        print(f"   {i:2}. {word:20} → score: {score:.4f}")
    print()
    
    print(f"📝 Sentences ({len(analysis['sentences'])} total):")
    for i, sentence in enumerate(analysis['sentences'], 1):
        print(f"   {i}. {sentence}")
    print()
    
    # Run chatbot with keyword extraction from text
    print("\nSTEP 3: CHATBOT INTERACTION")
    print("-" * 80)
    chatbot = SimpleChatbot()
    
    # Extract important words and create a chatbot prompt
    print("\n🤖 Chatbot Analysis of Your Text:")
    
    # Create a synthetic user prompt from important words
    important_words_list = list(analysis['important_words'].keys())
    if important_words_list:
        synthetic_input = " ".join(important_words_list[:5])
        print(f"Keywords detected: {', '.join(important_words_list[:5])}")
        response = chatbot.find_matching_response(synthetic_input)
    else:
        response = chatbot.find_matching_response(user_text)
    
    print(f"Chatbot: {response}")
    print()
    
    # Interactive chatbot loop
    print("\nLet's chat! (type 'exit' to finish)")
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            print("Chatbot: Thank you for chatting! Goodbye!")
            break
        
        if user_input:
            response = chatbot.find_matching_response(user_input)
            print(f"Chatbot: {response}")
    
    print()
    
    # Save results to file
    print("STEP 4: SAVE RESULTS")
    print("-" * 80)
    output_file = save_analysis_to_file(analysis)
    
    # Final summary
    print()
    print("=" * 80)
    print("NLP ANALYSIS COMPLETE".center(80))
    print("=" * 80)
    print(f"Total analysis time completed successfully.")
    print(f"Results saved to: {output_file}")


# ============================================================================
# SECTION 5: DEMONSTRATION WITH SAMPLE DATA
# ============================================================================

def run_demonstration():
    """
    Run the application with sample data for demonstration.
    
    This function demonstrates the NLP analysis without requiring
    user input. Useful for testing and showing capabilities.
    """
    # Sample text about machine learning
    sample_text = """
    Machine learning is a field of artificial intelligence that enables 
    computers to learn from data without being explicitly programmed. 
    Machine learning algorithms use historical data to predict future outcomes 
    and identify patterns in data.
    
    There are three main types of machine learning. Supervised learning uses 
    labeled data to train models that can predict outcomes for new data. 
    Unsupervised learning finds hidden patterns in unlabeled data. Reinforcement 
    learning trains agents to make sequences of decisions.
    
    Neural networks are inspired by how neurons work in biological brains. 
    Deep learning uses neural networks with many layers to learn complex 
    patterns. Convolutional neural networks are particularly effective for 
    computer vision tasks. Transformers have revolutionized natural language 
    processing.
    
    Applications of machine learning include image recognition, natural 
    language processing, recommendation systems, and autonomous vehicles. 
    Machine learning is transforming industries and creating new opportunities."
    """
    
    print("\n" + "=" * 80)
    print("NLP MINI APP - DEMONSTRATION MODE".center(80))
    print("=" * 80)
    print()
    print("This demonstration shows the application with sample text.")
    print()
    
    # Perform analysis
    print("Analyzing sample text about Machine Learning...")
    print()
    analysis = analyze_text(sample_text)
    
    # Display results
    print(f"📊 TEXT STATISTICS:")
    print(f"{'Characters':<30} {analysis['statistics']['total_characters']}")
    print(f"{'Words':<30} {analysis['statistics']['total_words']}")
    print(f"{'Unique words':<30} {analysis['statistics']['unique_words']}")
    print(f"{'Sentences':<30} {analysis['statistics']['total_sentences']}")
    print()
    
    print(f"🔤 MOST COMMON WORDS:")
    for i, (word, count) in enumerate(analysis['statistics']['word_frequencies'][:10], 1):
        print(f"   {i:2}. {word:30} {count} times")
    print()
    
    print(f"⭐ TOP IMPORTANT WORDS (TF-IDF):")
    for i, (word, score) in enumerate(analysis['important_words'].items(), 1):
        print(f"   {i:2}. {word:30} score: {score:.4f}")
    print()
    
    print(f"📝 SENTENCES:")
    for i, sentence in enumerate(analysis['sentences'], 1):
        print(f"   {i}. {sentence}")
    print()
    
    # Chatbot demonstration
    print(f"🤖 CHATBOT DEMONSTRATION:")
    chatbot = SimpleChatbot()
    
    test_inputs = [
        "hello",
        "how are you",
        "tell me about machine learning",
        "Can I ask for help?",
        "goodbye"
    ]
    
    for test_input in test_inputs:
        response = chatbot.find_matching_response(test_input)
        print(f"   You: {test_input}")
        print(f"   Bot: {response}")
        print()
    
    # Save results
    print("Saving analysis results...")
    output_file = save_analysis_to_file(analysis, "nlp_demo_results.txt")
    
    print()
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


# ============================================================================
# SECTION 6: APPLICATION ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the NLP application.
    
    Offers choice between interactive mode and demonstration mode.
    """
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "INTERACTIVE NLP TEXT ANALYSIS APPLICATION".center(78) + "║")
    print("║" + "TF-IDF, Tokenization, and Keyword-Based Chatbot".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    print("Choose mode:")
    print("  1. Interactive mode (input your own text)")
    print("  2. Demonstration mode (use sample text)")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_nlp_application()
    elif choice == "2":
        run_demonstration()
    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
