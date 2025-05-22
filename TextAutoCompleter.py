import re
import pickle
import numpy as np
from math import exp, log
import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from collections import Counter, defaultdict


# Download required NLTK data
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class TextAutoCompleter:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.unigram_model = {}
        self.bigram_model = {}
        self.trigram_model = {}
        self.l1, self.l2, self.l3 = 0.749, 0.010, 0.241  # Default optimized lambdas
        self.is_trained = False
        
    def clean_text(self, text):
        """Remove punctuation and non-ASCII characters"""
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text

    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        return ' '.join([word for word in text.split() if word.lower() not in self.stop_words])

    def additional_preprocessing(self, text):
        """Apply additional preprocessing steps"""
        text = text.lower()
        text = re.sub(r'\d+', '<NUM>', text)
        
        # Expand contractions
        contractions = {
            "n't": " not", "'re": " are", "'s": " is", "'d": " would", 
            "'ll": " will", "'ve": " have", "'m": " am",
            "won't": "will not", "can't": "cannot", 
            "shan't": "shall not", "let's": "let us", 
            "y'all": "you all", "o'clock": "of the clock", 
            "ma'am": "madam", "ain't": "is not", 
            "could've": "could have", "should've": "should have", 
            "would've": "would have", "it's": "it is", 
            "they're": "they are", "we're": "we are", 
            "you're": "you are", "I've": "I have", 
            "he's": "he is", "she's": "she is",
            "that's": "that is", "what's": "what is", 
            "who's": "who is"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])
        text = ' '.join(text.split())
        
        return text

    def preprocess_text(self, text):
        """Apply complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.additional_preprocessing(text)
        return text

    def tokenize(self, text):
        """Tokenize text into words"""
        return text.split()

    def generate_ngrams(self, tokens, n):
        """Generate n-grams from tokens"""
        return list(zip(*[tokens[i:] for i in range(n)]))

    def train_ngram_model(self, corpus, n, min_threshold=0.01):
        """Train n-gram model with MLE and Laplace smoothing"""
        article_count = len(corpus)
        ngram_counts = Counter()
        ngram_article_counts = defaultdict(set)

        for i, article in enumerate(corpus):
            tokens = self.tokenize(article)
            ngrams = self.generate_ngrams(tokens, n)
            ngram_counts.update(ngrams)
            for ngram in set(ngrams):
                ngram_article_counts[ngram].add(i)

        min_articles = max(1, int(min_threshold * article_count))
        vocab = {ngram for ngram, articles in ngram_article_counts.items() 
                if len(articles) >= min_articles}

        total_ngrams = sum(ngram_counts.values())
        vocab_size = len(vocab)
        
        model = {
            ngram: (ngram_counts[ngram] + 1) / (total_ngrams + vocab_size)
            for ngram in vocab
        }

        return model, vocab

    def train_models(self, texts):
        """Train unigram, bigram, and trigram models"""
        print("Preprocessing texts...")
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        print("Training n-gram models...")
        self.unigram_model, _ = self.train_ngram_model(processed_texts, 1)
        self.bigram_model, _ = self.train_ngram_model(processed_texts, 2)
        self.trigram_model, _ = self.train_ngram_model(processed_texts, 3)
        
        self.is_trained = True
        print("Models trained successfully!")

    def interpolation_model(self, word, history):
        """Calculate interpolated probability of word given history"""
        # Get trigram probability
        trigram_prob = self.trigram_model.get(
            (history[-2], history[-1], word), 1e-6
        ) if len(history) >= 2 else 1e-6
        
        # Get bigram probability
        bigram_prob = self.bigram_model.get(
            (history[-1], word), 1e-6
        ) if len(history) >= 1 else 1e-6
        
        # Get unigram probability
        unigram_prob = self.unigram_model.get((word,), 1e-6)

        # Calculate interpolated probability
        interpolated_prob = (self.l1 * trigram_prob +
                           self.l2 * bigram_prob +
                           self.l3 * unigram_prob)

        return interpolated_prob

    def get_word_candidates(self, prefix="", max_candidates=10):
        """Get word candidates based on unigram model"""
        if not self.is_trained:
            return []
            
        candidates = []
        
        # Get all words from unigram model
        all_words = [word[0] for word in self.unigram_model.keys()]
        
        if prefix:
            # Filter words that start with the prefix
            matching_words = [word for word in all_words if word.startswith(prefix.lower())]
        else:
            matching_words = all_words
        
        # Sort by probability and return top candidates
        word_probs = [(word, self.unigram_model.get((word,), 0)) for word in matching_words]
        word_probs.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, prob in word_probs[:max_candidates]]

    def get_next_word_predictions(self, text, max_predictions=5):
        """Get next word predictions using interpolated model"""
        if not self.is_trained:
            return []
            
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        tokens = self.tokenize(processed_text)
        
        # Get history (last 2 words for trigram context)
        history = tokens[-2:] if len(tokens) >= 2 else tokens
        
        # Get all possible next words from vocabulary
        all_words = set()
        for ngram in self.unigram_model.keys():
            all_words.add(ngram[0])
        
        # Calculate probabilities for each possible next word
        word_probs = []
        for word in all_words:
            prob = self.interpolation_model(word, history)
            word_probs.append((word, prob))
        
        # Sort by probability and return top predictions
        word_probs.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, prob in word_probs[:max_predictions]]

    def complete_text(self, partial_text, max_completions=5):
        """Complete partial text with word suggestions"""
        if not self.is_trained:
            return []
            
        words = partial_text.strip().split()
        if not words:
            return self.get_word_candidates(max_candidates=max_completions)
        
        last_word = words[-1]
        
        # If the last word seems incomplete (no space after), complete it
        if not partial_text.endswith(' '):
            return self.get_word_candidates(prefix=last_word, max_candidates=max_completions)
        else:
            # Predict next word
            return self.get_next_word_predictions(partial_text, max_predictions=max_completions)

    def save_models(self, filepath):
        """Save trained models to file"""
        models_data = {
            'unigram_model': self.unigram_model,
            'bigram_model': self.bigram_model,
            'trigram_model': self.trigram_model,
            'lambdas': (self.l1, self.l2, self.l3),
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(models_data, f)

    def load_models(self, filepath):
        """Load trained models from file"""
        with open(filepath, 'rb') as f:
            models_data = pickle.load(f)
        
        self.unigram_model = models_data['unigram_model']
        self.bigram_model = models_data['bigram_model']
        self.trigram_model = models_data['trigram_model']
        self.l1, self.l2, self.l3 = models_data['lambdas']
        self.is_trained = models_data['is_trained']