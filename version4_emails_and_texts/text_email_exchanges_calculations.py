import numpy as np
import re
import itertools
from scipy.stats import poisson

class PrimeComplexityEngine:
    def __init__(self):
        # Pre-compute primes for a-z
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
        self.char_map = {chr(i + 97): self.primes[i] for i in range(26)}
        self.baseline_lambda = 0.08  # Expected prime-sum density in random speech

    def is_prime(self, n: int) -> bool:
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    def get_word_prime_value(self, word: str) -> int:
        clean_word = re.sub(r'[^a-z]', '', word.lower())
        return sum(self.char_map.get(c, 0) for c in clean_word)

    def calculate_structural_complexity(self, text: str) -> tuple:
        """
        Implements the 'Prefix-Sum' and 'Combinatorial Window' logic.
        Checks: W1, W1+W2, W1+W2+W3... and deep subsets in recent window.
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())
        if not words: return 0.0, 0

        word_values = [self.get_word_prime_value(w) for w in words]
        prime_hits = 0
        total_tests = 0

        # 1. Prefix Sums (The 'Building' pattern: W1, W1+W2, W1+W2+W3...)
        current_prefix_sum = 0
        for val in word_values:
            current_prefix_sum += val
            total_tests += 1
            if self.is_prime(current_prefix_sum):
                prime_hits += 1

        # 2. Deep Combinatorial Windows (Limited to 10 words to prevent O(2^n) explosion)
        window_size = min(len(word_values), 10)
        recent_words = word_values[-window_size:]
        for r in range(2, window_size):
            for subset in itertools.combinations(recent_words, r):
                total_tests += 1
                if self.is_prime(sum(subset)):
                    prime_hits += 1

        density = prime_hits / total_tests if total_tests > 0 else 0.0
        return density, total_tests

    def calculate_p_value(self, observed_density: float, total_tests: int) -> float:
        """
        Calculates the probability that the observed prime density
        is a result of random chance (Poisson-based significance).
        """
        k = int(observed_density * total_tests)
        if k == 0: return 1.0
        try:
            # Probability of seeing k or more successes in a Poisson distribution
            # based on the natural speech lambda.
            # If poisson.cdf doesn't return a numerical value, how does this work?
            p_val = 1 - poisson.cdf(k, self.baseline_lambda * total_tests)
            return max(0.0, min(1.0, p_val))  # Clamp [0,1]
        except:
            return 1.0
        

class Message:
    def __init__(self):
        self.text_content = ""
        self.id = None
        self.sender_foreign_key = None
        self.conversation_thread_id = None
        self.position_in_conversation_thread = -1
        # If this message is a part of a conversation thread, but also replying 
        # to another message in the thread explicitly, reference that id here
        self.reply_to = -1


class TextBasedIntelligencePipeline:
    def __init__(self):
        
        self.prime_engine = PrimeComplexityEngine()
        # conversation_thread is a list of Message Objects
        self.conversation_thread = []

        # List of Semantic Reciprocities between various messengers in this thread of messages
        # That details out reciprocities between different senders, right now, though we want to 
        # produce these values and chart them as they change for really long threads (for example
        # long chains of WhatsApp messages that happen over multiple months or years)
    
    def process_text_interaction(self, message: Message ):
        density, tests = self.prime_engine.calculate_structural_complexity(message.text_content)
        p_val = self.prime_engine.calculate_p_value(density, tests)


