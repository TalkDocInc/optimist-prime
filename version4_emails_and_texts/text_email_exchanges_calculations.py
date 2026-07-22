import numpy as np
import re
import itertools
from scipy.stats import poisson
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class PatternSignature:
    """
    Represents the 'DNA' of a message's structural pattern.
    By comparing these vectors, we measure how much the 'style' of 
    prime-distribution matches, rather than just checking for identical numbers.
    """
    density: float          # Ratio of prime hits to total tests
    complexity_score: float # Logarithmic scale of total patterns tested
    prime_hit_count: int    # Absolute number of primes found
    
    def to_vector(self) -> np.ndarray:
        """Converts the signature into a numerical vector for similarity math."""
        return np.array([self.density, np.log1p(self.complexity_score), float(self.prime_hit_count)])

@dataclass
class Message:
    text_content: str
    sender_id: str
    thread_id: str
    timestamp: float
    reply_to: Optional[str] = None
    signature: Optional[PatternSignature] = None
    p_value : float = -1

class PrimeComplexityEngine:
    def __init__(self):
        # Alphabet prime mapping
        self.primes_map = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
        self.char_map = {chr(i + 97): self.primes_map[i] for i in range(26)}
        self.baseline_lambda = 0.15

    def is_prime(self, n: int) -> bool:
        if n < 2: return False
        for i in int(n**0.5) + 1: # Optimization: range check
            if n % i == 0: return False
        return True

    def get_word_prime_value(self, word: str) -> int:
        clean_word = re.sub(r'[^a-z]', '', word.lower())
        return sum(self.char_map.get(c, 0) for c in clean_word)

    def analyze_text(self, text: str) -> PatternSignature:
        """
        Performs the combinatorial analysis to generate a PatternSignature.
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())
        if not words:
            return PatternSignature(0.0, 0.0, 0)

        word_values = [self.get_word_prime_value(w) for w in words]
        prime_hits = 0
        total_tests = 0

        # 1. Prefix Sum Pattern (The 'Building' pattern)
        current_prefix_sum = 0
        for val in word_values:
            current_prefix_sum += val
            total_tests += 1
            if self._is_prime_internal(current_prefix_sum):
                prime_hits += 1

        # 2. Combinatorial Window Pattern (Checking deeper clusters)
        # We limit window to 12 to prevent O(2^n) exponential explosion on M4
        window_size = min(len(word_values), 12)
        recent_values = word_values[-window_size:]
        
        for r in range(2, window_size):
            for subset in itertools.combinations(recent_values, r):
                total_tests += 1
                if self._is_prime_internal(sum(subset)):
                    prime_hits += 1

        return PatternSignature(
            density=prime_hits / total_tests if total_tests > 0 else 0.0,
            complexity_score=float(total_tests),
            prime_hit_count=prime_hits
        )

    def _is_prime_internal(self, n: int) -> bool:
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    def calculate_p_value(self, signature: PatternSignature) -> float:
        """Calculates probability of density occurring by chance."""
        # Using total tests as the 'n' for the Poisson approximation
        expected_hits = self.baseline_lambda * signature.complexity_score
        if expected_hits == 0: return 1.0
        
        # Probability of seeing the observed hits or more (Significance)
        try:
            # We use the survival function (1 - CDF)
            p_val = 1 - poisson.cdf(signature.prime_hit_count, expected_hits)
            return max(0.0, min(1.0, p_val))
        except:
            return 1.0

class IntelligencePipeline:
    def __init__(self):
        self.engine = PrimeComplexityEngine()
        self.history: List[Message] = []
        # Stores: {(sender_a, sender_b): [list of similarity scores]}
        self.pattern_reciprocity: Dict[tuple, List[float]] = {}

    def process_message(self, new_msg: Message):
        """
        Processes a new message, calculates its signature, and 
        compares its pattern to all previous different senders.
        """
        # 1. Generate signature for the new message
        new_msg.signature = self.engine.analyze_text(new_msg.text_content)
        new_msg.p_value = self.engine.calculate_p_value(new_msg.signature)
        new_vec = new_msg.signature.to_vector()

        # 2. Compare against history
        for old_msg in self.history:
            if not old_msg:
                continue
            # We only care about reciprocity between DIFFERENT people
            if old_msg.sender_id != new_msg.sender_id:
                # Create a stable key for the pair (alphabetical order)
                pair_key = tuple(sorted([new_msg.sender_id, old_msg.sender_id]))
                
                if pair_key not in self.pattern_reciprocity:
                    self.pattern_reciprocity[pair_key] = []

                # 3. THE CORE INNOVATION: Pattern Vector Similarity
                # Instead of comparing primes, we compare the 'DNA' of the patterns
                old_vec = old_msg.signature.to_vector()
                
                # Cosine Similarity of the Pattern Signatures
                similarity = np.dot(new_vec, old_vec) / (
                    np.linalg.norm(new_vec) * np.linalg.norm(old_vec)
                )
                
                self.pattern_reciprocity[pair_key].append(similarity)

        # 4. Append to history
        self.history.append(new_msg)

    def get_reciprocity_report(self, sender_id: str) -> Dict:
        """Returns the average pattern alignment for a specific person."""
        report = {}
        for (p1, p2), scores in self.pattern_reciprocity.items():
            if sender_id in (p1, p2):
                report[f"Alignment with {p1 if p2 == sender_id else p2}"] = np.mean(scores)
        return report

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    pipeline = IntelligencePipeline()

    # Mocking a sequence of messages
    msg1 = Message(text_content=""" Hi James,

                It's been a while since the Rooster days—hope you’re doing well.

                I’m writing because I'm teaming up with Paul Gamble (former CTO/Cofounder of Hippocratic Health, $3.5B) to build a new healthcare AI company.

                We are building an AI Psychiatrist model—using human-in-the-loop AI to increase a provider's caseload capacity by 10x.

                We know "AI Services" can become a crowded trade, so we are architecting this as a Market Network. We use the AI workflow as a wedge to lock in supply, creating a unique data network effect on clinical decision-making that generic LLMs can't replicate.

                We'd love your pressure-test on the model given your thesis on market networks.

                Are you open to a brief chat?

                Best,

                Rishi
                """, sender_id="Rishi", thread_id="Thread_A", timestamp=1.0)
    msg2 = Message(text_content="""Rishi
                    Unfortunately, I'm not going to participate here.  But good luck.  Sounds like an important project which could help a lot of people.
                    James""", sender_id="James", thread_id="Thread_A", timestamp=2.0, reply_to="1")

    for m in [msg1, msg2]:
        pipeline.process_message(m)

    #print("Reciprocity Report for Rishi:")
    #print(pipeline.get_reciprocity_report(""))
    print("pattern reciprocity ")
    print(pipeline.pattern_reciprocity)
    print("P Value of each message")
    for message in pipeline.history:
        print(message.text_content, message.p_value)
    
    print(pipeline.history)
