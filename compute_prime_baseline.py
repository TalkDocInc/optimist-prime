# compute_prime_baseline.py
import random
import json
import time
import os
import re
import numpy as np
import ollama

# ——— CONFIG ———
WORDS_FILE = "/usr/share/dict/words"
N_SAMPLES = 100_000_000  # total word pairs/triplets to sample
BATCH_SIZE = 234_456   # process per iteration to save RAM
OLLAMA_MODEL = "gemma4:e4b"
OUTPUT_PATH = "baseline_lambda.json"
ACRONYM_THRESHOLD = 0.7  # minimum Gemma confidence to accept acronym
UNIQUE_PRIME_WORDS = set()
UNIQUE_PRIME_WORDS_TUPLES = set()

# Your existing char→prime mapping
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
char_to_prime = {chr(i + 97): p for i, p in enumerate(primes)}
ALPHABET = set(char_to_prime.keys())

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True

def word_prime_sum(word: str) -> int:
    clean = re.sub(r'[^a-zA-Z]', '', word).lower()
    return sum(char_to_prime.get(c, 0) for c in clean)

def get_first_letters(word: str) -> str:
    """Get first letter (lowercase)."""
    w = re.sub(r'[^a-zA-Z]', '', word).lower()
    return w[0] if w else ""

# ——— GEMMA-BASED ACRONYM DETECTOR ———
def is_plausible_acronym(letters: str, model=OLLAMA_MODEL):
    """
    Uses local Gemma to determine if `letters` (e.g., 'con') looks like a real acronym.
    Returns (is_acronym, confidence).
    """
    knowledge_domains = {
        "Technical & Scientific Foundation": [
            "Computer Science (Algorithms, architecture, and logic)",
            "Mathematics (The language of modeling and optimization)",
            "Artificial Intelligence & Machine Learning (The frontier of automated intelligence)",
            "Data Science & Analytics (Extracting actionable insights from data)",
            "Physics & Engineering (Fundamental principles for hardware and deep tech)",
            "Biotechnology & Life Sciences (Innovation in health and biological systems)",
            "Cybersecurity (Protection of digital assets and trust)",
            "Robotics & Automation (The intersection of software and physical motion)",
            "Cognitive Science (Understanding how intelligence functions)",
            "Systems Theory (Managing complex, interconnected parts)"
        ],
        "Product, Design & User Experience": [
            "Product Management (Defining the vision, roadmap, and 'why')",
            "User Experience (UX) Design (The logic of user journeys)",
            "User Interface (UI) Design (The visual and interactive layer)",
            "Web & Visual Design (Aesthetics and digital presence)",
            "Industrial/Physical Design (Ergonomics and physical form)",
            "Human-Computer Interaction (HCI) (The study of interface effectiveness)",
            "Content Strategy & Copywriting (Using language to guide and persuade)",
            "Human Factors Engineering (Designing for human error and capability)"
        ],
        "The Commercial Engine": [
            "Business Strategy (Competitive positioning and moat building)",
            "Marketing & Growth Hacking (Customer acquisition and brand awareness)",
            "Sales & Business Development (Revenue generation and partnerships)",
            "Economics (Micro & Macro) (Market dynamics and global trends)",
            "Finance & Accounting (Managing burn rate and cash flow)",
            "Venture Capital & Fundraising (Mechanics of equity and capital)",
            "Market Research (Identifying pain points and unmet needs)",
            "E-commerce & Digital Distribution (Modern channels of commerce)",
            "Consumer Behavior (Psychology of purchasing decisions)"
        ],
        "Operations & Execution": [
            "Operations Management (The day-to-day running of a business entity)",
            "Supply Chain & Logistics (Managing movement and dependencies)",
            "Project Management (Execution via Agile, Scrum, or Waterfall)",
            "Legal & Intellectual Property (Patents, trademarks, and compliance)",
            "Risk Management (Mitigating technical and financial threats)",
            "Strategic Studies/Military Science (Resource allocation and competition)",
            "Cloud Infrastructure & DevOps (Scalability of the digital backbone)"
        ],
        "The Human & Social Context": [
            "Leadership & Organizational Behavior (Managing teams and culture)",
            "Psychology (Social & Cognitive) (Understanding human and group dynamics)",
            "Sociology & Anthropology (Understanding cultural and societal shifts)",
            "Politics & Public Policy (Regulation and the impact of law)",
            "Ethics & Philosophy (The moral implications of innovation)",
            "History (Pattern recognition through human progress)"
        ]
    }

    str_knowledge_domains = str(knowledge_domains)

    if len(letters) < 2 or len(letters) > 6:
        return False, 0.0

    prompt = f"""
    Question 1:
    Is "{letters}" a plausible English acronym or initialism, and if so what acronyms can be formed with it?
    Examples: 'NATO', 'UN', 'GPT', 'FBI' → YES. 'xkq' → NO.

    Question 2: What categories in this comma seperated list of categories "{str_knowledge_domains}" is the word?

    For example return something like ['NATO' : ['confidence': 1.0,
        'knowledge_domains': [('The Human & Social Context', 0.5), ('Leadership and Organizational Behavior', 0.6)]
    ] , 'ANTO' : ['confidence': 0.1 , ....]
    """
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        content = response["message"]["content"].strip()
        import re
        m = re.search(r'<score:\s*([\d.]+)>', content)
        if m:
            score = float(m.group(1))
        else:
            # fallback: scan for any number 0–1
            nums = re.findall(r'(\d+\.\d+|\d+)', content)
            for n in nums:
                val = float(n)
                if 0 <= val <= 1:
                    score = val
                    break
            else:
                score = 0.1  # default conservative
        return score >= ACRONYM_THRESHOLD, score
    except Exception as e:
        print(f"[Acronym Checker] Error: {e}")
        return False, 0.0

def acronym_check_batch(letters_batch):
    """Batch acronym check — for speed, use local cache if Gemma slow."""
    results = []
    for letters in letters_batch:
        #results.append(is_plausible_acronym(letters))
        results.append(True)
    return results

# ——— CORE SAMPLING LOGIC ———
def sample_and_analyze(words, word_to_prime, n_samples, batch_size, seed=42):
    n_words = len(words)

    # Pre-fetch first letters for all words (saves O(N) time)
    first_letters = [get_first_letters(w) for w in words]

    print("Starting baseline computation...")
    start = time.time()

    whole_dictionary_processed_for_single_primes = False
    whole_word_tuples_processed_for_primality_sums = False

    if not whole_dictionary_processed_for_single_primes:
        for position in range(0, min(234_456, len(words))):
            val = word_to_prime[words[position]]
            if is_prime(val):
                UNIQUE_PRIME_WORDS.add(position)

       #print("UNIQUE PRIME WORDS COUNT")
        #print(len(UNIQUE_PRIME_WORDS))
        #print("UNIQUE PRIME WORDS LIST")
        #print([words[i] for i in UNIQUE_PRIME_WORDS])
    
        whole_dictionary_processed_for_single_primes = True
    
    considered_amount_update_size = 100_000
    considered = 0
    pairs = 0

    if not whole_word_tuples_processed_for_primality_sums:

        for position_word_1 in range(0, min(234_456, len(words))):
            for position_word_2 in range(0, min(234_456, len(words))):
                val = word_to_prime[words[position_word_1]] + word_to_prime[words[position_word_2]]
                if ((words[position_word_1], words[position_word_2],val ) in UNIQUE_PRIME_WORDS_TUPLES or 
                   (words[position_word_2], words[position_word_1],val ) in UNIQUE_PRIME_WORDS_TUPLES ):
                    continue
                if is_prime(val):
                    UNIQUE_PRIME_WORDS_TUPLES.add( (words[position_word_1], words[position_word_2],val ) )
                    pairs += 1
                considered += 1
                percentage = ((pairs) / considered) * 100
                if considered % considered_amount_update_size == 0 and pairs > 0 and considered > 0:
                    print(f"Considered {considered} pairs")
                    percentage = f"Percentage of Pairs that are prime so far is {percentage}"
                    print(percentage)

                if considered > 10_000_000:
                    break

        whole_word_tuples_processed_for_primality_sums = True

        #print("UNIQUE PRIME TUPLES WORDS")
        #print("UNIQUE_PRIME_WORDS_TUPLES words")
        #print(UNIQUE_PRIME_WORDS_TUPLES)


def calculate_primes_on_main_line(message, words):
    prime_pairs = 0
    total_pairs = 0

    word_to_prime = {w: word_prime_sum(w) for w in words}
    message_word_list = message.split(" ")
    for i in range(0, len(message_word_list) - 1):
        val = word_to_prime[words[i]] + word_to_prime[words[i+1]]
        total_pairs += 1
        if is_prime(val):
            prime_pairs += 1
    
    ratio = prime_pairs / total_pairs
    print(f"Ratio between prime_pairs and total_pairs in this message is {ratio}")
    return ratio




# ——— MAIN ———
def main():
    # Load words
    if not os.path.exists(WORDS_FILE):
        raise FileNotFoundError(f"Dictionary not found at {WORDS_FILE}")
    print("Loading dictionary...")
    with open(WORDS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        words = set()
        for line in f:
            w = re.sub(r'[^a-zA-Z]', '', line.strip()).lower()
            if len(w) >= 1 and all(c in ALPHABET for c in w):
                words.add(w)
    words = list(words)
    n_words = len(words)
    print(f"Loaded {n_words:,} valid words (alphabet-only, deduplicated)")

    # Precompute word prime sums
    print("Precomputing word prime sums...")
    word_to_prime = {w: word_prime_sum(w) for w in words}

    # Run sampling
    sample_and_analyze(words, word_to_prime, N_SAMPLES, BATCH_SIZE)
    
    message = """ 
Hi Rishi,

Ooshma looks forward to meeting you this morning! Is there any chance you can come at 11:30am instead? She has a meeting until 11am and I want to give her ample time in between. Please let me know if this does not work for you. Otherwise we look forward to seeing you at 11:30am!

Thank you kindly,

Allison Washburn
Executive Assistant 
GOBBLE
        """

    calculate_primes_on_main_line(message, words)
    
    # Save to JSON

if __name__ == "__main__":
    main()