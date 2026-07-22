# compute_prime_baseline.py
import random
import json
import time
import os
import re
import numpy as np

# Your existing char→prime mapping
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
char_to_prime = {chr(i + 97): p for i, p in enumerate(primes)}
ALPHABET = set(char_to_prime.keys())

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def word_prime_sum(word: str) -> int:
    clean = re.sub(r'[^a-zA-Z]', '', word).lower()
    return sum(char_to_prime.get(c, 0) for c in clean)

def main():
    WORDS_FILE = "/usr/share/dict/words"
    N_SAMPLES = 100_000_000  # 100 million
    BATCH_SIZE = 100_000   # process in chunks to avoid memory overload
    OUTPUT_PATH = "baseline_lambda.json"

    # Load words: filter to alphabetic only, lowercase, dedupe
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

    if n_words < 2:
        raise ValueError("Not enough words for pair sampling.")

    # Precompute word prime sums (caching avoids repeated work)
    word_to_prime = {w: word_prime_sum(w) for w in words}
    print("✅ Precomputed word prime sums")

    # Seed for reproducibility (optional)
    random.seed(42)

    # Track totals
    total_tests = 0
    prime_hits = 0

    print(f"Sampling {N_SAMPLES:,} random word pairs...")
    start = time.time()

    for batch_start in range(0, N_SAMPLES, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N_SAMPLES)
        batch_size = batch_end - batch_start

        # Sample pairs: two *independent* random words (repetition allowed)
        idx1 = np.random.randint(0, n_words, size=batch_size)
        idx2 = np.random.randint(0, n_words, size=batch_size)

        # Get sums: sum(word1_prime + word2_prime)
        primes_batch = [word_to_prime[words[i]] for i in idx1]
        primes_batch2 = [word_to_prime[words[i]] for i in idx2]
        sums = np.array(primes_batch) + np.array(primes_batch2)

        # Count primes in the sum (i.e., is sum(word1) + sum(word2) prime?)
        batch_hits = np.sum([is_prime(s) for s in sums])
        prime_hits += int(batch_hits)
        total_tests += batch_size

        # Progress report
        elapsed = time.time() - start
        rate = batch_end / elapsed
        print(f"[{batch_end:,}/{N_SAMPLES:,}] "
              f"Prime hits: {prime_hits:,} / {total_tests:,} "
              f"({prime_hits/total_tests:.5%}), "
              f"Rate: {rate:.0f} pairs/sec, "
              f"ETA: {(N_SAMPLES - batch_end) / rate / 60:.1f} min")

    # Final stats
    density = prime_hits / total_tests
    lambda_hat = density  # For Poisson, λ ≈ observed density

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total pairs tested     : {total_tests:,}")
    print(f"Prime-sum pairs found  : {prime_hits:,}")
    print(f"Empirical prime density: {density:.8f} (≈ {density*100:.4f}%)")
    print(f"Calibrated baseline λ  : {lambda_hat:.8f}")
    print(f"Time elapsed           : {(time.time() - start)/60:.1f} min")
    print("=" * 60)

    # Save calibrated baseline
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"baseline_lambda": lambda_hat, "n_samples": N_SAMPLES}, f, indent=2)
    print(f"\n✅ Baseline saved to {OUTPUT_PATH}")

    return lambda_hat

if __name__ == "__main__":
    main()