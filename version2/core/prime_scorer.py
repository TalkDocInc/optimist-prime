# prime_scorer.py
import string
import sympy

# Generate prime list for a–z (26 primes)
primes = list(sympy.primerange(2, 150))[:26]
char_to_prime = {c: p for c, p in zip(string.ascii_lowercase, primes)}

def word_to_prime_sum(word: str) -> int:
    return sum(char_to_prime.get(c, 0) for c in word.lower() if c.isalpha())

def is_prime(n: int) -> bool:
    return sympy.isprime(n)

def all_prime_substring_sums(text: str) -> list[int]:
    """Return all contiguous word-level substrings whose prime sum is prime."""
    words = [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]
    prime_sums = []
    n = len(words)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += word_to_prime_sum(words[j])
            if is_prime(s):
                prime_sums.append(s)
    return prime_sums