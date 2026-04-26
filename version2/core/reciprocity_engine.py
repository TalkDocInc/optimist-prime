# reciprocity_engine.py

# reciprocity_engine.py
import numpy as np
from scipy import stats
from collections import Counter
from typing import List, Dict, Tuple
import pickle
import os

NORM_CORPUS_PATH = "norm_prime_stats.pkl"

def build_norm_corpus(transcripts: List[str]) -> Dict:
    """
    Build baseline prime-density stats from corpus.
    Returns: {'mean': float, 'std': float, 'density_per_sentence': List[float]}
    """
    densities = []
    for t in transcripts:
        subs = all_prime_substring_sums(t)
        words = [w for w in t.split() if w.strip()]
        density = len(subs) / max(len(words), 1)
        densities.append(density)
    return {
        'mean': np.mean(densities),
        'std': np.std(densities),
        'p95': np.percentile(densities, 95),
        'density_list': densities
    }

def load_or_build_norm_corpus():
    if os.path.exists(NORM_CORPUS_PATH):
        with open(NORM_CORPUS_PATH, 'rb') as f:
            return pickle.load(f)
    # In practice: seed with 500+ real transcripts from public dialogues (e.g., Reddit, YouTube comments)
    print("⚠️ No norm corpus found. Build it with `build_norm_corpus()` on real data.")
    return {'mean': 0.07, 'std': 0.04, 'p95': 0.15, 'density_list': [0.07]*100}  # placeholder

def speaker_prime_profile(text: str, norm_stats: Dict = None) -> Dict:
    if norm_stats is None:
        norm_stats = load_or_build_norm_corpus()
    subs = all_prime_substring_sums(text)
    words = [w for w in text.split() if w.strip()]
    density = len(subs) / max(len(words), 1)
    z_score = (density - norm_stats['mean']) / max(norm_stats['std'], 1e-6)
    return {
        'prime_sum_list': subs,
        'prime_count': len(subs),
        'word_count': len(words),
        'prime_density': density,
        'z_score': z_score,
        'deviation_significance': stats.norm.sf(abs(z_score))  # p-value
    }

def reciprocal_pattern_score(s1_text: str, s2_text: str) -> float:
    """
    Returns 0–1 score: how similarly Speaker 2 mirrors Speaker 1's prime pattern.
    Measures:
    - #primes match (Jaccard)
    - density ratio match (s2/s1 ≈ 1 → high)
    - distribution similarity (KS test)
    """
    p1 = all_prime_substring_sums(s1_text)
    p2 = all_prime_substring_sums(s2_text)

    if not p1 or not p2:
        return 0.0

    # Jaccard overlap of prime sets
    set1, set2 = set(p1), set(p2)
    jaccard = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0

    # Density ratio (log scale)
    d1 = len(p1) / max(len(s1_text.split()), 1)
    d2 = len(p2) / max(len(s2_text.split()), 1)
    density_ratio = min(d1/d2, d2/d1) if d1 > 0 and d2 > 0 else 0

    # Distribution similarity (KS test on prime values)
    ks_stat, _ = stats.ks_2samp(p1, p2, method='exact')

    # Composite score (weighted)
    score = 0.4 * jaccard + 0.3 * density_ratio + 0.3 * (1 - min(ks_stat, 1))
    return max(0.0, min(1.0, score))

def cognitive_reciprocity_score(s1_text: str, s2_text: str, norm_stats: Dict = None) -> Dict:
    profile1 = speaker_prime_profile(s1_text, norm_stats)
    profile2 = speaker_prime_profile(s2_text, norm_stats)
    rec_score = reciprocal_pattern_score(s1_text, s2_text)
    return {
        'speaker1_profile': profile1,
        'speaker2_profile': profile2,
        'reciprocity_score': rec_score,
        'mutual_karma_estimate': rec_score * (1 - profile1['deviation_significance']) * (1 - profile2['deviation_significance'])  # heuristic
    }