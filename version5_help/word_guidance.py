"""
word_guidance.py — Optimist Prime, Version 5 (improved)

A program that provides guidance.

Design principles below were divined by running the original oracle 100 times
(1,200 words) on the question "how do I improve this program?", and reading the
tea leaves. The loudest signals, now encoded as rules:

  FUNCTION      the program must actually run (a NameError crashed the main
                entry point; is_prime iterated over an int; the Gita corpus
                was loaded but never used; paths broke outside the cwd)
  SIMPLIFY      one clear mechanism, spared of noise
  SIGNIFY       words must be signposts: speakable, significant, citable —
                not "cochlospermaceous" and "sphenethmoidal"
  COHERE        output arrives as themed collections (guidance / advice /
                action), not a disjointed flat list
  SANIFY        deterministic by default, deduplicated, validated — so the
                results can be checked (see meta/GOALS.md)
  THROTTLE      randomness is throttled: a seeded signature, with optional
                salt only when variety is explicitly requested
  WARMTH        words you could actually send to another human
  DISCOURSE     every word carries a gloss; every report ends in a reading
  ACTION        every reading ends in something to do (no "inactionist")

Typical use:

    from word_guidance import guidance_report
    report = guidance_report(LONG_CONTEXT_OF_THE_SITUATION, 2, RELEVANT_PEOPLE)
    print(report["reading"])

The original API is preserved and backward compatible:

    words_to_use_in_outgoing_messages_or_to_collect_advice_or_take_action(
        context, m_samples, persons_name)
    words_to_use_in_response(input_string, m_samples, persons_name)
"""

import os
import re
import random
from math import prod

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------

def _load_dictionary():
    """Full system dictionary. Used for VALIDATION only — never drawn from
    raw, because 235k unfiltered words are mostly unusable obscura."""
    for path in ("/usr/share/dict/words", "/usr/dict/words"):
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read().split()
    return []


def _load_gita(valid_words):
    """The Bhagavad Gita (Song Celestial translation) ships with this repo.
    The original program loaded it and never used it; it is now the wisdom
    layer of every report. Words are kept only if they are real dictionary
    English — the raw text's dehyphenated compounds ("fathersinlaw") are
    not signposts anyone can use."""
    path = os.path.join(_HERE, "bhagavad_gita.txt")
    if not os.path.exists(path):
        return [], []
    with open(path, "r") as f:
        text = f.read()
    all_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    archaic = {"thee", "thou", "thy", "thine", "hath", "doth", "unto", "ye",
               "art", "thyself", "'tis", "ere", "hither", "whence", "thence"}
    lexicon = sorted({w for w in (re.sub(r"[^a-z]", "", t.lower()) for t in text.split())
                      if len(w) >= 4 and w in valid_words and w not in archaic})
    return all_lines, lexicon


word_list = _load_dictionary()                      # kept for backward compat
_DICT_LOWER = set(w.lower() for w in word_list)
gita_lines, gita_lexicon = _load_gita(_DICT_LOWER)
gita_words = gita_lexicon                           # kept for backward compat


# --------------------------------------------------------------------------
# Curated thematic banks (word -> one-line gloss)
# --------------------------------------------------------------------------
# The oracle's own words, distilled: signify, simplify, cohere, friendship,
# think, throttle, sanity, action. Each bank is small, speakable, and warm —
# words you would actually put in an outgoing message.

GUIDANCE_BANK = {
    "clarity": "make the request, and the goal, unmistakably clear",
    "simplify": "remove everything that does not serve the purpose",
    "signpost": "point plainly at the next step",
    "focus": "one thing at a time, fully",
    "listen": "hear what is actually being said before answering",
    "measure": "check the result against reality",
    "patience": "let the answer ripen; do not force it",
    "honesty": "say the true thing, kindly",
    "context": "give the whole situation, not a fragment",
    "question": "the right question outranks a quick answer",
    "steady": "calm, even, unhurried progress",
    "verify": "trust, then check anyway",
    "signal": "attend to what repeats; ignore what flickers once",
    "quiet": "make room to think",
    "compass": "orient by values, not by moods",
    "threshold": "know when to cross from planning into doing",
    "mirror": "reflect the other person's words back to them",
    "horizon": "keep the long goal in view while taking the short step",
    "anchor": "hold to what does not change",
    "lantern": "light only the next few feet of the path",
    "spare": "spare the noise; say only what serves",
    "candor": "gentle directness, without varnish",
    "evidence": "prefer what is shown over what is assumed",
    "humility": "hold conclusions loosely",
    "warmth": "be the friendly presence in the exchange",
}

ADVICE_BANK = {
    "empathy": "stand in the other person's situation first",
    "specificity": "vague advice helps no one; name the exact thing",
    "brevity": "short advice gets followed",
    "timing": "the right counsel at the wrong moment is the wrong counsel",
    "reciprocity": "give the kind of response you hope to receive",
    "curiosity": "ask one more question than feels necessary",
    "courage": "name the difficult thing once, gently",
    "gratitude": "acknowledge what is already working",
    "restraint": "do not advise what was not asked about",
    "precedent": "look for what worked before, and reuse it",
    "smallness": "shrink the advice until it can be done today",
    "sincerity": "mean it, or do not say it",
    "perspective": "zoom out one level before concluding",
    "kindness": "correction lands only when wrapped in regard",
    "candor": "withhold nothing material; say it without cruelty",
    "patience": "let the other person finish their thought",
    "verification": "check the facts before forwarding them",
    "boundaries": "be clear about what you can and cannot offer",
    "followup": "circle back; advice without follow-through evaporates",
    "tone": "how it sounds is part of what it says",
    "listening": "most people want to be heard before they want to be helped",
    "consent": "offer, do not impose",
    "hope": "leave the other person more able than you found them",
    "practicality": "every suggestion should survive contact with Tuesday",
}

ACTION_BANK = {
    "ask": "pose the question directly, today",
    "write": "draft the message and send it",
    "test": "run the smallest experiment that could change your mind",
    "call": "use the phone; resolve in minutes what email stretches into weeks",
    "ship": "release the imperfect version now; improve it in the open",
    "rest": "sleep before deciding anything large",
    "review": "re-read what you wrote before sending",
    "schedule": "put it on the calendar or it does not exist",
    "delegate": "hand off what does not require you",
    "decline": "say no to one thing, cleanly, to protect the main thing",
    "begin": "start with two minutes of the actual work",
    "finish": "close one open loop before opening another",
    "document": "write down the decision and the reason",
    "apologize": "repair the small tear before it widens",
    "thank": "send the thank-you note you have been composing in your head",
    "measure": "decide what 'better' looks like in numbers, then look",
    "practice": "rehearse the hard conversation once aloud",
    "prune": "delete one commitment this week",
    "reply": "answer the oldest unanswered message",
    "walk": "take the thinking walk; return with the answer",
    "backup": "protect the work before extending it",
    "confirm": "verify the appointment, the address, the assumption",
    "celebrate": "mark the small win out loud",
    "repeat": "do the working thing again tomorrow",
}

_BANKS = {
    "guidance": GUIDANCE_BANK,
    "advice": ADVICE_BANK,
    "action": ACTION_BANK,
}


# --------------------------------------------------------------------------
# Prime engine (fixed)
# --------------------------------------------------------------------------

class PrimeComplexityEngine:
    def __init__(self):
        # Alphabet prime mapping
        self.primes_map = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                           43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
        self.char_map = {chr(i + 97): self.primes_map[i] for i in range(26)}
        with open(os.path.join(_HERE, "..", "baseline_lambda.json")) as f:
            import json
            self.baseline_lambda = json.load(f)["baseline_lambda"]

    def is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    def get_word_prime_value(self, word: str) -> int:
        clean_word = re.sub(r"[^a-z]", "", word.lower())
        return prod(self.char_map.get(c, 0) for c in clean_word)

    def context_signature(self, context: str, persons_name: str) -> int:
        """A deterministic integer fingerprint of the situation, built from
        the prime values of its words. Same question + same people → same
        signature → same reading. This is what makes results checkable."""
        sig = 0
        for idx, word in enumerate((context + " " + persons_name).split()):
            sig += (idx + 1) * (self.get_word_prime_value(word) % 1_000_003)
        return sig


_PCE = PrimeComplexityEngine()


# --------------------------------------------------------------------------
# Core mechanism (unchanged in spirit, throttled in practice)
# --------------------------------------------------------------------------

def _draw_words(context, m_samples, persons_name, lexicon, vary=False, salt=0):
    """Walk the context m_samples words at a time, accumulate prime values,
    and index into the lexicon — the original divination mechanism, with
    two repairs: the lexicon is curated (SIGNIFY), and the random salt is
    throttled — applied only when vary=True (THROTTLE)."""
    if not lexicon:
        return []
    seed = _PCE.context_signature(context, persons_name) + salt
    rng = random.Random() if vary else random.Random(seed)
    input_words = context.split()
    drawn = []
    seen = set()
    person_val = _PCE.get_word_prime_value(persons_name)
    for i in range(0, len(input_words), m_samples):
        sum_so_far = person_val
        if vary:
            sum_so_far += rng.randint(0, 1000)
        for q in range(i, min(len(input_words), i + m_samples)):
            sum_so_far += _PCE.get_word_prime_value(input_words[q])
        word = lexicon[sum_so_far % len(lexicon)]
        if word not in seen:                    # COHERE: no stutter
            seen.add(word)
            drawn.append(word)
    return drawn


def _is_valid_word(word):
    """SANIFY: every emitted word must be a real word (curated bank or
    full system dictionary)."""
    return word in _ALL_VALID


_ALL_VALID = set(_DICT_LOWER)
for _bank in _BANKS.values():
    _ALL_VALID.update(_bank.keys())
_ALL_VALID.update(gita_lexicon)


# --------------------------------------------------------------------------
# Definitions (uses define.py's macOS DictionaryServices when available,
# falls back to the built-in glosses — every bank word has one)
# --------------------------------------------------------------------------

def define_word(word):
    try:
        from DictionaryServices import DCSCopyTextDefinition
        result = DCSCopyTextDefinition(None, word, (0, len(word)))
        if result:
            return re.sub(r"\s+", " ", result).strip()
    except Exception:
        pass
    for bank in _BANKS.values():
        if word in bank:
            return bank[word]
    return None


# --------------------------------------------------------------------------
# Gita resonance — the wisdom layer the original program never used
# --------------------------------------------------------------------------

def gita_resonance(context, persons_name, vary=False):
    """Draw one word from the Gita lexicon and return it together with the
    verse line it lives in, so the reading has a root in the text."""
    if not gita_lexicon:
        return None
    drawn = _draw_words(context, 1, persons_name, gita_lexicon, vary=vary)
    if not drawn:
        return None
    word = drawn[0]
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.I)
    matches = [ln for ln in gita_lines if pattern.search(ln)]
    verse = next((ln for ln in matches if len(ln) > 20),
                 matches[0] if matches else None)
    return {"word": word, "verse": verse}


# --------------------------------------------------------------------------
# Public API — original signatures preserved, behavior improved
# --------------------------------------------------------------------------

def words_to_use_in_response(input_string, m_samples, persons_name, vary=False):
    """Words to weave into a reply. Drawn from the warm, speakable union of
    the curated banks; deterministic for a given input unless vary=True."""
    lexicon = sorted(set(GUIDANCE_BANK) | set(ADVICE_BANK) | set(ACTION_BANK))
    return _draw_words(input_string, m_samples, persons_name, lexicon, vary=vary)


def words_to_use_in_outgoing_messages_or_to_collect_advice_or_take_action(
        context, m_samples, persons_name, vary=False):
    """The original entry point, repaired: returns a deduplicated list of
    meaningful words, ordered guidance → advice → action."""
    report = guidance_report(context, m_samples, persons_name, vary=vary)
    combined = (report["words_to_use_in_outgoing_messages"]
                + report["advice_to_collect"]
                + report["actions_to_take"])
    return list(dict.fromkeys(combined))


def guidance_report(context, m_samples, persons_name, vary=False):
    """The flagship. Returns a coherent, validated, self-interpreting report:

      words_to_use_in_outgoing_messages  — signposts for your own words
      advice_to_collect                  — what to seek from others
      actions_to_take                    — what to actually do
      gita_resonance                     — one word + its verse from the Gita
      definitions                        — gloss for every word
      reading                            — the whole report in one paragraph
      signature                          — deterministic fingerprint (audit)
      valid                              — True iff every word passed validation
    """
    m = max(1, m_samples)
    salt_cycle = {"guidance": 0, "advice": 1, "action": 2}
    out = {
        "context": context.strip(),
        "persons": persons_name,
        "signature": _PCE.context_signature(context, persons_name),
    }
    for theme, bank in _BANKS.items():
        lexicon = sorted(bank)
        drawn = _draw_words(context, m, persons_name, lexicon,
                            vary=vary, salt=salt_cycle[theme])
        out[{
            "guidance": "words_to_use_in_outgoing_messages",
            "advice": "advice_to_collect",
            "action": "actions_to_take",
        }[theme]] = drawn

    out["gita_resonance"] = gita_resonance(context, persons_name, vary=vary)

    all_words = (out["words_to_use_in_outgoing_messages"]
                 + out["advice_to_collect"] + out["actions_to_take"])
    out["definitions"] = {w: define_word(w) for w in all_words}
    out["valid"] = all(_is_valid_word(w) for w in all_words)
    out["reading"] = _compose_reading(out)
    return out


def _compose_reading(report):
    g = report["words_to_use_in_outgoing_messages"]
    a = report["advice_to_collect"]
    d = report["actions_to_take"]
    res = report["gita_resonance"]

    def weave(words):
        if not words:
            return "silence"
        if len(words) == 1:
            return words[0]
        return ", ".join(words[:-1]) + " and " + words[-1]

    parts = [
        f"Speak with {weave(g[:3])}.",
        f"Seek {weave(a[:3])} from {report['persons']}.",
        f"Then: {weave(d[:3])}.",
    ]
    if res and res.get("verse"):
        parts.append(f'Hold the word "{res["word"]}" — "{res["verse"].strip()}"')
    return " ".join(parts)


# --------------------------------------------------------------------------
# Self-check (meta/GOALS.md: results must be feasible and not anomalous)
# --------------------------------------------------------------------------

def self_check():
    """Sanity validations. Returns a dict of check -> bool; all must be True."""
    probe = "Provide words that will guide me on a difficult decision"
    r1 = guidance_report(probe, 2, "Self Check")
    r2 = guidance_report(probe, 2, "Self Check")
    varied = {tuple(words_to_use_in_response(probe, 2, "Self Check", vary=True))
              for _ in range(5)}
    flat = (r1["words_to_use_in_outgoing_messages"]
            + r1["advice_to_collect"] + r1["actions_to_take"])
    return {
        "deterministic_same_input_same_output": r1["reading"] == r2["reading"],
        "vary_produces_variety": len(varied) > 1,
        "no_duplicate_words": len(flat) == len(set(flat)),
        "all_words_valid": r1["valid"],
        "all_words_have_definitions": all(r1["definitions"].values()),
        "is_prime_correct": all([_PCE.is_prime(2), _PCE.is_prime(3),
                                 _PCE.is_prime(5), _PCE.is_prime(7),
                                 not _PCE.is_prime(1), not _PCE.is_prime(4),
                                 not _PCE.is_prime(9), not _PCE.is_prime(0)]),
        "gita_resonance_present": r1["gita_resonance"] is not None,
        "banks_nonempty": all(_BANKS.values()),
        "dictionary_loaded": len(word_list) > 1000,
    }


# --------------------------------------------------------------------------
# CLI: python3 word_guidance.py "context" [m_samples] "persons"
# --------------------------------------------------------------------------

def _print_report(report):
    print("=" * 68)
    print("OPTIMIST PRIME v5 — GUIDANCE REPORT")
    print("=" * 68)
    print(f"For:       {report['persons']}")
    print(f"Signature: {report['signature']}  (valid: {report['valid']})")
    sections = [
        ("WORDS TO USE IN OUTGOING MESSAGES", "words_to_use_in_outgoing_messages"),
        ("ADVICE TO COLLECT", "advice_to_collect"),
        ("ACTIONS TO TAKE", "actions_to_take"),
    ]
    for title, key in sections:
        print(f"\n{title}:")
        for w in report[key]:
            gloss = report["definitions"].get(w) or ""
            print(f"  • {w:<14} {gloss}")
    res = report["gita_resonance"]
    if res:
        print(f"\nGITA RESONANCE: {res['word']}")
        if res.get("verse"):
            print(f"  “{res['verse'].strip()}”")
    print(f"\nREADING:\n  {report['reading']}")
    print("=" * 68)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] in ("--check", "self_check"):
        results = self_check()
        for k, v in results.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        sys.exit(0 if all(results.values()) else 1)
    _context = sys.argv[1] if len(sys.argv) > 1 else \
        "Provide words that will guide me on improving this program"
    _m = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    _persons = sys.argv[3] if len(sys.argv) > 3 else "the people involved"
    _print_report(guidance_report(_context, _m, _persons))
