"""
keyphrase_extraction.py

Information extraction: pulls out candidate key phrases from a paper
abstract. Implements RAKE (Rapid Automatic Keyword Extraction) -- a
classic, well-established, dependency-light extractive keyphrase algorithm
(Rose et al., 2010). No model downloads, no external corpora required:
just stopword-based phrase chunking + word co-occurrence scoring.

Algorithm
---------
1. Split text into candidate phrases at stopwords/punctuation boundaries.
2. Build a word co-occurrence graph across all candidate phrases.
3. Score each word as degree(word) / frequency(word) -- words that appear
   in many different phrase contexts (high degree) relative to how often
   they occur alone score higher.
4. Score each phrase as the sum of its words' scores.
5. Return the top-N phrases by score.
"""

import re
from collections import defaultdict
from typing import List, Tuple

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "of",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "out", "off", "over", "under", "again", "further",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "this", "that", "these", "those", "we", "our",
    "us", "it", "its", "as", "such", "which", "while", "also", "not", "no",
    "using", "used", "use", "based", "propose", "proposed", "present",
    "presented", "show", "shows", "shown", "results", "result", "study",
    "paper", "work", "method", "methods", "approach", "approaches",
    "compared", "comparison", "however", "both", "each", "across", "via",
    "one", "two", "several", "various", "including", "achieve", "achieves",
    "achieved", "provide", "provides", "provided", "allow", "allows",
    "allowing", "particular", "particularly",
}

_PHRASE_SPLIT_RE = re.compile(r"[.!?,;:()\[\]{}\"']|\s-\s")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def _split_into_phrases(text: str) -> List[List[str]]:
    """Splits text on punctuation, then splits each chunk into phrase
    candidates by breaking on stopwords."""
    chunks = _PHRASE_SPLIT_RE.split(text)
    phrases = []
    for chunk in chunks:
        words = _WORD_RE.findall(chunk.lower())
        current = []
        for w in words:
            if w in STOPWORDS:
                if current:
                    phrases.append(current)
                    current = []
            else:
                current.append(w)
        if current:
            phrases.append(current)
    return [p for p in phrases if p]


def extract_keyphrases(text: str, top_n: int = 8, max_phrase_len: int = 4) -> List[Tuple[str, float]]:
    """
    Returns up to top_n (phrase, score) pairs, highest score first.
    Phrases longer than max_phrase_len words are dropped (very long chains
    are usually parsing artifacts, not genuine key phrases).
    """
    if not text or not text.strip():
        return []

    phrases = [p for p in _split_into_phrases(text) if len(p) <= max_phrase_len]
    if not phrases:
        return []

    freq = defaultdict(int)
    degree = defaultdict(int)
    for phrase in phrases:
        deg = len(phrase) - 1
        for word in phrase:
            freq[word] += 1
            degree[word] += deg
    for word in freq:
        degree[word] += freq[word]  # word's own frequency counts toward degree

    word_scores = {w: degree[w] / freq[w] for w in freq}

    phrase_scores = {}
    for phrase in phrases:
        key = " ".join(phrase)
        score = sum(word_scores[w] for w in phrase)
        if key not in phrase_scores or score > phrase_scores[key]:
            phrase_scores[key] = score

    ranked = sorted(phrase_scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n]


if __name__ == "__main__":
    sample = (
        "We propose a low-rank adaptation technique that freezes the pretrained "
        "weights and injects trainable rank-decomposition matrices into each layer "
        "of the transformer architecture, drastically reducing the number of "
        "trainable parameters."
    )
    for phrase, score in extract_keyphrases(sample):
        print(f"{score:6.2f}  {phrase}")
