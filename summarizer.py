"""
summarizer.py

Extractive summarization via TextRank (Mihalcea & Tarau, 2004) -- a graph
-based ranking algorithm, the text analog of PageRank. Sentences are nodes;
edge weight is TF-IDF cosine similarity between sentences; each sentence's
importance is computed by running PageRank over that similarity graph, and
the top-ranked original sentences are returned as the summary.

Deliberately dependency-light and deterministic: no model downloads, works
offline, and -- importantly -- since sentences are pulled verbatim from the
abstract rather than generated, the summary can never contain a claim the
abstract didn't make. This complements (not replaces) the LLM-based
explanation step in explanation_engine.py, which is used for genuinely
generative work (explaining a concept in the user's own terms) rather than
condensing existing text.
"""

import re
from typing import List

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def summarize(text: str, num_sentences: int = 2) -> str:
    """
    Returns the top num_sentences sentences (by TextRank score), in their
    ORIGINAL order of appearance -- reordering by score alone tends to
    produce summaries that read like disconnected fragments.
    """
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(matrix)
    except ValueError:
        # e.g. all sentences reduced to nothing but stopwords
        return " ".join(sentences[:num_sentences])

    graph = nx.from_numpy_array(sim_matrix)
    try:
        scores = nx.pagerank(graph)
    except nx.PowerIterationFailedConvergence:
        return " ".join(sentences[:num_sentences])

    ranked_idx = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    ranked_idx.sort()  # restore original order
    return " ".join(sentences[i] for i in ranked_idx)


if __name__ == "__main__":
    abstract = (
        "Long documents pose a challenge for transformer-based summarization models "
        "due to the quadratic cost of self-attention with respect to sequence length. "
        "We propose a sparse attention mechanism that restricts attention computation "
        "to a fixed window combined with a small number of global attention tokens, "
        "enabling the model to process documents of several thousand tokens. "
        "Evaluation on scientific paper and news summarization benchmarks demonstrates "
        "improved ROUGE scores and substantially reduced memory usage compared to full "
        "self-attention baselines."
    )
    print(summarize(abstract, num_sentences=2))
