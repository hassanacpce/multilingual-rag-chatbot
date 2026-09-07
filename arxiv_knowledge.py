"""
arxiv_knowledge.py

Wires the standalone arXiv research-paper pipeline (arxiv_loader.py,
paper_retriever.py, keyphrase_extraction.py, summarizer.py) into the
chatbot as a THIRD knowledge source, alongside the company FAISS KB
(langchain_helper.py) and the MedQuAD medical KB (medquad_knowledge.py).

Mirrors medquad_knowledge.py's design exactly, for consistency:
  - Lazy: the raw arXiv metadata file must be present locally (see README --
    requires a Kaggle account, not fetchable automatically). Nothing here
    runs at import time; if the file isn't present, the rest of the chatbot
    keeps working exactly as before.
  - Cheap to call repeatedly: loaded once per process and cached.
  - Retrieval-only: this module never generates an answer. Retrieved paper
    context (extractive summary + key phrases, not raw abstracts) is handed
    to reasoning_engine.py's existing draft-generation + validation +
    self-correction loop, so research-paper answers are held to the exact
    same "don't say anything unsupported by context" standard as company
    and medical answers.

Usage
-----
    from arxiv_knowledge import get_research_context

    context, papers, available = get_research_context("explain low-rank adaptation")
"""

import os
from typing import List, Optional, Tuple

from paper_retriever import RetrievedPaper

ARXIV_RAW_PATH = os.getenv("ARXIV_RAW_PATH", "arxiv-metadata-oai-snapshot.json")
ARXIV_CATEGORIES = os.getenv("ARXIV_CATEGORIES", "cs.").split(",")
ARXIV_MAX_ROWS = int(os.getenv("ARXIV_MAX_ROWS", "50000"))
QA_CACHE = "cs_papers.csv"
INDEX_CACHE = "cs_papers_index.joblib"

_retriever = None
_load_attempted = False
_load_error: Optional[str] = None


def _try_load():
    global _retriever, _load_attempted, _load_error

    if _load_attempted:
        return

    _load_attempted = True
    try:
        from arxiv_loader import load_or_build_subset
        from paper_retriever import load_or_build_retriever

        papers_df = load_or_build_subset(ARXIV_RAW_PATH, QA_CACHE, ARXIV_CATEGORIES, max_rows=ARXIV_MAX_ROWS)
        _retriever = load_or_build_retriever(papers_df, INDEX_CACHE)
    except Exception as e:
        _load_error = str(e)
        _retriever = None


def is_available() -> bool:
    _try_load()
    return _retriever is not None


def load_error() -> Optional[str]:
    _try_load()
    return _load_error


def get_retriever():
    """Exposes the underlying retriever directly, for the Search Papers /
    Concept Map UI tabs, which need raw search results and coordinates
    rather than a pre-formatted LLM context block."""
    _try_load()
    return _retriever


def force_rebuild():
    global _retriever, _load_attempted, _load_error
    from arxiv_loader import load_or_build_subset
    from paper_retriever import load_or_build_retriever

    papers_df = load_or_build_subset(ARXIV_RAW_PATH, QA_CACHE, ARXIV_CATEGORIES, max_rows=ARXIV_MAX_ROWS, force_rebuild=True)
    _retriever = load_or_build_retriever(papers_df, INDEX_CACHE, force_rebuild=True)
    _load_attempted = True
    _load_error = None


def get_research_context(
    question: str, top_k: int = 2, min_score: float = 0.12, max_summary_papers: int = 2
) -> Tuple[str, List[RetrievedPaper], bool]:
    """
    Runs semantic (LSA) search against the arXiv paper corpus and builds a
    pre-digested context block: extractive summary + RAKE key phrases per
    paper, not the raw abstract. top_k/min_score kept small for the same
    reason as medquad_knowledge.get_medical_context -- this context gets
    resent on every LLM call in a turn, so token usage needs to stay bounded.

    Returns (context_block, papers, available).
    """
    if not is_available():
        return "", [], False

    results = _retriever.search(question, top_k=top_k)
    results = [r for r in results if r.score >= min_score]
    if not results:
        return "", [], True

    from keyphrase_extraction import extract_keyphrases
    from summarizer import summarize

    blocks = []
    for r in results[:max_summary_papers]:
        summary = summarize(r.abstract, num_sentences=2)
        phrases = [kp for kp, _ in extract_keyphrases(r.abstract, top_n=5)]
        blocks.append(
            f"[arXiv Paper / {r.primary_category}] {r.title}\n"
            f"Authors: {r.authors}\n"
            f"Summary: {summary}\n"
            f"Key phrases: {', '.join(phrases)}\n"
            f"Source: {r.arxiv_url}"
        )
    context_block = "\n\n".join(blocks)
    return context_block, results, True
