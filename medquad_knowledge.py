"""
medquad_knowledge.py

Wires the standalone MedQuAD retrieval + entity-recognition pipeline
(medquad_loader.py, medical_ner.py, retriever.py) into the existing
customer-service chatbot as a SECOND knowledge source, alongside the
company FAISS knowledge base built by langchain_helper.py.

Design goals:
  - Lazy: MedQuAD requires the dataset cloned locally (git clone
    https://github.com/abachaa/MedQuAD.git) and a one-time index build.
    Nothing here runs at import time -- if the dataset isn't present, the
    rest of the chatbot keeps working exactly as before (company KB only).
  - Cheap to call repeatedly: loaded once per process and cached.
  - Same trust model as the rest of reasoning_engine.py: this module only
    RETRIEVES text. It never generates an answer itself -- retrieved MedQuAD
    content is handed to the existing draft-generation + validation +
    self-correction loop in reasoning_engine.py, so medical answers are
    held to the exact same "don't say anything unsupported by context"
    standard as company-KB answers.

Usage
-----
    from medquad_knowledge import get_medical_context

    context, entities, available = get_medical_context("what are the symptoms of a fever?")
"""

import os
from typing import List, Optional, Tuple

MEDQUAD_ROOT = os.getenv("MEDQUAD_ROOT", "./MedQuAD")
QA_CACHE = "medquad_qa.csv"
FOCUS_CACHE = "medquad_focus_terms.csv"
INDEX_CACHE = "medquad_index.joblib"

_retriever = None
_recognizer = None
_load_attempted = False
_load_error: Optional[str] = None


def _try_load():
    """Attempt to load (or build) the MedQuAD retriever + recognizer once.
    Safe to call many times -- only does real work on the first call."""
    global _retriever, _recognizer, _load_attempted, _load_error

    if _load_attempted:
        return

    _load_attempted = True
    try:
        from medquad_loader import load_or_build_dataset, load_or_build_focus_terms
        from medical_ner import MedicalEntityRecognizer
        from retriever import load_or_build_retriever

        qa_df = load_or_build_dataset(MEDQUAD_ROOT, QA_CACHE)
        focus_df = load_or_build_focus_terms(MEDQUAD_ROOT, FOCUS_CACHE)
        _retriever = load_or_build_retriever(qa_df, INDEX_CACHE)
        _recognizer = MedicalEntityRecognizer(focus_df)
    except Exception as e:
        _load_error = str(e)
        _retriever = None
        _recognizer = None


def is_available() -> bool:
    """Whether the MedQuAD knowledge source is loaded and usable."""
    _try_load()
    return _retriever is not None and _recognizer is not None


def load_error() -> Optional[str]:
    _try_load()
    return _load_error


def force_rebuild():
    """Re-parse the MedQuAD XML and rebuild the index/recognizer from scratch."""
    global _retriever, _recognizer, _load_attempted, _load_error
    from medquad_loader import load_or_build_dataset, load_or_build_focus_terms
    from medical_ner import MedicalEntityRecognizer
    from retriever import load_or_build_retriever

    qa_df = load_or_build_dataset(MEDQUAD_ROOT, QA_CACHE, force_rebuild=True)
    focus_df = load_or_build_focus_terms(MEDQUAD_ROOT, FOCUS_CACHE, force_rebuild=True)
    _retriever = load_or_build_retriever(qa_df, INDEX_CACHE, force_rebuild=True)
    _recognizer = MedicalEntityRecognizer(focus_df)
    _load_attempted = True
    _load_error = None


def get_medical_context(question: str, top_k: int = 2, min_score: float = 0.22, max_answer_chars: int = 700) -> Tuple[str, List[dict], bool]:
    """
    Runs entity recognition + retrieval against the MedQuAD knowledge base.

    min_score filters out weak matches. TF-IDF cosine similarity is never
    exactly zero for unrelated text (shared stopword-adjacent tokens can
    produce scores around 0.10-0.16 even for completely irrelevant queries
    -- e.g. "Can I take this course?" weakly matching a disease "outlook"
    answer purely on the word "course"). A threshold keeps that noise out
    of company-only questions rather than relying on the LLM to notice it's
    irrelevant.

    top_k and max_answer_chars are kept small deliberately: this context
    block gets resent on EVERY LLM call in a turn (draft, validate, and any
    revise/re-validate retries), so a small number of trimmed answers keeps
    token usage bounded against rate limits like Groq's free-tier 8,000
    tokens/minute cap -- a handful of full-length NIH answers can exceed
    that on their own before any generation even happens.

    Returns (context_block, entities, available):
      context_block -- formatted text ready to drop into an LLM prompt,
                        or "" if nothing relevant enough was found / KB unavailable.
      entities      -- list of {"type": ..., "text": ...} dicts detected in
                        the question (diseases, medications, herbs, symptoms).
      available     -- whether the MedQuAD KB is loaded at all (lets the
                        caller distinguish "no KB" from "KB loaded, no match").
    """
    if not is_available():
        return "", [], False

    ner_result = _recognizer.recognize(question)
    entities = [{"type": e.entity_type, "text": e.text} for e in ner_result.entities]

    boost_terms = (
        ner_result.by_type("DISEASE")
        + ner_result.by_type("MEDICATION")
        + ner_result.by_type("HERB_SUPPLEMENT")
    )

    results = _retriever.search(question, top_k=top_k, boost_focus_terms=boost_terms)
    results = [r for r in results if r.score >= min_score]
    if not results:
        return "", entities, True

    blocks = []
    for r in results:
        answer_text = r.answer
        if len(answer_text) > max_answer_chars:
            answer_text = answer_text[:max_answer_chars].rsplit(" ", 1)[0] + "..."
        blocks.append(
            f"[MedQuAD / {r.source} / topic: {r.focus}]\n"
            f"Q: {r.question}\n"
            f"A: {answer_text}\n"
            f"Source: {r.url}"
        )
    context_block = "\n\n".join(blocks)
    return context_block, entities, True
