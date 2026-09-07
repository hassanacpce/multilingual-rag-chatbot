"""
retriever.py

Retrieval mechanism for the medical Q&A chatbot.

Approach: TF-IDF over each QA pair's question (question text weighted more
heavily than answer text, since users phrase queries as questions) with
cosine similarity search. This is deliberately dependency-light -- no
external model downloads required, fully deterministic, fast to build
(~16k QA pairs indexes in well under a second) and fast to query.

Entity-aware boosting: if medical_ner.py detects a DISEASE/MEDICATION/
HERB_SUPPLEMENT entity in the user's question, QA pairs whose `focus` field
matches that entity get a similarity boost. This lets "what is the outlook
for it" style follow-ups (where the entity was established earlier in
conversation) land on the right document even when the raw text overlap
with the question is thin.

The index is cached to disk (joblib) alongside the QA dataframe so it only
needs to be rebuilt when the underlying dataset changes.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedAnswer:
    question: str
    answer: str
    score: float
    focus: str
    source: str
    qtype: str
    url: str
    doc_id: str


class MedQuADRetriever:
    def __init__(self, qa_df: pd.DataFrame):
        self.df = qa_df.reset_index(drop=True)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def fit(self):
        # Question text carries most of the retrieval signal; folding in a
        # little answer text helps match queries phrased more like statements
        # than questions (e.g. "chest pain and shortness of breath").
        corpus = (self.df["question"] + " " + self.df["question"] + " " + self.df["answer"]).tolist()
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=1,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)
        return self

    def save(self, path: str):
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "df": self.df}, path)

    @classmethod
    def load(cls, path: str) -> "MedQuADRetriever":
        state = joblib.load(path)
        obj = cls(state["df"])
        obj.vectorizer = state["vectorizer"]
        obj.matrix = state["matrix"]
        return obj

    def search(
        self,
        query: str,
        top_k: int = 5,
        boost_focus_terms: Optional[List[str]] = None,
        boost_amount: float = 0.15,
    ) -> List[RetrievedAnswer]:
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Retriever not fitted/loaded. Call fit() or load() first.")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()

        if boost_focus_terms:
            lowered_terms = [t.lower() for t in boost_focus_terms]
            focus_lower = self.df["focus"].str.lower()
            for term in lowered_terms:
                mask = focus_lower.str.contains(term, regex=False, na=False)
                scores[mask.values] += boost_amount

        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            row = self.df.iloc[idx]
            results.append(
                RetrievedAnswer(
                    question=row["question"],
                    answer=row["answer"],
                    score=float(scores[idx]),
                    focus=row["focus"],
                    source=row["source"],
                    qtype=row["qtype"],
                    url=row["url"],
                    doc_id=str(row["doc_id"]),
                )
            )
        return results


def load_or_build_retriever(qa_df: pd.DataFrame, index_cache_path: str, force_rebuild: bool = False) -> MedQuADRetriever:
    if not force_rebuild and os.path.exists(index_cache_path):
        return MedQuADRetriever.load(index_cache_path)

    retriever = MedQuADRetriever(qa_df).fit()
    retriever.save(index_cache_path)
    return retriever


if __name__ == "__main__":
    from medquad_loader import load_or_build_dataset

    df = load_or_build_dataset("/home/claude/MedQuAD", "medquad_qa.csv")
    retriever = load_or_build_retriever(df, "medquad_index.joblib", force_rebuild=True)

    for q in [
        "What are the symptoms of diabetes?",
        "How is asthma treated?",
        "chest pain and shortness of breath",
        "is leukemia hereditary",
    ]:
        print(f"\n=== {q} ===")
        for r in retriever.search(q, top_k=3):
            print(f"  [{r.score:.3f}] ({r.source} / {r.qtype}) {r.question}")
            print(f"    -> {r.answer[:140]}...")
