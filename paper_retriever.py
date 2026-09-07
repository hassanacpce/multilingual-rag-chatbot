"""
paper_retriever.py

Semantic search over the paper corpus using Latent Semantic Analysis (LSA):
TF-IDF vectorization followed by TruncatedSVD, which projects each paper
into a lower-dimensional "concept space" where semantically related papers
land close together even without exact keyword overlap (the classic
strength of LSA over raw TF-IDF matching).

This does double duty for the assignment's two related requirements:
  - "paper searching": cosine similarity search in the LSA concept space.
  - "concept visualization": the first 2 LSA components ARE literally named
    "concepts" in LSA terminology, so projecting them directly gives a
    genuine 2D concept-space scatter plot, not a cosmetic add-on -- papers
    that cluster together in that plot are ones LSA judged conceptually
    related.

Deliberately dependency-light (scikit-learn only, no model downloads),
consistent with the same tradeoff made in the MedQuAD project's retriever.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedPaper:
    id: str
    title: str
    abstract: str
    authors: str
    primary_category: str
    arxiv_url: str
    score: float


class PaperRetriever:
    def __init__(self, papers_df: pd.DataFrame, n_components: int = 100):
        self.df = papers_df.reset_index(drop=True)
        self.n_components = min(n_components, max(2, len(self.df) - 1))
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.doc_vectors: Optional[np.ndarray] = None  # LSA-space vectors, one per paper

    def fit(self):
        corpus = (self.df["title"] + " " + self.df["title"] + " " + self.df["abstract"]).tolist()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.9, min_df=1)
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.doc_vectors = self.svd.fit_transform(tfidf_matrix)
        return self

    def save(self, path: str):
        joblib.dump(
            {"vectorizer": self.vectorizer, "svd": self.svd, "doc_vectors": self.doc_vectors, "df": self.df},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "PaperRetriever":
        state = joblib.load(path)
        obj = cls(state["df"])
        obj.vectorizer = state["vectorizer"]
        obj.svd = state["svd"]
        obj.doc_vectors = state["doc_vectors"]
        return obj

    def search(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[RetrievedPaper]:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("Retriever not fitted/loaded. Call fit() or load() first.")

        query_tfidf = self.vectorizer.transform([query])
        query_vec = self.svd.transform(query_tfidf)
        scores = cosine_similarity(query_vec, self.doc_vectors).ravel()

        candidate_idx = np.argsort(scores)[::-1]
        results = []
        for idx in candidate_idx:
            if len(results) >= top_k:
                break
            if scores[idx] <= 0:
                continue
            row = self.df.iloc[idx]
            if category_filter and not row["categories"].split()[0:1] == [category_filter] and category_filter not in row["categories"].split():
                continue
            results.append(
                RetrievedPaper(
                    id=row["id"],
                    title=row["title"],
                    abstract=row["abstract"],
                    authors=row["authors"],
                    primary_category=row["primary_category"],
                    arxiv_url=row["arxiv_url"],
                    score=float(scores[idx]),
                )
            )
        return results

    def concept_coordinates_2d(self) -> pd.DataFrame:
        """
        Returns a DataFrame with x, y (the first two LSA components -- the
        two dominant 'concepts' the corpus decomposes into), plus paper
        metadata, ready for a scatter-plot visualization colored by category.
        """
        if self.doc_vectors is None:
            raise RuntimeError("Retriever not fitted/loaded.")
        coords = self.doc_vectors[:, :2]
        out = self.df[["id", "title", "primary_category"]].copy()
        out["x"] = coords[:, 0]
        out["y"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
        return out


def load_or_build_retriever(papers_df: pd.DataFrame, index_cache_path: str, force_rebuild: bool = False) -> PaperRetriever:
    if not force_rebuild and os.path.exists(index_cache_path):
        return PaperRetriever.load(index_cache_path)

    retriever = PaperRetriever(papers_df).fit()
    retriever.save(index_cache_path)
    return retriever


if __name__ == "__main__":
    df = pd.read_csv("cs_papers.csv", keep_default_na=False)
    retriever = load_or_build_retriever(df, "cs_papers_index.joblib", force_rebuild=True)

    for q in ["fine-tuning large language models", "graph neural networks for chemistry", "attention mechanism summarization"]:
        print(f"\n=== {q} ===")
        for r in retriever.search(q, top_k=3):
            print(f"  [{r.score:.3f}] ({r.primary_category}) {r.title}")

    print("\nConcept coordinates (first 5):")
    print(retriever.concept_coordinates_2d().head())
