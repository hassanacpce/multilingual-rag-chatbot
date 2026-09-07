"""
arxiv_loader.py

Loads and filters the Kaggle arXiv metadata snapshot
(https://www.kaggle.com/datasets/Cornell-University/arxiv) into a working
subset for a specific field (e.g. computer science).

IMPORTANT DATASET NOTES
------------------------
- This is a METADATA-ONLY dataset: id, title, abstract, authors, categories,
  dates. It does NOT include full paper text/PDFs. Summarization and
  explanation in this project therefore operate on abstracts, not full
  papers -- a real scope boundary, not a bug. Fetching full text would
  require a separate pipeline against arXiv's own bulk access or the
  individual paper PDFs, which is out of scope here.
- The file is huge (~1M+ entries, several GB) and is JSON-LINES despite its
  `.json` extension -- one JSON object per line, NOT a single JSON array.
  Loading it with json.load() will exhaust memory; this module streams it
  line by line instead.
- Getting the file: requires a (free) Kaggle account and API credentials
  (kaggle.json). Not something this code can fetch automatically.
      pip install kaggle
      kaggle datasets download -d Cornell-University/arxiv
      unzip arxiv.zip   # -> arxiv-metadata-oai-snapshot.json

Usage
-----
    from arxiv_loader import load_or_build_subset

    df = load_or_build_subset(
        raw_path="arxiv-metadata-oai-snapshot.json",
        cache_path="cs_papers.csv",
        category_prefixes=["cs."],
        max_rows=50000,
    )
"""

import os
import json
from typing import Iterable, List, Optional

import pandas as pd


def _iter_raw_records(raw_path: str) -> Iterable[dict]:
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _matches_category(categories_field: str, prefixes: List[str]) -> bool:
    if not categories_field:
        return False
    cats = categories_field.split()
    return any(cat.startswith(p) for cat in cats for p in prefixes)


def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def parse_arxiv_subset(
    raw_path: str,
    category_prefixes: List[str],
    max_rows: Optional[int] = 50000,
    min_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Streams the raw arXiv metadata file and keeps only records matching the
    given category prefixes (e.g. ["cs.", "stat.ML"]), up to max_rows.

    max_rows exists because the real file has 1M+ entries -- for a chatbot
    knowledge base, a large-but-bounded subset (tens of thousands of papers)
    keeps indexing fast and the app responsive, without meaningfully
    limiting topic coverage within a field like "cs.*".
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"'{raw_path}' not found. Download the Kaggle arXiv dataset first "
            "(requires a free Kaggle account + API credentials):\n"
            "  pip install kaggle\n"
            "  kaggle datasets download -d Cornell-University/arxiv\n"
            "  unzip arxiv.zip"
        )

    rows = []
    for rec in _iter_raw_records(raw_path):
        categories = rec.get("categories", "") or ""
        if not _matches_category(categories, category_prefixes):
            continue

        update_date = rec.get("update_date", "") or ""
        if min_year is not None:
            try:
                year = int(update_date.split("-")[0])
                if year < min_year:
                    continue
            except (ValueError, IndexError):
                pass

        rows.append(
            {
                "id": rec.get("id", ""),
                "title": _clean_text(rec.get("title", "")),
                "abstract": _clean_text(rec.get("abstract", "")),
                "authors": _clean_text(rec.get("authors", "")),
                "categories": categories,
                "primary_category": categories.split()[0] if categories else "",
                "update_date": update_date,
                "arxiv_url": f"https://arxiv.org/abs/{rec.get('id', '')}",
            }
        )

        if max_rows is not None and len(rows) >= max_rows:
            break

    df = pd.DataFrame.from_records(rows)
    df.drop_duplicates(subset=["id"], inplace=True)
    df = df[df["abstract"].str.len() > 0].reset_index(drop=True)
    return df


def load_or_build_subset(
    raw_path: str,
    cache_path: str,
    category_prefixes: List[str],
    max_rows: Optional[int] = 50000,
    min_year: Optional[int] = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if not force_rebuild and os.path.exists(cache_path):
        return pd.read_csv(cache_path, keep_default_na=False)

    df = parse_arxiv_subset(raw_path, category_prefixes, max_rows=max_rows, min_year=min_year)
    df.to_csv(cache_path, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter the arXiv metadata snapshot to a category subset.")
    parser.add_argument("--raw", required=True, help="Path to arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--out", default="cs_papers.csv")
    parser.add_argument("--categories", nargs="+", default=["cs."], help="Category prefixes to keep, e.g. cs. stat.ML")
    parser.add_argument("--max-rows", type=int, default=50000)
    args = parser.parse_args()

    df = parse_arxiv_subset(args.raw, args.categories, max_rows=args.max_rows)
    df.to_csv(args.out, index=False)
    print(f"Kept {len(df):,} papers matching {args.categories} -> {args.out}")
    print(df["primary_category"].value_counts().head(15))
