"""
knowledge_base_manager.py

Adds dynamic, incremental updates to the FAISS knowledge base on top of the
existing langchain_helper.py setup, without changing how create_vector_db()
or get_qa_chain() work.

How it works
------------
- sources_config.json  -> list of knowledge sources to pull from + schedule
- ingested_hashes.json -> content hashes already embedded (prevents re-adding
                           the same text every run)
- update_log.json      -> history of update runs (timestamp, per-source
                           counts, errors)

Supported source types: "csv", "url", "text", "pdf", "directory".

Usage
-----
    from knowledge_base_manager import add_source, update_vector_store

    add_source({"id": "faq_page", "type": "url", "url": "https://example.com/faq"})
    summary = update_vector_store()   # pulls new docs, embeds only new ones
"""

import os
import json
import glob
import hashlib
from datetime import datetime

from langchain_community.document_loaders import (
    CSVLoader,
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_community.vectorstores import FAISS

from langchain_helper import embeddings, vectordb_file_path

CONFIG_PATH = "sources_config.json"
HASHES_PATH = "ingested_hashes.json"
LOG_PATH = "update_log.json"

DEFAULT_CONFIG = {
    "sources": [],
    "schedule": {"enabled": False, "interval_minutes": 60},
}


# -----------------------------
# Config / state persistence
# -----------------------------
def load_config():
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_hashes():
    if not os.path.exists(HASHES_PATH):
        return set()
    with open(HASHES_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f))


def save_hashes(hashes):
    with open(HASHES_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(hashes), f, indent=2)


def append_log(entry):
    log = get_log()
    log.append(entry)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log[-200:], f, indent=2)  # keep last 200 runs


def get_log():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


# -----------------------------
# Source management
# -----------------------------
def add_source(source: dict):
    """
    source examples:
      {"id": "faq_page", "type": "url", "url": "https://example.com/faq"}
      {"id": "extra_csv", "type": "csv", "path": "more_data.csv", "encoding": "utf-8"}
      {"id": "notes", "type": "text", "path": "notes.txt"}
      {"id": "manual", "type": "pdf", "path": "manual.pdf"}
      {"id": "docs", "type": "directory", "path": "knowledge_docs", "pattern": "*.txt"}
    """
    if "id" not in source or "type" not in source:
        raise ValueError("Source must include an 'id' and a 'type'.")

    config = load_config()
    if any(s["id"] == source["id"] for s in config["sources"]):
        raise ValueError(f"Source id '{source['id']}' already exists.")

    config["sources"].append(source)
    save_config(config)
    return config


def remove_source(source_id: str):
    config = load_config()
    config["sources"] = [s for s in config["sources"] if s["id"] != source_id]
    save_config(config)
    return config


def set_schedule(enabled: bool, interval_minutes: int):
    config = load_config()
    config["schedule"] = {"enabled": enabled, "interval_minutes": interval_minutes}
    save_config(config)
    return config


# -----------------------------
# Loading documents per source type
# -----------------------------
def _load_from_source(source: dict):
    stype = source["type"]

    if stype == "csv":
        loader = CSVLoader(
            file_path=source["path"],
            encoding=source.get("encoding", "utf-8"),
        )
        return loader.load()

    if stype == "url":
        loader = WebBaseLoader(source["url"])
        return loader.load()

    if stype == "text":
        loader = TextLoader(source["path"], encoding=source.get("encoding", "utf-8"))
        return loader.load()

    if stype == "pdf":
        loader = PyPDFLoader(source["path"])
        return loader.load()

    if stype == "directory":
        pattern = source.get("pattern", "*.txt")
        docs = []
        for path in glob.glob(os.path.join(source["path"], pattern)):
            docs.extend(TextLoader(path, encoding="utf-8").load())
        return docs

    raise ValueError(f"Unknown source type: {stype}")


def _get_vector_db():
    if os.path.exists(vectordb_file_path):
        return FAISS.load_local(
            vectordb_file_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


# -----------------------------
# Core update routine
# -----------------------------
def update_vector_store(source_ids=None):
    """
    Pulls fresh documents from configured sources, skips anything already
    embedded (by content hash of the chunk text), and incrementally adds
    only the new documents to the FAISS index. Safe to call repeatedly or
    on a schedule -- it never re-embeds unchanged content and never
    rebuilds the index from scratch.

    source_ids: optional list to restrict the run to specific source ids.
    Returns a summary dict describing what was added.
    """
    config = load_config()
    sources = config["sources"]
    if source_ids:
        sources = [s for s in sources if s["id"] in source_ids]

    seen_hashes = load_hashes()
    vectordb = _get_vector_db()

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sources": {},
        "total_added": 0,
    }

    for source in sources:
        added = 0
        error = None
        try:
            docs = _load_from_source(source)
            new_docs = []
            for doc in docs:
                h = content_hash(doc.page_content)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    new_docs.append(doc)

            if new_docs:
                if vectordb is None:
                    vectordb = FAISS.from_documents(new_docs, embeddings)
                else:
                    vectordb.add_documents(new_docs)
                added = len(new_docs)

        except Exception as e:
            error = str(e)

        summary["sources"][source["id"]] = {"added": added, "error": error}
        summary["total_added"] += added

    if vectordb is not None and summary["total_added"] > 0:
        vectordb.save_local(vectordb_file_path)
        save_hashes(seen_hashes)

    append_log(summary)
    return summary
