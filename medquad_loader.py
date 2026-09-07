"""
medquad_loader.py

Parses the MedQuAD dataset (https://github.com/abachaa/MedQuAD) from its raw
XML layout into a flat, cached table of individual question/answer records
that the rest of the app (retriever, NER, Streamlit UI) can work with.

MedQuAD layout (as of the abachaa/MedQuAD repo):

    <root>/
      1_CancerGov_QA/*.xml
      2_GARD_QA/*.xml
      3_GHR_QA/*.xml
      4_MPlus_Health_Topics_QA/*.xml
      5_NIDDK_QA/*.xml
      6_NINDS_QA/*.xml
      7_SeniorHealth_QA/*.xml
      8_NHLBI_QA_XML/*.xml
      9_CDC_QA/*.xml
      10_MPlus_ADAM_QA/*.xml
      11_MPlusDrugs_QA/*.xml
      12_MPlusHerbsSupplements_QA/*.xml

Each XML file looks like:

    <Document id="..." source="CancerGov" url="...">
      <Focus>Adult Acute Lymphoblastic Leukemia</Focus>
      <FocusAnnotations>
        <UMLS><SemanticGroup>Disorders</SemanticGroup>...</UMLS>
      </FocusAnnotations>
      <QAPairs>
        <QAPair pid="1">
          <Question qid="..." qtype="symptoms">What are the symptoms of X ?</Question>
          <Answer>...</Answer>
        </QAPair>
        ...
      </QAPairs>
    </Document>

Some QAPair blocks have an empty/missing <Answer> (a handful of files in the
official repo) -- those are skipped rather than stored as empty answers.

Usage
-----
    python medquad_loader.py --root ./MedQuAD --out medquad_qa.csv

    # or from Python:
    from medquad_loader import load_or_build_dataset
    df = load_or_build_dataset("./MedQuAD", "medquad_qa.csv")
"""

import os
import glob
import argparse
import xml.etree.ElementTree as ET

import pandas as pd

# Maps the MedQuAD source folder prefix -> a coarse entity type used later
# by medical_ner.py to build lexicons (disease/condition vs. drug vs. herb).
SOURCE_ENTITY_TYPE = {
    "CancerGov": "disease",
    "GARD": "disease",
    "GHR": "disease",
    "MPlusHealthTopics": "disease",
    "NIDDK": "disease",
    "NINDS": "disease",
    "NIHSeniorHealth": "disease",
    "NHLBI": "disease",
    "CDC": "disease",
    "ADAM": "disease",              # general health-encyclopedia topics
    "MPlusDrugs": "medication",
    "MPlusHerbsSupplements": "herb_supplement",
}


def _text_or_empty(elem) -> str:
    if elem is None or elem.text is None:
        return ""
    return " ".join(elem.text.split())


def parse_xml_file(path: str) -> list:
    """Parse a single MedQuAD XML file into a list of QA record dicts."""
    records = []
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return records

    root = tree.getroot()
    doc_id = root.get("id", "")
    source = root.get("source", "")
    doc_url = root.get("url", "")
    focus = _text_or_empty(root.find("Focus"))

    semantic_group = ""
    sg_elem = root.find("./FocusAnnotations/UMLS/SemanticGroup")
    if sg_elem is not None:
        semantic_group = _text_or_empty(sg_elem)

    entity_type = SOURCE_ENTITY_TYPE.get(source, "other")

    qa_pairs = root.find("QAPairs")
    if qa_pairs is None:
        return records

    for qa in qa_pairs.findall("QAPair"):
        q_elem = qa.find("Question")
        a_elem = qa.find("Answer")
        question = _text_or_empty(q_elem)
        answer = _text_or_empty(a_elem)

        if not question or not answer:
            continue

        records.append(
            {
                "doc_id": doc_id,
                "qid": q_elem.get("qid", "") if q_elem is not None else "",
                "qtype": q_elem.get("qtype", "") if q_elem is not None else "",
                "source": source,
                "entity_type": entity_type,
                "focus": focus,
                "semantic_group": semantic_group,
                "question": question,
                "answer": answer,
                "url": doc_url,
            }
        )

    return records


def parse_medquad(root_dir: str) -> pd.DataFrame:
    """Walk every XML file under root_dir and return one big DataFrame."""
    xml_paths = glob.glob(os.path.join(root_dir, "**", "*.xml"), recursive=True)
    if not xml_paths:
        raise FileNotFoundError(
            f"No .xml files found under '{root_dir}'. "
            "Clone the dataset first: git clone https://github.com/abachaa/MedQuAD.git"
        )

    all_records = []
    for path in xml_paths:
        all_records.extend(parse_xml_file(path))

    df = pd.DataFrame.from_records(all_records)
    df.drop_duplicates(subset=["qid", "question", "answer"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def parse_focus_terms(root_dir: str) -> pd.DataFrame:
    """
    Collects (focus_term, entity_type, source) for every MedQuAD document,
    INCLUDING the three sources whose answers were stripped for copyright
    reasons (ADAM, MPlusDrugs, MPlusHerbsSupplements). Those still carry
    real entity names (e.g. "Abacavir", "Activated Charcoal") that are
    useful for entity recognition even though there's no retrievable
    answer text for them in this dataset.
    """
    xml_paths = glob.glob(os.path.join(root_dir, "**", "*.xml"), recursive=True)
    rows = []
    for path in xml_paths:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        focus = _text_or_empty(root.find("Focus"))
        if not focus:
            continue
        source = root.get("source", "")
        rows.append({"focus": focus, "entity_type": SOURCE_ENTITY_TYPE.get(source, "other"), "source": source})

    df = pd.DataFrame.from_records(rows)
    df.drop_duplicates(subset=["focus", "entity_type"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_or_build_dataset(root_dir: str, cache_path: str, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Returns the parsed MedQuAD dataframe, using a CSV cache when available so
    the ~47k QA pairs don't get re-parsed from XML on every app restart.
    """
    if not force_rebuild and os.path.exists(cache_path):
        return pd.read_csv(cache_path, keep_default_na=False)

    df = parse_medquad(root_dir)
    df.to_csv(cache_path, index=False)
    return df


def load_or_build_focus_terms(root_dir: str, cache_path: str, force_rebuild: bool = False) -> pd.DataFrame:
    if not force_rebuild and os.path.exists(cache_path):
        return pd.read_csv(cache_path, keep_default_na=False)

    df = parse_focus_terms(root_dir)
    df.to_csv(cache_path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the MedQuAD XML dataset into a flat CSV.")
    parser.add_argument("--root", required=True, help="Path to the cloned MedQuAD repo root.")
    parser.add_argument("--out", default="medquad_qa.csv", help="Output CSV cache path for QA pairs.")
    parser.add_argument("--focus-out", default="medquad_focus_terms.csv", help="Output CSV cache path for entity focus terms.")
    args = parser.parse_args()

    df = parse_medquad(args.root)
    df.to_csv(args.out, index=False)
    print(f"Parsed {len(df):,} QA pairs from {df['doc_id'].nunique():,} documents -> {args.out}")
    print(df["source"].value_counts())
    print(df["qtype"].value_counts().head(10))

    focus_df = parse_focus_terms(args.root)
    focus_df.to_csv(args.focus_out, index=False)
    print(f"\nParsed {len(focus_df):,} unique entity focus terms -> {args.focus_out}")
    print(focus_df["entity_type"].value_counts())
