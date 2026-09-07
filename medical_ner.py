"""
medical_ner.py

Lightweight, dictionary-based medical entity recognition. Not a trained NLP
model -- a fast lexicon matcher (via flashtext's KeywordProcessor, which does
single-pass multi-pattern matching in O(len(text)) regardless of lexicon
size). This is intentionally "basic" as scoped: it recognizes entities the
chatbot already has knowledge about, which is exactly what's useful for
routing/boosting retrieval, rather than trying to be a general clinical NER
system.

Three entity types, two different sourcing strategies:

  - DISEASE / MEDICATION / HERB_SUPPLEMENT
      Built directly from MedQuAD's own `Focus` field across all 10,550
      documents (see medquad_loader.parse_focus_terms). This means the
      lexicon and the retrieval corpus are grounded in the same vocabulary
      the dataset actually uses.

  - SYMPTOM
      MedQuAD doesn't tag symptom entities directly (symptoms appear inside
      free-text "symptoms"-qtype answers, not as a Focus). A curated list of
      ~120 common symptom/finding terms is used instead. This is the
      "basic" part of "basic medical entity recognition" -- good enough to
      flag when a user is describing symptoms, not a substitute for a real
      clinical NER model.

Matching is case-insensitive, whole-word/phrase, and longest-match-wins per
span (flashtext handles overlap resolution internally).
"""

import os
from dataclasses import dataclass, field
from typing import List

import pandas as pd
from flashtext import KeywordProcessor

CURATED_SYMPTOMS = [
    "fever", "high temperature", "chills", "fatigue", "tiredness", "weakness",
    "headache", "migraine", "dizziness", "lightheadedness", "fainting",
    "nausea", "vomiting", "diarrhea", "constipation", "abdominal pain",
    "stomach pain", "stomach ache", "bloating", "heartburn", "indigestion",
    "loss of appetite", "weight loss", "weight gain",
    "cough", "dry cough", "shortness of breath", "difficulty breathing",
    "wheezing", "chest pain", "chest tightness", "palpitations",
    "rapid heartbeat", "irregular heartbeat",
    "sore throat", "runny nose", "stuffy nose", "congestion", "sneezing",
    "rash", "itching", "hives", "swelling", "redness", "bruising",
    "joint pain", "muscle pain", "muscle weakness", "back pain", "neck pain",
    "stiffness", "cramping", "cramps", "numbness", "tingling",
    "blurred vision", "vision loss", "double vision", "eye pain",
    "hearing loss", "ringing in the ears", "tinnitus",
    "confusion", "memory loss", "difficulty concentrating",
    "anxiety", "depression", "irritability", "mood swings",
    "insomnia", "difficulty sleeping", "excessive sleepiness",
    "seizures", "tremor", "muscle spasms", "loss of balance",
    "difficulty walking", "difficulty swallowing", "slurred speech",
    "night sweats", "excessive sweating", "dry skin", "hair loss",
    "jaundice", "yellowing of the skin", "yellowing of the eyes",
    "dark urine", "blood in urine", "blood in stool", "frequent urination",
    "painful urination", "excessive thirst", "excessive hunger",
    "swollen lymph nodes", "swollen glands", "enlarged lymph nodes",
    "difficulty breathing", "shortness of breath on exertion",
    "low blood pressure", "high blood pressure", "irregular periods",
    "cold hands and feet", "poor wound healing", "easy bruising",
    "easy bleeding", "recurrent infections", "delayed growth",
    "developmental delay", "failure to thrive", "poor coordination",
    "muscle atrophy", "loss of consciousness", "abdominal swelling",
    "nosebleeds", "sensitivity to light", "sensitivity to sound",
    "dry mouth", "mouth sores", "difficulty chewing", "loss of taste",
    "loss of smell", "vertigo", "unsteady gait",
]


@dataclass
class EntityMatch:
    text: str            # the exact surface text matched
    entity_type: str      # DISEASE | MEDICATION | HERB_SUPPLEMENT | SYMPTOM
    canonical: str = ""   # canonical/normalized form (from the lexicon)


@dataclass
class NERResult:
    entities: List[EntityMatch] = field(default_factory=list)

    def by_type(self, entity_type: str) -> List[str]:
        return [e.text for e in self.entities if e.entity_type == entity_type]

    def is_empty(self) -> bool:
        return len(self.entities) == 0


class MedicalEntityRecognizer:
    ENTITY_TYPE_LABELS = {
        "disease": "DISEASE",
        "medication": "MEDICATION",
        "herb_supplement": "HERB_SUPPLEMENT",
    }

    def __init__(self, focus_terms_df: pd.DataFrame, min_term_length: int = 3):
        self._processors = {}

        for raw_type, label in self.ENTITY_TYPE_LABELS.items():
            kp = KeywordProcessor(case_sensitive=False)
            terms = focus_terms_df.loc[focus_terms_df.entity_type == raw_type, "focus"]
            for term in terms:
                term = str(term).strip()
                if len(term) < min_term_length:
                    continue
                kp.add_keyword(term, term)  # clean_name = canonical form
            self._processors[label] = kp

        symptom_kp = KeywordProcessor(case_sensitive=False)
        for term in CURATED_SYMPTOMS:
            symptom_kp.add_keyword(term, term)
        self._processors["SYMPTOM"] = symptom_kp

    def recognize(self, text: str) -> NERResult:
        result = NERResult()
        if not text or not text.strip():
            return result

        for label, kp in self._processors.items():
            for canonical in kp.extract_keywords(text):
                result.entities.append(EntityMatch(text=canonical, entity_type=label, canonical=canonical))

        # De-duplicate (e.g. "Diabetes" matched both as substring of two
        # near-identical lexicon entries) while preserving first-seen order.
        seen = set()
        deduped = []
        for e in result.entities:
            key = (e.entity_type, e.canonical.lower())
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        result.entities = deduped
        return result


def build_recognizer(focus_terms_csv: str) -> MedicalEntityRecognizer:
    if not os.path.exists(focus_terms_csv):
        raise FileNotFoundError(
            f"'{focus_terms_csv}' not found. Run medquad_loader.py first to build it."
        )
    df = pd.read_csv(focus_terms_csv, keep_default_na=False)
    return MedicalEntityRecognizer(df)


if __name__ == "__main__":
    recognizer = build_recognizer("medquad_focus_terms.csv")
    samples = [
        "I have a bad headache and fever, could it be the flu?",
        "What are the side effects of Trazodone?",
        "My grandmother was just diagnosed with Adult Acute Lymphoblastic Leukemia, what's the treatment?",
        "Is Activated Charcoal safe to take with ibuprofen?",
    ]
    for s in samples:
        res = recognizer.recognize(s)
        print(f"\n{s}")
        for e in res.entities:
            print(f"  [{e.entity_type}] {e.text}")
