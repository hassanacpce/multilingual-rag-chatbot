"""
language_detection.py

Detects the language of user messages, with two design decisions driven by
real, measured limitations of per-message language detection (see
eval_language_detection.py for the numbers):

1. CONTEXT-AWARE FALLBACK. Raw per-message detection is unreliable on short
   text -- "thanks", "merci", "gracias" are all a handful of characters with
   thin statistical signal, and some short words are genuinely ambiguous
   across languages (e.g. "धन्यवाद" is the same spelling for "thank you" in
   Hindi, Marathi, AND Nepali -- no detector can resolve that from the word
   alone). Rather than trusting a shaky single-message guess, short/
   low-confidence messages fall back to the conversation's already-
   established language. This is also just correct conversational behavior:
   if someone has been chatting in Hindi for five turns, a two-word reply
   should be interpreted as Hindi, not re-guessed from scratch. This is
   what actually implements "context retention...throughout language
   switches" and "resolve ambiguous queries across languages" from the
   assignment, not a side effect.

2. MIXED-LANGUAGE (code-switch) DETECTION. Splits the message into clauses
   and detects each independently; if more than one language appears with
   real signal, the message is flagged "mixed" with the set of languages
   involved, while still picking a primary language for response purposes.

Deliberately offline/deterministic (langdetect, not an LLM call) for the
same reason sentiment detection is -- zero added latency/token cost on a
pipeline that already makes several sequential LLM calls per turn, and it's
actually testable.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # deterministic results across runs

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "es": "Spanish",
    "fr": "French",
}

# Below this word count, single-message detection is unreliable enough that
# falling back to conversation context is preferred over trusting the guess
# (see eval_language_detection.py for the accuracy numbers behind this).
SHORT_MESSAGE_WORD_THRESHOLD = 4
# Below this confidence, same treatment even for longer messages.
LOW_CONFIDENCE_THRESHOLD = 0.55

_CLAUSE_SPLIT_RE = re.compile(r"[.!?,;]|\s+(?:and|but|aur|आणि|y|et)\s+", re.IGNORECASE)


@dataclass
class LanguageResult:
    code: str                  # ISO code, e.g. "hi" -- may be outside SUPPORTED_LANGUAGES
    name: str                  # human-readable, "Hindi" / "Unknown (xx)" for unsupported codes
    confidence: float
    is_mixed: bool = False
    detected_languages: List[str] = field(default_factory=list)  # all codes seen, for mixed messages
    used_context_fallback: bool = False  # true if this result came from conversation history, not detection

    @property
    def is_supported(self) -> bool:
        return self.code in SUPPORTED_LANGUAGES


def _language_name(code: str) -> str:
    return SUPPORTED_LANGUAGES.get(code, f"Unknown ({code})")


def _detect_raw(text: str) -> List[tuple]:
    """Returns [(code, confidence), ...] sorted by confidence, or [] if undetectable."""
    try:
        return [(r.lang, r.prob) for r in detect_langs(text)]
    except LangDetectException:
        return []


def _detect_mixed(text: str) -> Optional[List[str]]:
    """
    Splits text into clauses and detects each independently. Returns the
    sorted list of distinct languages found (len > 1) if the message looks
    genuinely mixed, else None. Short clauses are skipped -- they'd just
    reintroduce the same short-text unreliability this module is designed
    to avoid.
    """
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]
    clauses = [c for c in clauses if len(c.split()) >= SHORT_MESSAGE_WORD_THRESHOLD]
    if len(clauses) < 2:
        return None

    found = set()
    for clause in clauses:
        raw = _detect_raw(clause)
        if raw and raw[0][1] >= LOW_CONFIDENCE_THRESHOLD:
            found.add(raw[0][0])

    return sorted(found) if len(found) > 1 else None


def detect_language(text: str, conversation_language: Optional[str] = None) -> LanguageResult:
    """
    conversation_language: the established language code from prior turns
    (e.g. tracked by the caller across a ConversationMemory), used as the
    fallback for short/ambiguous messages. Pass None for the first turn.
    """
    text = (text or "").strip()
    if not text:
        code = conversation_language or "en"
        return LanguageResult(code=code, name=_language_name(code), confidence=0.0, used_context_fallback=True)

    mixed_langs = _detect_mixed(text)
    raw = _detect_raw(text)
    word_count = len(text.split())

    if not raw:
        code = conversation_language or "en"
        return LanguageResult(code=code, name=_language_name(code), confidence=0.0, used_context_fallback=True)

    top_code, top_conf = raw[0]

    is_short = word_count < SHORT_MESSAGE_WORD_THRESHOLD
    is_low_confidence = top_conf < LOW_CONFIDENCE_THRESHOLD

    if (is_short or is_low_confidence) and conversation_language:
        return LanguageResult(
            code=conversation_language,
            name=_language_name(conversation_language),
            confidence=top_conf,
            used_context_fallback=True,
            detected_languages=[top_code],
        )

    return LanguageResult(
        code=top_code,
        name=_language_name(top_code),
        confidence=top_conf,
        is_mixed=mixed_langs is not None,
        detected_languages=mixed_langs or [top_code],
        used_context_fallback=False,
    )


if __name__ == "__main__":
    samples = [
        ("What are your business hours?", None),
        ("thanks", "en"),
        ("Cuáles son los horarios de atención?", None),
        ("यह product kharab hai, can you help mujhe?", None),  # genuine Hindi/English code-switch
        ("धन्यवाद", "hi"),
        ("धन्यवाद", "mr"),
        ("धन्यवाद", None),  # no context at all -- genuinely ambiguous
    ]
    for text, ctx in samples:
        result = detect_language(text, conversation_language=ctx)
        print(
            f"[{result.code:3s}] conf={result.confidence:.2f} "
            f"mixed={result.is_mixed} fallback={result.used_context_fallback}  "
            f"ctx={ctx!s:6s}  '{text}'"
        )
