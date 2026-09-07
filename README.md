# Multilingual Support Integration

Adds automatic language detection, cross-lingual retrieval, and in-language
responses across 5 languages: **English, Hindi, Marathi, Spanish, French**.
Two new files, two changed files — everything else in your project (all
three knowledge bases, sentiment analysis) is untouched and works alongside
this.

## What's new

| File | Status |
|---|---|
| `language_detection.py` | **New.** Context-aware language detection + mixed-language (code-switch) flagging. |
| `eval_language_detection.py` | **New.** Hand-labeled accuracy test — run it to reproduce the numbers below. |
| `reasoning_engine.py` | **Changed.** Language detected every turn with conversation-level "sticky" context; planning step also produces an English translation of the query for retrieval; drafts generated in the user's language; fail-closed fallback messages localized into all 5 languages. |
| `main.py` | **Changed.** Language badge next to each user message (flag + name, marked when resolved via context or detected as mixed); sidebar note on supported languages. |

## Install

```bash
pip install -r requirements.txt   # adds langdetect (on top of vaderSentiment from the sentiment integration)
```

Drop these 4 files into your project folder, overwriting the existing
`reasoning_engine.py` and `main.py`. Restart Streamlit.

## The core design decision: context-aware fallback, not just per-message detection

Raw per-message language detection is genuinely unreliable on short text —
tested this directly rather than assuming it works:

```bash
python eval_language_detection.py
```

**Result: raw per-message detection is 75% accurate on a realistic labeled
set (16 messages); with conversation-context fallback, 100%.**

Why raw detection fails on short messages:
- Single words ("thanks", "merci", "gracias") carry too little statistical
  signal for character-n-gram detectors.
- Some words are **genuinely ambiguous across languages** — "धन्यवाद" ("thank
  you") is spelled identically in Hindi, Marathi, *and* Nepali. No detector
  can resolve this from the word alone; it requires knowing what language
  the conversation has already been happening in.

The fix implemented here: `ConversationMemory` tracks an established
"sticky" language across turns. Short or low-confidence messages fall back
to that established language instead of re-guessing blind; only a genuinely
confident detection (a full sentence) updates the sticky language, so a
real language switch is still correctly picked up. This is what directly
implements the assignment's "resolve ambiguous queries across languages"
and "maintain context...throughout language switches" — not a separate
bolt-on feature, but the actual mechanism making detection accurate at all.

Verified through the real `answer()` orchestration (not just the isolated
module) across a 3-turn conversation:
1. Full Hindi sentence → establishes Hindi as the conversation language.
2. Ambiguous "धन्यवाद" → correctly resolved to Hindi *via context*, not a
   fresh guess (confirmed `used_context_fallback: True`).
3. Full French sentence → correctly detected as a genuine language switch,
   conversation language updates to French.

## A known, honestly-documented gap: Hinglish / romanized code-switching

Found while testing mixed-language detection: a message like *"यह product
kharab hai, can you help mujhe?"* — genuine Hindi/English code-switching,
extremely common in real Hindi/Marathi chat — was **misdetected as pure
English**. The reason: `kharab hai` ("is bad") is Hindi, but spelled in
**Latin script**, not Devanagari. `langdetect` is a character-script
statistical model; it has no way to distinguish romanized Hindi words from
English ones, since there's no script-level signal to key off. Native-script
code-switching (Devanagari Hindi mixed with English) *does* get correctly
flagged as mixed — verified separately.

This is a known, published limitation of n-gram-based language detectors,
not a bug specific to this integration. Properly handling Hinglish/romanized
code-mixed text would require either a dedicated code-mixed language model
or a lexicon-based approach — a meaningfully larger undertaking than fits
here. Mitigation already in place: the LLM itself (multilingual open-weight
model) is instructed to interpret the message "holistically" regardless of
what the deterministic detector concluded, since gpt-oss-120b's own
multilingual training makes it considerably better at understanding
code-mixed text than a lightweight statistical detector — the detector's
role is a fast, testable UI signal and a routing hint, not the sole
authority on comprehension.

## How cross-lingual retrieval actually works

All three knowledge bases (company KB, MedQuAD, arXiv) are **English-only**
TF-IDF/LSA indices. Searching them with a raw non-English query wouldn't
just underperform — it would return essentially nothing, since there's no
vocabulary overlap at all. The fix: the planning step (`plan_response`,
already an existing LLM call — no extra round-trip added) now also produces
an `english_query` field: an English translation of the user's underlying
question, used only internally for retrieval. The user never sees this
translation; they see a response generated and validated in their own
language, grounded in what that English query retrieved.

## What's genuinely tested vs. what isn't

Tested for real in this session:
- Language detection accuracy: 75% raw / **100% with context fallback**
  on a labeled set, with two honestly documented findings (short-word
  ambiguity, Hinglish/romanized code-switching).
- Mixed-language (native-script code-switch) detection — confirmed
  correctly flagged for genuine Devanagari+Latin mixing.
- The full `answer()` orchestration with a stubbed LLM across a real
  multi-turn conversation — confirmed sticky language context, correct
  language-switch detection, and correct per-language selection of the
  localized fallback message (all 5 languages verified individually).
- Full Streamlit app boot (HTTP 200) with language badges, no import/
  runtime errors, working alongside the existing sentiment/KB features.

**Not tested** (same limitation as every LLM-dependent piece throughout this
project — no network access to Groq's API from this sandbox):
- Actual translation quality of the `english_query` field.
- Whether generated responses are fluent, natural-sounding text in
  Hindi/Marathi/Spanish/French (vs. grammatically correct but stilted).
- Cross-lingual validation quality — `validate_answer` has to compare a
  non-English draft's factual claims against English-language retrieved
  context; this relies entirely on the LLM's own cross-lingual reasoning
  and could not be verified end-to-end here.

Worth deliberately testing with real Groq calls in all 5 languages once
running live — this is the one layer of the whole project that's
architecturally sound and unit-tested at every seam, but whose actual
output quality has never been observed.
