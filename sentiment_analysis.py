"""
sentiment_analysis.py

Detects sentiment in user messages using VADER (Valence Aware Dictionary and
sEntiment Reasoner) -- a lexicon- and rule-based sentiment analyzer tuned
specifically for short, informal text (social media, chat messages), which
is exactly the register customer-service/chat input tends to be in.

Deliberately not LLM-based, for the same reasons NER/retrieval in this
project are deterministic where possible: VADER is fast (no network call,
no added latency or token cost on top of the existing multi-call reasoning
pipeline), fully offline, deterministic (same input -> same output, so it's
actually testable), and free of the rate-limit pressure already documented
as a recurring issue with Groq's free tier elsewhere in this project.

Design: sentiment doesn't just classify positive/negative/neutral -- it
produces TONE GUIDANCE text that gets injected into the draft-generation
prompt, so the LLM adjusts HOW it frames a response (leading with empathy
for a frustrated user) without ever being told to alter WHAT facts it
states. This keeps sentiment-driven tone changes orthogonal to the
retrieval-grounding/validation system elsewhere in the pipeline -- tone is
a framing instruction, not a claim that needs to be "supported by context".
"""

from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Standard VADER convention for compound-score thresholds, plus a second
# tier on the negative side -- "appropriateness of response to different
# sentiments" (per the assignment's own evaluation criteria) means a mildly
# annoyed message and a genuinely angry one shouldn't get identical tone
# handling.
STRONGLY_NEGATIVE_THRESHOLD = -0.5
NEGATIVE_THRESHOLD = -0.05
POSITIVE_THRESHOLD = 0.05


@dataclass
class SentimentResult:
    label: str          # "strongly_negative" | "negative" | "neutral" | "positive"
    compound: float      # VADER compound score, range [-1, 1]
    pos: float
    neu: float
    neg: float

    @property
    def is_negative(self) -> bool:
        return self.label in ("negative", "strongly_negative")


def analyze_sentiment(text: str) -> SentimentResult:
    if not text or not text.strip():
        return SentimentResult(label="neutral", compound=0.0, pos=0.0, neu=1.0, neg=0.0)

    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound <= STRONGLY_NEGATIVE_THRESHOLD:
        label = "strongly_negative"
    elif compound <= NEGATIVE_THRESHOLD:
        label = "negative"
    elif compound >= POSITIVE_THRESHOLD:
        label = "positive"
    else:
        label = "neutral"

    return SentimentResult(label=label, compound=compound, pos=scores["pos"], neu=scores["neu"], neg=scores["neg"])


# Tone guidance injected into the draft-generation prompt. Written as
# instructions about FRAMING, explicitly not about facts, so the validator
# (which checks factual claims against retrieved context) has no reason to
# reject an empathetic opening line as "unsupported".
TONE_GUIDANCE = {
    "strongly_negative": (
        "The user's message expresses strong frustration or distress. Before addressing "
        "their question, open with a brief, genuine acknowledgment of how they're feeling "
        "and, if appropriate, a short apology for the inconvenience. Do not be dismissive "
        "or jump straight into information without first acknowledging the emotion. Keep "
        "the acknowledgment brief (one sentence) -- don't over-apologize or be performative."
    ),
    "negative": (
        "The user's message expresses mild frustration, disappointment, or annoyance. "
        "Acknowledge this briefly and warmly (a short empathetic phrase) before answering, "
        "without being effusive about it."
    ),
    "positive": (
        "The user's message expresses positive sentiment. You may briefly match that warmth "
        "in your tone, but stay focused and helpful rather than overdoing enthusiasm."
    ),
    "neutral": "",
}


def get_tone_guidance(sentiment: SentimentResult) -> str:
    return TONE_GUIDANCE.get(sentiment.label, "")


if __name__ == "__main__":
    samples = [
        "This is absolutely ridiculous, I've been waiting for a week and nobody has helped me!",
        "I'm a bit confused about how the billing works, could you explain?",
        "Thank you so much, this is exactly what I needed! You're amazing.",
        "What are your business hours?",
        "I am extremely frustrated and disappointed with this terrible service.",
        "great, thanks",
        "this is fine i guess",
    ]
    for s in samples:
        result = analyze_sentiment(s)
        print(f"[{result.label:17s}] compound={result.compound:+.3f}  {s}")
