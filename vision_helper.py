"""
vision_helper.py

Turns an image into *structured evidence* rather than a free-form caption,
so downstream reasoning/validation steps can decide how much to trust it.

Design choice: the vision model is instructed to separate what it can
literally read/see (high confidence) from what it is inferring/guessing
(low confidence), and to say when the image is unclear. That separation is
what lets reasoning_engine.py flag ambiguity instead of confidently
answering from a guess.
"""

import os
import io
import json
import base64

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load .env directly here rather than relying on langchain_helper.py having
# been imported first -- otherwise running this file standalone, or importing
# it before langchain_helper.py, fails with a missing GROQ_API_KEY error even
# though .env has it set.
load_dotenv()

# Groq's current multimodal preview model (see console.groq.com/docs/deprecations --
# llama-3.2/4 vision previews have been retired in favor of this one as of mid-2026).
# Kept as a separate model/instance from the text-only `llm` in langchain_helper.py
# so a vision-model deprecation doesn't break text-only chat, and vice versa.
VISION_MODEL_NAME = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

vision_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=VISION_MODEL_NAME,
    temperature=0,
)

_EXTRACTION_INSTRUCTIONS = """You are a visual evidence extractor for a customer service assistant.

Look at the image and return ONLY a JSON object (no markdown fences, no prose
outside the JSON) with this exact shape:

{
  "description": "<one factual sentence describing what the image literally shows>",
  "extracted_text": "<any text/numbers/labels visible in the image via OCR, empty string if none>",
  "objects": ["<short list of concrete objects/items/UI elements visibly present>"],
  "possible_issue": "<if this looks like a product/service problem, name it plainly; else empty string>",
  "confidence": "high" | "medium" | "low",
  "unclear_reason": "<if confidence is medium/low, briefly say why (blurry, cropped, ambiguous, no context); else empty string>"
}

Rules:
- Only put things in "description"/"extracted_text"/"objects" that you can
  actually see. Do not guess brand names, product identity, or root cause.
- If you are inferring rather than reading/seeing directly, that belongs in
  "possible_issue" and confidence must be "medium" or "low", never "high".
- If the image is unrelated to a product/support context, still describe it
  factually and leave "possible_issue" empty.
"""


def _to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _safe_parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        # Last resort: don't fabricate structured evidence from unparsable output.
        return {
            "description": text[:400],
            "extracted_text": "",
            "objects": [],
            "possible_issue": "",
            "confidence": "low",
            "unclear_reason": "Model output could not be parsed as structured evidence.",
        }


_OCR_ONLY_INSTRUCTIONS = """Transcribe ALL text visible in this image exactly as written,
top to bottom, left to right. Include headings, numbering, labels, and body text.
Preserve line breaks where they matter for meaning (e.g. separate questions/items).

Output ONLY the transcribed text. No JSON, no commentary, no "Here is the text:" preamble.
If there is no legible text in the image, output exactly: NO_TEXT_FOUND
"""


def _extract_raw_text(image_bytes: bytes, mime_type: str) -> str:
    """
    Dedicated OCR call with a plain-text (not JSON) response format.
    Transcription is a much more reliable task for a vision model when it
    isn't also forced to fit the output into a JSON schema at the same time
    -- important for text-dense images (documents, screenshots, exam papers)
    where a combined JSON response is prone to truncation/parse failures.
    """
    data_url = _to_data_url(image_bytes, mime_type)
    message = HumanMessage(
        content=[
            {"type": "text", "text": _OCR_ONLY_INSTRUCTIONS},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    try:
        response = vision_llm.invoke([message])
        text = response.content.strip()
        if text == "NO_TEXT_FOUND" or not text:
            return ""
        return text
    except Exception:
        return ""


def extract_image_evidence(image_bytes: bytes, mime_type: str = "image/png", user_question: str = "") -> dict:
    """
    Calls the vision model and returns a structured evidence dict:
      description, extracted_text, objects, possible_issue, confidence, unclear_reason

    `user_question` is optional context passed to the vision model so it can
    focus extraction (e.g. "is there a red light on this device?"), but the
    model is still instructed to only report what's literally visible.

    Runs two calls: a structured JSON call for description/objects/issue, and
    a dedicated plain-text OCR call for extracted_text. Keeping OCR separate
    from the JSON schema avoids losing text to JSON parse failures on
    text-dense images.
    """
    data_url = _to_data_url(image_bytes, mime_type)

    focus_hint = (
        f'The user is asking about this image in the context of: "{user_question}". '
        "Extract evidence relevant to that, but stay strictly factual."
        if user_question
        else ""
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": _EXTRACTION_INSTRUCTIONS + "\n" + focus_hint},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    try:
        response = vision_llm.invoke([message])
        evidence = _safe_parse_json(response.content)
    except Exception as e:
        evidence = {
            "description": "",
            "extracted_text": "",
            "objects": [],
            "possible_issue": "",
            "confidence": "low",
            "unclear_reason": f"Vision model call failed: {e}",
        }

    # Dedicated OCR pass -- overrides whatever (if anything) the structured
    # call put in extracted_text, since this call is more reliable for text.
    raw_text = _extract_raw_text(image_bytes, mime_type)
    if raw_text:
        evidence["extracted_text"] = raw_text
        # Finding real text is strong signal the image was legible even if
        # the structured call's confidence came back low/failed.
        if evidence.get("confidence", "low") == "low":
            evidence["confidence"] = "medium"
        if not evidence.get("description"):
            evidence["description"] = "An image containing text (see extracted_text)."

    evidence.setdefault("confidence", "low")
    evidence["is_usable"] = evidence["confidence"] in ("high", "medium") and bool(
        evidence.get("description") or evidence.get("extracted_text")
    )
    return evidence


def evidence_to_context_string(evidence: dict) -> str:
    """Renders evidence as a compact block that can be dropped into an LLM prompt."""
    if not evidence:
        return ""
    lines = [f"Image description: {evidence.get('description', '')}"]
    if evidence.get("extracted_text"):
        lines.append(f"Text visible in image: {evidence['extracted_text']}")
    if evidence.get("objects"):
        lines.append(f"Objects visible: {', '.join(evidence['objects'])}")
    if evidence.get("possible_issue"):
        lines.append(f"Possible issue suggested by image: {evidence['possible_issue']}")
    lines.append(f"Extraction confidence: {evidence.get('confidence', 'unknown')}")
    if evidence.get("unclear_reason"):
        lines.append(f"Note: {evidence['unclear_reason']}")
    return "\n".join(lines)