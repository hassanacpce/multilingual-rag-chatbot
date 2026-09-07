"""
reasoning_engine.py

Orchestrates a multi-step, branching reasoning pipeline.

    1. Update conversational memory with the new turn (text + image evidence)
    2. LANGUAGE       -- detect the language of this message, using the
                        conversation's established language as a fallback
                        for short/ambiguous text (language_detection.py,
                        deterministic/offline). See eval_language_detection.py
                        for why this matters: raw per-message detection is
                        75% accurate on a realistic labeled set; with
                        context-aware fallback, 100%.
    3. SENTIMENT      -- detect user sentiment (sentiment_analysis.py, VADER,
                        deterministic/offline) and derive tone guidance.
                        Neither language nor sentiment detection change WHAT
                        facts get stated -- only HOW and in WHICH LANGUAGE
                        the response is framed.
    4. PLAN            -- classify the turn (greeting / unclear-image /
                           ambiguous / answerable), and -- if the message
                           isn't in English -- get an English translation of
                           the underlying question for retrieval purposes
                           (the knowledge bases are English-only; retrieval
                           against them with a non-English query would fail
                           outright, not just underperform).
    5. Retrieval        -- ALWAYS runs for answerable turns (company FAISS KB,
                           MedQuAD medical KB, arXiv research KB), against the
                           English-translated query when the source message
                           wasn't in English.
    6. Draft generation  -- answer using KB context + image evidence + history
                           + sentiment tone guidance, IN THE USER'S LANGUAGE.
    7. Validation + SELF-CORRECTION LOOP -- an independent LLM call checks
       the draft's FACTUAL claims against the evidence (tone/empathy framing
       is explicitly excluded from this check).

Even the fail-closed fallback message (when nothing in the KB supports an
answer) is localized into all 5 supported languages -- "maintain consistent
responses regardless of the language used" applies to the worst case too,
not just the happy path.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage

from langchain_helper import llm, vectordb_file_path
from langchain_community.vectorstores import FAISS
from langchain_helper import embeddings

from vision_helper import extract_image_evidence, evidence_to_context_string
from medquad_knowledge import get_medical_context
from arxiv_knowledge import get_research_context
from sentiment_analysis import analyze_sentiment, get_tone_guidance
from language_detection import detect_language, SUPPORTED_LANGUAGES

MAX_HISTORY_TURNS = 6
MAX_VALIDATION_RETRIES = 1

MEDICAL_DISCLAIMER = (
    "\n\n_This includes general medical reference information from NIH/MedlinePlus "
    "sources (MedQuAD) — it isn't a diagnosis or personal medical advice. "
    "Please talk to a licensed healthcare professional about your specific situation._"
)

# Localized fail-closed fallback messages. Straightforward functional
# translations of a short, fixed sentence -- not LLM output, so they're
# consistent and testable, but worth a native-speaker review before
# production use, same as any non-professionally-translated UI string.
FALLBACK_MESSAGE = {
    "en": "I'm sorry, I couldn't find that information in my knowledge base.",
    "es": "Lo siento, no pude encontrar esa información en mi base de conocimientos.",
    "fr": "Je suis désolé, je n'ai pas trouvé cette information dans ma base de connaissances.",
    "hi": "मुझे खेद है, मुझे यह जानकारी अपने नॉलेज बेस में नहीं मिली।",
    "mr": "माफ करा, मला ही माहिती माझ्या नॉलेज बेसमध्ये सापडली नाही.",
}

FALLBACK_MESSAGE_EMPATHETIC = {
    "en": "I'm sorry for the trouble — I wasn't able to find that information in my knowledge base. Could you rephrase, or let me know more details so I can help?",
    "es": "Lamento las molestias — no pude encontrar esa información en mi base de conocimientos. ¿Podría reformular su pregunta o darme más detalles para poder ayudarle?",
    "fr": "Je suis désolé pour le désagrément — je n'ai pas trouvé cette information dans ma base de connaissances. Pourriez-vous reformuler votre question ou me donner plus de détails pour que je puisse vous aider ?",
    "hi": "असुविधा के लिए क्षमा करें — मुझे यह जानकारी अपने नॉलेज बेस में नहीं मिली। क्या आप अपना सवाल दोबारा बता सकते हैं, या मुझे मदद करने के लिए और विवरण दे सकते हैं?",
    "mr": "गैरसोयीबद्दल क्षमस्व — मला ही माहिती माझ्या नॉलेज बेसमध्ये सापडली नाही. कृपया तुमचा प्रश्न पुन्हा सांगा किंवा मदतीसाठी अधिक तपशील द्या.",
}


def _fallback_message(language_code: str, empathetic: bool) -> str:
    table = FALLBACK_MESSAGE_EMPATHETIC if empathetic else FALLBACK_MESSAGE
    return table.get(language_code, table["en"])


@dataclass
class Turn:
    role: str
    content: str
    image_evidence: Optional[dict] = None


@dataclass
class ConversationMemory:
    turns: list = field(default_factory=list)
    last_image_evidence: Optional[dict] = None
    language: Optional[str] = None  # established conversation language code, "sticky" across turns

    def add(self, role: str, content: str, image_evidence: Optional[dict] = None):
        self.turns.append(Turn(role, content, image_evidence))
        self.turns = self.turns[-(MAX_HISTORY_TURNS * 2):]
        if image_evidence is not None:
            self.last_image_evidence = image_evidence

    def as_prompt_block(self) -> str:
        if not self.turns:
            return "(no prior conversation)"
        lines = []
        for t in self.turns:
            prefix = "User" if t.role == "user" else "Assistant"
            line = f"{prefix}: {t.content}"
            if t.image_evidence:
                line += f"  [image shown: {t.image_evidence.get('description', 'n/a')}]"
            lines.append(line)
        return "\n".join(lines)


_cached_vectordb = None
_vectordb_load_attempted = False


def _get_vectordb():
    global _cached_vectordb, _vectordb_load_attempted
    import os

    if _vectordb_load_attempted:
        return _cached_vectordb

    _vectordb_load_attempted = True
    if os.path.exists(vectordb_file_path):
        _cached_vectordb = FAISS.load_local(
            vectordb_file_path, embeddings, allow_dangerous_deserialization=True
        )
    return _cached_vectordb


def reload_vectordb():
    global _vectordb_load_attempted
    _vectordb_load_attempted = False


def _ask_llm_json(prompt: str) -> dict:
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {}


# -----------------------------
# Step 4: PLANNING (also produces the English retrieval query for non-English turns)
# -----------------------------
def plan_response(
    question: str,
    history_block: str,
    evidence_block: str,
    image_provided: bool,
    image_evidence: Optional[dict],
    language_name: str,
    is_mixed_language: bool,
) -> dict:
    if image_provided and image_evidence is not None and not image_evidence.get("is_usable", False):
        reason = image_evidence.get("unclear_reason") or "the image content could not be reliably read"
        return {
            "response_type": "unclear_image",
            "needs_retrieval": False,
            "clarifying_question": (
                f"I had trouble reading that image ({reason}). "
                "Could you upload a clearer photo/screenshot, or describe in words what you'd like help with?"
            ),
            "english_query": question,
            "reasoning": "Vision extraction confidence too low to use as evidence.",
        }

    mixed_note = (
        " The user's message appears to mix multiple languages -- interpret it holistically "
        "rather than getting confused by the mixture."
        if is_mixed_language
        else ""
    )

    prompt = f"""You are the planning step of a customer-service assistant, deciding what
kind of response this turn needs BEFORE any answer is generated.

The user is writing in {language_name}.{mixed_note}

Conversation so far:
{history_block}

New question: "{question}"

Image evidence available for this question:
{evidence_block or "(no image provided)"}

Classify this turn into exactly one response_type:
- "greeting": pure greeting / small talk / "who are you" -- no knowledge base or image lookup needed.
- "ambiguous": the question is genuinely underspecified given everything above
  (e.g. vague reference to something not established in history, no image and
  a vague question). The clarifying_question you write MUST be grounded only
  in what's actually missing -- do not invent unrelated products, scenarios,
  or examples not mentioned anywhere above. Write the clarifying_question IN
  {language_name}, matching the user's language.
- "answerable": there's enough here to attempt a real answer.

Also provide english_query: an English translation of the user's underlying
question/request, for internal knowledge-base search purposes only (the
knowledge bases are English-only). If the user is already writing in
English, just repeat the question as english_query.

Respond with ONLY JSON:
{{"response_type": "greeting"|"ambiguous"|"answerable", "needs_retrieval": true/false, "clarifying_question": "<empty string unless ambiguous, written in {language_name}>", "english_query": "<English translation of the question, for search>", "reasoning": "<one short sentence>"}}
"""
    result = _ask_llm_json(prompt)
    return {
        "response_type": result.get("response_type", "answerable"),
        "needs_retrieval": bool(result.get("needs_retrieval", True)),
        "clarifying_question": result.get("clarifying_question", ""),
        "english_query": result.get("english_query") or question,
        "reasoning": result.get("reasoning", ""),
    }


# -----------------------------
# Step 5: Retrieval (always runs for answerable turns; uses English query)
# -----------------------------
def retrieve_company_context(query: str, evidence_block: str, k: int = 3) -> str:
    vectordb = _get_vectordb()
    if vectordb is None:
        return ""
    full_query = query
    if evidence_block:
        full_query = f"{query}\n\nRelated visual evidence:\n{evidence_block}"
    docs = vectordb.as_retriever(search_kwargs={"k": k}).invoke(full_query)
    return "\n\n".join(d.page_content for d in docs)


def retrieve_context(english_query: str, evidence_block: str, k: int = 3) -> dict:
    """
    Pulls from THREE knowledge sources using the ENGLISH-translated query
    (all three knowledge bases -- company, MedQuAD, arXiv -- are English-only
    TF-IDF/LSA indices; searching them with a non-English query would return
    nothing, not just perform worse).
    """
    company_context = retrieve_company_context(english_query, evidence_block, k=k)
    medical_context, entities, medical_available = get_medical_context(english_query, top_k=k)
    research_context, papers, research_available = get_research_context(english_query, top_k=k)

    blocks = []
    if company_context:
        blocks.append(f"=== Company knowledge base ===\n{company_context}")
    if medical_context:
        blocks.append(f"=== Medical reference (MedQuAD / NIH / MedlinePlus) ===\n{medical_context}")
    if research_context:
        blocks.append(f"=== Research papers (arXiv) ===\n{research_context}")

    return {
        "combined": "\n\n".join(blocks),
        "has_medical_content": bool(medical_context),
        "has_research_content": bool(research_context),
        "medical_entities": entities,
        "medical_kb_available": medical_available,
        "research_papers": papers,
        "research_kb_available": research_available,
    }


# -----------------------------
# Step 6: Draft generation
# -----------------------------
def generate_draft(
    question: str,
    history_block: str,
    kb_context: str,
    evidence_block: str,
    language_name: str,
    tone_guidance: str = "",
) -> str:
    tone_rule = (
        f"\n10. Tone guidance for this specific message: {tone_guidance}"
        if tone_guidance
        else ""
    )
    prompt = f"""You are a friendly, professional Customer Service AI Assistant that can also
answer general medical/health questions (NIH/MedlinePlus reference source) and
explain computer-science research topics using retrieved arXiv paper summaries.

Rules:
1. Answer ONLY using the knowledge base context and/or image evidence below.
2. If the user is just greeting you or asking who you are, respond naturally without needing evidence.
3. Never invent facts not present in the context or image evidence.
4. When you use a fact from the image, say so explicitly (e.g. "Based on the image you shared...").
5. If the answer isn't supported by the context or image evidence, say so plainly (in {language_name}).
6. The knowledge base context below may contain a "=== Company knowledge base ==="
   section, a "=== Medical reference (MedQuAD / NIH / MedlinePlus) ===" section,
   a "=== Research papers (arXiv) ===" section, or any combination. Only use
   each section for questions it's actually relevant to -- don't mix company
   policy into a medical answer, or a paper finding into a policy answer.
7. For medical/health questions, summarize what the medical reference material
   says. Never diagnose, never recommend a specific dosage or treatment beyond
   what's explicitly stated in the reference text, and never claim more
   certainty than the source material supports.
8. For research/technical questions, explain concepts in plain language
   before using jargon, and name the specific paper(s) you're drawing from so
   the user can look them up. The provided "Summary" and "Key phrases" per
   paper were extracted from that paper's abstract -- treat them as the
   ground truth for what that paper says, and don't extrapolate findings the
   abstract doesn't state.
9. Respond in {language_name} -- the SAME language the user wrote in. If
   their message mixed languages, respond primarily in {language_name} but
   you may naturally mirror a short phrase from the other language if it
   keeps the reply conversational. Keep any technical terms that don't
   translate well in their original form.{tone_rule}

Conversation so far:
{history_block}

Knowledge base context:
{kb_context or "(no matching knowledge base content found)"}

Image evidence:
{evidence_block or "(no image provided)"}

Question: {question}

Answer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def revise_draft(
    question: str,
    history_block: str,
    kb_context: str,
    evidence_block: str,
    previous_draft: str,
    objection: str,
    language_name: str,
    tone_guidance: str = "",
) -> str:
    tone_rule = f"\n\nTone guidance to keep in the revision: {tone_guidance}" if tone_guidance else ""
    prompt = f"""You previously drafted this customer-service answer:

"{previous_draft}"

A fact-checker reviewed it and found this problem:
"{objection}"

Rewrite the answer to fix that specific problem, using ONLY the knowledge base
context and/or image evidence below. Keep responding in {language_name}. If,
once you remove the unsupported part, nothing left is actually supported,
say so plainly (in {language_name}).{tone_rule}

Conversation so far:
{history_block}

Knowledge base context:
{kb_context or "(no matching knowledge base content found)"}

Image evidence:
{evidence_block or "(no image provided)"}

Question: {question}

Revised answer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# -----------------------------
# Step 7: Validation
# -----------------------------
def validate_answer(draft_answer: str, kb_context: str, evidence_block: str) -> dict:
    """
    Checks FACTUAL claims regardless of what language the draft is written
    in -- relies on the LLM's cross-lingual reasoning to compare a
    non-English draft against English-language retrieved context. This is
    the one piece of the multilingual pipeline that can't be verified
    without a real LLM call (see README).
    """
    prompt = f"""You are a strict fact-checker reviewing a draft customer-service answer
BEFORE it is sent to the user. The draft may be written in a language other
than the knowledge base context below -- compare the MEANING of its claims
against the context, not the literal language/wording.

Knowledge base context available:
{kb_context or "(none)"}

Image evidence available:
{evidence_block or "(none)"}

Draft answer:
{draft_answer}

Check every FACTUAL claim in the draft against the context/evidence above.
Do NOT flag empathetic acknowledgments, apologies for inconvenience,
greetings, or general tone/emotional framing sentences as unsupported --
those are not factual claims and don't need to trace back to the context.
Only flag claims about the company, medical information, or research
findings that aren't backed by the provided context.

Respond with ONLY JSON:
{{"supported": true/false, "final_answer": "<the answer to send if supported, otherwise leave as draft>", "objection": "<if not supported, describe SPECIFICALLY which factual claim is unsupported and why -- this will be used to ask for a rewrite; empty string if supported>"}}
"""
    result = _ask_llm_json(prompt)
    if not result or "supported" not in result:
        return {
            "supported": False,
            "final_answer": draft_answer,
            "objection": "Validator response could not be parsed.",
        }
    return result


# -----------------------------
# Orchestration entry point
# -----------------------------
def answer(
    question: str,
    memory: ConversationMemory,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/png",
) -> dict:
    """
    Runs the full pipeline for one turn. Returns:
      {
        "final_answer": str,
        "clarifying_question": str | None,
        "image_evidence": dict | None,
        "sentiment": dict,
        "language": dict,   # {"code": ..., "name": ..., "confidence": ..., "is_mixed": ..., "used_context_fallback": ...}
        "trace": {...}
      }
    Also updates `memory` in place with this turn, including memory.language.
    """
    # --- Step 2: LANGUAGE (deterministic, offline, context-aware fallback) ---
    lang_result = detect_language(question, conversation_language=memory.language)
    language_code = lang_result.code if lang_result.is_supported else "en"
    language_name = SUPPORTED_LANGUAGES.get(language_code, "English")
    language_info = {
        "code": language_code,
        "name": language_name,
        "confidence": lang_result.confidence,
        "is_mixed": lang_result.is_mixed,
        "used_context_fallback": lang_result.used_context_fallback,
    }
    # "Sticky" conversation language: only update from a genuine (non-fallback)
    # detection, so a short/ambiguous later message doesn't reset context.
    if not lang_result.used_context_fallback:
        memory.language = language_code

    # --- Step 3: SENTIMENT (deterministic, offline, no LLM call) ---
    sentiment = analyze_sentiment(question)
    tone_guidance = get_tone_guidance(sentiment)
    sentiment_info = {"label": sentiment.label, "compound": sentiment.compound}

    image_evidence = None
    evidence_block = ""
    if image_bytes:
        image_evidence = extract_image_evidence(image_bytes, image_mime, user_question=question)
        evidence_block = evidence_to_context_string(image_evidence)
    elif memory.last_image_evidence is not None:
        image_evidence = memory.last_image_evidence
        evidence_block = evidence_to_context_string(image_evidence)

    history_block = memory.as_prompt_block()

    # --- Step 4: PLAN (also produces the English retrieval query) ---
    plan = plan_response(
        question, history_block, evidence_block, image_evidence is not None, image_evidence,
        language_name=language_name, is_mixed_language=lang_result.is_mixed,
    )

    if plan["response_type"] == "greeting":
        draft = generate_draft(question, history_block, "", evidence_block, language_name=language_name, tone_guidance=tone_guidance)
        memory.add("user", question, image_evidence)
        memory.add("assistant", draft)
        return {
            "final_answer": draft,
            "clarifying_question": None,
            "image_evidence": image_evidence,
            "sentiment": sentiment_info,
            "language": language_info,
            "trace": {"plan": plan, "sentiment": sentiment_info, "language": language_info, "kb_context": None, "attempts": [{"draft": draft, "validation": None}]},
        }

    if plan["response_type"] in ("unclear_image", "ambiguous") and plan["clarifying_question"]:
        memory.add("user", question, image_evidence)
        memory.add("assistant", plan["clarifying_question"])
        return {
            "final_answer": None,
            "clarifying_question": plan["clarifying_question"],
            "image_evidence": image_evidence,
            "sentiment": sentiment_info,
            "language": language_info,
            "trace": {"plan": plan, "sentiment": sentiment_info, "language": language_info, "kb_context": None, "attempts": []},
        }

    retrieval = retrieve_context(plan["english_query"], evidence_block)
    kb_context = retrieval["combined"]

    draft = generate_draft(question, history_block, kb_context, evidence_block, language_name=language_name, tone_guidance=tone_guidance)
    validation = validate_answer(draft, kb_context, evidence_block)
    attempts = [{"draft": draft, "validation": validation}]

    retries = 0
    while not validation.get("supported") and retries < MAX_VALIDATION_RETRIES:
        objection = validation.get("objection") or "Claim not supported by context/evidence."
        draft = revise_draft(question, history_block, kb_context, evidence_block, draft, objection, language_name=language_name, tone_guidance=tone_guidance)
        validation = validate_answer(draft, kb_context, evidence_block)
        attempts.append({"draft": draft, "validation": validation})
        retries += 1

    if validation.get("supported"):
        final_answer = validation.get("final_answer", draft)
        if retrieval["has_medical_content"]:
            final_answer += MEDICAL_DISCLAIMER
        if retrieval["has_research_content"]:
            paper_titles = ", ".join(f'"{p.title}"' for p in retrieval["research_papers"][:2])
            final_answer += f"\n\n_Source paper(s): {paper_titles} (arXiv)._"
    else:
        final_answer = _fallback_message(language_code, empathetic=sentiment.is_negative)

    memory.add("user", question, image_evidence)
    memory.add("assistant", final_answer)

    return {
        "final_answer": final_answer,
        "clarifying_question": None,
        "image_evidence": image_evidence,
        "sentiment": sentiment_info,
        "language": language_info,
        "trace": {
            "plan": plan,
            "sentiment": sentiment_info,
            "language": language_info,
            "kb_context": kb_context,
            "attempts": attempts,
            "retries_used": retries,
            "medical_entities": retrieval["medical_entities"],
            "medical_kb_available": retrieval["medical_kb_available"],
            "research_papers": [p.title for p in retrieval["research_papers"]],
            "research_kb_available": retrieval["research_kb_available"],
        },
    }
