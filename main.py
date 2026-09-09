import os
import streamlit as st
from langchain_helper import create_vector_db
from knowledge_base_manager import update_vector_store, load_config
from scheduler import start_scheduler
from reasoning_engine import ConversationMemory, answer, reload_vectordb
import medquad_knowledge
import arxiv_knowledge
from keyphrase_extraction import extract_keyphrases
from summarizer import summarize
import feedback_log

st.set_page_config(
    page_title="Reference Assistant",
    page_icon="📘",
    layout="wide",
)

SENTIMENT_STYLE = {
    "strongly_negative": ("var(--warn)", "Strongly negative"),
    "negative": ("var(--warn-soft)", "Negative"),
    "neutral": ("var(--ink-soft)", "Neutral"),
    "positive": ("var(--medical)", "Positive"),
}

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "es": "Spanish",
    "fr": "French",
}

# ------------------------
# Custom CSS
# ------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
    --paper: #F2F4F3;
    --panel: #FFFFFF;
    --ink: #1C2624;
    --ink-soft: #5B6B67;
    --line: #D8DEDB;
    --support: #35506B;
    --medical: #3F7A5E;
    --research: #A6752E;
    --warn: #B5533C;
    --warn-soft: #C98A5B;
}

html, body, [class*="css"]{
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}

.main, [data-testid="stAppViewContainer"]{
    background-color: var(--paper);
}

[data-testid="stSidebar"]{
    background-color: var(--panel);
    border-right: 1px solid var(--line);
}

h1, h2, h3, .app-title{
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    letter-spacing: -0.01em;
}

/* Reduced-motion respect */
@media (prefers-reduced-motion: reduce){
    *{ animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; }
}

/* Visible keyboard focus, since we're overriding a lot of default chrome */
button:focus-visible, input:focus-visible, textarea:focus-visible{
    outline: 2px solid var(--support);
    outline-offset: 2px;
}

/* ---- App header ---- */
.app-header{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--line);
    padding-bottom: 14px;
    margin-bottom: 18px;
}
.app-title{
    font-size: 1.6rem;
    margin: 0;
}
.app-subtitle{
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink-soft);
    font-size: 0.92rem;
    margin-top: 4px;
}

/* ---- Sidebar section labels (sentence case, not all-caps) ---- */
.panel-label{
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 0.8rem;
    color: var(--ink-soft);
    margin: 18px 0 6px 0;
    padding-top: 12px;
    border-top: 1px solid var(--line);
}
.panel-label:first-of-type{ border-top: none; padding-top: 0; }

.status-row{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.88rem;
    padding: 3px 0;
}
.status-dot{
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 7px;
}

/* ---- Chat turns: no bubbles, no shadow-card kit -- typographic distinction + a thin domain rule ---- */
.turn{
    padding: 10px 0 10px 14px;
    margin-bottom: 4px;
    border-left: 3px solid var(--line);
}
.turn.user{
    border-left-color: var(--ink-soft);
}
.turn.assistant{
    border-left-color: var(--speaker-color, var(--support));
    background: var(--panel);
    padding: 14px 16px 14px 16px;
    margin-bottom: 14px;
    border-radius: 0 4px 4px 0;
}
.turn-meta{
    font-size: 0.78rem;
    color: var(--ink-soft);
    margin-bottom: 4px;
}
.turn-meta b{ color: var(--ink); font-weight: 600; }

.chip{
    display: inline-block;
    padding: 1px 9px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-left: 6px;
    color: var(--panel);
}

/* ---- Buttons: flat, quiet, one accent ---- */
.stButton>button{
    width: 100%;
    border-radius: 4px;
    border: 1px solid var(--line);
    background: var(--panel);
    color: var(--ink);
    font-size: 0.92rem;
    height: 42px;
}
.stButton>button:hover{
    border-color: var(--support);
    color: var(--support);
}
[data-testid="stFormSubmitButton"] button{
    background: var(--support);
    color: var(--panel);
    border: none;
}
[data-testid="stFormSubmitButton"] button:hover{
    background: var(--ink);
}

/* ---- Tabs: plain text, underline for active state, no emoji chrome ---- */
[data-testid="stTabs"] button[role="tab"]{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.95rem;
}
[data-testid="stTabs"] button[aria-selected="true"]{
    color: var(--support);
    border-bottom-color: var(--support) !important;
}

/* ---- Research/paper cards: monospace only for genuine identifiers ---- */
.paper-card{
    background: var(--panel);
    border-left: 3px solid var(--research);
    border-radius: 0 4px 4px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.paper-meta{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: var(--ink-soft);
}
.paper-title{
    font-family: 'Source Serif 4', serif;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 4px 0 6px 0;
}
.keyphrase-chip{
    display: inline-block;
    background: transparent;
    border: 1px solid var(--research);
    color: var(--research);
    padding: 1px 9px;
    border-radius: 3px;
    font-size: 0.74rem;
    margin: 2px 5px 2px 0;
}
</style>
""", unsafe_allow_html=True)


def render_meta_chips(sentiment: dict, language: dict) -> str:
    chips = ""
    if sentiment:
        color, label = SENTIMENT_STYLE.get(sentiment.get("label", "neutral"), ("var(--ink-soft)", "Neutral"))
        chips += f"<span class='chip' style='background:{color}'>{label} {sentiment.get('compound', 0.0):+.2f}</span>"
    if language:
        name = LANGUAGE_NAMES.get(language.get("code", "en"), language.get("name", "English"))
        suffix = ""
        if language.get("used_context_fallback"):
            suffix = " · from context"
        elif language.get("is_mixed"):
            suffix = " · mixed"
        chips += f"<span class='chip' style='background:var(--ink-soft)'>{name}{suffix}</span>"
    return chips


def domain_for_trace(trace: dict) -> tuple:
    """Picks the dominant source domain for a turn so the assistant panel's
    accent rule reflects what the answer actually drew on, rather than
    decorating every message identically."""
    if not trace:
        return ("support", "var(--support)")
    if trace.get("medical_entities"):
        return ("medical", "var(--medical)")
    if trace.get("research_papers"):
        return ("research", "var(--research)")
    return ("support", "var(--support)")


# ------------------------
# Sidebar
# ------------------------

with st.sidebar:

    st.markdown("<div class='panel-label' style='border-top:none;padding-top:0;'>Console</div>", unsafe_allow_html=True)
    show_trace = st.toggle("Show reasoning trace", value=False)

    st.markdown("<div class='panel-label'>Sources</div>", unsafe_allow_html=True)

    def _status_row(name: str, ready: bool, color: str) -> None:
        dot_color = color if ready else "var(--warn)"
        state = "ready" if ready else "not loaded"
        st.markdown(
            f"<div class='status-row'><span><span class='status-dot' style='background:{dot_color}'></span>{name}</span>"
            f"<span style='color:var(--ink-soft)'>{state}</span></div>",
            unsafe_allow_html=True,
        )

    _status_row("Company knowledge base", True, "var(--support)")
    _status_row("Medical (MedQuAD)", medquad_knowledge.is_available(), "var(--medical)")
    _status_row("Research (arXiv)", arxiv_knowledge.is_available(), "var(--research)")

    if not medquad_knowledge.is_available():
        err = medquad_knowledge.load_error()
        st.caption(f"Medical KB: {err}" if err else "Medical KB not built yet.")
    if not arxiv_knowledge.is_available():
        err = arxiv_knowledge.load_error()
        st.caption(f"Research KB: {err}" if err else "Research KB not built yet.")

    st.markdown("<div class='panel-label'>Actions</div>", unsafe_allow_html=True)

    if st.button("Rebuild company KB"):
        with st.spinner("Rebuilding company knowledge base..."):
            create_vector_db()
            reload_vectordb()
        st.success("Company knowledge base rebuilt.")

    if st.button("Pull updates from sources"):
        with st.spinner("Checking configured sources for new content..."):
            summary = update_vector_store()
            reload_vectordb()
        if summary["total_added"] > 0:
            st.success(f"Added {summary['total_added']} new item(s).")
        else:
            st.caption("No new content found — already up to date.")
        for source_id, result in summary["sources"].items():
            if result["error"]:
                st.error(f"{source_id}: {result['error']}")

    if not medquad_knowledge.is_available():
        if st.button("Build medical KB"):
            with st.spinner("Parsing MedQuAD and building the medical index..."):
                try:
                    medquad_knowledge.force_rebuild()
                    st.success("Medical knowledge base built.")
                except Exception as e:
                    st.error(f"Could not build medical KB: {e}")

    if not arxiv_knowledge.is_available():
        if st.button("Build research KB"):
            with st.spinner("Parsing arXiv metadata and building the research index..."):
                try:
                    arxiv_knowledge.force_rebuild()
                    st.success("Research knowledge base built.")
                except Exception as e:
                    st.error(f"Could not build research KB: {e}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.memory = ConversationMemory()

    st.markdown("<div class='panel-label'>Satisfaction</div>", unsafe_allow_html=True)
    stats = feedback_log.get_feedback_stats()
    if stats is None:
        st.caption("No feedback yet — rate answers below to track this.")
    else:
        st.metric("Rated positively", f"{stats['overall_satisfaction_rate']*100:.0f}%",
                   help=f"{stats['up']} up / {stats['down']} down out of {stats['total']} rated")
        with st.expander("By detected sentiment"):
            for label, counts in sorted(stats["by_sentiment"].items()):
                st.write(f"**{label.replace('_', ' ')}** — {counts['up']}/{counts['total']} ({counts['satisfaction_rate']*100:.0f}%)")

    st.markdown("<div class='panel-label'>About</div>", unsafe_allow_html=True)
    st.caption(
        "Answers draw on the company knowledge base, MedQuAD medical reference "
        "data, and arXiv research papers, plus any attached image. Detects "
        "sentiment and adapts tone; supports English, Hindi, Marathi, Spanish, "
        "and French with context retained across language switches."
    )
    st.caption("LangChain · FAISS · Groq · MedQuAD · arXiv · VADER · langdetect")

# ------------------------
# Header
# ------------------------

_sources_ready = sum([
    True,  # company KB
    medquad_knowledge.is_available(),
    arxiv_knowledge.is_available(),
])

st.markdown(
    f"""
<div class="app-header">
  <div>
    <p class="app-title">Reference Assistant</p>
    <p class="app-subtitle">Company support, medical reference, and research paper questions — in your language, with sources</p>
  </div>
  <div style="text-align:right; font-size:0.82rem; color:var(--ink-soft); white-space:nowrap;">
    {_sources_ready}/3 sources connected
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------
# State
# ------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

# Idempotent: safe to call on every rerun. Only actually starts a background
# job if sources_config.json has schedule.enabled = true.
if load_config().get("schedule", {}).get("enabled", False):
    start_scheduler()

tab_chat, tab_search, tab_concepts = st.tabs(["Chat", "Search papers", "Concept map"])

# ------------------------
# Tab 1: Chat (company + medical + research, unified, sentiment-aware)
# ------------------------
with tab_chat:

    for idx, message in enumerate(st.session_state.messages):

        if message["role"] == "user":
            meta_chips = render_meta_chips(message.get("sentiment"), message.get("language"))
            st.markdown(
                f"<div class='turn user'><div class='turn-meta'><b>You</b>{meta_chips}</div>{message['content']}</div>",
                unsafe_allow_html=True,
            )
            if message.get("image"):
                st.image(message["image"], width=220)

        else:
            domain_label, domain_color = domain_for_trace(message.get("trace"))
            st.markdown(
                f"<div class='turn assistant' style='--speaker-color:{domain_color}'>"
                f"<div class='turn-meta'><b>Assistant</b> · {domain_label}</div>{message['content']}</div>",
                unsafe_allow_html=True,
            )

            # Thumbs up/down feedback -- logged once per message, tied to
            # the sentiment detected on the PRECEDING user turn, so
            # satisfaction can be broken down by sentiment (see sidebar).
            if not message.get("feedback_logged"):
                fb_col1, fb_col2, _ = st.columns([1, 1, 8])
                with fb_col1:
                    if st.button("\U0001f44d", key=f"up_{idx}"):
                        prior_user_msg = st.session_state.messages[idx - 1] if idx > 0 else {}
                        sentiment = prior_user_msg.get("sentiment") or {"label": "neutral", "compound": 0.0}
                        feedback_log.log_feedback(
                            question=prior_user_msg.get("content", ""),
                            sentiment_label=sentiment.get("label", "neutral"),
                            sentiment_compound=sentiment.get("compound", 0.0),
                            response=message["content"],
                            feedback="up",
                        )
                        message["feedback_logged"] = True
                        st.rerun()
                with fb_col2:
                    if st.button("\U0001f44e", key=f"down_{idx}"):
                        prior_user_msg = st.session_state.messages[idx - 1] if idx > 0 else {}
                        sentiment = prior_user_msg.get("sentiment") or {"label": "neutral", "compound": 0.0}
                        feedback_log.log_feedback(
                            question=prior_user_msg.get("content", ""),
                            sentiment_label=sentiment.get("label", "neutral"),
                            sentiment_compound=sentiment.get("compound", 0.0),
                            response=message["content"],
                            feedback="down",
                        )
                        message["feedback_logged"] = True
                        st.rerun()
            else:
                st.caption("Thanks for the feedback.")

            if show_trace and message.get("trace"):
                entities = message["trace"].get("medical_entities") or []
                if entities:
                    chips = " ".join(
                        f"<span class='chip' style='background:var(--medical)'>{e['type']}: {e['text']}</span>"
                        for e in entities
                    )
                    st.markdown(f"**Medical entities detected:** {chips}", unsafe_allow_html=True)
                papers = message["trace"].get("research_papers") or []
                if papers:
                    paper_chips = " ".join(
                        f"<span class='chip' style='background:var(--research)'>{p}</span>"
                        for p in papers
                    )
                    st.markdown(f"**Papers referenced:** {paper_chips}", unsafe_allow_html=True)
                with st.expander("Reasoning trace"):
                    st.json(message["trace"])

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input("Type your question...")
        uploaded_image = st.file_uploader(
            "Attach an image (optional)", type=["png", "jpg", "jpeg", "webp"]
        )
        submitted = st.form_submit_button("Send")

    if submitted and (question or uploaded_image):

        image_bytes = uploaded_image.read() if uploaded_image else None
        image_mime = uploaded_image.type if uploaded_image else "image/png"

        with st.spinner("Thinking..."):
            result = answer(
                question=question or "What can you tell me about this image?",
                memory=st.session_state.memory,
                image_bytes=image_bytes,
                image_mime=image_mime,
            )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question or "(image only)",
                "image": image_bytes,
                "sentiment": result.get("sentiment"),
                "language": result.get("language"),
            }
        )

        if result["clarifying_question"]:
            reply = result["clarifying_question"]
        else:
            reply = result["final_answer"]

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "trace": result["trace"],
                "feedback_logged": False,
            }
        )

        st.rerun()

# ------------------------
# Tab 2: Search Papers (direct semantic search over arXiv corpus)
# ------------------------
with tab_search:
    retriever = arxiv_knowledge.get_retriever()
    if retriever is None:
        st.warning("Research knowledge base not loaded yet. Build it from the sidebar first.")
    else:
        col_q, col_cat = st.columns([3, 1])
        with col_q:
            search_query = st.text_input("Search papers by topic...", key="paper_search_input")
        with col_cat:
            categories = ["(any)"] + sorted(retriever.df["primary_category"].unique().tolist())
            category_filter = st.selectbox("Category", categories)

        search_top_k = st.slider("Results", 1, 15, 6)

        if search_query.strip():
            cat = None if category_filter == "(any)" else category_filter
            results = retriever.search(search_query, top_k=search_top_k, category_filter=cat)
            if not results:
                st.warning("No matching papers found. Try a broader query.")
            for r in results:
                summary = summarize(r.abstract, num_sentences=2)
                phrases = [kp for kp, _ in extract_keyphrases(r.abstract, top_n=6)]
                chips = "".join(f"<span class='keyphrase-chip'>{p}</span>" for p in phrases)
                st.markdown(
                    f"""
<div class="paper-card">
  <div class="paper-meta">{r.primary_category} · match {r.score:.2f}</div>
  <div class="paper-title">{r.title}</div>
  <div class="paper-meta" style="margin-bottom:8px;">
    {r.authors} · <a href="{r.arxiv_url}" target="_blank" style="color:var(--research);">{r.arxiv_url}</a>
  </div>
  <p>{summary}</p>
  <div>{chips}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Enter a topic above to search the research paper corpus.")

# ------------------------
# Tab 3: Concept Map (2D LSA concept-space visualization)
# ------------------------
with tab_concepts:
    retriever = arxiv_knowledge.get_retriever()
    if retriever is None:
        st.warning("Research knowledge base not loaded yet. Build it from the sidebar first.")
    else:
        import plotly.express as px

        st.caption(
            "Each point is a paper, projected onto the first two LSA (Latent Semantic "
            "Analysis) components \u2014 the two dominant 'concepts' the corpus decomposes "
            "into. Papers close together were judged conceptually related, even without "
            "sharing exact keywords."
        )
        coords = retriever.concept_coordinates_2d()
        fig = px.scatter(
            coords, x="x", y="y", color="primary_category", hover_data=["title"],
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(legend_title_text="Category", xaxis_title="Concept 1", yaxis_title="Concept 2")
        st.plotly_chart(fig, use_container_width=True)