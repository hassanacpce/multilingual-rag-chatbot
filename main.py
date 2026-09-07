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
    page_title="Customer Service Chatbot",
    page_icon="🤖",
    layout="wide",
)

SENTIMENT_STYLE = {
    "strongly_negative": ("😠", "#C0392B"),
    "negative": ("🙁", "#D9822B"),
    "neutral": ("😐", "#8A8A8A"),
    "positive": ("🙂", "#2E8B57"),
}

LANGUAGE_STYLE = {
    "en": ("🇬🇧", "#2C3E50"),
    "hi": ("🇮🇳", "#FF9933"),
    "mr": ("🇮🇳", "#B34700"),
    "es": ("🇪🇸", "#C0392B"),
    "fr": ("🇫🇷", "#2E4A9E"),
}

# ------------------------
# Custom CSS
# ------------------------
st.markdown("""
<style>

.main{
    background-color:#f4f7fb;
}

.title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#1f77ff;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:25px;
}

.user{
    background:#1f77ff;
    color:white;
    padding:12px;
    border-radius:15px;
    margin-top:10px;
}

.bot{
    background:white;
    color:black;
    padding:12px;
    border-radius:15px;
    border-left:6px solid #1f77ff;
    margin-bottom:20px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.08);
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:45px;
    font-size:17px;
}

.sentiment-badge{
    display:inline-block;
    padding:2px 10px;
    border-radius:999px;
    color:white;
    font-size:0.75rem;
    font-weight:600;
    margin-left:8px;
}

</style>
""", unsafe_allow_html=True)


def render_sentiment_badge(sentiment: dict) -> str:
    if not sentiment:
        return ""
    emoji, color = SENTIMENT_STYLE.get(sentiment.get("label", "neutral"), ("😐", "#8A8A8A"))
    label = sentiment.get("label", "neutral").replace("_", " ")
    return (
        f"<span class='sentiment-badge' style='background:{color}'>"
        f"{emoji} {label} ({sentiment.get('compound', 0.0):+.2f})</span>"
    )


def render_language_badge(language: dict) -> str:
    if not language:
        return ""
    flag, color = LANGUAGE_STYLE.get(language.get("code", "en"), ("🌐", "#555555"))
    name = language.get("name", "English")
    suffix = ""
    if language.get("used_context_fallback"):
        suffix = " · from context"
    elif language.get("is_mixed"):
        suffix = " · mixed"
    return (
        f"<span class='sentiment-badge' style='background:{color}'>"
        f"{flag} {name}{suffix}</span>"
    )


# ------------------------
# Sidebar
# ------------------------

with st.sidebar:

    st.title("⚙️ Control Panel")

    if st.button(" Create Knowledge Base"):

        with st.spinner("Creating Vector Database..."):
            create_vector_db()
            reload_vectordb()

        st.success("Knowledge Base Created Successfully!")

    if st.button("🔄 Update Knowledge Base"):

        with st.spinner("Pulling new content from configured sources..."):
            summary = update_vector_store()
            reload_vectordb()

        if summary["total_added"] > 0:
            st.success(f"Added {summary['total_added']} new chunk(s) to the knowledge base.")
        else:
            st.info("No new content found — knowledge base is already up to date.")

        for source_id, result in summary["sources"].items():
            if result["error"]:
                st.error(f"{source_id}: {result['error']}")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory = ConversationMemory()

    show_trace = st.toggle("Show reasoning trace", value=False)

    st.markdown("---")
    st.subheader("🩺 Medical Knowledge Base")
    if medquad_knowledge.is_available():
        st.success("MedQuAD loaded — medical questions are answered from NIH/MedlinePlus data.")
    else:
        err = medquad_knowledge.load_error()
        st.warning("MedQuAD not loaded. Medical questions will fall back to the company KB (likely 'not found').")
        if err:
            st.caption(f"Reason: {err}")
        st.caption(
            "Clone it first: `git clone https://github.com/abachaa/MedQuAD.git` "
            "into `./MedQuAD` (or set `MEDQUAD_ROOT`), then click below."
        )

    if st.button("🔁 Build/Rebuild Medical KB"):
        with st.spinner("Parsing MedQuAD XML and building the medical retrieval index..."):
            try:
                medquad_knowledge.force_rebuild()
                st.success("Medical knowledge base built.")
            except Exception as e:
                st.error(f"Could not build medical KB: {e}")

    st.markdown("---")
    st.subheader("📄 Research Paper Knowledge Base (arXiv)")
    if arxiv_knowledge.is_available():
        st.success("arXiv papers loaded — research questions are answered with paper summaries + citations.")
    else:
        err = arxiv_knowledge.load_error()
        st.warning("arXiv papers not loaded. Research questions will fall back to other knowledge sources.")
        if err:
            st.caption(f"Reason: {err}")
        st.caption(
            "Requires the Kaggle arXiv metadata file locally (needs a free Kaggle account): "
            "`pip install kaggle && kaggle datasets download -d Cornell-University/arxiv && unzip arxiv.zip`, "
            "then set `ARXIV_RAW_PATH` (or place `arxiv-metadata-oai-snapshot.json` here) and click below."
        )

    if st.button("🔁 Build/Rebuild Research KB"):
        with st.spinner("Parsing arXiv metadata and building the paper retrieval index..."):
            try:
                arxiv_knowledge.force_rebuild()
                st.success("Research knowledge base built.")
            except Exception as e:
                st.error(f"Could not build research KB: {e}")

    st.markdown("---")
    st.subheader("📊 Sentiment & Satisfaction")
    stats = feedback_log.get_feedback_stats()
    if stats is None:
        st.caption("No feedback logged yet. Use the 👍/👎 buttons under each answer to start tracking satisfaction.")
    else:
        st.metric("Overall satisfaction", f"{stats['overall_satisfaction_rate']*100:.0f}%", help=f"{stats['up']} up / {stats['down']} down out of {stats['total']} rated responses")
        with st.expander("Breakdown by detected sentiment"):
            for label, counts in sorted(stats["by_sentiment"].items()):
                emoji, _ = SENTIMENT_STYLE.get(label, ("😐", "#8A8A8A"))
                st.write(f"{emoji} **{label.replace('_', ' ')}**: {counts['up']}/{counts['total']} = {counts['satisfaction_rate']*100:.0f}%")

    st.markdown("---")

    st.info("""
This chatbot answers questions
using your company knowledge base,
general medical reference info (MedQuAD),
computer-science research papers (arXiv),
and any image you attach. It also detects
sentiment and language, and adapts its
tone and response language accordingly.

Supports English, Hindi, Marathi, Spanish,
and French — automatically, with context
retained across language switches.

Powered by

 LangChain

 FAISS

 Groq (text + vision)

 MedQuAD (NIH / MedlinePlus)

 arXiv (Cornell University)

 VADER (sentiment analysis)

 langdetect (language detection)
""")

# ------------------------
# Title
# ------------------------

st.markdown("<div class='title'> Customer Service Chatbot</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>Ask about your account, general medical questions, or computer-science research \u2014 attach a screenshot or photo if it helps</div>",
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

tab_chat, tab_search, tab_concepts = st.tabs(["\U0001f4ac Chat", "\U0001f50d Search Papers", "\U0001f5fa\ufe0f Concept Map"])

# ------------------------
# Tab 1: Chat (company + medical + research, unified, sentiment-aware)
# ------------------------
with tab_chat:

    for idx, message in enumerate(st.session_state.messages):

        if message["role"] == "user":
            sentiment_badge = render_sentiment_badge(message.get("sentiment"))
            language_badge = render_language_badge(message.get("language"))
            st.markdown(
                f"<div class='user'>\U0001f9d1 <b>You</b>{sentiment_badge}{language_badge}<br>{message['content']}</div>",
                unsafe_allow_html=True,
            )
            if message.get("image"):
                st.image(message["image"], width=220)

        else:
            st.markdown(
                f"<div class='bot'>\U0001f916 <b>Assistant</b><br>{message['content']}</div>",
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
                st.caption("Thanks for the feedback!")

            if show_trace and message.get("trace"):
                entities = message["trace"].get("medical_entities") or []
                if entities:
                    chips = " ".join(
                        f"<span style='background:#4C8C6B;color:white;padding:2px 8px;"
                        f"border-radius:999px;font-size:0.75rem;margin-right:4px;'>"
                        f"{e['type']}: {e['text']}</span>"
                        for e in entities
                    )
                    st.markdown(f"**Medical entities detected:** {chips}", unsafe_allow_html=True)
                papers = message["trace"].get("research_papers") or []
                if papers:
                    paper_chips = " ".join(
                        f"<span style='background:#7A3B2E;color:white;padding:2px 8px;"
                        f"border-radius:999px;font-size:0.75rem;margin-right:4px;'>{p}</span>"
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
        st.warning(
            "Research knowledge base not loaded yet. Build it from the sidebar "
            "(\U0001f4c4 Research Paper Knowledge Base) first."
        )
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
                chips = "".join(
                    f'<span style="display:inline-block;background:#E8A33D;color:#20293A;'
                    f'padding:2px 9px;border-radius:3px;font-size:0.76rem;margin:2px 4px 2px 0;'
                    f'font-weight:600;">{p}</span>'
                    for p in phrases
                )
                st.markdown(
                    f"""
<div style="background:white;border:1px solid #E3E3E3;border-left:4px solid #7A3B2E;
border-radius:4px;padding:16px 20px;margin-bottom:14px;">
  <div style="font-family:monospace;font-size:0.78rem;color:#6B6252;">{r.primary_category} \u00b7 match {r.score:.2f}</div>
  <div style="font-size:1.05rem;font-weight:700;margin:4px 0 6px 0;">{r.title}</div>
  <div style="font-family:monospace;font-size:0.78rem;color:#6B6252;margin-bottom:8px;">
    {r.authors} \u00b7 <a href="{r.arxiv_url}" target="_blank">{r.arxiv_url}</a>
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
        st.warning(
            "Research knowledge base not loaded yet. Build it from the sidebar "
            "(\U0001f4c4 Research Paper Knowledge Base) first."
        )
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
