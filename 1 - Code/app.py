import streamlit as st
#import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchAI",
    page_icon="🔬",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.hero {
    text-align: center;
    padding: 50px 20px 30px 20px;
}

.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
}

.hero p {
    font-size: 1rem;
    color: #a0a0c0;
    margin-bottom: 30px;
}

/* Expander styling */
div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    margin-bottom: 14px !important;
}

div[data-testid="stExpander"] summary {
    color: #c4b5fd !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
}

div[data-testid="stExpander"] p,
div[data-testid="stExpander"] li {
    color: #e2e8f0 !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
}

/* Text input */
div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 1rem !important;
}

/* Submit button */
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
}
/* Hide "Press Enter to submit form" helper text */
div[data-testid="InputInstructions"] {
    display: none !important;
}

/* Download button */
div[data-testid="stDownloadButton"] button {
    background: rgba(255,255,255,0.07) !important;
    color: #c4b5fd !important;
    border: 1px solid rgba(196,181,253,0.3) !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    width: 100% !important;
}

.ref-item {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
    color: #94a3b8;
    font-size: 0.83rem;
}

.ref-item a { color: #818cf8; text-decoration: none; }

.left-panel {
    padding-right: 20px;
    border-right: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────────────────────
load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_resources():
    embeddings = np.load(os.path.join(BASE_DIR, "embeddings.npy"))
    with open(os.path.join(BASE_DIR, "papers_metadata.pkl"), "rb") as f:
        data = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    client = Groq(api_key=groq_key)
    return embeddings, data, model, client

embeddings, data, model, client = load_resources()
papers_list = data['papers']

def retrieve(query, top_k=8):
    query_vec = model.encode([query])[0]
    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_vec)
    similarities = np.dot(embeddings, query_vec) / (norms * query_norm + 1e-10)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [papers_list[i] for i in top_indices if i < len(papers_list)]

def build_context(results):
    context = ""
    for r in results:
        context += f"\n---\nTitle: {r['title']} ({r.get('year','')})\nAbstract: {r['abstract']}\n"
    return context

def run_llm(prompt, max_tokens=1500, temperature=0.4):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content

# ── Session state ─────────────────────────────────────────────────────────────
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
if "synthesis" not in st.session_state:
    st.session_state.synthesis = ""
if "gaps" not in st.session_state:
    st.session_state.gaps = ""
if "hypotheses" not in st.session_state:
    st.session_state.hypotheses = ""
if "refs" not in st.session_state:
    st.session_state.refs = []

# ── Layout ────────────────────────────────────────────────────────────────────
if not st.session_state.results_ready:
    # ── Centered hero before first run ────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🔬 ResearchAI</h1>
        <p>Enter a research topic. Get a synthesis, gap analysis, and novel hypotheses — instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("search_form"):
            topic = st.text_input("", placeholder="e.g. pre-strain influence on toughness in high-strength steel")
            run = st.form_submit_button("Generate Research Report →")

    if run and topic.strip():
        results = retrieve(topic)
        context = build_context(results)

        with st.spinner("Synthesizing field..."):
            st.session_state.synthesis = run_llm(f"""You are a senior researcher. Synthesize the current state of the field on:
TOPIC: {topic}
PAPERS: {context}
Cover: key findings, methodologies, connections between studies, contradictions. Write at PhD level.""")

        with st.spinner("Identifying research gaps..."):
            st.session_state.gaps = run_llm(f"""Identify research gaps based on these papers about: {topic}
{context}
Cover: underexplored areas, methodological gaps, contradictions, industry gaps, emerging opportunities.""")

        with st.spinner("Generating novel hypotheses..."):
            st.session_state.hypotheses = run_llm(f"""Generate 5 novel, testable PhD-level research hypotheses based on these gaps:
{st.session_state.gaps}
For each: hypothesis statement, gap addressed, why novel, proposed approach, expected impact.""",
            temperature=0.7, max_tokens=2000)

        st.session_state.refs = results
        st.session_state.results_ready = True
        st.rerun()

else:
    # ── Two-column layout after results ───────────────────────────────────────
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)
        st.markdown("### 🔬 ResearchAI")
        st.markdown("<p style='color:#a0a0c0; font-size:0.9rem;'>Enter a new topic to generate a fresh report.</p>", unsafe_allow_html=True)

        with st.form("search_form"):
            topic = st.text_input("", placeholder="e.g. pre-strain influence on toughness...")
            run = st.form_submit_button("Generate Research Report →")

        if run and topic.strip():
            results = retrieve(topic)
            context = build_context(results)

            with st.spinner("Synthesizing field..."):
                st.session_state.synthesis = run_llm(f"""You are a senior researcher. Synthesize the current state of the field on:
TOPIC: {topic}
PAPERS: {context}
Cover: key findings, methodologies, connections between studies, contradictions. Write at PhD level.""")

            with st.spinner("Identifying research gaps..."):
                st.session_state.gaps = run_llm(f"""Identify research gaps based on these papers about: {topic}
{context}
Cover: underexplored areas, methodological gaps, contradictions, industry gaps, emerging opportunities.""")

            with st.spinner("Generating novel hypotheses..."):
                st.session_state.hypotheses = run_llm(f"""Generate 5 novel, testable PhD-level research hypotheses based on these gaps:
{st.session_state.gaps}
For each: hypothesis statement, gap addressed, why novel, proposed approach, expected impact.""",
                temperature=0.7, max_tokens=2000)

            st.session_state.refs = results
            st.rerun()

        # ── Chat ──────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💬 Deep Dive")
        st.markdown("<p style='color:#a0a0c0; font-size:0.85rem;'>Ask follow-up questions about the report.</p>", unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chat history display
        # chat_container = st.container()
        # with chat_container:
        #     for msg in st.session_state.chat_history:
        #         with st.chat_message(msg["role"]):
        #             st.markdown(msg["content"])
        # Chat history display — scrollable
        st.markdown("""
        <style>
        .chat-scroll {
            height: 700px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            background: rgba(255,255,255,0.03);
            margin-bottom: 10px;
        }
        .chat-bubble-user {
            background: rgba(99,102,241,0.2);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 8px;
            color: #e2e8f0;
            font-size: 0.88rem;
        }
        .chat-bubble-ai {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 8px;
            color: #e2e8f0;
            font-size: 0.88rem;
        }
        .chat-label {
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 4px;
            color: #a0a0c0;
        }
        
        </style>
        """, unsafe_allow_html=True)

        # Build scrollable HTML chat bubbles
        chat_html = '<div class="chat-scroll">'
        if not st.session_state.chat_history:
            chat_html += '<p style="color:#64748b; font-size:0.85rem; text-align:center; margin-top:160px;">Ask a question to start the conversation...</p>'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f'<div class="chat-label">You</div><div class="chat-bubble-user">{msg["content"]}</div>'
            else:
                chat_html += f'<div class="chat-label">ResearchAI</div><div class="chat-bubble-ai">{msg["content"]}</div>'
        chat_html += '</div>'

        st.markdown(chat_html, unsafe_allow_html=True)


        # Chat input
        user_input = st.chat_input("Ask about the research...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Build verified references list from actual retrieved papers
            verified_refs = ""
            for i, paper in enumerate(st.session_state.refs, 1):
                verified_refs += f"{i}. \"{paper['title']}\" ({paper.get('year', 'n.d.')}) — {paper.get('url', 'no URL')}\n"

            report_context = f"""You are an expert research assistant. The user has received this AI-generated research report.

                    ## Field Synthesis
                    {st.session_state.synthesis}

                    ## Research Gaps
                    {st.session_state.gaps}

                    ## Novel Hypotheses
                    {st.session_state.hypotheses}

                    ## Verified Papers (the ONLY sources you may cite)
                    {verified_refs}

                    STRICT RULES — you must follow these at all times:
                    - You may ONLY cite papers from the "Verified Papers" list above.
                    - NEVER invent, fabricate, or guess paper titles, authors, years, or URLs.
                    - When asked about a hypothesis (e.g. "tell me more about Hypothesis 3"), refer directly to the text above.
                    - Help the user evaluate, critique, extend, or refine the hypotheses.
                    - Suggest experimental designs, related methodologies, or potential challenges for any hypothesis.
                    - If the answer requires a source not in the list, say: "I don't have a verified source for this in the current dataset."
                    - When citing, use the exact title from the list, e.g. (Source: "Title of Paper", Year).
                    - Think at PhD level, but stay strictly within the verified sources when referencing literature."""

            messages = [{"role": "system", "content": report_context}]
            for msg in st.session_state.chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=1000,
                temperature=0.5
            )
            reply = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown("### 📄 Research Report")

        # ── Field Synthesis ───────────────────────────────────────────────────
        with st.expander("📖 Field Synthesis", expanded=True):
            st.markdown(st.session_state.synthesis)
            st.download_button(
                label="⬇ Download Synthesis",
                data=st.session_state.synthesis,
                file_name="field_synthesis.md",
                mime="text/markdown"
            )

        # ── Research Gaps ─────────────────────────────────────────────────────
        with st.expander("🔍 Research Gaps", expanded=True):
            st.markdown(st.session_state.gaps)
            st.download_button(
                label="⬇ Download Gaps",
                data=st.session_state.gaps,
                file_name="research_gaps.md",
                mime="text/markdown"
            )

        # ── Novel Hypotheses ──────────────────────────────────────────────────
        with st.expander("💡 Novel Hypotheses", expanded=True):
            st.markdown(st.session_state.hypotheses)
            st.download_button(
                label="⬇ Download Hypotheses",
                data=st.session_state.hypotheses,
                file_name="novel_hypotheses.md",
                mime="text/markdown"
            )

        # ── References ────────────────────────────────────────────────────────
        with st.expander("📚 References", expanded=False):
            for i, paper in enumerate(st.session_state.refs, 1):
                url = paper.get('url', '')
                link = f' — [↗]({url})' if url else ''
                st.markdown(f'<div class="ref-item">{i}. <b>{paper["title"]}</b> ({paper.get("year","n.d.")}){link}</div>',
                    unsafe_allow_html=True)