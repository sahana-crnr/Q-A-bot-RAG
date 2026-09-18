"""
app.py — Document Intelligence Assistant
Clean, professional UI for asking questions about uploaded documents.
"""

import os
import streamlit as st
from rag.document_processor import (
    load_and_split_document,
    load_and_split_pdf,
    get_document_stats,
    SUPPORTED_EXTENSIONS,
)
from rag.embeddings import create_vectorstore, load_vectorstore, get_collection_name, collection_exists
from rag.chain import (
    build_rag_chain, ask_with_timing, get_sources,
    DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL,
)
from rag.history import save_session, load_session, list_sessions, delete_session, format_saved_at

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clarity — Document Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Professional, clean design
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit header/footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Main background */
    .stApp {
        background: #f9fafb;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #6b7280;
        font-size: 0.82rem;
    }

    /* ── App header ── */
    .app-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 20px 0 16px 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 24px;
    }
    .app-logo {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        color: white;
        font-weight: 700;
        padding: 6px;
        text-align: center;
        line-height: 1;
    }
    .app-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.78rem;
        color: #9ca3af;
        margin: 0;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 20px 0 8px 0;
    }

    /* ── Document card ── */
    .doc-card {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }
    .doc-card-name {
        font-weight: 600;
        font-size: 0.88rem;
        color: #111827;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .doc-card-meta {
        font-size: 0.76rem;
        color: #6b7280;
        margin-top: 3px;
    }

    /* ── Stat grid ── */
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin: 10px 0;
    }
    .stat-item {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
    }
    .stat-label {
        font-size: 0.68rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    /* ── Model badge ── */
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #ede9fe;
        color: #5b21b6;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-bottom: 6px;
    }

    /* ── Chat bubbles ── */
    .chat-user-wrap {
        display: flex;
        justify-content: flex-end;
        margin: 16px 0 8px 60px;
    }
    .chat-user-bubble {
        background: #6366f1;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        font-size: 0.9rem;
        line-height: 1.5;
        max-width: 100%;
    }
    .chat-ai-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }
    .chat-ai-avatar {
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 13px;
        flex-shrink: 0;
    }
    .chat-ai-name {
        font-size: 0.78rem;
        font-weight: 600;
        color: #374151;
    }
    .chat-ai-time {
        font-size: 0.72rem;
        color: #9ca3af;
        margin-left: 4px;
    }
    .chat-ai-bubble {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 14px 16px;
        border-radius: 4px 18px 18px 18px;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #111827;
        margin-left: 36px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .faster-tag {
        background: #d1fae5;
        color: #065f46;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 1px 7px;
        border-radius: 20px;
        margin-left: 4px;
    }

    /* ── Typing indicator ── */
    .typing-wrap {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin: 12px 0;
    }
    .typing-avatar {
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 13px;
        flex-shrink: 0;
    }
    .typing-bubble {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 4px 18px 18px 18px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .typing-label {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-right: 4px;
    }
    .dot {
        width: 7px;
        height: 7px;
        background: #6366f1;
        border-radius: 50%;
        animation: bounce 1.2s infinite ease-in-out;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
        40%            { transform: scale(1.1); opacity: 1;   }
    }

    /* ── Compare column header ── */
    .compare-col-head {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }
    .compare-col-model {
        font-weight: 600;
        font-size: 0.9rem;
        color: #111827;
    }
    .compare-col-speed {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 2px;
    }

    /* ── Source pills ── */
    .source-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.72rem;
        font-weight: 500;
        margin: 3px 3px 0 0;
    }

    /* ── Welcome card ── */
    .welcome-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin: 30px auto;
        max-width: 540px;
    }
    .welcome-icon {
        font-size: 48px;
        margin-bottom: 16px;
    }
    .welcome-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 8px;
    }
    .welcome-sub {
        font-size: 0.9rem;
        color: #6b7280;
        line-height: 1.6;
    }
    .step-list {
        text-align: left;
        margin: 20px auto;
        max-width: 300px;
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px 0;
        font-size: 0.88rem;
        color: #374151;
    }
    .step-num {
        width: 22px;
        height: 22px;
        background: #6366f1;
        color: white;
        border-radius: 50%;
        font-size: 0.72rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    /* Streamlit overrides */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stSelectbox label, .stMultiselect label, .stFileUploader label {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }

    /* ── Bottom Fixed Input Area & Integrated Model Selector ── */
    div[data-testid="stBottom"] {
        background: linear-gradient(180deg, rgba(249,250,251,0) 0%, #f9fafb 25%) !important;
        padding-top: 10px !important;
        padding-bottom: 20px !important;
    }
    .prompt-toolbar {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-bottom: 1px dashed #e5e7eb;
        border-radius: 16px 16px 0 0;
        padding: 5px 14px 2px 14px;
        margin-bottom: 0px !important;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.02);
    }
    div[data-testid="stChatInput"] {
        border-top: none !important;
        padding-top: 0px !important;
    }
    div[data-testid="stChatInput"] > div {
        border-top: none !important;
        border-top-left-radius: 0px !important;
        border-top-right-radius: 0px !important;
        border-color: #d1d5db !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
    }
    div[data-testid="stChatInput"] textarea {
        padding-top: 6px !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Friendly model name mappings (hide technical names)
# ---------------------------------------------------------------------------
LLM_DISPLAY = {
    "llama3.2":  ("LLaMA 3.2",  "Fast & efficient · 3B"),
    "mistral":   ("Mistral",    "Precise & thorough · 7B"),
    "gemma2":    ("Gemma 2",    "Google · 9B"),
    "phi3":      ("Phi-3 Mini", "Microsoft · 3.8B"),
    "qwen2.5":   ("Qwen 2.5",  "Multilingual · 7B"),
}

# Build LLM option list: Display name → model id
LLM_OPTIONS = {v[0]: k for k, v in LLM_DISPLAY.items()}

# Rich model names formatted for prompt dropdown
LLM_FORMATTED_NAMES = {
    "llama3.2": "✦ LLaMA 3.2 (3B · Fast & efficient)",
    "mistral":  "✦ Mistral (7B · Best reasoning & accuracy)",
    "phi3":     "✦ Phi-3 Mini (3.8B · Compact & quick)",
    "qwen2.5":  "✦ Qwen 2.5 (7B · Multilingual)",
    "gemma2":   "✦ Gemma 2 (9B · Google)",
}
LLM_FORMATTED_TO_ID = {v: k for k, v in LLM_FORMATTED_NAMES.items()}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "chat_history": [],
    "vectorstore": None,
    "current_doc": None,
    "doc_stats": None,
    "selected_embed_model": DEFAULT_EMBED_MODEL,
    "selected_llm_models": [DEFAULT_LLM_MODEL],
    "compare_mode": False,
    "session_id": None,       # Tracks the current save file
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    # ── Brand ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <div class="app-logo">✦</div>
        <div>
            <div class="app-title">Clarity</div>
            <div class="app-subtitle">Document Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Your Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a file here",
        type=SUPPORTED_EXTENSIONS,
        help="Supported formats: PDF, Word (.docx), Excel (.xlsx, .xls), CSV, Text (.txt, .md), JSON",
        label_visibility="collapsed",
    )
    st.caption("Supports **PDF, Word, Excel, CSV, Text, JSON**")

    # ── Process button ────────────────────────────────────────────────────────
    # Embedding model is fixed to nomic-embed-text (best default, no config needed)
    selected_embed_model = DEFAULT_EMBED_MODEL

    if uploaded_file:
        is_new = (
            st.session_state.current_doc != uploaded_file.name
        )
        if is_new:
            if st.button("Analyze Document →", type="primary", use_container_width=True):
                collection_name = get_collection_name(uploaded_file.name, selected_embed_model)
                with st.spinner("Reading & parsing document…"):
                    chunks = load_and_split_document(uploaded_file)
                    stats  = get_document_stats(chunks)

                if collection_exists(collection_name):
                    with st.spinner("Loading…"):
                        vectorstore = load_vectorstore(collection_name, selected_embed_model)
                else:
                    with st.spinner("Indexing document… (first time only)"):
                        vectorstore = create_vectorstore(chunks, collection_name, selected_embed_model)

                st.session_state.vectorstore         = vectorstore
                st.session_state.current_doc         = uploaded_file.name
                st.session_state.selected_embed_model = selected_embed_model
                st.session_state.doc_stats           = stats
                st.session_state.chat_history        = []
                st.rerun()

    # ── Document info card ────────────────────────────────────────────────────
    if st.session_state.current_doc:
        s = st.session_state.doc_stats or {}
        ext = os.path.splitext(st.session_state.current_doc)[1].lower()
        icon = {
            ".pdf": "📕",
            ".docx": "📘",
            ".txt": "📄",
            ".md": "📝",
            ".csv": "📊",
            ".xlsx": "📈",
            ".xls": "📈",
            ".json": "🗂️",
        }.get(ext, "📄")
        unit_label = s.get("unit_label", "Sections")
        unit_count = s.get("total_units", s.get("total_pages", "–"))
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-card-name">{icon} {st.session_state.current_doc}</div>
            <div class="doc-card-meta">{unit_count} {unit_label} · {s.get('total_chunks', '–')} chunks · {s.get('total_characters', 0):,} characters</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Current conversation controls ─────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("---")
        col_new, col_clear = st.columns(2)
        with col_new:
            if st.button("＋ New", use_container_width=True):
                # Save current before clearing
                if st.session_state.current_doc:
                    save_session(
                        st.session_state.current_doc,
                        st.session_state.chat_history,
                        st.session_state.session_id,
                    )
                st.session_state.chat_history = []
                st.session_state.session_id   = None
                st.rerun()
        with col_clear:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.session_id   = None
                st.rerun()

    # ── Past Conversations ────────────────────────────────────────────────────
    sessions = list_sessions()
    if sessions:
        st.markdown("---")
        st.markdown('<div class="section-label">Past Conversations</div>', unsafe_allow_html=True)

        for s in sessions[:8]:  # Show max 8 recent sessions
            sid        = s["session_id"]
            doc_short  = os.path.splitext(s["document_name"])[0][:22]
            time_label = format_saved_at(s["saved_at"])
            msg_count  = s["message_count"]
            is_active  = sid == st.session_state.session_id

            # Highlight active session
            border = "2px solid #6366f1" if is_active else "1px solid #e5e7eb"
            ext = os.path.splitext(s["document_name"])[1].lower()
            icon = {
                ".pdf": "📕",
                ".docx": "📘",
                ".txt": "📄",
                ".md": "📝",
                ".csv": "📊",
                ".xlsx": "📈",
                ".xls": "📈",
                ".json": "🗂️",
            }.get(ext, "📄")
            st.markdown(f"""
            <div style="background:{'#f5f3ff' if is_active else '#f9fafb'};
                        border:{border}; border-radius:8px;
                        padding:8px 10px; margin-bottom:6px;">
                <div style="font-weight:600;font-size:0.82rem;color:#111827;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                    {icon} {doc_short}
                </div>
                <div style="font-size:0.72rem;color:#9ca3af;margin-top:2px">
                    {time_label} · {msg_count} question{"s" if msg_count != 1 else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                if st.button("Load", key=f"load_{sid}", use_container_width=True):
                    data = load_session(sid)
                    if data:
                        doc_name = data.get("document_name", "")
                        # Restore chat history
                        st.session_state.chat_history = data["messages"]
                        st.session_state.session_id   = sid
                        st.session_state.current_doc  = doc_name

                        # Restore vectorstore from ChromaDB if it was indexed before
                        collection_name = get_collection_name(doc_name, DEFAULT_EMBED_MODEL)
                        if collection_exists(collection_name):
                            try:
                                st.session_state.vectorstore = load_vectorstore(
                                    collection_name, DEFAULT_EMBED_MODEL
                                )
                            except Exception:
                                st.session_state.vectorstore = None
                        else:
                            # Document not indexed on this machine — chat is read-only
                            st.session_state.vectorstore = None

                        st.rerun()
            with btn_col2:
                if st.button("✕", key=f"del_{sid}", use_container_width=True):
                    delete_session(sid)
                    if st.session_state.session_id == sid:
                        st.session_state.chat_history = []
                        st.session_state.session_id   = None
                    st.rerun()


# ===========================================================================
# MAIN AREA
# ===========================================================================

# ── Helper: render sources as pills ─────────────────────────────────────────
def render_sources(sources):
    if not sources:
        return
    pills = []
    for s in sources:
        loc = s.get("location") or f"Page {s.get('page', 1)}"
        pills.append(f'<span class="source-pill">📌 {loc}</span>')
    st.markdown(f'<div style="margin-top:8px">{"".join(pills)}</div>', unsafe_allow_html=True)


# ── Helper: typing indicator ─────────────────────────────────────────────────
def show_typing_indicator(placeholder, label: str = ""):
    """Show animated 3-dot typing bubble in a st.empty() placeholder."""
    placeholder.markdown(f"""
    <div class="typing-wrap">
        <div class="typing-avatar">✦</div>
        <div class="typing-bubble">
            <span class="typing-label">{label}</span>
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Helper: render one AI response block ────────────────────────────────────
def render_ai_response(model_id, answer, elapsed, sources, faster=False):
    label = LLM_DISPLAY.get(model_id, (model_id,))[0]
    faster_tag = '<span class="faster-tag">Faster</span>' if faster else ""
    st.markdown(f"""
    <div class="chat-ai-header">
        <div class="chat-ai-avatar">✦</div>
        <span class="chat-ai-name">{label}</span>{faster_tag}
        <span class="chat-ai-time">⏱ {elapsed}s</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="chat-ai-bubble">{answer}</div>', unsafe_allow_html=True)
    render_sources(sources)


# ── No document loaded → Welcome screen ─────────────────────────────────────
# ── Welcome screen or Chat History ──────────────────────────────────────────
if st.session_state.vectorstore is None and not st.session_state.chat_history:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">✦</div>
        <div class="welcome-title">Ask anything across any document</div>
        <div class="welcome-sub">
            Upload any <b>PDF, Word (.docx), Excel, CSV, Text, or JSON</b> file on the left.
            Ask questions in plain English and get grounded answers with exact source citations.
        </div>
        <div class="step-list">
            <div class="step-item"><span class="step-num">1</span> Upload document or dataset</div>
            <div class="step-item"><span class="step-num">2</span> Click "Analyze Document"</div>
            <div class="step-item"><span class="step-num">3</span> Ask questions & compare models</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Test across different types of data & domains:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info('📄 **Formal SOW / Contracts:**\n*"What are the project deliverables & payment milestones?"*')
    with c2:
        st.info('📊 **Spreadsheets / CSV Catalogs:**\n*"Which shoe model costs less than $100 and is in stock?"*')
    with c3:
        st.info('🍽️ **Menus & Policies:**\n*"What vegan dishes are listed, and what is the return policy?"*')

else:
    # ── Render chat history ──────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-user-wrap">
                <div class="chat-user-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            responses = msg.get("responses", [])
            if len(responses) == 1:
                r = responses[0]
                render_ai_response(r["model_id"], r["answer"], r["elapsed"], r.get("sources", []))
            elif len(responses) == 2:
                col1, col2 = st.columns(2)
                for col, r in zip([col1, col2], responses):
                    with col:
                        render_ai_response(
                            r["model_id"], r["answer"], r["elapsed"],
                            r.get("sources", []), faster=r.get("is_faster", False)
                        )
            st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    if st.session_state.vectorstore is None and st.session_state.chat_history:
        st.info(
            f"📂 You're viewing a past conversation for **{st.session_state.current_doc}**. "
            "To ask new questions, upload and re-analyze the same document from the sidebar."
        )

# ── Bottom Fixed Question Box & Model Selector (ALWAYS VISIBLE) ─────────────
has_doc = st.session_state.vectorstore is not None

with st.bottom:
    llm_display_choices = list(LLM_FORMATTED_NAMES.values())

    st.markdown('<div class="prompt-toolbar">', unsafe_allow_html=True)
    t_col1, t_col2 = st.columns([3, 1])

    with t_col2:
        compare_mode = st.toggle("⚡ Compare 2 models", value=st.session_state.compare_mode, key="prompt_compare_toggle")
        st.session_state.compare_mode = compare_mode

    with t_col1:
        if compare_mode:
            default_labels = [LLM_FORMATTED_NAMES.get(m, llm_display_choices[0]) for m in st.session_state.selected_llm_models]
            if len(default_labels) < 2:
                default_labels = llm_display_choices[:2]
            elif len(default_labels) > 2:
                default_labels = default_labels[:2]

            selected_labels = st.multiselect(
                "Pick 2 models to compare",
                options=llm_display_choices,
                default=default_labels,
                max_selections=2,
                label_visibility="collapsed",
                placeholder="Choose 2 models to compare side-by-side…",
                key="prompt_models_multi",
            )
            if len(selected_labels) != 2:
                st.caption("⚠️ *Select exactly 2 models to compare.*")
            st.session_state.selected_llm_models = [LLM_FORMATTED_TO_ID[l] for l in selected_labels] if selected_labels else [DEFAULT_LLM_MODEL]
        else:
            cur_id = st.session_state.selected_llm_models[0] if st.session_state.selected_llm_models else DEFAULT_LLM_MODEL
            cur_fmt = LLM_FORMATTED_NAMES.get(cur_id, llm_display_choices[0])
            def_idx = llm_display_choices.index(cur_fmt) if cur_fmt in llm_display_choices else 0

            selected_label = st.selectbox(
                "Model",
                options=llm_display_choices,
                index=def_idx,
                label_visibility="collapsed",
                key="prompt_model_single",
            )
            st.session_state.selected_llm_models = [LLM_FORMATTED_TO_ID[selected_label]]

    st.markdown('</div>', unsafe_allow_html=True)

    models = st.session_state.selected_llm_models
    question = st.chat_input(
        "Ask a question about your document…" if has_doc else "Upload a document in the sidebar to start asking questions…",
        disabled=not has_doc,
    )

if question and has_doc:
    # Show user message
    st.markdown(f"""
    <div class="chat-user-wrap">
        <div class="chat-user-bubble">{question}</div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": question})

    vs      = st.session_state.vectorstore
    sources = get_sources(vs, question)

    if not st.session_state.compare_mode or len(models) == 1:
        # ── Single model ─────────────────────────────────────────────
        model_id    = models[0]
        model_label = LLM_DISPLAY.get(model_id, (model_id,))[0]

        # Show typing indicator while model generates
        typing_placeholder = st.empty()
        show_typing_indicator(typing_placeholder, model_label)

        try:
            chain  = build_rag_chain(vs, model_id)
            result = ask_with_timing(chain, question)

            # Clear typing indicator, render actual answer
            typing_placeholder.empty()
            render_ai_response(model_id, result["answer"], result["elapsed_seconds"], sources)

            st.session_state.chat_history.append({
                "role": "assistant",
                "responses": [{
                    "model_id": model_id,
                    "answer":   result["answer"],
                    "elapsed":  result["elapsed_seconds"],
                    "sources":  sources,
                }],
            })
            # Auto-save conversation to disk
            st.session_state.session_id = save_session(
                st.session_state.current_doc,
                st.session_state.chat_history,
                st.session_state.session_id,
            )
        except Exception as e:
            typing_placeholder.empty()
            st.error(f"Something went wrong. Make sure Ollama is running and the model is downloaded.\n\n`{e}`")

    else:
        # ── Side-by-side comparison ───────────────────────────────────
        # Step 1: Collect both results (sequential, with typing indicator)
        results      = []
        model_labels = [LLM_DISPLAY.get(m, (m,))[0] for m in models]
        typing_ph    = st.empty()

        for i, model_id in enumerate(models):
            label = model_labels[i]
            show_typing_indicator(
                typing_ph,
                f"{label}  ({i+1} of {len(models)})"
            )
            try:
                chain  = build_rag_chain(vs, model_id)
                result = ask_with_timing(chain, question)
                results.append({
                    "model_id": model_id,
                    "answer":   result["answer"],
                    "elapsed":  result["elapsed_seconds"],
                    "sources":  sources,
                })
            except Exception as e:
                results.append({
                    "model_id": model_id,
                    "answer":   f"⚠️ Error: Make sure Ollama is running and **{label}** is downloaded (`ollama pull {model_id}`).\n\n`{e}`",
                    "elapsed":  0,
                    "sources":  [],
                })

        typing_ph.empty()  # Clear the typing indicator

        # Step 2: Mark the faster model
        if len(results) == 2 and results[0]["elapsed"] > 0 and results[1]["elapsed"] > 0:
            idx = 0 if results[0]["elapsed"] <= results[1]["elapsed"] else 1
            results[idx]["is_faster"] = True

        # Step 3: Render both results side by side cleanly
        col1, col2 = st.columns(2)
        for col, r in zip([col1, col2], results):
            with col:
                render_ai_response(
                    r["model_id"], r["answer"], r["elapsed"],
                    r.get("sources", []), faster=r.get("is_faster", False)
                )

        # Save to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "responses": results,
        })
        # Auto-save conversation to disk
        st.session_state.session_id = save_session(
            st.session_state.current_doc,
            st.session_state.chat_history,
            st.session_state.session_id,
        )
