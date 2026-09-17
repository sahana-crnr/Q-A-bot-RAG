"""
app.py — Document Intelligence Assistant
Clean, professional UI for asking questions about uploaded documents.
"""

import streamlit as st
from rag.document_processor import load_and_split_pdf, get_document_stats
from rag.embeddings import create_vectorstore, load_vectorstore, get_collection_name, collection_exists
from rag.chain import (
    build_rag_chain, ask_with_timing, get_sources,
    AVAILABLE_LLM_MODELS, AVAILABLE_EMBED_MODELS,
    DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL,
)

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
    div[data-testid="stChatInput"] {
        border-top: 1px solid #e5e7eb;
        padding-top: 12px;
    }
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
EMBED_DISPLAY = {
    "nomic-embed-text":  ("Balanced",   "Best for most documents"),
    "mxbai-embed-large": ("Thorough",   "Higher precision, slower"),
    "all-minilm":        ("Fast",       "Quickest search, lighter"),
}

# Build option lists
LLM_OPTIONS   = {v[0]: k for k, v in LLM_DISPLAY.items()}   # Display name → model id
EMBED_OPTIONS = {v[0]: k for k, v in EMBED_DISPLAY.items()}  # Display name → model id


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
        "Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )

    # ── Search precision (embedding model) ───────────────────────────────────
    st.markdown('<div class="section-label">Search Precision</div>', unsafe_allow_html=True)
    embed_choice = st.radio(
        "Search mode",
        options=list(EMBED_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
        help="Controls how the document is indexed. 'Balanced' works great for most cases.",
    )
    selected_embed_model = EMBED_OPTIONS[embed_choice]
    embed_hint = EMBED_DISPLAY[selected_embed_model][1]
    st.caption(f"↳ {embed_hint}")

    # ── Process button ────────────────────────────────────────────────────────
    if uploaded_file:
        is_new = (
            st.session_state.current_doc != uploaded_file.name
            or st.session_state.selected_embed_model != selected_embed_model
        )
        if is_new:
            if st.button("Analyze Document →", type="primary", use_container_width=True):
                collection_name = get_collection_name(uploaded_file.name, selected_embed_model)
                with st.spinner("Reading document…"):
                    chunks = load_and_split_pdf(uploaded_file)
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
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-card-name">📄 {st.session_state.current_doc}</div>
            <div class="doc-card-meta">{s.get('total_pages','–')} pages · {s.get('total_chunks','–')} sections · {s.get('total_characters',0):,} characters</div>
        </div>
        """, unsafe_allow_html=True)

    # ── AI Model ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">AI Model</div>', unsafe_allow_html=True)
    compare_mode = st.toggle("Compare two models", value=False)
    st.session_state.compare_mode = compare_mode

    llm_display_names = list(LLM_OPTIONS.keys())

    if compare_mode:
        selected_labels = st.multiselect(
            "Pick two models to compare",
            options=llm_display_names,
            default=llm_display_names[:2],
            max_selections=2,
            label_visibility="collapsed",
        )
        if len(selected_labels) != 2:
            st.warning("Select exactly 2 models to compare.")
        selected_llm_models = [LLM_OPTIONS[l] for l in selected_labels]
    else:
        selected_label = st.selectbox(
            "Model",
            options=llm_display_names,
            index=0,
            label_visibility="collapsed",
        )
        hint = LLM_DISPLAY[LLM_OPTIONS[selected_label]][1]
        st.caption(f"↳ {hint}")
        selected_llm_models = [LLM_OPTIONS[selected_label]]

    st.session_state.selected_llm_models = selected_llm_models

    # ── Clear ─────────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ===========================================================================
# MAIN AREA
# ===========================================================================

# ── Helper: render sources as pills ─────────────────────────────────────────
def render_sources(sources):
    if not sources:
        return
    pills = "".join(
        f'<span class="source-pill">📄 Page {s["page"]}</span>'
        for s in sources
    )
    st.markdown(f'<div style="margin-top:8px">{pills}</div>', unsafe_allow_html=True)


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
if st.session_state.vectorstore is None:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">✦</div>
        <div class="welcome-title">Ask anything about your document</div>
        <div class="welcome-sub">
            Upload a PDF on the left, then ask questions in plain English.
            Get precise answers with exact page references.
        </div>
        <div class="step-list">
            <div class="step-item"><span class="step-num">1</span> Upload a PDF document</div>
            <div class="step-item"><span class="step-num">2</span> Click "Analyze Document"</div>
            <div class="step-item"><span class="step-num">3</span> Ask questions naturally</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**You can ask things like…**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info('"What are the return policy details?"')
    with c2:
        st.info('"Summarize the key findings on page 5."')
    with c3:
        st.info('"What vegetarian options are available?"')

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

    # ── Chat input ───────────────────────────────────────────────────────────
    models = st.session_state.selected_llm_models
    question = st.chat_input("Ask a question about your document…")

    if question:
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
            # ── Single model ─────────────────────────────────────────────────
            model_id    = models[0]
            model_label = LLM_DISPLAY.get(model_id, (model_id,))[0]
            with st.spinner(f"{model_label} is thinking…"):
                try:
                    chain  = build_rag_chain(vs, model_id)
                    result = ask_with_timing(chain, question)
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
                except Exception as e:
                    st.error(f"Something went wrong. Make sure Ollama is running and the model is downloaded.\n\n`{e}`")

        else:
            # ── Side-by-side comparison ───────────────────────────────────────
            # Step 1: Collect both results first (sequential, under one spinner)
            results = []
            model_labels = [LLM_DISPLAY.get(m, (m,))[0] for m in models]
            status_text  = st.empty()

            for i, model_id in enumerate(models):
                label = model_labels[i]
                status_text.markdown(
                    f"⏳ Running **{label}** ({i+1} of {len(models)})…"
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

            status_text.empty()  # Clear the status line

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
