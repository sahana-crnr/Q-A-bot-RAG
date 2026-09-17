"""
chain.py
Builds the LangChain RAG chain using LCEL (LangChain Expression Language).
Retrieves relevant document chunks and generates grounded answers.
Supports dynamic model selection for multi-model comparison.
"""

import time
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma

# Number of relevant chunks to retrieve per query
TOP_K = 4

# Available LLM models via Ollama — (id, display name, description)
AVAILABLE_LLM_MODELS = [
    ("llama3.2",   "LLaMA 3.2 (3B)",    "Meta's fast, compact model. Great all-rounder."),
    ("mistral",    "Mistral 7B",         "Strong reasoning, excellent for document Q&A."),
    ("gemma2",     "Gemma 2 (9B)",       "Google's model. Excellent instruction following."),
    ("phi3",       "Phi-3 Mini (3.8B)",  "Microsoft's tiny but surprisingly capable model."),
    ("qwen2.5",    "Qwen 2.5 (7B)",      "Alibaba's model. Strong multilingual support."),
]

# Available embedding models — (id, display name, description)
AVAILABLE_EMBED_MODELS = [
    ("nomic-embed-text",  "Nomic Embed Text",    "Best balance of speed & quality. Recommended."),
    ("mxbai-embed-large", "MxBai Embed Large",   "Higher accuracy, slightly slower."),
    ("all-minilm",        "All-MiniLM (L6-v2)",  "Fastest & smallest. Good for quick testing."),
]

DEFAULT_LLM_MODEL   = "llama3.2"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based strictly on the provided document context.

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer ONLY based on the context provided above.
- If the answer is not found in the context, say "I couldn't find this information in the uploaded document."
- Be concise and clear.
- If relevant, mention which part of the document supports your answer.

ANSWER:"""

PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


# ---------------------------------------------------------------------------
# Helper to format retrieved docs
# ---------------------------------------------------------------------------
def format_docs(docs):
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        location = doc.metadata.get("location")
        if not location:
            page = doc.metadata.get("page", "?")
            location = f"Page {page + 1}" if isinstance(page, int) else f"Page {page}"
        formatted.append(f"[Chunk {i} | {location}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# ---------------------------------------------------------------------------
# Build the RAG chain (dynamic model)
# ---------------------------------------------------------------------------
def build_rag_chain(vectorstore: Chroma, llm_model: str = DEFAULT_LLM_MODEL):
    """
    Build a RAG chain from a vectorstore with a specific LLM model.

    Args:
        vectorstore: ChromaDB vectorstore instance.
        llm_model: Ollama model name (e.g. "llama3.2", "mistral", "gemma2").

    Returns:
        A callable LCEL chain.
    """
    llm = ChatOllama(model=llm_model, temperature=0.1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def ask_with_timing(chain, question: str) -> dict:
    """
    Invoke a RAG chain and measure response time.

    Returns:
        dict with 'answer' (str) and 'elapsed_seconds' (float).
    """
    start = time.perf_counter()
    answer = chain.invoke(question)
    elapsed = round(time.perf_counter() - start, 2)
    return {"answer": answer, "elapsed_seconds": elapsed}


def get_sources(vectorstore: Chroma, query: str) -> list:
    """
    Retrieve source chunks for a query (used to display citations).

    Returns:
        List of dicts with location, page number, and content snippet.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(query)
    sources = []
    seen_locations = set()
    for doc in docs:
        location = doc.metadata.get("location")
        page = doc.metadata.get("page", 1)
        if not location:
            page_display = page + 1 if isinstance(page, int) else page
            location = f"Page {page_display}"

        if location not in seen_locations:
            seen_locations.add(location)
            sources.append({
                "page": page,
                "location": location,
                "snippet": doc.page_content[:160].strip() + "...",
                "filename": doc.metadata.get("source_filename", "document"),
            })
    return sources
