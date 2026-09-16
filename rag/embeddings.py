"""
embeddings.py
Manages ChromaDB vector store creation and retrieval using Ollama embeddings.
Supports dynamic embedding model selection for comparison.
"""

import hashlib
import os
from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rag.chain import DEFAULT_EMBED_MODEL

# Directory where ChromaDB persists data between sessions
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")


def get_embeddings(embed_model: str = DEFAULT_EMBED_MODEL) -> OllamaEmbeddings:
    """Return an Ollama embeddings instance for the given model."""
    return OllamaEmbeddings(model=embed_model)


def get_collection_name(filename: str, embed_model: str = DEFAULT_EMBED_MODEL) -> str:
    """
    Generate a unique, ChromaDB-safe collection name from a filename + embedding model.
    Each embedding model gets its own separate collection (different vector spaces).
    ChromaDB collection names must be 3-63 chars, alphanumeric + underscore/hyphen.
    """
    combined = f"{filename}::{embed_model}"
    name_hash = hashlib.md5(combined.encode()).hexdigest()[:8]
    base = os.path.splitext(filename)[0]
    base = "".join(c if c.isalnum() else "_" for c in base)[:20]
    # Shorten embed model name for readability
    model_short = embed_model.replace("-", "_").replace(".", "_")[:12]
    return f"doc_{base}_{model_short}_{name_hash}"


def create_vectorstore(
    chunks: List[Document],
    collection_name: str,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> Chroma:
    """
    Embed documents and store in ChromaDB (persisted to disk).

    Args:
        chunks: List of Document chunks to embed.
        collection_name: Unique name for this document's collection.
        embed_model: Ollama embedding model name to use.

    Returns:
        A Chroma vectorstore instance ready for retrieval.
    """
    embeddings = get_embeddings(embed_model)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vectorstore


def load_vectorstore(collection_name: str, embed_model: str = DEFAULT_EMBED_MODEL) -> Chroma:
    """
    Load an already-persisted ChromaDB collection (skip re-embedding).

    Args:
        collection_name: The collection name used when it was created.
        embed_model: Ollama embedding model name used during creation.

    Returns:
        A Chroma vectorstore instance.
    """
    embeddings = get_embeddings(embed_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vectorstore


def collection_exists(collection_name: str) -> bool:
    """Check if a collection already exists in the persisted ChromaDB."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        existing = [col.name for col in client.list_collections()]
        return collection_name in existing
    except Exception:
        return False
