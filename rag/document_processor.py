"""
document_processor.py
Handles PDF loading and text chunking.
"""

import tempfile
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_split_pdf(uploaded_file, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Load a PDF from a Streamlit UploadedFile object and split it into chunks.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks (keeps context).

    Returns:
        List of LangChain Document objects with page metadata.
    """
    # PyPDFLoader needs a real file path, so we write to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()  # Each page is a Document with page metadata

        # Attach the original filename to metadata
        for page in pages:
            page.metadata["source_filename"] = uploaded_file.name

        # Split pages into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(pages)

        return chunks

    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)


def get_document_stats(chunks: List[Document]) -> dict:
    """Return basic stats about the processed document."""
    if not chunks:
        return {}

    pages = set(chunk.metadata.get("page", 0) for chunk in chunks)
    total_chars = sum(len(chunk.page_content) for chunk in chunks)

    return {
        "total_chunks": len(chunks),
        "total_pages": len(pages),
        "total_characters": total_chars,
        "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
    }

