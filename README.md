# 📚 Clarity — Document Intelligence & Multi-Model RAG

A fully **local, privacy-first, zero-API-key** RAG system. Upload documents or datasets across **multiple file formats and data types**, query them in natural English, and compare open-source AI models **side-by-side**.

**Stack:** Ollama + ChromaDB + LangChain + Streamlit

---

## 📁 Supported File Formats & Data Types

| Format | Extension | Data Type | Parsing Strategy | Source Citations |
|---|---|---|---|---|
| **PDF** | `.pdf` | Unstructured / Reports / Contracts | PyPDFLoader with text chunking | `Page X` |
| **Word** | `.docx` | Formatted specs / SOWs / Tables | Paragraph & Markdown table extraction | `Section X` |
| **Excel** | `.xlsx`, `.xls` | Tabular / Multi-sheet spreadsheets | Sheet-wise record serialization | `Sheet 'Name', Rows X-Y` |
| **CSV** | `.csv` | Tabular datasets / Catalogs | Header-preserved record rows | `Rows X-Y` |
| **Text** | `.txt`, `.md` | Documentation / Logs / Notes | Recursive character splitting | `Section X` |
| **JSON** | `.json` | Structured records / API data | Object-level record batching | `Items X-Y` |

---

## 🤖 Supported Open-Source Models

| Model | Size | Best For |
|---|---|---|
| `llama3.2` | 3B | **Fastest response** — direct factual extraction |
| `mistral` | 7B | **Best accuracy** — polished, comprehensive answers |
| `gemma2` | 9B | High instruction following (Google) |
| `phi3` | 3.8B | Compact, efficient for low-spec PCs (Microsoft) |
| `qwen2.5` | 7B | Strong multilingual & code understanding (Alibaba) |

**Embedding Model:** `nomic-embed-text` (8192-token context window, high-density vector representation).

---

## 🚀 Setup & Run (3 Steps)

### Step 1: Install Ollama
Download and run the installer from **https://ollama.ai**

### Step 2: Pull Models
```bash
# Required
ollama pull llama3.2
ollama pull nomic-embed-text

# Optional (for comparison)
ollama pull mistral
ollama pull phi3
```

### Step 3: Install Dependencies & Launch
```bash
pip install -r requirements.txt
streamlit run app.py
```
App opens at **http://localhost:8501**

---

## 🎯 Key Features

1. **Multi-Format Support:** Drag and drop PDFs, Word files, Excel sheets, CSV spreadsheets, or JSON files.
2. **Side-by-Side Model Comparison:** Compare two LLMs answering the same question simultaneously with latency timers (⏱️) and speed badges (🏆).
3. **Exact Source Attribution:** Every answer provides clickable source citations with exact page numbers, row ranges, or section numbers.
4. **Persistent Chat Sessions:** Auto-saves conversations to `chat_history/` as JSON; reload or delete anytime from the sidebar.
5. **100% Offline & Private:** Zero external API calls, zero tracking, all data stays on your local machine.

---

## 📂 Project Structure

```
Q-A-bot-RAG/
├── app.py                    # Streamlit UI with comparison & history
├── rag/
│   ├── document_processor.py # Multi-format parser (PDF, DOCX, CSV, Excel, TXT, JSON)
│   ├── embeddings.py         # ChromaDB vector store + nomic-embed-text
│   ├── chain.py              # Dynamic LCEL RAG pipeline & citations
│   └── history.py            # Local JSON chat session persistence
├── chroma_db/                # Persisted vector database (local only)
├── chat_history/             # Saved chat sessions (local only)
├── requirements.txt          # Python dependencies
└── README.md
```
