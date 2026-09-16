# 📚 RAG Q&A Bot — Multi-Model Comparison

A fully **local, free, no-API-key** RAG application. Upload any PDF, pick open-source models, and compare answers **side-by-side**.

**Stack:** Ollama + ChromaDB + LangChain + Streamlit

---

## 🤖 Supported Open-Source Models

### LLM Models (for answer generation)
| Model | Size | Best For |
|---|---|---|
| `llama3.2` | 3B | Fast, great all-rounder |
| `mistral` | 7B | Strong document Q&A reasoning |
| `gemma2` | 9B | Excellent instruction following (Google) |
| `phi3` | 3.8B | Tiny but surprisingly capable (Microsoft) |
| `qwen2.5` | 7B | Strong multilingual support (Alibaba) |

### Embedding Models (for vector search)
| Model | Notes |
|---|---|
| `nomic-embed-text` | Best speed/quality balance — recommended |
| `mxbai-embed-large` | Higher accuracy, slightly slower |
| `all-minilm` | Fastest & smallest |

---

## 🚀 Setup (3 steps)

### Step 1: Install Ollama
Download from **https://ollama.ai** (Windows installer)

### Step 2: Pull models
```bash
# Required (default)
ollama pull llama3.2
ollama pull nomic-embed-text

# Optional — pull more to compare
ollama pull mistral
ollama pull gemma2
ollama pull phi3
ollama pull qwen2.5
ollama pull mxbai-embed-large
ollama pull all-minilm
```

### Step 3: Install dependencies & run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## 🎯 How to Use

### Single Model Mode
1. Upload PDF → Process → Ask questions → Get answers with source citations

### Comparison Mode ⚡
1. Toggle **"Side-by-side comparison mode"** in the sidebar
2. Select **2 LLM models**
3. Ask a question → Both models answer simultaneously
4. See which is **faster** (🏆) and compare answer quality

---

##  What Gets Compared

| Metric | How to observe |
|---|---|
| **Answer quality** | Read both responses side by side |
| **Speed** | Response time shown in seconds (⏱️), faster model marked 🏆 |
| **Embedding quality** | Re-process same PDF with different embedding model, ask same question |

---

## 📂 Project Structure

```
Q-A-bot-RAG/
├── app.py                    # Streamlit UI with comparison mode
├── rag/
│   ├── document_processor.py # PDF loading + chunking
│   ├── embeddings.py         # ChromaDB + dynamic embedding model
│   └── chain.py              # RAG chain + model list + timing
├── chroma_db/                # Persisted embeddings (auto-created)
├── requirements.txt
└── README.md
```

---

## 🔒 Privacy
Everything runs **100% locally**. No data sent anywhere.
