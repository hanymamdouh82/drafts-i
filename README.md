# Drafts-I

### Offline Semantic Search + Local LLM Q&A over Markdown Notes

Drafts-I is a high-performance, fully offline RAG (Retrieval-Augmented Generation) system that turns your Markdown notes into a searchable semantic knowledge base.

It embeds your notes using SentenceTransformer models, indexes them via **Qdrant**, performs fast ANN (approximate nearest-neighbor) retrieval, and feeds the relevant context to a **local LLM** powered by `llama-cli`.  
Everything runs locally, with zero external calls.

---

## 🚀 Features

- **Fast semantic search** using Qdrant ANN indexing
- **Offline embeddings** with SentenceTransformer (`gte-small` by default)
- **Local LLM answers** using `llama-cli` (Qwen2.5 recommended)
- **Markdown ingestion pipeline** with incremental updates
- **Supports hundreds of files / thousands of chunks**
- **Clean, deterministic output** with optimized llama-cli flags

---

## 🧩 Architecture Overview

Drafts-I consists of three major components:

1. **Ingestion Pipeline**
   - Recursively scans a directory of Markdown files
   - Splits documents into semantic chunks
   - Generates embeddings for each chunk
   - Stores chunks and vectors inside `Qdrant` 

2. **Semantic Retrieval Engine**
   - Converts the user query to an embedding
   - Retrieves the top-K most relevant chunks

3. **LLM Answering Layer**
   - Builds a context prompt containing retrieved chunks
   - Calls `llama-cli` with optimized flags (`--simple-io`, `--no-display-prompt`, `-st`)
   - Produces a clean, grounded answer

---

## 📦 Project Structure

```

drafts-i/
│
├── src/drafts_i/
│ ├── cli.py # Entry point (ask / ingest)
│ ├── query.py # Semantic retrieval + llama-cli interaction
│ ├── ingest.py # Markdown ingestion + embedding generation
│ ├── config.py # Paths + settings
│ └── ...
│
├── pyproject.toml
└── README.md

```

---

## 🛠 Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- `Qdrant` server (can be Docker container for portability)
- GPU optional (recommended for larger LLM models)
- Local LLM model in GGUF format (e.g., _Qwen2.5-3B-Instruct-Q4_K_M_)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/hanymamdouh82/drafts-i.git
cd drafts-i
uv sync
```

---

## 📥 Ingest Notes

Point `NOTES_DIR` in `config.py` to your markdown directory, then run:

```bash
uv run cli ingest
```

This will:

- scan all `.md` files
- chunk them
- embed them
- insert vectors into Qdrant

Re-running it will update only changed files based on modification time.

---

## 🔍 Ask Questions

To retrieve relevant notes and generate an answer:

```bash
uv run cli ask
```

or simply:

```
uv run cli
```

Example:

```
Ask your question: What is FleetFix?
```

You’ll get a clean, contextual answer from your notes powered by your local LLM.

---

## ⚡ Performance Notes

- Vectors strored in Qdrant for fast cosine similarty search
- llama-cli flags are tuned for deterministic, clean output

---

## 🧪 Advanced Usage

You can experiment with different components:

### Change Embedding Model

Edit in `query.py` and `ingest.py`:

```python
SentenceTransformer("thenlper/gte-small")
```

Recommended alternatives:

- `all-MiniLM-L6-v2` (fastest CPU)
- `gte-base` (better accuracy)
- `mxbai-embed-large` (high accuracy, slower)

### Change LLM Model

Edit `LLAMA_MODEL_PATH` in `config.py`.

Models Tested:

- Qwen2.5 (Best one)
- Phi-3 GGUF
- Llama-3-Instruct
- Mistral (requires high CPU and GPU)

---

## 🧿 Why Drafts-I?

Most note-search tools depend on:

- cloud embedding APIs
- proprietary vector databases
- server processes
- complex infrastructure

Drafts-I is:

- simple
- portable
- fully offline
- blazing fast
- easy to extend

It’s designed to be a **developer’s personal knowledge engine**.

---

## 📄 License

MIT License. Use freely.

---

## 🙌 Author

Created by **Hany Mamdouh**
A technologist, builder, and diver who likes his tools **fast, offline, and under his control**.
