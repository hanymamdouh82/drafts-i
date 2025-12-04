# Ingestion Config -----------------------------------------------------------
# NOTES_DIR = "/mnt/repos/amms/amms_project_docs/"
NOTES_DIR = "/mnt/repos/drafts/"
EMBED_MODEL_NAME = "thenlper/gte-small"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
EMBED_BATCH_SIZE = 32
# -----------------------------------------------------------------------------
# Query Config ----------------------------------------------------------------
DB_PATH = "/mnt/repos/misc/drafts-i/notes.db"
EMBED_MODEL_NAME = "thenlper/gte-small"
# EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
MAX_CONTEXT = 8000

LLAMA_CLI_PATH = "/mnt/repos/llama.cpp/build/bin/llama-cli"
LLAMA_MODEL_PATH = "/mnt/extra/models/gguf/qwen/Qwen2.5-3B-Instruct-Q4_K_M-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf"
LLAMA_CTX = "8192"
# -----------------------------------------------------------------------------
