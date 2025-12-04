#!/usr/bin/env python3
"""
query.py

Query sqlite-vec vec0 database using NumPy float32 embeddings,
and generate final answers using llama-cli (no server required).
"""

import sqlite3
import sqlite_vec
import numpy as np
import subprocess
from drafts_i.config import (
    DB_PATH,
    EMBED_MODEL_NAME,
    TOP_K,
    MAX_CONTEXT,
    LLAMA_CLI_PATH,
    LLAMA_MODEL_PATH,
    LLAMA_CTX,
)


def open_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def embed_query(model, question: str):
    vec = model.encode([question], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)


def ann_search(conn, query_vec, k: int):
    sql = f"""
        WITH knn AS (
            SELECT rowid, distance
            FROM chunks_index
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT {k}
        )
        SELECT c.text, c.file_path, c.chunk_index, knn.distance
        FROM knn
        JOIN chunks AS c ON c.id = knn.rowid
        ORDER BY knn.distance ASC;
    """
    cursor = conn.execute(sql, (query_vec,))
    return cursor.fetchall()


def build_context(chunks):
    ctx = ""
    for text, fp, idx, dist in chunks:
        entry = f"[{fp} / chunk {idx} / dist={dist:.4f}]\n{text}\n\n"
        if len(ctx) + len(entry) > MAX_CONTEXT:
            break
        ctx += entry
    return ctx


def ask_llama_cli(context, question):
    prompt = (
        "You are a factual assistant. Use ONLY the provided context. "
        "If the answer is not found, say: 'Not found in context.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:\n"
    )

    result = subprocess.run(
        [
            LLAMA_CLI_PATH,
            "-m",
            LLAMA_MODEL_PATH,
            "-c",
            LLAMA_CTX,
            "-p",
            prompt,
            "--temp",
            "0.0",
            "--threads",
            "4",
            "--threads-batch",
            "2",
            "--gpu-layers",
            "20",
            "--no-display-prompt",
            "--simple-io",
            "-st",
        ],
        capture_output=True,
        text=True,
    )

    return "\n" + result.stdout.strip()


def query():
    question = input("Ask your question: ").strip()
    if not question:
        print("Empty question.")
        return

    # print("Loading embedding model...")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # print("Embedding query...")
    q_vec = embed_query(embedder, question)

    # print("Opening DB...")
    conn = open_conn()

    # print("Performing ANN search...")
    hits = ann_search(conn, q_vec, TOP_K)

    if not hits:
        print("No results.")
        return

    # print("\n=== Retrieved chunks ===")
    # for text, fp, idx, dist in hits:
    #     print(f"[{fp} | chunk {idx} | dist={dist:.4f}]")
    #     print(text[:200] + "...\n")

    context = build_context(hits)

    # print("\n=== Running llama-cli ===")
    answer = ask_llama_cli(context, question)

    # print("\n=== FINAL ANSWER ===")
    print(answer)


def ask(question: str):
    if not question:
        print("Empty question.")
        return

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    q_vec = embed_query(embedder, question)

    conn = open_conn()

    hits = ann_search(conn, q_vec, TOP_K)

    if not hits:
        print("No results.")
        return

    context = build_context(hits)

    answer = ask_llama_cli(context, question)

    print(answer)
