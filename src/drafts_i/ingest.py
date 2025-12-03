#!/usr/bin/env python3
"""
ingest.py

Ingest markdown notes into sqlite-vec powered SQLite DB.

- Uses sqlite-vec vec0 virtual table
- Embeddings stored as NumPy float32 arrays (buffer protocol → BLOB)
- Chunking via llama-index
- Fast incremental ingestion using file mtime
"""

import sqlite3
import sqlite_vec
import numpy as np
import os
import time
from pathlib import Path
from hashlib import sha256
from sentence_transformers import SentenceTransformer
from llama_index.readers.file import MarkdownReader
from llama_index.core.node_parser import SimpleNodeParser
from drafts_i.config import (
    DB_PATH,
    EMBED_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_BATCH_SIZE,
    NOTES_DIR,
)


# SQL schema
CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    file_mtime REAL NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
"""

CREATE_INDEX_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_index
USING vec0(embedding FLOAT[384]);
"""


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime


def chunk_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def open_conn(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def ensure_schema(conn):
    conn.execute(CREATE_CHUNKS_TABLE)
    conn.execute(CREATE_INDEX_TABLE)
    conn.commit()


def gather_files(notes_dir: str):
    return sorted(Path(notes_dir).rglob("*.md"))


def chunk_file(path: Path, reader: MarkdownReader, parser: SimpleNodeParser):
    docs = reader.load_data(path)
    nodes = parser.get_nodes_from_documents(docs)

    chunks = []
    for i, node in enumerate(nodes):
        text = node.get_content().strip()
        if text:
            chunks.append((i, text))
    return chunks


def embed_texts_batch(model, texts, batch_size):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True)
        for v in vecs:
            results.append(np.asarray(v, dtype=np.float32))
    return results


def delete_file_chunks(conn, file_path):
    conn.execute(
        "DELETE FROM chunks_index WHERE rowid IN (SELECT id FROM chunks WHERE file_path = ?);",
        (file_path,),
    )
    conn.execute("DELETE FROM chunks WHERE file_path = ?;", (file_path,))
    conn.commit()


def insert_chunks(conn, file_path, chunks_with_emb, mtime):
    sql = """INSERT INTO chunks (file_path, chunk_index, chunk_hash, file_mtime, text, embedding)
             VALUES (?, ?, ?, ?, ?, ?);"""
    cur = conn.cursor()

    for chunk_index, text, emb in chunks_with_emb:
        cur.execute(
            sql,
            (
                file_path,
                chunk_index,
                chunk_hash(text),
                mtime,
                text,
                emb,  # raw NumPy float32 vector → stored as BLOB
            ),
        )

    conn.commit()


def populate_index_for_file(conn, file_path):
    conn.execute(
        """
        INSERT INTO chunks_index(rowid, embedding)
        SELECT id, embedding FROM chunks WHERE file_path = ?;
    """,
        (file_path,),
    )
    conn.commit()


def ingest_all(notes_dir, db_path):
    start = time.time()

    conn = open_conn(db_path)
    ensure_schema(conn)

    reader = MarkdownReader()
    parser = SimpleNodeParser.from_defaults(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    model = SentenceTransformer(EMBED_MODEL_NAME)

    files = gather_files(notes_dir)
    print(f"Found {len(files)} markdown files under {notes_dir}")

    for f in files:
        fp = str(f.resolve())
        mtime = file_mtime(f)

        cur = conn.execute(
            "SELECT MAX(file_mtime) FROM chunks WHERE file_path = ?;", (fp,)
        )
        row = cur.fetchone()
        old_mtime = row[0] if row and row[0] is not None else None

        if old_mtime is not None and abs(old_mtime - mtime) < 1e-6:
            print(f"[SKIP] {fp}")
            continue

        print(f"[UPDATE] {fp}")
        delete_file_chunks(conn, fp)

        chunks = chunk_file(f, reader, parser)
        if not chunks:
            print("  No chunks.")
            continue

        texts = [t for (_, t) in chunks]
        print(f"  Chunk count: {len(chunks)} — embedding...")

        embeddings = embed_texts_batch(model, texts, EMBED_BATCH_SIZE)

        chunks_with_emb = [
            (idx, text, emb) for (idx, text), emb in zip(chunks, embeddings)
        ]

        insert_chunks(conn, fp, chunks_with_emb, mtime)
        populate_index_for_file(conn, fp)

        print(f"  Inserted {len(chunks_with_emb)} chunks.")

    conn.close()
    print(f"Ingestion complete ({time.time() - start:.1f}s)")

def main():
    ingest_all(NOTES_DIR, DB_PATH)
