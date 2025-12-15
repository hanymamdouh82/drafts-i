#!/usr/bin/env python3

import time
from pathlib import Path

# from hashlib import sha256
from typing import List
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from uuid import uuid4

from llama_index.readers.file import MarkdownReader
from llama_index.core.node_parser import SimpleNodeParser
from drafts_i.config import (
    EMBED_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SOURCE_DIRS,
    COLLECTION_NAME,
)


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime


# def chunk_hash(text: str) -> str:
#     return sha256(text.encode("utf-8")).hexdigest()


def gather_files(dirs: List[str]):
    files = []
    for dir in dirs:
        files.extend(Path(dir).rglob("*.md"))
    return sorted(files)
    # return sorted(Path(notes_dir).rglob("*.md"))


def chunk_file(path: Path, reader: MarkdownReader, parser: SimpleNodeParser) -> List[str]:
    docs = reader.load_data(path)
    nodes = parser.get_nodes_from_documents(docs)

    chunks = []
    for i, node in enumerate(nodes):
        text = node.get_content().strip()
        if text:
            chunks.append(text)
    return chunks


def ingest_all():
    start = time.time()

    reader = MarkdownReader()
    parser = SimpleNodeParser.from_defaults(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    client = QdrantClient(host="localhost", port=6333)

    print("Deleting Collection in Qdrant")
    client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    files = gather_files(SOURCE_DIRS)
    print(f"Found {len(files)} markdown files under {SOURCE_DIRS}")

    # prepare model
    model = TextEmbedding(EMBED_MODEL_NAME)

    # iterate over files in target dir
    for f in files:
        print(f"Ingesting file: {f.absolute()}")

        fp = str(f.resolve())
        mtime = file_mtime(f)

        chunks = chunk_file(f, reader, parser)
        if not chunks:
            print("  No chunks.")
            continue

        vectors = model.embed(chunks, normalize_embeddings=True)

        # creating points
        points = []
        for i, v in enumerate(vectors):
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector=v,
                    payload={
                        "filename": fp,
                        "chunk": chunks[i],
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(f"  Inserted {len(chunks)} chunks.")

    print(f"Ingestion complete ({time.time() - start:.1f}s)")
