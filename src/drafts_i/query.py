#!/usr/bin/env python3

import subprocess
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from drafts_i.config import (
    EMBED_MODEL_NAME,
    MAX_CONTEXT,
    LLAMA_CLI_PATH,
    LLAMA_MODEL_PATH,
    LLAMA_CTX,
    COLLECTION_NAME,
)


def build_context(points):
    ctx = ""
    for i, point in enumerate(points):
        score = point.score
        text = point.payload.get("chunk")
        entry = f"[chunk {i} / score={score:.4f}]\n{text}\n\n"
        if len(ctx) + len(entry) > MAX_CONTEXT:
            break
        ctx += entry
    return ctx


def ask_llama_cli(context, question, isLong: bool):
    guide_short = "You are a factual assistant. Use ONLY the provided context. If the answer is not found, say: 'Not found in context.'\n\n"

    guide_long = "You are a factual assistant. Use ONLY the provided context to provide a comprehensive answer. If the answer is not found, say: 'Not found in context.'\n\n"

    guide = guide_short
    if isLong:
        guide = guide_long

    prompt = f"{guide}Context:\n{context}\n\nQuestion: {question}\nAnswer:\n"

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


def query(question: str, isLong=False):
    embedder = TextEmbedding(EMBED_MODEL_NAME)
    qClient = QdrantClient(host="localhost", grpc_port=6333)

    q_vec = next(embedder.embed(question, normalize_embeddings=True))

    results = qClient.query_points(collection_name=COLLECTION_NAME, query=q_vec.tolist(), limit=10)

    if not results:
        print("No results.")
        return

    context = build_context(results.points)

    answer = ask_llama_cli(context, question, isLong)

    # prepare references
    file_names = []
    for result in results.points:
        file_names.append(f"-{result.payload.get('filename')}")
    file_names = list(set(file_names))
    references_text = "\n".join(file_names)

    final_answer = f"""
## Answer:
{answer}

## References:
{references_text}
"""

    print(final_answer)


def interactive():
    question = input("Ask your question: ").strip()
    if not question:
        print("Empty question.")
        return
    query(question, isLong=True)


def ask(question: str):
    if not question:
        print("Empty question.")
        return
    query(question, isLong=False)


def explain(question: str):
    if not question:
        print("Empty question.")
        return
    query(question, isLong=True)
