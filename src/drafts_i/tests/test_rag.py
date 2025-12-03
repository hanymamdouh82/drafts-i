import numpy as np
from sentence_transformers import SentenceTransformer
from llama_index.readers.file import MarkdownReader
from llama_index.core.node_parser import SimpleNodeParser
from pathlib import Path


def load_markdown_files(directory: str):
    return list(Path(directory).rglob("*.md"))


def embed(texts, model):
    """Return embeddings as numpy arrays (batch encoded)."""
    return np.array(model.encode(texts, normalize_embeddings=True))


def cosine_similarity(a, b):
    return np.dot(a, b)


def main():
    notes_dir = "/mnt/repos/drafts/"
    # query = "What is the command required to run FleetFix for dev?"
    query = "What is FleetFix?"

    print("Loading Markdown files...")
    files = load_markdown_files(notes_dir)

    reader = MarkdownReader()
    parser = SimpleNodeParser.from_defaults(
        chunk_size=400,
        chunk_overlap=80,
    )

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_chunks = []
    all_texts = []

    print("Chunking documents...")
    for f in files:
        docs = reader.load_data(f)
        nodes = parser.get_nodes_from_documents(docs)

        for node in nodes:
            text = node.get_content().strip()
            if len(text) > 0:
                all_chunks.append((f, text))
                all_texts.append(text)

    print(f"Total chunks: {len(all_chunks)}")

    print("Embedding chunks (batch mode)...")
    chunk_embeddings = embed(all_texts, model)

    print("Embedding query...")
    query_embedding = embed([query], model)[0]

    print("Computing similarities...")
    sims = chunk_embeddings @ query_embedding

    # Get top 5 results
    top_k = 5
    top_indices = sims.argsort()[-top_k:][::-1]

    print("\n=== Top Matches ===")
    for idx in top_indices:
        file_path, chunk_text = all_chunks[idx]
        score = sims[idx]

        print("\n----------------------------------------")
        print(f"Score: {score:.4f}")
        print(f"File: {file_path}")
        print("Chunk:")
        print(chunk_text)
        print("----------------------------------------")


if __name__ == "__main__":
    main()
