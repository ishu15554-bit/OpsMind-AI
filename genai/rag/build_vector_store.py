
from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Knowledge base and vector store paths
KNOWLEDGE_BASE = BASE_DIR / "docs" / "knowledge_base"
VECTOR_STORE = BASE_DIR / "data" / "processed" / "vector_store"

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

# Chunk settings
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def load_documents():
    """Read all Markdown documents from the knowledge base."""

    documents = []

    for file_path in KNOWLEDGE_BASE.glob("*.md"):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "source": file_path.name,
                "text": text
            })

    if not documents:
        raise ValueError(
            "No Markdown documents found in docs/knowledge_base/"
        )

    return documents


def split_into_chunks(text):
    """Split document text into overlapping chunks."""

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def build_vector_store():
    """Create embeddings and save the FAISS index."""

    VECTOR_STORE.mkdir(parents=True, exist_ok=True)

    documents = load_documents()

    all_chunks = []

    for document in documents:
        chunks = split_into_chunks(document["text"])

        for chunk_number, chunk in enumerate(chunks):
            all_chunks.append({
                "source": document["source"],
                "chunk_number": chunk_number,
                "text": chunk
            })

    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(all_chunks)}")

    # Load the embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    texts = [chunk["text"] for chunk in all_chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Create FAISS similarity index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save FAISS index
    index_path = VECTOR_STORE / "index.faiss"
    faiss.write_index(index, str(index_path))

    # Save chunk metadata
    metadata_path = VECTOR_STORE / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, indent=4, ensure_ascii=False)

    print("\nVector store created successfully!")
    print(f"Total vectors: {index.ntotal}")
    print(f"Embedding dimensions: {dimension}")
    print(f"Index saved: {index_path}")
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    build_vector_store()
