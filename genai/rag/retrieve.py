
from pathlib import Path
import json
import re
from functools import lru_cache

import faiss
from sentence_transformers import SentenceTransformer


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_STORE = BASE_DIR / "data" / "processed" / "vector_store"

INDEX_PATH = VECTOR_STORE / "index.faiss"
METADATA_PATH = VECTOR_STORE / "metadata.json"

# Same embedding model used during vector store creation
MODEL_NAME = "all-MiniLM-L6-v2"

# Minimum semantic similarity
SIMILARITY_THRESHOLD = 0.40

# Common words that do not help identify a topic
STOP_WORDS = {
    "the", "is", "are", "was", "were", "a", "an",
    "and", "or", "to", "of", "in", "on", "for",
    "how", "can", "i", "my", "me", "what", "why",
    "unable", "user", "please", "help", "with",
    "this", "that", "it", "do", "does", "issue"
}


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load the embedding model once and reuse it."""

    return SentenceTransformer(MODEL_NAME)


def load_vector_store():
    """Load saved FAISS index and metadata."""

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Vector store files not found.")

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise ValueError("FAISS index and metadata size mismatch.")

    return index, metadata


def extract_keywords(text):
    """Extract useful keywords from text."""

    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    keywords = set()

    for word in words:

        if word in STOP_WORDS or len(word) < 3:
            continue

        # Basic plural normalization
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"

        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]

        keywords.add(word)

    return keywords


def retrieve_documents(question, top_k=3):
    """Retrieve relevant knowledge-base chunks."""

    if not question or not question.strip():
        return []

    index, metadata = load_vector_store()

    # Handle an empty vector store
    if index.ntotal == 0:
        return []

    model = load_embedding_model()

    # Convert question into an embedding
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Search a wider set before filtering
    search_k = min(max(top_k * 3, 5), index.ntotal)

    scores, indices = index.search(query_embedding, search_k)

    question_keywords = extract_keywords(question)

    results = []
    seen_sources = set()

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        similarity = float(score)

        # Remove weak semantic matches
        if similarity < SIMILARITY_THRESHOLD:
            continue

        document = metadata[idx]

        source = document["source"]
        content = document["text"]

        # Check keyword relevance
        document_keywords = extract_keywords(content)
        keyword_overlap = question_keywords.intersection(document_keywords)

        # Reject unrelated documents
        if not keyword_overlap:
            continue

        # Avoid duplicate chunks from the same source
        if source in seen_sources:
            continue

        results.append({
            "source": source,
            "text": content,
            "similarity_score": round(similarity, 4)
        })

        seen_sources.add(source)

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":

    test_questions = [
        "How can I troubleshoot a password reset issue?",
        "How can I fix network connectivity problems?",
        "Why am I unable to send or receive emails?",
        "How can I troubleshoot a printer hardware issue?"
    ]

    for question in test_questions:

        results = retrieve_documents(question)

        print("\n" + "=" * 60)
        print("USER QUESTION:", question)
        print("=" * 60)

        if not results:
            print("No relevant documents found.")
            continue

        for result in results:
            print("\nSource:", result["source"])
            print("Similarity:", result["similarity_score"])
            print("Content:", result["text"][:300])
