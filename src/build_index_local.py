"""
Local embedding script for creating Chroma index (no API key required).

Input:
    - data/processed/course_cards.jsonl

Output:
    - ChromaDB persist directory (under data/chroma_db_local)
    - Collection name: courses_index_local

This script does NOT use REAL semantic embeddings.
Purpose:
    - Test ETL -> course cards -> embedding -> Chroma index pipeline
      without OpenAI API key.
    - Validate RAG retrieval logic.

NOTE: This script is for TESTING only.
      In production, src/build_index.py (OpenAI embedding) should be used.
"""

import hashlib
import json
from pathlib import Path

import chromadb
import numpy as np
from tqdm import tqdm

import config


# =============================================================================
# LOCAL CONFIGURATION
# =============================================================================

# Separate persist directory and collection name for local testing
CHROMA_PERSIST_DIR_LOCAL = "data/chroma_db_local"
CHROMA_COLLECTION_NAME_LOCAL = "courses_index_local"

# Fake embedding dimension (same as OpenAI text-embedding-3-small: 1536)
EMBEDDING_DIM = 1536


# =============================================================================
# DATA LOADING FUNCTION
# =============================================================================


def load_course_cards(path: Path) -> list[dict]:
    """
    Loads course cards from JSONL file.

    Args:
        path: Path to course_cards.jsonl file

    Returns:
        Course card list

    Raises:
        FileNotFoundError: If file is not found
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Course cards file not found: {path}")

    cards = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                card = json.loads(line)
                cards.append(card)

    return cards


# =============================================================================
# LOCAL EMBEDDING FUNCTION
# =============================================================================


def compute_local_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Produces deterministic, fake embedding for text.

    This function is NOT a REAL semantic embedding.
    Used for testing purposes only.

    Algorithm:
        1. Hash the text with SHA256
        2. Use hash as seed to generate numpy random vector
        3. Normalize vector (unit vector)

    Args:
        text: Text to compute embedding for
        dim: Embedding dimension (default: 1536)

    Returns:
        Normalized embedding vector (list[float])
    """
    # Hash the text (for deterministic seed)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Use first 8 characters of hash as integer seed
    seed = int(text_hash[:8], 16)

    # Create random generator with seed
    rng = np.random.default_rng(seed)

    # Generate random vector
    embedding = rng.standard_normal(dim)

    # Normalize (unit vector)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()


def embed_texts_local(texts: list[str]) -> list[list[float]]:
    """
    Calculates local embeddings for given texts.

    Args:
        texts: List of texts to compute embeddings for

    Returns:
        List of embedding vectors
    """
    embeddings = []
    for text in texts:
        emb = compute_local_embedding(text)
        embeddings.append(emb)
    return embeddings


# =============================================================================
# CHROMADB FUNCTIONS
# =============================================================================


def get_chroma_collection_local() -> chromadb.Collection:
    """
    Returns ChromaDB collection for local testing.

    Returns:
        ChromaDB Collection instance
    """
    # Create persist directory
    persist_dir = Path(CHROMA_PERSIST_DIR_LOCAL)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Create persistent client
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Get or create collection
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME_LOCAL,
        metadata={"description": "TUM Course Cards Index (LOCAL TEST)"},
    )

    return collection


# =============================================================================
# INDEX CREATION FUNCTION
# =============================================================================


def build_index_local() -> None:
    """
    Loads course cards, calculates LOCAL embedding and writes to ChromaDB.

    This function does not use OpenAI API, for testing purposes only.
    """
    print("=" * 60)
    print("TUM Smart Course Assistant - LOCAL Index Creation (TEST)")
    print("=" * 60)
    print()
    print("WARNING: This script does NOT use REAL semantic embeddings!")
    print("WARNING: Should only be used for pipeline testing.")
    print()

    # 1. Load course cards
    jsonl_path = Path(config.COURSE_CARDS_JSONL_PATH)
    print(f"1. Loading course cards: {jsonl_path}")

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Course cards file not found: {jsonl_path}\n"
            "First run 'python src/build_course_cards.py'."
        )

    cards = load_course_cards(jsonl_path)
    print(f"   Total course cards: {len(cards)}")

    if not cards:
        print("   No course cards found, cannot create index.")
        return

    # 2. Calculate local embeddings
    print("\n2. Calculating local embeddings...")
    print(f"   Embedding dimension: {EMBEDDING_DIM}")

    texts = [card["raw_text"] for card in cards]
    embeddings = []

    for text in tqdm(texts, desc="   Embedding"):
        emb = compute_local_embedding(text)
        embeddings.append(emb)

    print(f"   Total embeddings: {len(embeddings)}")

    # 3. Write to ChromaDB
    print("\n3. Writing to ChromaDB...")
    collection = get_chroma_collection_local()

    # Clear existing data first
    existing_count = collection.count()
    if existing_count > 0:
        print(f"   Deleting {existing_count} existing records...")
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

    # Prepare data
    ids = [card["course_id"] for card in cards]
    documents = [card["raw_text"] for card in cards]
    metadatas = [
        {
            "course_id": card["course_id"],
            "title": card["title"],
            "pages": json.dumps(card["pages"]),
        }
        for card in cards
    ]

    # Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"   Collection: {CHROMA_COLLECTION_NAME_LOCAL}")
    print(f"   Persist directory: {CHROMA_PERSIST_DIR_LOCAL}")
    print(f"   Records added: {collection.count()}")

    # 4. Simple test query
    print("\n4. Running test query...")
    test_query = "product management innovation"
    query_embedding = compute_local_embedding(test_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    print(f"   Query: '{test_query}'")
    print("   Top 3 results:")
    for i, (doc_id, metadata) in enumerate(
        zip(results["ids"][0], results["metadatas"][0])
    ):
        print(f"      {i + 1}. [{doc_id}] {metadata['title']}")

    print()
    print("=" * 60)
    print("LOCAL index creation completed!")
    print()
    print("NOTE: This index does NOT perform REAL semantic search.")
    print("      For production use 'python src/build_index.py'.")
    print("=" * 60)


# =============================================================================
# MAIN FLOW
# =============================================================================

if __name__ == "__main__":
    build_index_local()
