"""
TUM Smart Course Assistant - ChromaDB Index Creation Module (OpenAI API)

This module implements the "Embedding + Vector Database" step described in README.md:
    Course Cards -> OpenAI Embeddings -> ChromaDB

Architectural Principle (from README):
    - 1 chunk = 1 course card
    - Each course card is processed as a single meaningful unit
    - Context retrieved during RAG is directly course cards

Purpose:
    - Read course cards from data/processed/course_cards.jsonl
    - Calculate OpenAI embedding for each course card
    - Write embeddings and metadata to ChromaDB persistent collection

Input:
    - data/processed/course_cards.jsonl (COURSE_CARDS_JSONL_PATH)

Output:
    - ChromaDB persistent directory (CHROMA_PERSIST_DIR)
    - Collection name: CHROMA_COLLECTION_NAME

NOTE: This module does not perform RAG retrieval or LLM answer generation.
      It only handles index creation.

NOTE: For local testing, build_index_local.py can be used.
"""

import json
from pathlib import Path

import chromadb
from openai import OpenAI
from tqdm import tqdm

from . import config
from .config import get_openai_client

# =============================================================================
# CONSTANTS - Use centralized config values
# =============================================================================

# Embedding batch size is now in config.EMBEDDING_BATCH_SIZE


# =============================================================================
# DATA LOADING FUNCTION
# =============================================================================


def load_course_cards(path: Path) -> list[dict]:
    """
    Loads course cards from JSONL file.

    Args:
        path: Path to course_cards.jsonl file

    Returns:
        Course card list:
        [
            {
                "course_id": "WI000116",
                "title": "Lead User Project",
                "pages": [1, 2, 3],
                "raw_text": "..."
            },
            ...
        ]

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
# OPENAI CLIENT - Use centralized singleton from config
# =============================================================================
# Imported from config.get_openai_client for DRY compliance


# =============================================================================
# EMBEDDING FUNCTION
# =============================================================================


def embed_texts(
    texts: list[str],
    client: OpenAI,
    model: str
) -> list[list[float]]:
    """
    Calculates OpenAI embeddings for given texts.

    Args:
        texts: List of texts to calculate embeddings for
        client: OpenAI client instance
        model: Embedding model name (e.g., text-embedding-3-small)

    Returns:
        List of embedding vectors (each is list[float])

    Note:
        - This function does not batch process, sends all texts at once
        - For large lists, external batching loop should be used
        - OpenAI API may not return results in order, matched by index
    """
    if not texts:
        return []

    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    # Extract embedding vectors from response
    # OpenAI API does not guarantee results in index order
    # So we match by index
    embeddings = [None] * len(texts)
    for item in response.data:
        embeddings[item.index] = item.embedding

    return embeddings


# =============================================================================
# CHROMADB FUNCTIONS
# =============================================================================


def get_chroma_collection() -> chromadb.Collection:
    """
    Returns ChromaDB persistent client and collection.

    Creates collection if it doesn't exist, returns existing one otherwise.

    Returns:
        ChromaDB Collection instance
    """
    # Create persist directory
    persist_dir = Path(config.CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Create persistent client
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Get or create collection
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={
            "description": "TUM Course Cards Index",
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "hnsw:space": "cosine",  # Use cosine similarity
        },
    )

    return collection


# =============================================================================
# INDEX CREATION FUNCTION
# =============================================================================


def build_index() -> None:
    """
    Loads course cards, calculates OpenAI embeddings and writes to ChromaDB.

    Steps:
        1. Load course_cards.jsonl file
        2. Create OpenAI client
        3. Calculate embedding for each course card (in batches)
        4. Clear ChromaDB collection and add new data
        5. Run test query

    Raises:
        FileNotFoundError: If course cards file is not found
        RuntimeError: If OpenAI API key is not defined
    """
    print("=" * 60)
    print("TUM Smart Course Assistant - OpenAI Index Creation")
    print("=" * 60)
    print()

    # ---------------------------------------------------------------------
    # 1. Load course cards
    # ---------------------------------------------------------------------
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

    # Check for duplicate course_ids
    course_ids = [card["course_id"] for card in cards]
    unique_ids = set(course_ids)
    if len(course_ids) != len(unique_ids):
        duplicate_count = len(course_ids) - len(unique_ids)
        print(f"   {duplicate_count} duplicate course_ids found!")
        print("   -> Duplicates will be merged (last one wins)")
        
        # Clean duplicates (last one wins)
        seen = {}
        for card in cards:
            seen[card["course_id"]] = card
        cards = list(seen.values())
        print(f"   -> Cleaned card count: {len(cards)}")

    # ---------------------------------------------------------------------
    # 2. Create OpenAI client
    # ---------------------------------------------------------------------
    print("\n2. Creating OpenAI client...")
    client = get_openai_client()
    print(f"   Embedding model: {config.EMBEDDING_MODEL_NAME}")

    # ---------------------------------------------------------------------
    # 3. Calculate embeddings (in batches)
    # ---------------------------------------------------------------------
    print(f"\n3. Calculating embeddings (batch_size={config.EMBEDDING_BATCH_SIZE})...")
    
    # Create embedding text: title + content + learning_outcomes combination
    # This way boilerplate text (exam details, prerequisites) doesn't pollute embeddings
    def build_embedding_text(card: dict) -> str:
        """
        Creates optimized text for embedding from course card.
        
        Priority order:
            1. title + content + learning_outcomes (if available)
            2. title + content (if learning_outcomes not available)
            3. raw_text[:2000] (fallback if sections not found)
        """
        title = card.get("title", "")
        content = card.get("content")
        learning_outcomes = card.get("learning_outcomes")
        
        # Use sections if available
        if content or learning_outcomes:
            parts = [title]
            if content:
                parts.append(content)
            if learning_outcomes:
                parts.append(learning_outcomes)
            return "\n\n".join(parts)
        else:
            # Fallback: first 2000 characters of raw_text
            return card.get("raw_text", "")[:2000]
    
    texts = [build_embedding_text(card) for card in cards]
    
    # Section extraction statistics
    cards_with_content = sum(1 for c in cards if c.get("content"))
    cards_with_outcomes = sum(1 for c in cards if c.get("learning_outcomes"))
    print(f"   -> Cards with Content: {cards_with_content}/{len(cards)}")
    print(f"   -> Cards with Learning Outcomes: {cards_with_outcomes}/{len(cards)}")
    
    all_embeddings = []

    # tqdm ile batch'leri işle
    num_batches = (len(texts) + config.EMBEDDING_BATCH_SIZE - 1) // config.EMBEDDING_BATCH_SIZE
    
    with tqdm(total=num_batches, desc="   Embedding batches") as pbar:
        for i in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
            batch_texts = texts[i:i + config.EMBEDDING_BATCH_SIZE]
            batch_embeddings = embed_texts(
                texts=batch_texts,
                client=client,
                model=config.EMBEDDING_MODEL_NAME
            )
            all_embeddings.extend(batch_embeddings)
            pbar.update(1)

    print(f"   Total embeddings: {len(all_embeddings)}")

    # ---------------------------------------------------------------------
    # 4. Write to ChromaDB
    # ---------------------------------------------------------------------
    print("\n4. Writing to ChromaDB...")
    collection = get_chroma_collection()

    # Clear existing data first
    existing_count = collection.count()
    if existing_count > 0:
        print(f"   -> Deleting {existing_count} existing records...")
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
            # pages list stored as JSON string (consistent with local version)
            "pages": json.dumps(card["pages"]),
            # New metadata fields (from merge_course_cards.py)
            "category": card.get("category", "unknown"),
            "also_in_electives": card.get("also_in_electives", False),
            "sources": json.dumps(card.get("sources", [])),
            # Domain field: management (MMT) or technology (Informatics)
            "domain": card.get("domain", "unknown"),
            # Structured metadata fields (for metadata filtering)
            "level": card.get("level") or "unknown",
            "language": card.get("language") or "unknown",
            "duration": card.get("duration") or "unknown",
            "frequency": card.get("frequency") or "unknown",
            "sws": card.get("sws") or 0,
            # New section fields (content and learning_outcomes)
            "content": card.get("content") or "",
            "learning_outcomes": card.get("learning_outcomes") or "",
        }
        for card in cards
    ]

    # Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"   Collection: {config.CHROMA_COLLECTION_NAME}")
    print(f"   Persist directory: {config.CHROMA_PERSIST_DIR}")
    print(f"   Records added: {collection.count()}")

    # ---------------------------------------------------------------------
    # 5. Test query
    # ---------------------------------------------------------------------
    print("\n5. Running test query...")
    test_query = "product management innovation"
    print(f"   Query: '{test_query}'")

    # Use OpenAI embedding for test query too
    query_embedding = embed_texts(
        texts=[test_query],
        client=client,
        model=config.EMBEDDING_MODEL_NAME
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["metadatas", "distances"],
    )

    print("   Top 3 results:")
    for i, (doc_id, metadata, distance) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ):
        similarity = 1 - distance  # Cosine distance -> similarity
        print(f"      {i + 1}. [{doc_id}] {metadata['title']}")
        print(f"         Similarity: {similarity:.4f}")

    # ---------------------------------------------------------------------
    # Completed
    # ---------------------------------------------------------------------
    print()
    print("=" * 60)
    print("OpenAI Index creation completed!")
    print()
    print("Next step: RAG pipeline (src/rag_pipeline.py)")
    print("=" * 60)


# =============================================================================
# MAIN FLOW
# =============================================================================

if __name__ == "__main__":
    build_index()
