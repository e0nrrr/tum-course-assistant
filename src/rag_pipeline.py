"""
TUM Smart Course Assistant - RAG Pipeline (Retrieval Only)

This module performs course search using OpenAI embedding + ChromaDB.

Architecture (from README):
    1. User query -> OpenAI embedding
    2. Similarity search in ChromaDB
    3. Return most relevant course cards

This file ONLY performs retrieval:
    - No LLM reasoning yet
    - Only for "Is RAG retrieval working correctly?" validation

Usage:
    python -m src.rag_pipeline

NOTE: For local testing, src/rag_local.py can be used.
      This module uses REAL OpenAI embedding.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb
from openai import OpenAI

import config


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CourseResult:
    """
    Course information returned as search result.
    
    Attributes:
        course_id: Course code (e.g., "WI000116")
        title: Course title
        pages: Page numbers where course is found
        raw_text: Full text of course card
        distance: Distance from ChromaDB (lower = more similar)
        similarity: 1 - distance (higher = more similar)
        category: Course category ("advanced_seminar" or "elective")
        also_in_electives: Is this course also in elective category?
        content: Course content (from Content section)
        learning_outcomes: Course learning outcomes (from Intended Learning Outcomes section)
        language: Course language (e.g., "German", "English", "German/English")
    """

    course_id: str
    title: str
    pages: list[int]
    raw_text: str
    distance: float
    similarity: float
    category: str = "unknown"
    also_in_electives: bool = False
    content: str | None = None
    learning_outcomes: str | None = None
    language: str = "English"  # Default to English

    def __str__(self) -> str:
        """Nicely formatted string representation."""
        # Format page range
        if not self.pages:
            pages_str = "[?]"
        elif len(self.pages) == 1:
            pages_str = f"[{self.pages[0]}]"
        else:
            pages_str = f"[{self.pages[0]}..{self.pages[-1]}]"
        
        # Category badge
        if self.also_in_electives:
            cat_str = " [Seminar+Elective]"
        elif self.category == "advanced_seminar":
            cat_str = " [Seminar]"
        else:
            cat_str = ""
        
        return (
            f"[{self.course_id}] {self.title}{cat_str} "
            f"(pages: {pages_str}, similarity: {self.similarity:.4f})"
        )

    def to_dict(self) -> dict:
        """
        Convert to dict format (for sending to LLM).
        
        Returns:
            Dict containing course information
        """
        return {
            "course_id": self.course_id,
            "title": self.title,
            "pages": self.pages,
            "raw_text": self.raw_text,
        }


# =============================================================================
# OPENAI CLIENT - Use centralized singleton from config
# =============================================================================
# Import get_openai_client from config for DRY compliance
from config import get_openai_client


# =============================================================================
# CHROMA COLLECTION HELPER
# =============================================================================


def get_chroma_collection() -> chromadb.Collection:
    """
    Opens and returns existing ChromaDB collection.

    This function does NOT create new index, only opens existing one.
    Use build_index.py to create index.

    Returns:
        ChromaDB Collection instance

    Raises:
        ValueError: If collection is not found
    """
    persist_dir = Path(config.CHROMA_PERSIST_DIR)

    if not persist_dir.exists():
        raise ValueError(
            f"ChromaDB directory not found: {persist_dir}\n"
            "First run 'python -m src.build_index'."
        )

    # Create persistent client
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Get collection (if exists)
    # Using get_or_create instead of get_collection for consistency
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={
            "description": "TUM Course Cards Index (PROD)",
            "hnsw:space": "cosine",
        },
    )

    # Warn if collection is empty
    if collection.count() == 0:
        raise ValueError(
            f"Collection '{config.CHROMA_COLLECTION_NAME}' is empty!\n"
            "First run 'python -m src.build_index'."
        )

    return collection


# =============================================================================
# EMBEDDING HELPER
# =============================================================================


def _is_likely_english(text: str) -> bool:
    """
    Quick heuristic to detect if text is likely English.
    Returns True if text appears to be English, False otherwise.
    
    Checks:
    1. Non-ASCII ratio (Turkish/German special chars: ö, ü, ş, ğ, ı, ç, ä, ß)
    2. Common non-English words
    """
    import re
    
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    # Turkish-EXCLUSIVE characters (ş, ğ, ı are NOT used in German)
    turkish_only_chars = set('şğı')
    turkish_words = {'ders', 'dersler', 'hakkında', 'istiyorum', 'arıyorum', 'bul', 
                     'göster', 'ilgili', 'için', 've', 'ile', 'nasıl', 'nedir',
                     'yapay', 'zeka', 'öğrenmek'}
    
    # German-specific words (NOT also English)
    german_words = {'und', 'ich', 'möchte', 'suche', 'kurse', 'kurs', 'über',
                    'finden', 'zeigen', 'lernen', 'wie', 'können', 'interessiert',
                    'welche', 'gibt', 'nicht', 'auch', 'oder', 'für', 'mit',
                    'vorlesung', 'fach', 'seminar', 'bitte', 'danke'}
    
    # Check Turkish-exclusive chars
    if any(c in turkish_only_chars for c in text_lower):
        return False
    
    # Check Turkish words
    if words & turkish_words:
        return False
    
    # Check German words
    if words & german_words:
        return False
    
    # Check German-exclusive chars (ä, ß)
    if any(c in set('äß') for c in text_lower):
        return False
    
    # ü/ö without German words context — could be Turkish
    if any(c in set('üöç') for c in text_lower):
        return False
    
    # If mostly ASCII and no known foreign words, assume English
    return True


def _expand_query(query: str, client: OpenAI) -> str:
    """
    Expands a short/vague query with related academic terms using LLM.
    
    For example:
        "AI courses" -> "artificial intelligence AI machine learning deep learning
                         neural networks computer vision NLP courses"
    
    This dramatically improves embedding similarity for broad topic queries
    by matching more terms in course descriptions.
    
    Args:
        query: English query (already translated if needed)
        client: OpenAI client
    
    Returns:
        Expanded query string
    """
    import re
    word_count = len(re.findall(r'\b\w+\b', query))
    
    if not config.QUERY_EXPANSION_ENABLED:
        return query
    
    if word_count >= config.QUERY_EXPANSION_MIN_WORDS:
        print(f"   Query has {word_count} words, skipping expansion")
        return query
    
    print(f"   Short query ({word_count} words), expanding...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a university course search assistant. "
                        "Given a short course search query, expand it with 5-8 closely related "
                        "academic keywords and synonyms that would appear in university course "
                        "descriptions. Include both formal academic terms and common alternatives.\n\n"
                        "Rules:\n"
                        "- Return ONLY the expanded query as a single line\n"
                        "- Keep the original words and ADD related terms\n"
                        "- Focus on terms likely found in course titles and descriptions\n"
                        "- Do NOT add explanations or formatting\n\n"
                        "Examples:\n"
                        "Input: AI courses\n"
                        "Output: artificial intelligence AI machine learning deep learning neural networks computer vision natural language processing data science courses\n\n"
                        "Input: entrepreneurship\n"
                        "Output: entrepreneurship startup business venture creation innovation new business development technology entrepreneurship\n\n"
                        "Input: software engineering\n"
                        "Output: software engineering software development programming systems design software architecture agile development code quality testing"
                    )
                },
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=100
        )
        expanded = response.choices[0].message.content.strip()
        print(f"   Query expanded: '{query}' -> '{expanded}'")
        return expanded
    except Exception as e:
        print(f"   Query expansion failed, using original: {e}")
        return query


def detect_and_translate_query(query: str, client: OpenAI) -> str:
    """
    Detects query language, translates if not English, then expands short queries.
    
    Pipeline:
        1. Language detection (heuristic)
        2. Translation to English (if needed, via LLM)
        3. Query expansion with related terms (if short query, via LLM)
    
    Since course descriptions in ChromaDB are in English/German,
    we translate multilingual queries to English for embedding,
    then expand short queries with academic synonyms for better recall.
    
    Args:
        query: User query (in any language)
        client: OpenAI client
    
    Returns:
        English query, potentially expanded with related terms
    """
    # Step 1: Translation (if needed)
    if _is_likely_english(query):
        print(f"   Query detected as English, skipping translation")
        english_query = query
    else:
        print(f"   Non-English query detected, translating...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Translate the following query to English. "
                            "Return ONLY the English translation, nothing else."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0,
                max_tokens=200
            )
            english_query = response.choices[0].message.content.strip()
            print(f"   Query translated: '{query}' -> '{english_query}'")
        except Exception as e:
            print(f"   Translation failed, using original query: {e}")
            english_query = query
    
    # Step 2: Query expansion (for short queries)
    expanded_query = _expand_query(english_query, client)
    
    return expanded_query


def embed_query(query: str, client: OpenAI) -> list[float]:
    """
    Calculates OpenAI embedding for a single query.

    Args:
        query: Query text to calculate embedding for
        client: OpenAI client instance

    Returns:
        Embedding vector (list[float])
    """
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL_NAME,
        input=[query],
    )

    # Return first (and only) embedding
    return response.data[0].embedding


# =============================================================================
# METADATA FILTER DETECTION
# =============================================================================


def detect_metadata_filters(query: str) -> dict | None:
    """
    Detects metadata filters from user query.
    
    Supported filters:
        - language: "German courses", "English", "Almanca dersler", "Ingilizce"
        - level: "Master courses", "Bachelor"
        - domain: "management specialization", "technology courses", "informatics"
    
    NOTE: language field in ChromaDB may contain values like "German", "English", 
    "German/English". Using $in operator to match multiple values.
    
    Args:
        query: User's natural language query
        
    Returns:
        ChromaDB where filter dict or None (if no filter found)
        
    Example:
        >>> detect_metadata_filters("German language courses")
        {"language": {"$in": ["German", "German/English", "Deutsch"]}}
        >>> detect_metadata_filters("technology specialization courses")
        {"domain": "technology"}
    """
    query_lower = query.lower()
    filters = {}
    
    # Domain detection - management vs technology
    # Technology keywords (Informatics, CS, Software, etc.)
    if any(kw in query_lower for kw in [
        "technology", "informatics", "computer science", "software", 
        "programming", "coding", "tech specialization", "teknoloji",
        "bilgisayar", "yazılım", "cs courses", "technical"
    ]):
        filters["domain"] = "technology"
    # Management keywords (MMT, Business, Innovation, etc.)
    elif any(kw in query_lower for kw in [
        "management", "business", "innovation", "entrepreneurship",
        "marketing", "strategy", "leadership", "mmt", "yönetim",
        "işletme", "girişimcilik", "management specialization"
    ]):
        filters["domain"] = "management"
    
    # Language detection - use $in to match multiple values
    # English keywords
    if any(kw in query_lower for kw in [
        "english", "in english", "english course", "english courses",
        "ingilizce", "Ingilizce"
    ]):
        # English or German/English - all courses containing these
        filters["language"] = {"$in": ["English", "German/English", "German/english", "english"]}
    # German keywords
    elif any(kw in query_lower for kw in [
        "german", "in german", "german course", "german courses",
        "deutsch", "almanca", "Almanca"
    ]):
        # German or German/English - all courses containing these
        filters["language"] = {"$in": ["German", "German/English", "German/english", "Deutsch", "german", "deutsch"]}
    
    # NOTE: Level detection disabled - causes false positives and confusion
    # Users rarely search by level, and "graduate/master" words appear in other contexts
    
    # ChromaDB requires $and operator for multiple filters
    if not filters:
        return None
    elif len(filters) == 1:
        # Single filter - return as-is
        return filters
    else:
        # Multiple filters - wrap with $and operator
        and_conditions = []
        for key, value in filters.items():
            and_conditions.append({key: value})
        return {"$and": and_conditions}


# =============================================================================
# MAIN SEARCH FUNCTION
# =============================================================================


def search_courses(
    query: str,
    n_results: int = 30,
    where_filter: dict | None = None,
    auto_detect_filters: bool = True
) -> list[CourseResult]:
    """
    Searches courses in ChromaDB and returns most relevant results.
    
    With metadata filtering support, queries like "German courses"
    automatically apply language filtering.
    
    MULTILINGUAL SUPPORT: Query is automatically translated to English
    since course descriptions in ChromaDB are in English/German.

    Args:
        query: User's natural language query (any language)
               (e.g., "product management innovation", "yapay zeka dersleri")
        n_results: Maximum number of results to return (default: 30)
        where_filter: Manual ChromaDB where filter (e.g., {"language": "German"})
        auto_detect_filters: If True, automatically detects filters from query

    Returns:
        CourseResult list (sorted by similarity, most similar first)

    Raises:
        RuntimeError: If API key is not defined
        ValueError: If collection is not found or empty

    Example:
        >>> results = search_courses("AI and machine learning", n_results=3)
        >>> results = search_courses("yapay zeka dersleri", n_results=50)  # Turkish
        >>> for r in results:
        ...     print(r)
        
        >>> # Manual filter
        >>> results = search_courses("courses", where_filter={"language": "German"})
        
        >>> # Auto-detect: "German courses" -> filters by language=German
        >>> results = search_courses("German courses about innovation")
    """
    import time as _time
    _start_total = _time.time()
    
    # 1. Get OpenAI client
    print(f"[DEBUG] RAG search started for: '{query[:50]}...'")
    client = get_openai_client()

    # 2. Get Chroma collection
    _t0 = _time.time()
    collection = get_chroma_collection()
    print(f"[DEBUG] ChromaDB collection loaded: {_time.time() - _t0:.2f}s")

    # 3. Translate query to English (if needed)
    _t0 = _time.time()
    query_translated = detect_and_translate_query(query, client)
    print(f"[DEBUG] Query translation: {_time.time() - _t0:.2f}s")

    # 4. Calculate query embedding (with translated query)
    _t0 = _time.time()
    query_embedding = embed_query(query_translated, client)
    print(f"[DEBUG] Embedding calculation: {_time.time() - _t0:.2f}s")

    # 5. Determine metadata filters
    effective_filter = where_filter
    if auto_detect_filters and effective_filter is None:
        detected = detect_metadata_filters(query)
        if detected:
            effective_filter = detected
            # Debug log (can be removed in production)
            print(f"   Auto-detected filter: {detected}")

    # 6. Search in ChromaDB
    _t0 = _time.time()
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    
    if effective_filter:
        query_params["where"] = effective_filter
    
    results = collection.query(**query_params)
    print(f"[DEBUG] ChromaDB search ({n_results} results): {_time.time() - _t0:.2f}s")

    # 7. Convert results to CourseResult list
    course_results = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc_id, doc, metadata, distance in zip(ids, documents, metadatas, distances):
        # pages stored as JSON string, parse it
        try:
            pages = json.loads(metadata.get("pages", "[]"))
        except (json.JSONDecodeError, TypeError):
            pages = []

        # Calculate similarity (for cosine distance)
        # Cosine distance can be 0-2, so we clamp
        similarity = max(0.0, 1.0 - distance)
        
        # Get category metadata
        category = metadata.get("category", "unknown")
        also_in_electives = metadata.get("also_in_electives", False)
        
        # Get new section fields
        content = metadata.get("content") or None
        learning_outcomes = metadata.get("learning_outcomes") or None
        language = metadata.get("language", "English")

        course_result = CourseResult(
            course_id=doc_id,
            title=metadata.get("title", "Unknown Course"),
            pages=pages,
            raw_text=doc,
            distance=distance,
            similarity=similarity,
            category=category,
            also_in_electives=also_in_electives,
            content=content,
            learning_outcomes=learning_outcomes,
            language=language,
        )
        course_results.append(course_result)

    print(f"[DEBUG] RAG pipeline total: {_time.time() - _start_total:.2f}s | Found {len(course_results)} courses")
    return course_results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def print_search_results(
    results: list[CourseResult],
    show_text: bool = False,
    max_preview_chars: int = 300,
) -> None:
    """
    Prints search results in formatted output to terminal.

    Args:
        results: CourseResult list
        show_text: If True, shows raw_text preview
        max_preview_chars: Maximum characters for preview
    """
    if not results:
        print("\nNo results found.")
        return

    print(f"\n{len(results)} courses found:\n")
    print("-" * 70)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")

        if show_text:
            # raw_text preview
            preview = result.raw_text[:max_preview_chars]
            # Clean line breaks
            preview = preview.replace("\n", " ").replace("  ", " ").strip()
            if len(result.raw_text) > max_preview_chars:
                preview += "..."
            print(f"   Preview: {preview}")

        print()


def get_index_stats() -> dict:
    """
    Returns statistics about ChromaDB index.

    Returns:
        {
            "collection_name": str,
            "document_count": int,
            "persist_dir": str,
            "embedding_model": str,
        }
    """
    collection = get_chroma_collection()
    return {
        "collection_name": collection.name,
        "document_count": collection.count(),
        "persist_dir": str(config.CHROMA_PERSIST_DIR),
        "embedding_model": config.EMBEDDING_MODEL_NAME,
    }


# =============================================================================
# INTERACTIVE DEMO
# =============================================================================


def interactive_demo() -> None:
    """
    Interactive course search demo from command line.
    
    Performs real semantic search using OpenAI embedding.
    """
    print("=" * 70)
    print("TUM Smart Course Assistant - RAG Retrieval Demo (OpenAI Embedding)")
    print("=" * 70)
    print()

    # Show index statistics
    try:
        stats = get_index_stats()
        print(f"Collection: {stats['collection_name']}")
        print(f"Total courses: {stats['document_count']}")
        print(f"Embedding model: {stats['embedding_model']}")
        print(f"Persist directory: {stats['persist_dir']}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    print()
    print("Type 'q', 'quit', or 'exit' to exit.")
    print("-" * 70)

    while True:
        print()
        try:
            query = input("Your query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not query:
            print("   Empty query, try again.")
            continue

        if query.lower() in ("q", "quit", "exit"):
            print("\nGoodbye!")
            break

        # Perform search and measure time
        try:
            start_time = time.time()
            results = search_courses(query, n_results=5)
            elapsed = time.time() - start_time

            print(f"   Search time: {elapsed:.2f}s")
            print_search_results(results, show_text=True)

        except RuntimeError as e:
            print(f"   API Error: {e}")
        except ValueError as e:
            print(f"   Index Error: {e}")
        except Exception as e:
            print(f"   Unexpected Error: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    interactive_demo()
