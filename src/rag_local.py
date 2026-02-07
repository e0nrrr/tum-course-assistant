"""
Course search on LOCAL Chroma index (RAG helper).

This module:
    - Uses LOCAL index created by src/build_index_local.py
    - Reuses compute_local_embedding function
    - Embeds user query and retrieves most relevant courses from Chroma

NOTE:
    - This is for TESTING only.
    - Real semantic quality is not expected (using fake embedding).
"""

import json
from dataclasses import dataclass

from build_index_local import (
    compute_local_embedding,
    get_chroma_collection_local,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CourseResult:
    """Course information returned as search result."""

    course_id: str
    title: str
    pages: list[int]
    raw_text: str
    distance: float  # Lower = more similar

    def __str__(self) -> str:
        pages_str = f"[{self.pages[0]}..{self.pages[-1]}]" if len(self.pages) > 1 else f"[{self.pages[0]}]"
        return f"[{self.course_id}] {self.title} (pages: {pages_str}, distance: {self.distance:.4f})"


# =============================================================================
# SEARCH FUNCTION
# =============================================================================


def search_courses_local(
    query: str,
    n_results: int = 5,
) -> list[CourseResult]:
    """
    Searches courses in LOCAL Chroma index.

    Args:
        query: User's natural language query
        n_results: Maximum number of results to return

    Returns:
        CourseResult list (sorted by distance, most similar first)

    Example:
        >>> results = search_courses_local("product management innovation", n_results=3)
        >>> for r in results:
        ...     print(r)
    """
    # Embed the query
    query_embedding = compute_local_embedding(query)

    # Get Chroma collection
    collection = get_chroma_collection_local()

    # Run query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Convert results to CourseResult list
    course_results = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc_id, doc, metadata, distance in zip(ids, documents, metadatas, distances):
        # pages stored as JSON string, parse it
        pages = json.loads(metadata["pages"])

        course_result = CourseResult(
            course_id=doc_id,
            title=metadata["title"],
            pages=pages,
            raw_text=doc,
            distance=distance,
        )
        course_results.append(course_result)

    return course_results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def print_search_results(results: list[CourseResult], show_text: bool = False) -> None:
    """
    Prints search results in formatted output.

    Args:
        results: CourseResult list
        show_text: If True, also shows first 300 characters of raw_text
    """
    if not results:
        print("No results found.")
        return

    print(f"\n{len(results)} courses found:\n")
    print("-" * 60)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")

        if show_text:
            # Show first 300 characters
            preview = result.raw_text[:300].replace("\n", " ")
            print(f"   Preview: {preview}...")

        print()


def get_index_stats() -> dict:
    """
    Returns statistics about LOCAL Chroma index.

    Returns:
        {"collection_name": str, "document_count": int}
    """
    collection = get_chroma_collection_local()
    return {
        "collection_name": collection.name,
        "document_count": collection.count(),
    }


# =============================================================================
# INTERACTIVE DEMO
# =============================================================================


def interactive_demo() -> None:
    """
    Interactive course search demo from command line.
    """
    print("=" * 60)
    print("TUM Smart Course Assistant - LOCAL RAG Demo")
    print("=" * 60)
    print()
    print("WARNING: This demo does NOT perform REAL semantic search (fake embedding).")
    print("WARNING: Should only be used for pipeline testing.")
    print()

    # Show index statistics
    stats = get_index_stats()
    print(f"Index: {stats['collection_name']}")
    print(f"Total courses: {stats['document_count']}")
    print()
    print("Type 'q' or 'quit' to exit.")
    print("-" * 60)

    while True:
        print()
        query = input("Your query: ").strip()

        if not query:
            print("   Empty query, try again.")
            continue

        if query.lower() in ("q", "quit", "exit"):
            print("\nGoodbye!")
            break

        # Perform search
        try:
            results = search_courses_local(query, n_results=5)
            print_search_results(results, show_text=True)
        except Exception as e:
            print(f"   Error: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    interactive_demo()
