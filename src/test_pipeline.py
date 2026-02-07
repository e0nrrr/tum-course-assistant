"""Quick test to diagnose pipeline issues."""

import traceback

def main():
    print("=== CONFIG ===")
    try:
        from config import CHROMA_PERSIST_DIR, OPENAI_API_KEY, CHROMA_COLLECTION_NAME
        print(f"CHROMA_PERSIST_DIR: {CHROMA_PERSIST_DIR}")
        print(f"CHROMA_COLLECTION_NAME: {CHROMA_COLLECTION_NAME}")
        api_key_status = f"{OPENAI_API_KEY[:20]}..." if OPENAI_API_KEY else "NOT SET"
        print(f"OPENAI_API_KEY: {api_key_status}")
        
        from pathlib import Path
        p = Path(CHROMA_PERSIST_DIR)
        print(f"\nChromaDB dir exists: {p.exists()}")
    except Exception as e:
        print(f"CONFIG ERROR: {e}")
        traceback.print_exc()
        return
    
    print("\n=== TESTING CHROMA ===")
    try:
        from rag_pipeline import get_chroma_collection
        col = get_chroma_collection()
        print(f"Collection count: {col.count()}")
    except Exception as e:
        print(f"CHROMA ERROR: {e}")
        traceback.print_exc()
        return
    
    print("\n=== TESTING OPENAI ===")
    try:
        from rag_pipeline import get_openai_client, embed_query
        client = get_openai_client()
        emb = embed_query("test query", client)
        print(f"Embedding length: {len(emb)}")
    except Exception as e:
        print(f"OPENAI ERROR: {e}")
        traceback.print_exc()
        return
    
    print("\n=== TESTING SEARCH ===")
    try:
        from rag_pipeline import search_courses
        results = search_courses("AI courses", n_results=3)
        print(f"Search results: {len(results)} courses found")
        for r in results:
            title = r.title[:50] if len(r.title) > 50 else r.title
            print(f"  - {r.course_id}: {title}")
    except Exception as e:
        print(f"SEARCH ERROR: {e}")
        traceback.print_exc()
        return
    
    print("\n=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    main()
