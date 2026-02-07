"""
TUM Smart Course Assistant - Course Cards Merge Script

This script merges course_cards files from different PDFs.
For duplicate courses, category information is preserved as metadata.

Usage:
    python -m src.merge_course_cards

Output:
    data/processed/course_cards.jsonl (merged, with added metadata)
"""

import json
from pathlib import Path
from collections import defaultdict


def load_course_cards(path: Path) -> list[dict]:
    """Loads course cards from JSONL file."""
    cards = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cards.append(json.loads(line))
    return cards


def merge_course_cards(
    cards_by_source: dict[str, list[dict]],
) -> list[dict]:
    """
    Merges course cards from different sources.
    
    For duplicate courses:
    - First found content is used
    - All sources are stored in 'sources' list
    - 'also_in_electives' flag is added
    
    Args:
        cards_by_source: {"innovation": [...], "advanced": [...]} format
        
    Returns:
        Merged course cards list with added metadata
    """
    # course_id -> {card_data, sources} mapping
    merged = {}
    
    # Source -> Domain mapping
    # management: MMT (Management & Technology) courses
    # technology: Informatics specialization courses
    SOURCE_TO_DOMAIN = {
        "innovation": "management",
        "advanced": "management", 
        "informatics": "technology",
        # Newly added sources (December 2025)
        "chemistry": "technology",
        "industrial": "technology",
        "infotech": "technology",
        "econometri": "management",
        "finance": "management",
        "operations": "management",
    }
    
    # Priority order: Elective > Advanced Seminar > Informatics > New Sources
    # (Elective descriptions are generally more comprehensive)
    source_priority = [
        "innovation", "advanced", "informatics",
        "chemistry", "industrial", "infotech",
        "econometri", "finance", "operations"
    ]
    
    for source in source_priority:
        if source not in cards_by_source:
            continue
            
        for card in cards_by_source[source]:
            course_id = card["course_id"]
            
            if course_id not in merged:
                # First time seeing this - determine domain from source
                domain = SOURCE_TO_DOMAIN.get(source, "other")
                merged[course_id] = {
                    "card": card.copy(),
                    "sources": [source],
                    "domain": domain,
                }
            else:
                # Duplicate - just add source
                merged[course_id]["sources"].append(source)
    
    # Create final list
    result = []
    for course_id, data in merged.items():
        card = data["card"]
        sources = data["sources"]
        
        # Add metadata
        card["sources"] = sources
        card["domain"] = data["domain"]  # management or technology
        
        # If found in multiple sources, add also_in_electives flag
        if len(sources) > 1:
            card["also_in_electives"] = True
        else:
            card["also_in_electives"] = False
        
        # Determine category (from title)
        title_lower = card["title"].lower()
        if "advanced seminar" in title_lower:
            card["category"] = "advanced_seminar"
        else:
            card["category"] = "elective"
        
        result.append(card)
    
    return result


def save_course_cards(cards: list[dict], output_path: Path) -> None:
    """Saves course cards in JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("TUM Smart Course Assistant - Course Cards Merge")
    print("=" * 60)
    print()
    
    # Define source files
    processed_dir = Path("data/processed")
    sources = {
        "innovation": processed_dir / "course_cards_innovation.jsonl",
        "advanced": processed_dir / "course_cards_advanced.jsonl",
        "informatics": processed_dir / "course_cards_informatics.jsonl",
        # New sources (December 2025)
        "chemistry": processed_dir / "course_cards_chemistry.jsonl",
        "industrial": processed_dir / "course_cards_industrial.jsonl",
        "infotech": processed_dir / "course_cards_infotech.jsonl",
        "econometri": processed_dir / "course_cards_econometri.jsonl",
        "finance": processed_dir / "course_cards_finance.jsonl",
        "operations": processed_dir / "course_cards_operations.jsonl",
    }
    
    # Load files
    cards_by_source = {}
    total_cards = 0
    
    print("1. Loading source files:")
    for source_name, path in sources.items():
        if path.exists():
            cards = load_course_cards(path)
            cards_by_source[source_name] = cards
            total_cards += len(cards)
            print(f"   {source_name}: {len(cards)} course cards")
        else:
            print(f"   {source_name}: File not found ({path})")
    
    if not cards_by_source:
        print("\nNo source files found!")
        return
    
    print(f"\n   Total (before merge): {total_cards} cards")
    
    # Merge
    print("\n2. Merging...")
    merged_cards = merge_course_cards(cards_by_source)
    
    # Statistics
    duplicates = [c for c in merged_cards if c["also_in_electives"]]
    advanced_only = [c for c in merged_cards if c["category"] == "advanced_seminar" and not c["also_in_electives"]]
    elective_only = [c for c in merged_cards if c["category"] == "elective" and not c["also_in_electives"]]
    
    print(f"   Unique course count: {len(merged_cards)}")
    print(f"   Advanced Seminar only: {len(advanced_only)}")
    print(f"   Elective only: {len(elective_only)}")
    print(f"   Found in both: {len(duplicates)}")
    
    if duplicates:
        print("\n   Courses found in both categories:")
        for card in duplicates:
            print(f"      - [{card['course_id']}] {card['title'][:60]}...")
    
    # Save
    output_path = processed_dir / "course_cards.jsonl"
    print(f"\n3. Saving: {output_path}")
    save_course_cards(merged_cards, output_path)
    
    print()
    print("=" * 60)
    print("Merge completed!")
    print()
    print("Next step: python -m src.build_index")
    print("=" * 60)


if __name__ == "__main__":
    main()
