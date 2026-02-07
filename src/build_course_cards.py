import json
import re
from pathlib import Path


# =============================================================================
# JSONL READING FUNCTION
# =============================================================================


def load_pages_from_jsonl(path: str) -> list[dict]:
    """
    Reads page data from a JSONL file.

    Args:
        path: Path to the JSONL file

    Returns:
        List sorted by page number: [{"page": int, "text": str}, ...]
    """
    pages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                page_data = json.loads(line)
                pages.append(page_data)

    # Sort by page number
    pages.sort(key=lambda x: x["page"])
    return pages


# =============================================================================
# COURSE HEADER DETECTION FUNCTION
# =============================================================================

# Course code pattern: 2-4 uppercase letters + 3-9 digits
# Examples: WI000116, IN2339, SOT10028, MGT001308, CIT4230001
COURSE_HEADER_PATTERN = re.compile(r"^\s*([A-Z]{2,4}\d{3,9}):\s*(.+)$")


def detect_course_header(text: str) -> tuple[str | None, str | None]:
    """
    Searches for course header pattern in the first lines of page text.

    TUM module handbook format is typically:
        "WI000116: Lead User Project | Lead User Projekt"
        "SOT10028: Some Course Title"
        "MGT001308: Another Course"

    Args:
        text: Page text

    Returns:
        (course_id, title) tuple. Returns (None, None) if not found.
        title: English title before "|" character (if present)
    """
    lines = text.split("\n")

    # Check first 8 lines (sometimes header may be slightly below)
    for line in lines[:8]:
        line = line.strip()
        if not line:
            continue

        match = COURSE_HEADER_PATTERN.match(line)
        if match:
            course_id = match.group(1)
            full_title = match.group(2).strip()

            # If "|" exists, take the first part (usually English title)
            if "|" in full_title:
                title = full_title.split("|")[0].strip()
            else:
                title = full_title

            return course_id, title

    return None, None


# =============================================================================
# METADATA EXTRACTION FUNCTIONS
# =============================================================================

# Metadata patterns (TUM module handbook format)
METADATA_PATTERNS = {
    "level": re.compile(r"Module Level:\s*\n?\s*(\w+)", re.IGNORECASE),
    "language": re.compile(r"Language:\s*\n?\s*([\w/]+)", re.IGNORECASE),  # Added "/" for German/English
    "duration": re.compile(r"Duration:\s*\n?\s*([^\n]+)", re.IGNORECASE),
    "frequency": re.compile(r"Frequency:\s*\n?\s*([^\n]+)", re.IGNORECASE),
    "sws": re.compile(r"(\d+)\s*SWS", re.IGNORECASE),
}


def extract_metadata(raw_text: str) -> dict:
    """
    Extracts course metadata from raw_text.
    
    Extracted fields:
        - level: Master, Bachelor, etc.
        - language: English, German, etc.
        - duration: one semester, two semesters, etc.
        - frequency: winter semester, summer semester, etc.
        - sws: Semester Weekly Hours (e.g., 4)
    
    Args:
        raw_text: Raw text of the course card
    
    Returns:
        Metadata dictionary (fields not found return as None)
    """
    metadata = {}
    
    for field, pattern in METADATA_PATTERNS.items():
        match = pattern.search(raw_text)
        if match:
            value = match.group(1).strip()
            # Some cleanup operations
            if field == "level":
                # Normalize to "Master" or "Bachelor"
                value = value.capitalize()
            elif field == "language":
                # Normalize to "English" or "German"
                value = value.capitalize()
            elif field == "sws":
                # Convert to number
                try:
                    value = int(value)
                except ValueError:
                    value = None
            metadata[field] = value
        else:
            metadata[field] = None
    
    return metadata


# =============================================================================
# SECTION EXTRACTION FUNCTIONS (Content, Learning Outcomes)
# =============================================================================

# Section patterns: Extract text between Content and Learning Outcomes, Learning Outcomes and Teaching Methods
SECTION_PATTERNS = {
    # Content: Text between "Content:" and "Intended Learning Outcomes:"
    "content": re.compile(
        r"Content:\s*\n(.+?)(?=Intended Learning Outcomes:|Teaching and Learning Methods:|$)",
        re.IGNORECASE | re.DOTALL
    ),
    # Learning Outcomes: Text between "Intended Learning Outcomes:" and "Teaching and Learning Methods:"
    "learning_outcomes": re.compile(
        r"Intended Learning Outcomes:\s*\n(.+?)(?=Teaching and Learning Methods:|Media:|Reading List:|$)",
        re.IGNORECASE | re.DOTALL
    ),
}


def extract_sections(raw_text: str) -> dict:
    """
    Extracts Content and Intended Learning Outcomes sections from raw_text.
    
    This function parses structured sections in the TUM module handbook:
        - Content: Course content and topics
        - Intended Learning Outcomes: Learning outcomes of the course
    
    Args:
        raw_text: Raw text of the course card
    
    Returns:
        Sections dictionary:
            {
                "content": "Course content...",
                "learning_outcomes": "Learning outcomes..."
            }
        Fields not found return as None.
    """
    sections = {}
    
    for field, pattern in SECTION_PATTERNS.items():
        match = pattern.search(raw_text)
        if match:
            value = match.group(1).strip()
            # Clean up excessive line breaks
            value = re.sub(r'\n{3,}', '\n\n', value)
            # Clean leading/trailing whitespace
            value = value.strip()
            # Only save if meaningful content exists (at least 50 characters)
            if len(value) >= 50:
                sections[field] = value
            else:
                sections[field] = None
        else:
            sections[field] = None
    
    return sections


# =============================================================================
# COURSE CARD CREATION FUNCTION
# =============================================================================


def build_course_cards(pages: list[dict]) -> list[dict]:
    """
    Creates course cards from page list.

    Algorithm:
        - Iterate through pages sequentially
        - Search for course header on each page
        - Merge if same course spans multiple pages
        - Pages without headers are added to previous course
        - Metadata (level, language, duration, sws) is extracted

    Args:
        pages: Page list in [{"page": int, "text": str}, ...] format

    Returns:
        Course card list:
        [
            {
                "course_id": "WI000116",
                "title": "Lead User Project",
                "pages": [1, 2, 3],
                "raw_text": "merged text...",
                "level": "Master",
                "language": "German",
                "duration": "one semester",
                "frequency": "winter/summer semester",
                "sws": 4
            },
            ...
        ]
    """
    course_cards = []
    current_course = None
    skipped_pages = []  # To log skipped pages

    for page in pages:
        page_num = page["page"]
        page_text = page["text"]

        course_id, title = detect_course_header(page_text)

        if course_id is not None:
            # New course header found
            if current_course is None:
                # First course
                current_course = {
                    "course_id": course_id,
                    "title": title,
                    "pages": [page_num],
                    "raw_text": page_text,
                }
            elif current_course["course_id"] == course_id:
                # Same course continues (rare but possible)
                current_course["pages"].append(page_num)
                current_course["raw_text"] += "\n\n" + page_text
            else:
                # Different course started
                # First add metadata and sections to current course and save
                metadata = extract_metadata(current_course["raw_text"])
                current_course.update(metadata)
                sections = extract_sections(current_course["raw_text"])
                current_course.update(sections)
                course_cards.append(current_course)
                # Start new course
                current_course = {
                    "course_id": course_id,
                    "title": title,
                    "pages": [page_num],
                    "raw_text": page_text,
                }
        else:
            # Course header not found
            if current_course is not None:
                # Add as continuation of current course
                current_course["pages"].append(page_num)
                current_course["raw_text"] += "\n\n" + page_text
            else:
                # No course has started yet, skip this page
                # (probably table of contents, preface, etc.)
                skipped_pages.append(page_num)

    # Loop finished, add metadata and sections to last course and save
    if current_course is not None:
        metadata = extract_metadata(current_course["raw_text"])
        current_course.update(metadata)
        sections = extract_sections(current_course["raw_text"])
        current_course.update(sections)
        course_cards.append(current_course)

    # Log skipped pages
    if skipped_pages:
        print(f"  Warning: {len(skipped_pages)} pages skipped (before course header): {skipped_pages[:10]}...")

    return course_cards


# =============================================================================
# COURSE CARD SAVING FUNCTION
# =============================================================================


def save_course_cards_to_jsonl(cards: list[dict], output_path: str) -> None:
    """
    Saves course card list in JSONL format.

    Args:
        cards: Course card list
        output_path: Output file path

    Output Format (each line):
        {"course_id": "...", "title": "...", "pages": [...], "raw_text": "..."}
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")


# =============================================================================
# VALIDATION AND STATISTICS FUNCTIONS
# =============================================================================


def print_statistics(cards: list[dict]) -> None:
    """
    Prints summary statistics about course cards.

    Helps with validation checkpoints mentioned in README:
    - Very short (< 200 characters) cards
    - Very long (> 10,000 characters) cards
    """
    if not cards:
        print("  No course cards were created!")
        return

    text_lengths = [len(card["raw_text"]) for card in cards]
    page_counts = [len(card["pages"]) for card in cards]

    print(f"\n  Total course cards: {len(cards)}")
    print(f"  Average text length: {sum(text_lengths) / len(text_lengths):.0f} characters")
    print(f"  Min text length: {min(text_lengths)} characters")
    print(f"  Max text length: {max(text_lengths)} characters")
    print(f"  Average pages/course: {sum(page_counts) / len(page_counts):.1f}")

    # Detect potentially problematic cards
    short_cards = [c for c in cards if len(c["raw_text"]) < 200]
    long_cards = [c for c in cards if len(c["raw_text"]) > 10000]

    if short_cards:
        print(f"\n  Very short cards (< 200 characters): {len(short_cards)}")
        for card in short_cards[:5]:
            print(f"      - {card['course_id']}: {card['title'][:50]}... ({len(card['raw_text'])} chars)")

    if long_cards:
        print(f"\n  Very long cards (> 10,000 characters): {len(long_cards)}")
        for card in long_cards[:5]:
            print(f"      - {card['course_id']}: {card['title'][:50]}... ({len(card['raw_text'])} chars)")


def print_sample_cards(cards: list[dict], n: int = 5) -> None:
    """
    Prints summary of first n course cards (for manual verification).
    """
    print(f"\n  First {min(n, len(cards))} course card preview:")
    print("  " + "-" * 60)

    for card in cards[:n]:
        print(f"  [{card['course_id']}] {card['title']}")
        print(f"      Pages: {card['pages']}")
        # Show first 150 characters
        preview = card["raw_text"][:150].replace("\n", " ")
        print(f"      Preview: {preview}...")
        print()


# =============================================================================
# MAIN FLOW
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # STEPS FOR ADDING A NEW PDF:
    # 1. Set INPUT_JSONL to the file output from preprocessing
    # 2. Change OUTPUT_JSONL to a new name (e.g., course_cards_advanced.jsonl)
    # 3. Run: python -m src.build_course_cards
    # 4. Then merge all course_cards:
    #    cat data/processed/course_cards_*.jsonl > data/processed/course_cards.jsonl
    # 5. Update index: python -m src.build_index
    # =========================================================================
    
    # Input and output file paths
    INPUT_JSONL = "data/processed/pages_advanced.jsonl"
    OUTPUT_JSONL = "data/processed/course_cards_advanced.jsonl"
    
    # Example for new PDF (uncomment and comment out above):
    # INPUT_JSONL = "data/processed/pages_innovation.jsonl"
    # OUTPUT_JSONL = "data/processed/course_cards_innovation.jsonl"

    print("=" * 60)
    print("TUM Smart Course Assistant - Course Card Creation (ETL Step 2)")
    print("=" * 60)
    print()

    # 1. Load pages from JSONL
    print(f"1. Loading pages: {INPUT_JSONL}")
    pages = load_pages_from_jsonl(INPUT_JSONL)
    print(f"   Total pages: {len(pages)}")

    # 2. Create course cards
    print(f"\n2. Creating course cards...")
    cards = build_course_cards(pages)

    # 3. Show statistics
    print("\n3. Statistics:")
    print_statistics(cards)

    # 4. Show sample cards (for manual verification)
    print_sample_cards(cards, n=5)

    # 5. Save to JSONL
    print(f"4. Saving course cards: {OUTPUT_JSONL}")
    save_course_cards_to_jsonl(cards, OUTPUT_JSONL)

    print()
    print("=" * 60)
    print("ETL Step 2 completed!")
    print(f"Output file: {OUTPUT_JSONL}")
    print("Next step: Embedding and ChromaDB indexing")
    print("=" * 60)
