"""
TUM Smart Course Assistant - PDF Preprocessing Module (ETL - Step 1)

This module performs the FIRST step of the ETL process described in README.md:
    PDF -> Clean Page Text

Purpose:
    - Extract raw text page by page from module handbook PDF
    - Clean unnecessary patterns like header/footer and page numbers
    - Write each page as a separate JSON line to JSONL file

NOTE: This module does not perform course card chunking yet.
      Chunking will be done in a separate step.
"""

import json
import re
from pathlib import Path

import pymupdf  # PyMuPDF


def extract_pages_to_jsonl(
    input_pdf_path: str,
    output_jsonl_path: str,
    page_start: int | None = None,
    page_end: int | None = None,
    apply_cleaning: bool = True,
) -> None:
    """
    Extracts text page by page from PDF file and saves in JSONL format.

    This function is the first step of the ETL process. Writes each page as a
    separate JSON line, enabling page-based processing in subsequent steps.

    Args:
        input_pdf_path: Path to source PDF file
        output_jsonl_path: Path to output JSONL file
        page_start: Start page number (1-indexed, inclusive). None starts from first page.
        page_end: End page number (1-indexed, inclusive). None processes until last page.
        apply_cleaning: If True, applies clean_page_text function

    Output Format (each line):
        {"page": <page_number>, "text": "<text>"}
    """
    # Ensure output directory exists
    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open PDF
    doc = pymupdf.open(input_pdf_path)
    total_pages = len(doc)

    # Determine page range (convert from 1-indexed to 0-indexed)
    start_idx = (page_start - 1) if page_start is not None else 0
    end_idx = page_end if page_end is not None else total_pages

    # Check bounds
    start_idx = max(0, start_idx)
    end_idx = min(total_pages, end_idx)

    print(f"PDF opened: {input_pdf_path}")
    print(f"Total pages: {total_pages}")
    print(f"Page range to process: {start_idx + 1} - {end_idx}")

    # Write to JSONL file
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for page_idx in range(start_idx, end_idx):
            page = doc[page_idx]
            raw_text = page.get_text("text")

            # Apply cleaning (if requested)
            if apply_cleaning:
                cleaned_text = clean_page_text(raw_text)
            else:
                cleaned_text = raw_text

            # Create and write JSON line
            page_data = {
                "page": page_idx + 1,  # 1-indexed page number
                "text": cleaned_text,
            }
            f.write(json.dumps(page_data, ensure_ascii=False) + "\n")

            # Show progress (every 100 pages)
            if (page_idx + 1) % 100 == 0:
                print(f"  Pages processed: {page_idx + 1}/{end_idx}")

    doc.close()
    print(f"Completed! Output file: {output_jsonl_path}")
    print(f"Total pages processed: {end_idx - start_idx}")


def clean_page_text(raw_text: str) -> str:
    """
    Cleans header/footer and unnecessary patterns from page text.

    This function ONLY performs safe cleaning:
    - Header/footer lines (Module Catalog, Generated on, etc.)
    - Page numbers (Page X of Y)
    - Unnecessary whitespace and empty lines

    NOTE: No aggressive cleaning is done, course headers are preserved.

    Args:
        raw_text: Raw page text extracted from PDF

    Returns:
        Cleaned text
    """
    text = raw_text

    # 1. Clean header patterns
    # Lines starting with "Module Catalog of the study program"
    text = re.sub(
        r"^Module Catalog of the study program.*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 2. Clean footer patterns
    # Lines starting with "Generated on"
    text = re.sub(
        r"^Generated on.*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 3. Clean page number patterns
    # "Page X of Y" or "X of Y" format (e.g., "Page 49 of 991" or "49 of 991")
    text = re.sub(
        r"^\s*(Page\s+)?\d+\s+of\s+\d+\s*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Lines with only page numbers (e.g., "123" or "- 123 -")
    text = re.sub(
        r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # 4. Clean unnecessary whitespace
    # Trailing whitespace on lines
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Reduce 3 or more consecutive empty lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Clean leading and trailing whitespace
    text = text.strip()

    return text


# =============================================================================
# MAIN FLOW
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # STEPS FOR ADDING A NEW PDF:
    # 1. Change INPUT_PDF to the path of new PDF
    # 2. Change OUTPUT_JSONL to a new name (e.g., pages_advanced.jsonl)
    # 3. Run: python -m src.preprocessing
    # 4. Then do the same in build_course_cards.py
    # =========================================================================
    
    # Input and output file paths
    INPUT_PDF = "data/raw/mmt_advanced_seminars_raw.pdf"
    OUTPUT_JSONL = "data/processed/pages_advanced.jsonl"
    
    # Example for new PDF (uncomment and comment out above):
    # INPUT_PDF = "data/raw/mmt_innovation_electives_raw.pdf"
    # OUTPUT_JSONL = "data/processed/pages_innovation.jsonl"

    # Page range (None = all pages)
    PAGE_START = None
    PAGE_END = None

    print("=" * 60)
    print("TUM Smart Course Assistant - PDF Preprocessing (ETL Step 1)")
    print("=" * 60)
    print()

    # Extract text from PDF and write to JSONL
    extract_pages_to_jsonl(
        input_pdf_path=INPUT_PDF,
        output_jsonl_path=OUTPUT_JSONL,
        page_start=PAGE_START,
        page_end=PAGE_END,
        apply_cleaning=True,
    )

    print()
    print("ETL Step 1 completed!")
    print("Next step: Course card chunking")
