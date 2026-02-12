"""
TUM Smart Course Assistant - LLM Reasoning Layer

This module evaluates RAG retrieval results with LLM and
generates ranked recommendations based on student preferences.

Architecture (README Section 7):
    1. Course cards from RAG
    2. To LLM: student preferences + course list
    3. From LLM: ranked recommendations + justifications (JSON)
    4. Hallucination check
    5. Validated result

Usage:
    from rag_pipeline import search_courses
    from llm_reasoning import rank_courses_with_llm

    retrieved = search_courses("product management innovation", n_results=8)
    result = rank_courses_with_llm("I don't like finance, I want project-based courses", retrieved)

Demo:
    cd src && python llm_reasoning.py
"""

import json
from typing import Any

from openai import OpenAI

import config
from config import get_openai_client
from rag_pipeline import CourseResult, search_courses


# =============================================================================
# CONSTANTS - Use centralized config values
# =============================================================================

# Maximum characters for course description (token savings)
# Now imported from config.MAX_RAW_TEXT_CHARS for DRY compliance

# System prompt (aligned with README rules)
SYSTEM_PROMPT: str = """You are a friendly TUM (Technische Universität München) course advisor chatbot.

CRITICAL RULES:

1. COURSE RELEVANCE & RANKING:
   - The courses below have been pre-filtered by semantic search and metadata filters.
   - Each course has a "retrieval_similarity" score (0.0-1.0).
   - TWO types of queries exist:

   A) BROAD/DISCOVERY queries (e.g., "management courses", "German courses", "courses in German language"):
      - The user wants to EXPLORE available courses in a category.
      - The courses have ALREADY been filtered by domain/language metadata.
      - Almost ALL provided courses are relevant — recommend generously.
      - Rank by how interesting/useful each course seems, but include most of them.
      - Give confidence >= 0.6 to all courses that belong to the requested category.

   B) SPECIFIC TOPIC queries (e.g., "AI courses", "entrepreneurship courses", "data science"):
      - The user wants courses about a SPECIFIC topic.
      - Courses whose TITLE or CONTENT directly match the topic should get HIGHER confidence.
      - Courses that only tangentially mention the topic should get LOWER confidence.
      - Example: For "AI courses", "Fundamentals of AI" > "Marketing Seminar mentioning AI".
      - If a course is NOT related to the topic, give it confidence < 0.5.

   - Do NOT say "no courses found" if there are genuinely relevant courses in the list.

2. RESPONSE LANGUAGE:
   - The response_language field tells you which language to use.
   - You MUST write your ENTIRE response in that language.
   - Do NOT confuse course teaching language with response language.
   - "show me German courses" = English sentence → respond in English.

3. TONE: Warm, friendly, conversational. Use "you/your" directly.

4. RECOMMENDATIONS:
   - ONLY use courses from the provided list.
   - Do NOT invent course codes or names.
   - Write match_reason as 1-3 friendly sentences.
   - evidence_quote: short quote from description, or null.

OUTPUT FORMAT (JSON only):
{
  "student_summary": "Friendly intro sentence",
  "recommendations": [
    {
      "course_id": "CODE",
      "title": "Title",
      "match_reason": "Why this fits",
      "evidence_quote": "quote or null",
      "confidence": 0.85
    }
  ],
  "no_match_explanation": null
}

confidence: 0.0-1.0. Include ONLY courses with confidence >= 0.5.
Sort from best to lowest match.
If NO course truly matches the user's request, set no_match_explanation."""

USER_PROMPT_TEMPLATE: str = """response_language: {response_language}

USER REQUEST: {user_query}

APPLIED FILTERS: {applied_filters}

RETRIEVED COURSES ({num_courses} courses):
{courses_json}

IMPORTANT:
- If the user asked for a BROAD category (e.g., "management courses", "German courses"), these courses have already been filtered — recommend most of them with confidence >= 0.6.
- If the user asked for a SPECIFIC topic, only recommend genuinely relevant courses.
- Write your ENTIRE response in {response_language}.
- Include up to {max_recommendations} courses with confidence >= 0.5."""


# =============================================================================
# OPENAI CLIENT - Use centralized singleton from config
# =============================================================================
# Imported from config.get_openai_client for DRY compliance


# =============================================================================
# HALLUCINATION CHECK FUNCTION
# =============================================================================


def validate_llm_response(
    response: dict,
    retrieved_courses: dict[str, CourseResult],
) -> dict:
    """
    Validates LLM response in two stages (README Section 7.3).

    Stage 1: Is course_id in the retrieval list?
    Stage 2: Does evidence_quote actually appear in course content?

    Args:
        response: Parsed JSON from LLM
        retrieved_courses: course_id -> CourseResult mapping

    Returns:
        Validated response dict (_validation field added)
    """
    valid_recommendations = []
    hallucinated_ids = []
    suspicious_quotes = []

    for rec in response.get("recommendations", []):
        course_id = rec.get("course_id", "")

        # Stage 1: course_id check
        if course_id not in retrieved_courses:
            hallucinated_ids.append(course_id)
            continue  # Skip this recommendation

        # Stage 2: evidence_quote check (if present)
        evidence = rec.get("evidence_quote")
        if evidence and evidence != "null":
            course = retrieved_courses[course_id]
            # Search in combination of content, learning_outcomes and raw_text
            search_text = " ".join(filter(None, [
                course.content,
                course.learning_outcomes,
                course.raw_text
            ]))
            # Simple substring check (case-insensitive)
            # Note: LLM may slightly modify quote, just log for now
            if evidence.lower() not in search_text.lower():
                suspicious_quotes.append({
                    "course_id": course_id,
                    "quote": evidence,
                })
                # Don't skip recommendation for now, just log

        valid_recommendations.append(rec)

    # Update result
    response["recommendations"] = valid_recommendations
    response["_validation"] = {
        "hallucinated_ids": hallucinated_ids,
        "suspicious_quotes": suspicious_quotes,
    }

    return response


# =============================================================================
# MAIN LLM REASONING FUNCTION
# =============================================================================


def rank_courses_with_llm(
    user_query: str,
    retrieved_courses: list[CourseResult],
    max_recommendations: int = 5,
    applied_filters: dict | None = None,
) -> dict:
    """
    Evaluates RAG results with LLM and generates ranked recommendations.

    Args:
        user_query: User's preferences in natural language
                   (e.g., "I don't like finance, I want project-based courses")
        retrieved_courses: CourseResult list from RAG
        max_recommendations: Maximum number of recommendations

    Returns:
        Dict conforming to JSON schema:
        {
            "student_summary": "...",
            "recommendations": [...],
            "no_match_explanation": null,
            "_validation": {...}
        }

    Raises:
        RuntimeError: If API key is not defined
        ValueError: If LLM returns invalid JSON

    Example:
        >>> from src.rag_pipeline import search_courses
        >>> retrieved = search_courses("AI machine learning", n_results=8)
        >>> result = rank_courses_with_llm("I want AI courses", retrieved)
        >>> print(result["recommendations"][0]["title"])
    """
    import time as _time
    _start_total = _time.time()
    print(f"[DEBUG] LLM ranking started with {len(retrieved_courses)} courses")
    
    if not retrieved_courses:
        return {
            "student_summary": "No courses found in search results.",
            "recommendations": [],
            "no_match_explanation": "No courses found during retrieval stage.",
            "_validation": {"hallucinated_ids": [], "suspicious_quotes": []},
        }

    # 1. Detect response language from the user query
    def detect_response_language(query: str) -> str:
        """Detect what language the user WROTE in (not what they're asking about).
        
        IMPORTANT: German and Turkish share ü/ö characters.
        We must check German words FIRST to avoid misclassifying German as Turkish.
        Turkish-exclusive chars (ş, ğ, ı, ç) are reliable Turkish indicators.
        """
        import re
        q = query.lower()
        words = set(re.findall(r'\b\w+\b', q))
        
        # Turkish-EXCLUSIVE characters (NOT shared with German)
        turkish_only_chars = set('şğı')
        # Turkish-specific words (NOT shared with German)
        turkish_words = {'ders', 'dersler', 'hakkında', 'istiyorum', 'arıyorum',
                         'göster', 'ilgili', 'için', 'nasıl', 'nedir', 'yapay',
                         'zeka', 'bul', 'bana', 'ile', 'mı', 'mi', 'mu', 'mü',
                         'var', 'yok', 'hangi', 'lütfen', 'teşekkür'}
        # German-specific words
        german_words = {'und', 'ich', 'möchte', 'suche', 'kurse', 'über', 'kurs',
                        'finden', 'zeigen', 'lernen', 'wie', 'können', 'welche',
                        'gibt', 'nicht', 'auch', 'oder', 'aber', 'haben',
                        'für', 'mit', 'von', 'ein', 'eine', 'der', 'die', 'das',
                        'ist', 'sind', 'was', 'gibt', 'bitte', 'danke',
                        'vorlesung', 'vorlesungen', 'fach', 'fächer', 'seminar',
                        'softwareentwicklung', 'informatik', 'veranstaltung'}
        
        # Check German FIRST (German words are very distinctive)
        if words & german_words:
            return "German"
        # German-exclusive chars: ä, ß (Turkish doesn't use these)
        if any(c in set('äß') for c in q):
            return "German"
        
        # Then check Turkish
        if any(c in turkish_only_chars for c in q):
            return "Turkish"
        if words & turkish_words:
            return "Turkish"
        # ç with Turkish words context (ç exists in both but rare in German)
        if 'ç' in q and not (words & german_words):
            return "Turkish"
        
        return "English"

    response_language = detect_response_language(user_query)
    print(f"[DEBUG] Detected response language: {response_language}")

    # 2. Prepare course list for LLM
    def build_course_description(c: CourseResult) -> str:
        """Creates description text to send to LLM for a course."""
        parts = [f"Teaching Language: {c.language}"]
        if c.content:
            parts.append(f"Content: {c.content}")
        if c.learning_outcomes:
            parts.append(f"Learning Outcomes: {c.learning_outcomes}")
        if len(parts) == 1:
            # Only language line, no content/outcomes - use raw_text
            parts.append(c.raw_text[:config.MAX_RAW_TEXT_CHARS])
        return "\n".join(parts)
    
    courses_payload = [
        {
            "course_id": c.course_id,
            "title": c.title,
            "teaching_language": c.language,
            "retrieval_similarity": round(c.similarity, 3),
            "description": build_course_description(c),
        }
        for c in retrieved_courses
    ]

    # 2. Dict map for quick access
    courses_by_id: dict[str, CourseResult] = {
        c.course_id: c for c in retrieved_courses
    }

    # 3. Create messages
    # Build human-readable filter description for LLM
    if applied_filters:
        filter_parts = []
        if isinstance(applied_filters, dict):
            # Handle $and wrapped filters
            filter_items = applied_filters.get('$and', [applied_filters])
            for f in filter_items:
                for k, v in f.items():
                    if k == 'domain':
                        filter_parts.append(f"Domain: {v}")
                    elif k == 'language':
                        filter_parts.append(f"Language: German (courses taught in German)")
        applied_filters_str = ", ".join(filter_parts) if filter_parts else "None"
    else:
        applied_filters_str = "None (pure semantic search)"
    
    user_content = USER_PROMPT_TEMPLATE.format(
        max_recommendations=max_recommendations,
        user_query=user_query,
        response_language=response_language,
        num_courses=len(courses_payload),
        courses_json=json.dumps(courses_payload, ensure_ascii=False, indent=2),
        applied_filters=applied_filters_str,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # 4. OpenAI API call
    client = get_openai_client()
    _t0 = _time.time()
    print(f"[DEBUG] Calling LLM API ({config.LLM_MODEL_NAME})...")

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=config.LLM_TEMPERATURE_REASONING,  # Centralized config
        )
        print(f"[DEBUG] LLM API response: {_time.time() - _t0:.2f}s")
    except Exception as e:
        print(f"[DEBUG] LLM API FAILED after {_time.time() - _t0:.2f}s: {e}")
        raise RuntimeError(f"OpenAI API error: {e}")

    # 5. Parse response
    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}\nContent: {content[:500]}")

    # 6. Add default fields (if LLM omits them)
    if "student_summary" not in parsed:
        parsed["student_summary"] = "Student preferences analyzed."
    if "recommendations" not in parsed:
        parsed["recommendations"] = []
    if "no_match_explanation" not in parsed:
        parsed["no_match_explanation"] = None

    # 7. Normalize confidence values
    for rec in parsed.get("recommendations", []):
        if "confidence" in rec and rec["confidence"] is not None:
            try:
                rec["confidence"] = float(rec["confidence"])
            except (ValueError, TypeError):
                rec["confidence"] = 0.5
        else:
            rec["confidence"] = None

    # 8. Hallucination check
    validated = validate_llm_response(parsed, courses_by_id)

    print(f"[DEBUG] LLM ranking total: {_time.time() - _start_total:.2f}s | Recommended {len(validated.get('recommendations', []))} courses")
    return validated


# =============================================================================
# HELPER PRINT FUNCTION
# =============================================================================


def print_reasoning_result(result: dict) -> None:
    """
    Prints LLM reasoning result in formatted output.

    Args:
        result: Output from rank_courses_with_llm()
    """
    print("\n" + "=" * 70)
    print("LLM REASONING RESULT")
    print("=" * 70)

    # Student summary
    print(f"\nStudent Profile:")
    print(f"   {result.get('student_summary', 'N/A')}")

    # Recommendations
    recommendations = result.get("recommendations", [])
    if recommendations:
        print(f"\nRecommended Courses ({len(recommendations)} total):")
        print("-" * 70)

        for i, rec in enumerate(recommendations, 1):
            course_id = rec.get("course_id", "?")
            title = rec.get("title", "Unknown Course")
            match_reason = rec.get("match_reason", "No reason specified.")
            evidence = rec.get("evidence_quote")
            confidence = rec.get("confidence")

            print(f"\n{i}. [{course_id}] {title}")
            print(f"   Reason: {match_reason}")

            if evidence and evidence != "null":
                # Shorten quote
                if len(evidence) > 150:
                    evidence = evidence[:150] + "..."
                print(f"   Quote: \"{evidence}\"")

            if confidence is not None:
                conf_bar = "*" * int(confidence * 10) + "-" * (10 - int(confidence * 10))
                print(f"   Confidence: [{conf_bar}] {confidence:.0%}")
    else:
        print("\nNo recommendations found.")
        no_match = result.get("no_match_explanation")
        if no_match:
            print(f"   Reason: {no_match}")

    # Validation warnings
    validation = result.get("_validation", {})
    hallucinated = validation.get("hallucinated_ids", [])
    suspicious = validation.get("suspicious_quotes", [])

    if hallucinated or suspicious:
        print("\n" + "-" * 70)
        print("VALIDATION WARNINGS:")

        if hallucinated:
            print(f"   Hallucination (fabricated course codes): {hallucinated}")

        if suspicious:
            print(f"   Suspicious quotes ({len(suspicious)} total):")
            for sq in suspicious[:3]:  # Show first 3
                print(f"      - {sq['course_id']}: \"{sq['quote'][:50]}...\"")

    print("\n" + "=" * 70)


# =============================================================================
# DEMO FUNCTION
# =============================================================================


def demo_reasoning() -> None:
    """
    Runs full demo of LLM reasoning pipeline.

    Steps:
        1. Sample user query
        2. RAG retrieval (search_courses)
        3. LLM reasoning (rank_courses_with_llm)
        4. Print results
    """
    print("=" * 70)
    print("TUM Smart Course Assistant - LLM Reasoning Demo")
    print("=" * 70)

    # Sample user query
    user_query = (
        "I don't like finance and accounting. "
        "I want courses focused on product management, innovation and entrepreneurship. "
        "Preferably project-based courses with team work."
    )

    print(f"\nUser Query:")
    print(f"   \"{user_query}\"")

    # 1. RAG Retrieval
    print("\nPerforming RAG Retrieval...")
    try:
        retrieved = search_courses(user_query, n_results=8)
        print(f"   {len(retrieved)} courses found.")

        print("\n   Found courses:")
        for i, c in enumerate(retrieved, 1):
            print(f"   {i}. [{c.course_id}] {c.title} (sim: {c.similarity:.3f})")

    except Exception as e:
        print(f"   Retrieval error: {e}")
        return

    # 2. LLM Reasoning
    print("\nPerforming LLM Reasoning...")
    try:
        result = rank_courses_with_llm(
            user_query=user_query,
            retrieved_courses=retrieved,
            max_recommendations=3,
        )
        print("   LLM response received.")

    except RuntimeError as e:
        print(f"   API error: {e}")
        return
    except ValueError as e:
        print(f"   Parse error: {e}")
        return

    # 3. Print results
    print_reasoning_result(result)

    # 4. Show raw JSON (for debug)
    print("\nRaw JSON Output (debug):")
    print("-" * 70)
    # Show without _validation
    output = {k: v for k, v in result.items() if k != "_validation"}
    print(json.dumps(output, ensure_ascii=False, indent=2))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo_reasoning()
