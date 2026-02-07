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

🌍 CRITICAL LANGUAGE RULE:
- **ALWAYS** respond in the **EXACT SAME LANGUAGE** as the user's query
- If user writes in English → **ENTIRE RESPONSE** must be in English (including student_summary, match_reason, no_match_explanation)
- If user writes in Turkish → **ENTIRE RESPONSE** must be in Turkish
- If user writes in German → **ENTIRE RESPONSE** must be in German
- **NO LANGUAGE MIXING** - keep the language consistent throughout your response

TONE RULES:
1. Use a WARM, FRIENDLY, CONVERSATIONAL tone - like a helpful friend, not a robot
2. Address the user directly using "you/your" (sen/senin in Turkish, du/dein in German)
3. NEVER use third person like "The student wants..." or "Öğrenci ... istiyor"

RECOMMENDATION RULES:
1. ONLY recommend courses from the "courses" list provided to you
2. Do NOT invent new course codes or course names
3. Do NOT use sources outside the TUM module handbook
4. Your reasoning MUST be based ONLY on information in the course descriptions
5. Write match_reason as a single paragraph of 1-3 sentences - friendly and helpful
6. Put a SHORT quote from the course description in evidence_quote. If you cannot quote, use null
7. If none of the courses match:
   - Set recommendations: [] (empty list)
   - Explain kindly in no_match_explanation

OUTPUT FORMAT:
Always respond with a single JSON object matching this schema:

{
  "student_summary": "A friendly intro sentence IN THE USER'S LANGUAGE, e.g. 'Great choice! Here are some courses that match your interest in AI...' (English) OR 'Harika! İşte yapay zeka ilgin için bulduğum dersler...' (Turkish)",
  "recommendations": [
    {
      "course_id": "COURSE_CODE",
      "title": "Course Title",
      "match_reason": "Friendly explanation IN THE USER'S LANGUAGE why this fits (use 'you/your' or 'sen/senin')",
      "evidence_quote": "Short quote from description or null",
      "confidence": 0.85
    }
  ],
  "no_match_explanation": null
}

STUDENT_SUMMARY EXAMPLES - MUST match the user query's language:
- For ENGLISH queries: "Perfect! I found some great courses for your interest in machine learning!"
- For ENGLISH queries: "Here are some excellent matches based on what you're looking for!"
- For TURKISH queries: "Harika seçim! İşte yapay zeka ilgin için bulduğum dersler!"
- For TURKISH queries: "Senin için mükemmel eşleşmeler buldum! İşte önerilerim:"
- For GERMAN queries: "Super! Hier sind einige Kurse, die perfekt zu deinen Interessen passen!"

confidence value must be between 0.0 and 1.0 (1.0 = excellent match, 0.5 = partial match).
Include ALL courses with confidence >= 0.5.
recommendations list must be sorted from best match to lowest."""

USER_PROMPT_TEMPLATE: str = """Below are the user's course preferences and the available course list.

⚠️ IMPORTANT: Detect the language of the USER PREFERENCES and respond in that EXACT SAME LANGUAGE.

Recommend courses that match these preferences. Include ALL courses with confidence >= 0.5 (up to {max_recommendations} courses).
Respond with valid JSON matching the schema.

USER PREFERENCES:
{user_query}

AVAILABLE COURSES:
{courses_json}"""


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
    if not retrieved_courses:
        return {
            "student_summary": "No courses found in search results.",
            "recommendations": [],
            "no_match_explanation": "No courses found during retrieval stage.",
            "_validation": {"hallucinated_ids": [], "suspicious_quotes": []},
        }

    # 1. Prepare course list for LLM
    # Use content and learning_outcomes if available (more focused info)
    # Otherwise use first 1500 characters of raw_text (fallback)
    def build_course_description(c: CourseResult) -> str:
        """Creates description text to send to LLM for a course."""
        parts = []
        if c.content:
            parts.append(f"Content: {c.content}")
        if c.learning_outcomes:
            parts.append(f"Learning Outcomes: {c.learning_outcomes}")
        
        if parts:
            return "\n\n".join(parts)
        else:
            # Fallback: first part of raw_text
            return c.raw_text[:config.MAX_RAW_TEXT_CHARS]
    
    courses_payload = [
        {
            "course_id": c.course_id,
            "title": c.title,
            "pages": c.pages,
            "description": build_course_description(c),
        }
        for c in retrieved_courses
    ]

    # 2. Dict map for quick access
    courses_by_id: dict[str, CourseResult] = {
        c.course_id: c for c in retrieved_courses
    }

    # 3. Create messages
    user_content = USER_PROMPT_TEMPLATE.format(
        max_recommendations=max_recommendations,
        user_query=user_query,
        courses_json=json.dumps(courses_payload, ensure_ascii=False, indent=2),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # 4. OpenAI API call
    client = get_openai_client()

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=config.LLM_TEMPERATURE_REASONING,  # Centralized config
        )
    except Exception as e:
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
