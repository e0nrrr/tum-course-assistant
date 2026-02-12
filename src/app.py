"""
TUM Course Assistant - Chainlit Application

A clean, minimal chat interface for TUM course recommendations.
Uses RAG + LLM pipeline with intent-based conversation flow.

Run:
    cd src && chainlit run app.py

Architecture:
    User Message -> Intent Classification -> Route based on intent
        GREETING/CHITCHAT/HELP/OUT_OF_SCOPE: Conversational response (no search)
        SEARCH: RAG retrieval -> LLM ranking -> Display courses
        FOLLOWUP: Answer from context (no RAG)
"""

import chainlit as cl

from async_wrappers import (
    async_search_courses,
    async_rank_courses,
    async_classify_intent,
    async_answer_from_context,
    async_conversational_response
)
from rag_pipeline import CourseResult, detect_metadata_filters

# Constants
INITIAL_VISIBLE_COUNT = 10  # Show more courses initially for better discovery
MAX_RECOMMENDATIONS = 10  # Max courses to show to user
RAG_RETRIEVAL_COUNT = 30   # Retrieve more from ChromaDB, let LLM filter


# =============================================================================
# STARTERS
# =============================================================================

@cl.set_starters
async def set_starters():
    """
    Define example queries shown at chat start.
    Professional, no emojis, clear descriptions.
    """
    return [
        cl.Starter(
            label="AI and Machine Learning",
            message="Find courses about artificial intelligence and machine learning"
        ),
        cl.Starter(
            label="Innovation and Entrepreneurship",
            message="Show me courses on innovation, startups, and entrepreneurship"
        ),
        cl.Starter(
            label="Project-based Learning",
            message="I want project-based courses with practical experience"
        ),
        cl.Starter(
            label="German Language Courses",
            message="Find courses taught in German"
        ),
    ]


# =============================================================================
# HELPERS
# =============================================================================

def format_course_card(rec: dict, index: int) -> str:
    """
    Format a single course recommendation as clean Markdown.
    No emojis, professional appearance.
    """
    course_id = rec.get("course_id", "N/A")
    title = rec.get("title", "Untitled")
    reason = rec.get("match_reason", "")
    confidence = rec.get("confidence", 0)
    
    # Format confidence as percentage
    conf_pct = int(confidence * 100)
    
    lines = [
        f"### {index}. {title}",
        f"**Code:** `{course_id}` | **Match:** {conf_pct}%",
        "",
        reason,
    ]
    
    return "\n".join(lines)


def format_all_courses(recommendations: list[dict], visible_count: int) -> str:
    """
    Format visible courses as a single Markdown message.
    """
    visible = recommendations[:visible_count]
    
    parts = []
    for i, rec in enumerate(visible, 1):
        parts.append(format_course_card(rec, i))
    
    return "\n\n---\n\n".join(parts)


def build_course_context(recommendations: list[dict], visible_count: int) -> str:
    """
    Build context string from visible courses for follow-up questions.
    """
    visible = recommendations[:visible_count]
    
    context_parts = []
    for i, rec in enumerate(visible, 1):
        course_id = rec.get("course_id", "N/A")
        title = rec.get("title", "")
        reason = rec.get("match_reason", "")
        raw_text = rec.get("raw_text", "")
        
        part = f"""Course {i}: [{course_id}] {title}
Match Reason: {reason}
Details: {raw_text[:800]}"""
        context_parts.append(part)
    
    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# CHAINLIT LIFECYCLE
# =============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Initialize session state when chat begins."""
    cl.user_session.set("all_recommendations", [])
    cl.user_session.set("visible_count", INITIAL_VISIBLE_COUNT)
    cl.user_session.set("current_course_context", "")
    cl.user_session.set("chat_history", [])


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages with smart intent routing."""
    user_input = message.content.strip()
    
    if not user_input:
        return
    
    # Get session state
    all_recs = cl.user_session.get("all_recommendations", [])
    visible_count = cl.user_session.get("visible_count", INITIAL_VISIBLE_COUNT)
    course_context = cl.user_session.get("current_course_context", "")
    chat_history = cl.user_session.get("chat_history", [])
    
    has_context = bool(course_context)
    
    # Build recent messages for intent classification
    recent_msgs = chat_history[-6:] if chat_history else None
    
    # SAFETY CHECK: New course search queries must ALWAYS go through SEARCH
    # This prevents them from being answered from stale context (FOLLOWUP)
    import re
    _force_search_re = re.compile(
        r'('
        # Language preference queries
        r'\b(german|deutsch|almanca|english|ingilizce|englisch)\b.*\b(course|courses|class|ders|dersler|kurs|taught|language)\b|'
        r'\b(course|courses|class|ders|dersler|kurs|taught|language)\b.*\b(german|deutsch|almanca|english|ingilizce|englisch)\b|'
        r'\b(in|on)\s+(german|deutsch|almanca|english|ingilizce|englisch)\b|'
        r'\balmanca\b|\bingilizce\b|'
        # Multilingual course search phrases (EN/DE/TR)
        r'\b(show|find|search|recommend|any)\b.*\b(course|courses|class|classes)\b|'
        r'\b(welche|zeig|finde|suche|gibt\s*es)\b.*\b(kurs|kurse|vorlesung|vorlesungen|fach|fächer|veranstaltung|seminar)\b|'
        r'\b(göster|bul|ara|öner)\b.*\b(ders|dersler|kurs|kurslar)\b|'
        r'\b(ders|dersler|kurs|kurslar)\b.*\b(hakkında|var\s*mı|arıyorum|istiyorum)\b'
        r')',
        re.IGNORECASE
    )
    if _force_search_re.search(user_input):
        intent = "SEARCH"
    else:
        # Classify intent with fail-forward error handling
        try:
            intent = await async_classify_intent(user_input, has_context, recent_msgs)
        except Exception as e:
            # Fail-forward: default to SEARCH on any error
            # User came here for courses - that's the safe assumption
            print(f"[WARN] Intent classification failed, defaulting to SEARCH: {e}")
            intent = "SEARCH"
    
    # Route based on intent
    if intent in ["GREETING", "CHITCHAT", "HELP", "OUT_OF_SCOPE"]:
        # Conversational response - no course search
        response = await async_conversational_response(user_input, intent, chat_history)
        
        # Update chat history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        cl.user_session.set("chat_history", chat_history)
        
        await cl.Message(content=response).send()
    
    elif intent == "FOLLOWUP" and has_context:
        # Answer from context without RAG
        response = await async_answer_from_context(
            user_input,
            course_context,
            chat_history
        )
        
        # Update chat history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        cl.user_session.set("chat_history", chat_history)
        
        await cl.Message(content=response).send()
    
    else:
        # SEARCH - New course search via RAG pipeline
        await handle_search(user_input, chat_history)


async def handle_search(query: str, chat_history: list):
    """
    Execute RAG search and display results.
    """
    # Detect filters from query - let rag_pipeline handle $and wrapping
    filters = detect_metadata_filters(query)
    
    # Note: Filters are applied silently in the background
    # No need to show technical details to the user
    
    # Show loading indicator
    msg = cl.Message(content="Searching courses...")
    await msg.send()
    
    try:
        # RAG retrieval - retrieve broadly, LLM will filter for relevance
        retrieved = await async_search_courses(
            query,
            n_results=RAG_RETRIEVAL_COUNT,
            filter_dict=filters
        )
    except Exception as e:
        print(f"[ERROR] RAG search failed: {e}")
        msg.content = f"Search error: {str(e)}"
        await msg.update()
        return
    
    if not retrieved:
        msg.content = "No courses found matching your criteria. Please try a different search."
        await msg.update()
        return
    
    # Filter out very low similarity results before sending to LLM
    # This prevents LLM from wasting tokens on clearly irrelevant courses
    MIN_SIMILARITY = 0.38
    filtered = [r for r in retrieved if r.similarity >= MIN_SIMILARITY]
    if not filtered:
        # If all below threshold, keep top 5 anyway
        filtered = sorted(retrieved, key=lambda x: x.similarity, reverse=True)[:5]
    retrieved = filtered
    print(f"[DEBUG] After similarity filter (>={MIN_SIMILARITY}): {len(retrieved)} courses")
    
    # LLM ranking
    msg.content = "Analyzing courses..."
    await msg.update()
    
    try:
        result = await async_rank_courses(query, retrieved, MAX_RECOMMENDATIONS, filters)
    except Exception as e:
        print(f"[ERROR] LLM ranking failed: {e}")
        msg.content = f"Analysis error: {str(e)}"
        await msg.update()
        return
    
    recommendations = result.get("recommendations", [])
    student_summary = result.get("student_summary", "")
    no_match = result.get("no_match_explanation")
    
    if no_match or not recommendations:
        msg.content = no_match or "No matching courses found for your preferences."
        await msg.update()
        return
    
    # Enrich recommendations with raw_text for context
    for rec in recommendations:
        for course in retrieved:
            if course.course_id == rec.get("course_id"):
                rec["raw_text"] = course.raw_text
                break
    
    # Store in session
    cl.user_session.set("all_recommendations", recommendations)
    cl.user_session.set("visible_count", INITIAL_VISIBLE_COUNT)
    
    # Build context for follow-ups
    context = build_course_context(recommendations, INITIAL_VISIBLE_COUNT)
    cl.user_session.set("current_course_context", context)
    
    # Update chat history
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": f"Found {len(recommendations)} courses"})
    cl.user_session.set("chat_history", chat_history)
    
    # Format and display results
    visible_count = INITIAL_VISIBLE_COUNT
    total_count = len(recommendations)
    
    # Summary line
    summary_line = f"**{student_summary}**\n\nShowing {min(visible_count, total_count)} of {total_count} courses:\n\n"
    
    # Course cards
    courses_text = format_all_courses(recommendations, visible_count)
    
    # Build final message
    full_content = summary_line + courses_text
    
    # Add Load More action if there are more courses
    actions = []
    if total_count > visible_count:
        remaining = total_count - visible_count
        actions.append(
            cl.Action(
                name="load_more",
                label=f"Load More ({remaining} remaining)",
                payload={"new_count": min(visible_count + 10, total_count)}
            )
        )
    
    msg.content = full_content
    msg.actions = actions
    await msg.update()


@cl.action_callback("load_more")
async def on_load_more(action: cl.Action):
    """Handle Load More button click."""
    new_count = action.payload.get("new_count", INITIAL_VISIBLE_COUNT)
    
    all_recs = cl.user_session.get("all_recommendations", [])
    total_count = len(all_recs)
    
    if not all_recs:
        return
    
    # Update visible count
    cl.user_session.set("visible_count", new_count)
    
    # Update context with newly visible courses
    context = build_course_context(all_recs, new_count)
    cl.user_session.set("current_course_context", context)
    
    # Format updated course list
    summary_line = f"Showing {new_count} of {total_count} courses:\n\n"
    courses_text = format_all_courses(all_recs, new_count)
    full_content = summary_line + courses_text
    
    # Build actions
    actions = []
    if total_count > new_count:
        remaining = total_count - new_count
        actions.append(
            cl.Action(
                name="load_more",
                label=f"Load More ({remaining} remaining)",
                payload={"new_count": min(new_count + 10, total_count)}
            )
        )
    
    await cl.Message(content=full_content, actions=actions).send()
    
    # Remove the action from the original message
    await action.remove()
