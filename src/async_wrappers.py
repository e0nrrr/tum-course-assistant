"""
TUM Course Assistant - Async Wrappers for Chainlit

This module wraps synchronous backend functions with asyncio.to_thread()
to make them compatible with Chainlit's async architecture.

Backend modules (rag_pipeline, llm_reasoning, intent_router) use synchronous
OpenAI API calls. Chainlit requires async functions for non-blocking UI.

Usage:
    from async_wrappers import (
        async_search_courses,
        async_rank_courses,
        async_classify_intent,
        async_answer_from_context,
        async_conversational_response
    )
    
    # In async context:
    results = await async_search_courses("AI courses", n_results=10)
"""

import asyncio
from typing import Literal

from intent_router import classify_intent, answer_from_context, generate_conversational_response, IntentType
from rag_pipeline import search_courses, CourseResult
from llm_reasoning import rank_courses_with_llm


async def async_search_courses(
    query: str,
    n_results: int = 15,
    filter_dict: dict | None = None
) -> list[CourseResult]:
    """
    Async wrapper for RAG course search.
    
    Args:
        query: User's search query
        n_results: Maximum number of results to return
        filter_dict: Optional metadata filters (language, level, domain)
    
    Returns:
        List of CourseResult objects
    """
    return await asyncio.to_thread(
        search_courses,
        query,
        n_results,
        filter_dict
    )


async def async_rank_courses(
    query: str,
    retrieved_courses: list[CourseResult],
    max_recommendations: int = 15
) -> dict:
    """
    Async wrapper for LLM course ranking.
    
    Args:
        query: User's original query
        retrieved_courses: Courses from RAG search
        max_recommendations: Maximum courses to recommend
    
    Returns:
        Dict with 'student_summary', 'recommendations', 'no_match_explanation'
    """
    return await asyncio.to_thread(
        rank_courses_with_llm,
        query,
        retrieved_courses,
        max_recommendations
    )


async def async_classify_intent(
    message: str,
    has_context: bool,
    recent_messages: list[dict] | None = None
) -> IntentType:
    """
    Async wrapper for intent classification.
    
    Args:
        message: Current user message
        has_context: Whether there are previous recommendations in memory
        recent_messages: Optional recent chat history
    
    Returns:
        One of: GREETING, CHITCHAT, HELP, SEARCH, FOLLOWUP, OUT_OF_SCOPE
    """
    return await asyncio.to_thread(
        classify_intent,
        message,
        has_context,
        recent_messages
    )


async def async_answer_from_context(
    question: str,
    course_context: str,
    chat_history: list[dict] | None = None
) -> str:
    """
    Async wrapper for context-based answering.
    
    Args:
        question: User's follow-up question
        course_context: Formatted text of visible courses
        chat_history: Previous conversation messages
    
    Returns:
        Natural language response
    """
    return await asyncio.to_thread(
        answer_from_context,
        question,
        course_context,
        chat_history
    )


async def async_conversational_response(
    message: str,
    intent: IntentType,
    chat_history: list[dict] | None = None
) -> str:
    """
    Async wrapper for conversational responses (non-search intents).
    
    Args:
        message: User's message
        intent: Classified intent type
        chat_history: Previous conversation messages
    
    Returns:
        Natural conversational response
    """
    return await asyncio.to_thread(
        generate_conversational_response,
        message,
        intent,
        chat_history
    )
