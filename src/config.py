"""
TUM Smart Course Assistant - Configuration File
RAG-Based Course Selection Chatbot

This file contains the core configuration settings for the project.
All settings can be overridden via .env file or environment variables.

Centralized Configuration:
    - OpenAI client singleton with timeout support
    - All LLM/embedding parameters in one place
    - DRY principle: no duplicate constants across modules
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# LOAD .ENV FILE
# =============================================================================
# Load .env file from project root directory
# override=False: Existing environment variables are preserved
load_dotenv(override=False)

# =============================================================================
# PROJECT ROOT DIRECTORY
# =============================================================================
# Parent of config.py's directory = project root
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()

# =============================================================================
# OPENAI API KEY
# =============================================================================
# Read from OPENAI_API_KEY environment variable
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# =============================================================================
# EMBEDDING MODEL SETTINGS
# =============================================================================
# .env: OPENAI_EMBEDDING_MODEL or EMBEDDING_MODEL_NAME
EMBEDDING_MODEL_NAME: str = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
)

# =============================================================================
# LLM MODEL SETTINGS
# =============================================================================
# .env: OPENAI_CHAT_MODEL or LLM_MODEL_NAME
LLM_MODEL_NAME: str = os.getenv(
    "OPENAI_CHAT_MODEL",
    os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
)

# =============================================================================
# LLM PARAMETERS (Centralized)
# =============================================================================
# Timeout for all OpenAI API calls (seconds)
API_TIMEOUT: float = float(os.getenv("OPENAI_API_TIMEOUT", "30.0"))

# LLM temperature settings by use case
LLM_TEMPERATURE_REASONING: float = 0.3      # For course ranking (more consistent)
LLM_TEMPERATURE_CONVERSATION: float = 0.7   # For followup answers
LLM_TEMPERATURE_CREATIVE: float = 0.8       # For greetings/chitchat
LLM_TEMPERATURE_CLASSIFICATION: float = 0.1 # For intent classification (very low)

# Token limits
LLM_MAX_TOKENS_REASONING: int = 4096        # For course ranking responses
LLM_MAX_TOKENS_CONVERSATION: int = 800      # For followup/context answers
LLM_MAX_TOKENS_CREATIVE: int = 200          # For greetings/chitchat
LLM_MAX_TOKENS_CLASSIFICATION: int = 50     # For intent classification

# RAG parameters
RAG_TOP_K_RESULTS: int = int(os.getenv("RAG_TOP_K_RESULTS", "5"))
RAG_MAX_RECOMMENDATIONS: int = int(os.getenv("RAG_MAX_RECOMMENDATIONS", "5"))

# Query Expansion settings
QUERY_EXPANSION_ENABLED: bool = True
QUERY_EXPANSION_MIN_WORDS: int = 5  # Expand queries shorter than this

# Embedding batch size for index building
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Max characters for course description in LLM context
MAX_RAW_TEXT_CHARS: int = 1500

# =============================================================================
# CHROMA DATABASE SETTINGS
# =============================================================================
# Collection name and persist directory can be overridden via .env
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "tum_courses")

_chroma_dir = os.getenv("CHROMA_PERSIST_DIR")
CHROMA_PERSIST_DIR: Path = (
    Path(_chroma_dir).resolve() if _chroma_dir 
    else PROJECT_ROOT / "data" / "chroma_db"
)

# =============================================================================
# DATA FILES
# =============================================================================
# Course card JSONL file
_cards_path = os.getenv("COURSE_CARDS_JSONL_PATH")
COURSE_CARDS_JSONL_PATH: Path = (
    Path(_cards_path).resolve() if _cards_path
    else PROJECT_ROOT / "data" / "processed" / "course_cards.jsonl"
)

# Page-based text (may be used later)
_pages_path = os.getenv("PAGES_JSONL_PATH")
PAGES_JSONL_PATH: Path = (
    Path(_pages_path).resolve() if _pages_path
    else PROJECT_ROOT / "data" / "processed" / "module_handbook_text.jsonl"
)

# =============================================================================
# OPENAI CLIENT SINGLETON
# =============================================================================
# Thread-safe singleton pattern for OpenAI client
_openai_client: Optional[OpenAI] = None


def get_openai_api_key() -> str | None:
    """
    Returns the OpenAI API key.

    Returns:
        str | None: Value of OPENAI_API_KEY environment variable.
                    Returns None if not defined.
    """
    return OPENAI_API_KEY


def get_openai_client() -> OpenAI:
    """
    Returns a singleton OpenAI client instance with configured timeout.
    
    This is the SINGLE source of truth for OpenAI client creation.
    All modules should import and use this function instead of creating
    their own client instances.
    
    Returns:
        OpenAI: Configured client instance with timeout
        
    Raises:
        RuntimeError: If OPENAI_API_KEY is not set
        
    Example:
        from config import get_openai_client
        client = get_openai_client()
        response = client.chat.completions.create(...)
    """
    global _openai_client
    
    if _openai_client is None:
        api_key = get_openai_api_key()
        
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not defined!\n"
                "Please check your .env file:\n"
                "  OPENAI_API_KEY=sk-your-api-key-here"
            )
        
        _openai_client = OpenAI(
            api_key=api_key,
            timeout=API_TIMEOUT,  # Global timeout for all requests
        )
    
    return _openai_client


def reset_openai_client() -> None:
    """
    Resets the OpenAI client singleton (useful for testing).
    """
    global _openai_client
    _openai_client = None
