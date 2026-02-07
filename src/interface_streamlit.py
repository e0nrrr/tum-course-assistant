"""
TUM Course Assistant - Streamlit Chat Interface

Chat-style course recommendation interface with follow-up support.

Logo: Place your TUM logo in src/assets/ folder as:
      - tum_logo.svg (preferred) or
      - tum_logo.png (fallback)

Usage:
    cd src && streamlit run interface_streamlit.py
"""

import re
import sys
from pathlib import Path

import streamlit as st
from openai import OpenAI

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_pipeline import search_courses, get_index_stats, CourseResult
from llm_reasoning import rank_courses_with_llm
from intent_router import classify_intent, answer_from_context
import config


# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="TUM Course Assistant",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS - TUM Minimalist Theme with Chat Support
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* TUM Color Palette */
    :root {
        --tum-blue: #0065BD;
        --tum-blue-dark: #003359;
        --border-color: #E2E8F0;
        --text-primary: #1A202C;
        --text-secondary: #4A5568;
        --background: #FFFFFF;
        --card-background: #F7FAFC;
        --user-bubble: #0065BD;
        --assistant-bubble: #F7FAFC;
    }

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: var(--background);
        color: var(--text-primary);
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}

    /* Main Container */
    [data-testid="block-container"] {
        padding: 1rem 2rem 6rem 2rem;
        max-width: 900px;
        margin: 0 auto;
    }

    /* Logo Header */
    .tum-header {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--tum-blue);
    }

    .tum-logo {
        height: 40px;
        width: auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .tum-logo svg {
        height: 40px;
        width: auto;
    }

    .tum-text-logo {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--tum-blue);
        letter-spacing: 0.05em;
    }

    /* Hero Section - Compact for chat */
    .hero-area {
        text-align: center;
        padding: 1rem 0;
    }

    .hero-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--tum-blue);
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
    }

    /* Chat Container */
    .chat-container {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        background: #FAFAFA;
    }

    /* Fix chat message text visibility */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: var(--text-primary) !important;
        opacity: 1 !important;
    }

    [data-testid="stChatMessage"] strong {
        color: var(--tum-blue) !important;
        font-weight: 600;
    }

    /* Course Card */
    .result-card {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }

    .result-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
        border-color: var(--tum-blue);
    }

    .course-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }

    .course-info {
        flex: 1;
    }

    .course-code {
        display: inline-block;
        background: var(--tum-blue);
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        letter-spacing: 0.3px;
    }

    .course-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.4;
        margin: 0.4rem 0;
    }

    .course-meta {
        color: var(--text-secondary);
        font-size: 0.82rem;
        margin-top: 0.4rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }

    .match-score {
        color: var(--tum-blue);
        font-weight: 600;
        font-size: 0.85rem;
        white-space: nowrap;
        padding: 0.35rem 0.7rem;
        background: #EBF8FF;
        border-radius: 8px;
        border: 1px solid #BEE3F8;
    }

    .reasoning-text {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.6;
        margin-top: 0.6rem;
    }

    /* Recommendation Header */
    .recommendation-header {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }

    /* Student Summary */
    .student-summary {
        background: #F0F7FF;
        border-left: 4px solid var(--tum-blue);
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.25rem;
        font-size: 0.95rem;
        color: var(--text-primary);
        line-height: 1.6;
    }

    /* Empty State */
    .empty-state {
        color: var(--text-secondary);
        text-align: center;
        margin: 2rem 0;
        font-size: 0.95rem;
        padding: 2rem;
        background: var(--card-background);
        border-radius: 8px;
    }

    /* Clear Chat Button */
    .clear-chat-btn {
        position: absolute;
        top: 1rem;
        right: 1rem;
    }

    /* Chat Input - Simple clean style */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] form,
    [data-testid="stChatInput"] form > div {
        background-color: white !important;
        background: white !important;
    }

    /* Main input container */
    [data-testid="stChatInput"] > div {
        border: 2px solid var(--tum-blue) !important;
        border-radius: 8px !important;
    }

    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--tum-blue-dark) !important;
    }

    /* Textarea */
    [data-testid="stChatInput"] textarea {
        background-color: white !important;
        background: white !important;
        color: var(--text-primary) !important;
        border: none !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-secondary) !important;
    }

    /* Bottom container - force white background */ */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] > div,
    section[data-testid="stBottom"] {
        background-color: white !important;
        background: white !important;
    }

    /* Override ALL dark backgrounds in chat input area */
    .stChatInput,
    .stChatInput > div,
    .stChatInput > div > div,
    div[class*="stChatInput"],
    div[class*="st-emotion-cache"][class*="e4man"] {
        background-color: white !important;
        background: white !important;
    }

    /* Remove any red/colored outlines */
    *:focus {
        outline-color: var(--tum-blue) !important;
    }

    /* Footer */
    body::after {
        content: "TUM · Programming in Python for Business and Life Science Analytics (MGT001437) - Edanur Öner";
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 0.75rem 1rem;
        background: white;
        border-top: 1px solid #E2E8F0;
        font-size: 0.8rem;
        color: #0065BD;
        font-weight: 600;
        letter-spacing: 0.01em;
        z-index: 999999;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Padding for footer */
    [data-testid="stVerticalBlock"] {
        padding-bottom: 60px !important;
    }

    section[data-testid="stMain"] {
        padding-bottom: 80px !important;
    }

    /* Small button styling - TUM Blue */
    .stButton > button,
    button[kind="secondary"],
    [data-testid="baseButton-secondary"] {
        background-color: var(--tum-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.4rem 1rem !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    .stButton > button:hover,
    button[kind="secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover {
        background-color: var(--tum-blue-dark) !important;
        color: white !important;
    }

    /* Chat Input Container - Fix dark background comprehensively */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInputContainer"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] > div,
    .stChatInput,
    .st-emotion-cache-hzygls,
    div[class*="stChatInput"],
    div[class*="e4man11"] {
        background-color: white !important;
        background: white !important;
    }

    /* Chat input textarea container */
    [data-testid="stChatInput"] > div > div {
        background-color: white !important;
        border-color: var(--tum-blue) !important;
    }

    /* Override any dark theme on bottom section */
    section[data-testid="stBottom"] {
        background-color: white !important;
        background: white !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables for chat."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # All recommendations from last search (up to 15)
    if "all_recommendations" not in st.session_state:
        st.session_state.all_recommendations = []
    
    # How many courses are currently visible (pagination)
    if "visible_count" not in st.session_state:
        st.session_state.visible_count = 5
    
    # Combined context of all VISIBLE courses (for follow-up questions)
    if "current_course_context" not in st.session_state:
        st.session_state.current_course_context = None
    
    # Legacy - keep for backward compatibility during transition
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    
    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_logo_exists() -> tuple[bool, str]:
    """Check if TUM logo exists in assets folder."""
    assets_dir = Path(__file__).parent / "assets"
    
    svg_path = assets_dir / "tum_logo.svg"
    if svg_path.exists():
        return True, str(svg_path)
    
    png_path = assets_dir / "tum_logo.png"
    if png_path.exists():
        return True, str(png_path)
    
    return False, ""


def extract_course_metadata(raw_text: str) -> dict:
    """Extract course metadata from raw_text using regex."""
    metadata = {
        "level": "",
        "language": "",
        "duration": "",
        "sws": ""
    }
    
    level_match = re.search(r"Module Level:\s*\n\s*(\w+)", raw_text, re.IGNORECASE)
    if level_match:
        metadata["level"] = level_match.group(1).strip()
    
    lang_match = re.search(r"Language:\s*\n\s*([\w/]+)", raw_text, re.IGNORECASE)
    if lang_match:
        metadata["language"] = lang_match.group(1).strip()
    
    duration_match = re.search(r"Duration:\s*\n\s*(one semester|two semesters?|\d+ semesters?)", raw_text, re.IGNORECASE)
    if duration_match:
        metadata["duration"] = duration_match.group(1).strip()
    
    sws_match = re.search(r"(\d+)\s*SWS", raw_text, re.IGNORECASE)
    if sws_match:
        metadata["sws"] = f"{sws_match.group(1)} SWS"
    
    return metadata


def format_metadata_line(metadata: dict) -> str:
    """Format metadata as a single line with bullet separators."""
    parts = []
    if metadata.get("level"):
        parts.append(metadata["level"])
    if metadata.get("language"):
        parts.append(metadata["language"])
    if metadata.get("duration"):
        parts.append(metadata["duration"])
    if metadata.get("sws"):
        parts.append(metadata["sws"])
    return " • ".join(parts) if parts else ""


def check_index_status() -> tuple[bool, str, int]:
    """Check if ChromaDB index is available."""
    try:
        stats = get_index_stats()
        if not stats:
            return False, "Index stats not available", 0
        count = stats.get("document_count") or stats.get("total_courses") or 0
        if count > 0:
            return True, "Index ready", count
        return False, "Index is empty", 0
    except Exception as e:
        return False, str(e), 0


def process_query(query: str, n_results: int = 15, max_recommendations: int = 15) -> dict:
    """Process user query through RAG + LLM pipeline.
    
    Now fetches up to 15 recommendations for Load More pagination.
    """
    retrieved = search_courses(query, n_results=n_results)
    
    if not retrieved:
        return {
            "success": False,
            "error": "No courses found in the database.",
            "retrieved": []
        }
    
    result = rank_courses_with_llm(
        user_query=query,
        retrieved_courses=retrieved,
        max_recommendations=max_recommendations
    )
    
    retrieved_map = {course.course_id: course for course in retrieved}
    
    for rec in result.get("recommendations", []):
        course_id = rec.get("course_id")
        if course_id and course_id in retrieved_map:
            rec["raw_text"] = retrieved_map[course_id].raw_text
    
    return {
        "success": True,
        "result": result,
        "retrieved": retrieved
    }


# =============================================================================
# COURSE CONTEXT BUILDER (FOR FOLLOW-UP QUESTIONS)
# =============================================================================

def build_course_context(recommendations: list[dict], count: int) -> str:
    """
    Build a formatted context string from visible course recommendations.
    
    This context is sent to the LLM for follow-up questions so it knows
    about all courses the user can see.
    
    Args:
        recommendations: List of recommendation dicts with raw_text
        count: Number of visible courses (pagination)
    
    Returns:
        Formatted string with all visible courses numbered
    """
    visible = recommendations[:count]
    
    if not visible:
        return ""
    
    parts = []
    for i, rec in enumerate(visible, 1):
        course_id = rec.get("course_id", "Unknown")
        title = rec.get("title", "Unknown Course")
        raw_text = rec.get("raw_text", "")[:2500]  # Limit per course
        match_reason = rec.get("match_reason", "")
        
        part = f"""
=== Course {i}: {course_id} - {title} ===
Match Reason: {match_reason}

Description:
{raw_text}
"""
        parts.append(part)
    
    return "\n".join(parts)


# =============================================================================
# RESPONSE FORMATTING
# =============================================================================

def display_course_results(recommendations: list[dict], visible_count: int, container) -> None:
    """Display course recommendations with Load More pagination.
    
    Args:
        recommendations: Full list of all recommendations
        visible_count: How many to show currently
        container: Streamlit container to render into
    """
    if not recommendations:
        container.info("No matching courses found. Try a different description.")
        return
    
    visible = recommendations[:visible_count]
    
    # Friendly intro message
    container.markdown("**Here are some courses that match your interests:**")
    
    for i, rec in enumerate(visible):
        course_id = rec.get("course_id", "N/A")
        title = rec.get("title", "Unknown Course")
        reason = rec.get("match_reason", "")
        confidence = rec.get("confidence")
        raw_text = rec.get("raw_text", "")
        
        # Format confidence
        if isinstance(confidence, (int, float)):
            score_pct = int(round(confidence * 100))
            conf_label = f"{score_pct}% Match"
        else:
            conf_label = "Recommended"
        
        # Extract metadata
        metadata = extract_course_metadata(raw_text)
        metadata_line = format_metadata_line(metadata)
        
        # Create a visual card using expander
        with container.expander(f"**{i+1}. {title}** — `{course_id}` ({conf_label})", expanded=True):
            if metadata_line:
                st.caption(f"📋 {metadata_line}")
            st.markdown(f"**Why this course?** {reason}")
    
    # Show "Load More" button if there are more courses
    remaining = len(recommendations) - visible_count
    if remaining > 0:
        container.markdown("---")
        if container.button(f"📚 Load More ({remaining} more courses)", key="load_more_btn"):
            st.session_state.visible_count += 5
            # Update context with newly visible courses
            st.session_state.current_course_context = build_course_context(
                st.session_state.all_recommendations,
                st.session_state.visible_count
            )
            st.rerun()


# =============================================================================
# MAIN UI
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    
    # Logo/Header
    logo_exists, logo_path = check_logo_exists()
    
    if logo_exists:
        try:
            with open(logo_path, 'r') as f:
                svg_content = f.read()
            st.markdown(
                f'<div class="tum-header"><div class="tum-logo">{svg_content}</div></div>',
                unsafe_allow_html=True,
            )
        except:
            st.markdown(
                '<div class="tum-header"><span class="tum-text-logo">TUM</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="tum-header"><span class="tum-text-logo">TUM</span></div>',
            unsafe_allow_html=True,
        )
    
    # Hero Section (compact)
    st.markdown(
        """
        <div class="hero-area">
            <h1 class="hero-title">Course Assistant</h1>
            <p class="hero-subtitle">Chat with me about your learning goals</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Index Status check (no display)
    index_ok, index_msg, course_count = check_index_status()
    if not index_ok:
        st.error(f"Database not available: {index_msg}")
        return
    
    # Clear chat button
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button("Clear", help="Start a new conversation"):
            st.session_state.messages = []
            st.session_state.all_recommendations = []
            st.session_state.visible_count = 5
            st.session_state.current_course_context = None
            st.session_state.last_result = None
            st.session_state.last_retrieved = []
            st.rerun()
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=None):
                if message.get("is_course_result") and message.get("recommendations"):
                    # Re-render course cards with pagination
                    display_course_results(
                        message["recommendations"],
                        message.get("visible_count", 5),
                        st
                    )
                else:
                    st.markdown(message["content"])
    
    # Welcome message if no messages
    if not st.session_state.messages:
        with chat_container:
            with st.chat_message("assistant", avatar=None):
                st.markdown("""**Welcome to TUM Course Assistant**

I can help you find courses that match your interests and goals. Describe what you're looking for and I'll recommend relevant courses from the TUM catalog.

**Example queries:**

- *\"I'm interested in AI and machine learning courses\"*
- *\"I want project-based courses about entrepreneurship\"*
- *\"Show me German language courses about finance\"*
""")
    
    # Chat input
    if prompt := st.chat_input("Describe your learning goals or ask about courses..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with chat_container:
            with st.chat_message("user", avatar=None):
                st.markdown(prompt)
        
        # Process the message using Intent Router
        with chat_container:
            with st.chat_message("assistant", avatar=None):
                # === INTENT ROUTER: Classify as SEARCH or FOLLOWUP ===
                has_context = st.session_state.current_course_context is not None
                
                # Hardened intent classification with fallback
                try:
                    with st.spinner("Thinking..."):
                        intent = classify_intent(
                            message=prompt,
                            has_context=has_context,
                            recent_messages=st.session_state.messages[-5:]
                        )
                except Exception as e:
                    # Fail-forward: default to SEARCH on any error
                    # User came here for courses - that's the safe assumption
                    print(f"[WARN] Intent classification failed, defaulting to SEARCH: {e}")
                    intent = "SEARCH"
                
                if intent == "FOLLOWUP" and has_context:
                    # === FOLLOWUP: Answer from context, no RAG search ===
                    with st.spinner("Answering..."):
                        response = answer_from_context(
                            question=prompt,
                            course_context=st.session_state.current_course_context,
                            chat_history=st.session_state.messages
                        )
                    
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                
                else:
                    # === SEARCH: Run RAG pipeline ===
                    with st.spinner("Searching courses..."):
                        result = process_query(prompt)
                    
                    if result["success"]:
                        recommendations = result["result"].get("recommendations", [])
                        
                        # Update session state with new recommendations
                        st.session_state.all_recommendations = recommendations
                        st.session_state.visible_count = 5
                        st.session_state.last_result = result["result"]
                        st.session_state.last_retrieved = result["retrieved"]
                        
                        # Build context from visible courses
                        st.session_state.current_course_context = build_course_context(
                            recommendations,
                            st.session_state.visible_count
                        )
                        
                        # Display courses with Load More
                        display_course_results(
                            recommendations,
                            st.session_state.visible_count,
                            st
                        )
                        
                        # Handle no match explanation
                        no_match = result["result"].get("no_match_explanation")
                        if no_match:
                            st.info(no_match)
                        
                        # Save to history for re-rendering
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Course recommendations displayed",
                            "is_course_result": True,
                            "recommendations": recommendations,
                            "visible_count": st.session_state.visible_count
                        })
                    else:
                        error_msg = result.get('error', 'An error occurred')
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })


# Run the app
if __name__ == "__main__":
    main()
