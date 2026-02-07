"""
TUM Course Assistant - Intent Router Module

LLM-based intent classification for conversational flow.
Classifies user messages into multiple categories for natural conversation.

Architecture:
    User Message → classify_intent() → Intent Category
        GREETING → Friendly welcome response
        CHITCHAT → General conversation response  
        HELP → Explain bot capabilities
        SEARCH → RAG Pipeline (find courses)
        FOLLOWUP → Answer from context (no RAG)
        OUT_OF_SCOPE → Politely redirect to courses

Usage:
    from intent_router import classify_intent, answer_from_context, generate_conversational_response
    
    intent = classify_intent("hello", has_context=False)
    if intent == "GREETING":
        response = generate_conversational_response("hello", "GREETING")
"""

import json
import re
from typing import Literal

from openai import OpenAI

import config
from config import get_openai_client


# =============================================================================
# FAST PATTERN MATCHING (Zero LLM cost)
# =============================================================================

# Regex patterns for obvious greetings - no need to call LLM
GREETING_PATTERNS = re.compile(
    r'^\s*(hi|hello|hey|hallo|merhaba|selam|selamlar|greetings|good\s*(morning|afternoon|evening)|'
    r'günaydın|iyi\s*(günler|akşamlar)|hola|bonjour|ciao)[\s!?.,:;]*$',
    re.IGNORECASE
)

# Regex patterns for obvious out-of-scope requests
# Be conservative here - only catch clear non-course requests
# Let LLM handle ambiguous cases (defaults to SEARCH anyway)
OUT_OF_SCOPE_PATTERNS = re.compile(
    r'^.*(weather forecast|hava durumu nasıl|kaç derece|'
    r'what is \d+\s*[\+\-\*\/]\s*\d+|'  # Math: "what is 2+2"
    r'\d+\s*[\+\-\*\/]\s*\d+\s*(kaç|nedir|=)|'  # Math: "2+2 kaç"
    r'write me (a|an) (essay|poem|story|article)|'
    r'bana (makale|şiir|hikaye) yaz).*$',
    re.IGNORECASE
)

# Regex patterns for help requests
HELP_PATTERNS = re.compile(
    r'^\s*(help|yardım|hilfe|ne\s*yapabilirsin|what\s*can\s*you\s*do|'
    r'nasıl\s*(kullanılır|çalışır)|how\s*(does\s*this|do\s*you)\s*work)[\s!?.]*$',
    re.IGNORECASE
)

# Regex patterns for obvious followup questions (when has_context=true)
# NOTE: These should NOT match new topic requests - see NEW_TOPIC_PATTERNS below
FOLLOWUP_PATTERNS = re.compile(
    r'(tell\s*me\s*more|more\s*details|daha\s*fazla|detay|compare|karşılaştır|'
    r'(first|second|third|1st|2nd|3rd|ilk|ikinci|üçüncü)\s*(one|course|ders)?|'
    r'is\s*it\s*(hard|easy|difficult)|zor\s*mu|kolay\s*mı|'
    r'what\s*about\s*(it|this|that|them)|bunlar?\s*hakkında)',
    re.IGNORECASE
)

# Regex patterns for NEW course topics - these should trigger SEARCH even with context
# When user mentions a new academic topic, they want NEW course recommendations
NEW_TOPIC_PATTERNS = re.compile(
    r'^\s*(product\s*management|innovation|entrepreneurship|startup|'
    r'finance|fintech|marketing|strategy|leadership|business|'
    r'machine\s*learning|deep\s*learning|artificial\s*intelligence|ai\b|'
    r'data\s*science|analytics|robotics|automation|'
    r'software|programming|web\s*development|cloud|devops|'
    r'blockchain|crypto|cybersecurity|security|'
    r'sustainability|energy|climate|environment|'
    r'health|biotech|medical|pharma|'
    r'design|ux|ui|human.computer|'
    r'economics|accounting|consulting|operations|'
    r'supply\s*chain|logistics|manufacturing|'
    r'kommunikation|digitalisierung|management|'
    r'yönetim|pazarlama|finans|girişimcilik)[\s!?,.:;]*$',
    re.IGNORECASE
)


# =============================================================================
# INTENT TYPES
# =============================================================================

IntentType = Literal["GREETING", "CHITCHAT", "HELP", "SEARCH", "FOLLOWUP", "OUT_OF_SCOPE"]


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a TUM university course assistant chatbot.

CRITICAL CONTEXT: This is a COURSE RECOMMENDATION bot. Users ONLY come here to find courses.
DEFAULT BEHAVIOR: Return SEARCH unless you are 100% certain it's something else.

Classify into ONE category:

**SEARCH** (DEFAULT - use this when uncertain!)
- Topics: "AI", "robotics", "finance", "management", "data science", etc.
- Interests: "I'm interested in...", "I like...", "I want..."
- Languages: "English", "German", "Almanca", "İngilizce", "Deutsch" (= course language preference!)
- Short keywords that could be course topics
- Vague requests: "something about X", "tell me more" (without context)
- WHEN IN DOUBT → SEARCH

**FOLLOWUP** (ONLY if has_context=true)
- References to shown courses: "the first one", "compare them", "is it hard?"
- Phrases like "tell me more", "more details", "what about..." when has_context=true
- Must have has_context=true, otherwise return SEARCH

**OUT_OF_SCOPE** (ONLY for requests that CANNOT possibly be about courses)
- Explicit weather: "what's the weather", "hava durumu nasıl"
- Explicit math: "2+2=?", "calculate 5*3"
- Explicit essay writing: "write me an essay about..."
- NOT language names alone (those are course preferences!)

**GREETING** (ONLY pure greetings with NOTHING else)
- Just "hi", "hello", "merhaba" (nothing more)

**CHITCHAT/HELP** (very rare)
- "who made you?", "help me" (explicit meta questions)

RULES:
1. Language names (English, German, İngilizce, Almanca) = SEARCH (course language preference)
2. Single words/short phrases about topics = SEARCH
3. "tell me more" without context = SEARCH (not OUT_OF_SCOPE)
4. UNCERTAIN? → SEARCH
5. OUT_OF_SCOPE only for EXPLICIT non-course requests

Respond with ONLY a JSON object: {"intent": "CATEGORY", "confidence": 0.0-1.0}"""


def classify_intent(
    message: str,
    has_context: bool,
    recent_messages: list[dict] | None = None
) -> IntentType:
    """
    Classify user intent - fast regex first, then LLM if needed.
    
    Action-oriented design: defaults to SEARCH when uncertain.
    User came to this bot to find courses - assume that's their intent.
    
    Args:
        message: Current user message
        has_context: Whether there are previous course recommendations in memory
        recent_messages: Optional recent chat history for context
    
    Returns:
        One of: GREETING, CHITCHAT, HELP, SEARCH, FOLLOWUP, OUT_OF_SCOPE
    """
    msg_clean = message.strip()
    
    # =================================================================
    # FAST PATH: Regex matching for obvious cases (zero LLM cost)
    # =================================================================
    
    # Pure greeting with nothing else
    if GREETING_PATTERNS.match(msg_clean):
        return "GREETING"
    
    # Obvious out-of-scope (weather, math, etc.)
    if OUT_OF_SCOPE_PATTERNS.search(msg_clean) and len(msg_clean) < 50:
        return "OUT_OF_SCOPE"
    
    # Pure help request
    if HELP_PATTERNS.match(msg_clean):
        return "HELP"
    
    # NEW TOPIC DETECTION - MUST come BEFORE followup check!
    # If user mentions a new academic topic, they want a NEW search
    # even if there's existing context from previous recommendations
    if NEW_TOPIC_PATTERNS.match(msg_clean):
        return "SEARCH"
    
    # Followup patterns (only when context exists AND not a new topic)
    if has_context and FOLLOWUP_PATTERNS.search(msg_clean):
        return "FOLLOWUP"
    
    # =================================================================
    # LLM PATH: For ambiguous cases
    # =================================================================
    
    # Build context for LLM
    context_info = f"has_context={'true' if has_context else 'false'}"
    
    # Include recent conversation if available
    conversation_hint = ""
    if recent_messages and len(recent_messages) > 0:
        recent = recent_messages[-3:]  # Last 3 messages
        conversation_hint = "\n\nRecent conversation:\n"
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:100]
            conversation_hint += f"- {role}: {content}...\n"
    
    user_prompt = f"""Context: {context_info}
{conversation_hint}
User's message: "{message}"

Classify this message into one of: GREETING, CHITCHAT, HELP, SEARCH, FOLLOWUP, OUT_OF_SCOPE"""

    try:
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=config.LLM_TEMPERATURE_CLASSIFICATION,
            max_tokens=config.LLM_MAX_TOKENS_CLASSIFICATION,
        )
        
        result = json.loads(response.choices[0].message.content)
        intent = result.get("intent", "SEARCH").upper()
        
        # Validate intent
        valid_intents = ["GREETING", "CHITCHAT", "HELP", "SEARCH", "FOLLOWUP", "OUT_OF_SCOPE"]
        if intent not in valid_intents:
            return "SEARCH"  # Default to action - user wants courses
        
        # FOLLOWUP only valid with context - otherwise treat as new search
        if intent == "FOLLOWUP" and not has_context:
            return "SEARCH"  # User probably wants to search, just phrased it oddly
            
        return intent
        
    except Exception as e:
        # On error, default to SEARCH - user came here for courses!
        print(f"Intent classification error: {e}")
        return "SEARCH"


# =============================================================================
# CONTEXT-BASED ANSWERING (NO RAG)
# =============================================================================

ANSWER_SYSTEM_PROMPT = """You are a friendly TUM course advisor chatbot.

You are answering follow-up questions about courses that were previously recommended to the user.
The course information is provided below - use ONLY this information to answer.

PERSONALITY & TONE:
1. Be warm, friendly, and helpful - like a knowledgeable friend
2. Answer in the SAME LANGUAGE as the user's question (Turkish, English, German)
3. Address the user directly using "you/your" (sen/senin, du/dein)
4. NEVER use third person ("The student...", "Öğrenci...")
5. Be conversational and natural

RULES:
1. Be helpful, specific, and concise
2. If comparing courses, only compare from the provided list
3. If the information isn't in the course descriptions, say you don't have that detail
4. Do NOT invent information not present in the course descriptions
5. Do NOT recommend new courses - only discuss the ones in context

Keep responses natural and conversational - you're helping a friend choose courses!"""


def answer_from_context(
    question: str,
    course_context: str,
    chat_history: list[dict] | None = None
) -> str:
    """
    Answer a follow-up question using only the course context (no RAG search).
    
    Args:
        question: User's follow-up question
        course_context: Combined text of all visible course recommendations
        chat_history: Recent chat messages for conversational context
    
    Returns:
        Natural language response to the user's question
    """
    # Build conversation context
    conversation = ""
    if chat_history and len(chat_history) > 0:
        recent = chat_history[-4:]  # Last 4 messages
        conversation = "Recent conversation:\n"
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:300]
            conversation += f"{role}: {content}\n"
        conversation += "\n"
    
    user_prompt = f"""{conversation}COURSE INFORMATION (from previous recommendations):
{course_context}

---

User's question: {question}

Please answer based on the course information above."""

    try:
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.LLM_TEMPERATURE_CONVERSATION,
            max_tokens=config.LLM_MAX_TOKENS_CONVERSATION,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"


# =============================================================================
# CONVERSATIONAL RESPONSES (NON-SEARCH)
# =============================================================================

CONVERSATIONAL_SYSTEM_PROMPT = """You are a friendly TUM Course Assistant chatbot named "TUM Buddy".

You help students find courses at Technical University of Munich (TUM).
Right now, you're having a general conversation (not searching for courses).

PERSONALITY:
- Warm, friendly, and approachable - like a helpful friend
- Enthusiastic about helping students
- Use casual but professional language
- Add personality with occasional expressions like "Great!", "Awesome!", "Harika!" etc.

CRITICAL RULES:
1. ALWAYS respond in the SAME LANGUAGE as the user (English, Turkish, German, etc.)
2. Keep responses brief and natural (1-3 sentences)
3. Address the user directly - use "you/your" (sen/senin, du/dein)
4. NEVER use third person ("The student...", "Öğrenci...")
5. Be conversational, not robotic

RESPONSE GUIDELINES BY INTENT:
- GREETING: Be warm! Welcome them, introduce yourself briefly, ask what they're interested in
- HELP: Explain what you can do in a friendly way - finding courses by topic, language, type
- CHITCHAT: Be friendly, but gently guide them towards exploring courses
- OUT_OF_SCOPE: Kindly explain you focus on TUM courses, offer to help with that

EXAMPLE RESPONSES:
- Greeting (English): "Hey there! 👋 I'm your TUM course buddy. What subjects are you interested in exploring?"
- Greeting (Turkish): "Merhaba! 👋 Ben TUM ders asistanınım. Hangi konularda ders arıyorsun?"
- Help (English): "I can help you discover amazing courses at TUM! Just tell me what topics excite you, or if you prefer courses in a specific language."
- Help (Turkish): "TUM'da harika dersler bulmana yardım edebilirim! Hangi konular ilgini çekiyor?"

NEVER make up course information - you can only give real recommendations during a search."""


def generate_conversational_response(
    message: str,
    intent: IntentType,
    chat_history: list[dict] | None = None
) -> str:
    """
    Generate a conversational response for non-search intents.
    
    Args:
        message: User's message
        intent: Classified intent (GREETING, CHITCHAT, HELP, OUT_OF_SCOPE)
        chat_history: Recent chat history for context
    
    Returns:
        Natural conversational response
    """
    # Build context hint based on intent
    intent_hints = {
        "GREETING": "User is greeting you. Welcome them warmly and invite them to search for courses.",
        "CHITCHAT": "User is making small talk. Be friendly but guide them towards course topics.",
        "HELP": "User wants to know what you can do. Explain your capabilities clearly.",
        "OUT_OF_SCOPE": "User asked something outside your domain. Politely redirect to courses.",
    }
    
    hint = intent_hints.get(intent, "Respond helpfully.")
    
    # Build conversation context
    conversation = ""
    if chat_history and len(chat_history) > 0:
        recent = chat_history[-4:]
        conversation = "Recent conversation:\n"
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:200]
            conversation += f"{role}: {content}\n"
        conversation += "\n"
    
    user_prompt = f"""{conversation}Intent: {intent}
Guidance: {hint}

User's message: "{message}"

Respond naturally and briefly."""

    try:
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": CONVERSATIONAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.LLM_TEMPERATURE_CREATIVE,
            max_tokens=config.LLM_MAX_TOKENS_CREATIVE,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Fallback responses
        fallbacks = {
            "GREETING": "Hello! I'm the TUM Course Assistant. I can help you find courses at TUM. What topics interest you?",
            "CHITCHAT": "I'm here to help you find TUM courses! What subjects are you interested in?",
            "HELP": "I can help you find TUM courses! Just tell me what topics interest you, or ask for courses in a specific language or format.",
            "OUT_OF_SCOPE": "I specialize in TUM course recommendations. Would you like me to help you find some courses?",
        }
        return fallbacks.get(intent, "How can I help you find courses today?")


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    # Test intent classification
    print("=== Intent Router Test ===\n")
    
    test_cases = [
        ("hello", False),
        ("hi there!", False),
        ("merhaba", False),
        ("how are you?", False),
        ("what can you do?", False),
        ("help", False),
        ("Show me AI courses", False),
        ("I want courses about machine learning", False),
        ("Find me German language courses", False),
        ("what's the weather?", False),
        ("Is it beginner friendly?", True),
        ("Tell me more about the second one", True),
        ("Compare the first and third options", True),
    ]
    
    for message, has_ctx in test_cases:
        intent = classify_intent(message, has_ctx)
        ctx_label = "with context" if has_ctx else "no context"
        print(f"[{intent:12}] ({ctx_label:12}) \"{message}\"")
