# TUM Course Assistant

**AI-powered course recommendation system for TUM students**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Chainlit](https://img.shields.io/badge/Chainlit-Chat%20UI-6366F1.svg)](https://chainlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991.svg)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange.svg)](https://www.trychroma.com/)

A RAG-based intelligent course advisor that helps TUM students find courses matching their interests, career goals, and preferences using natural language queries. Features an **action-oriented intent router** for efficient, token-saving conversations.

<p align="center">
  <img src="src/assets/tum_logo.svg" alt="TUM Logo" width="200"/>
</p>

---

## Quick Start

```bash
# Clone and setup
cd tum_project
python -m venv projecttum && source projecttum/bin/activate
pip install -r requirements.txt

# Configure API key
echo "OPENAI_API_KEY=your-key-here" > .env

# Launch the application
cd src && chainlit run app.py --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Features

### Core Capabilities
- **Natural Language Search** - Ask questions like "I want project-based courses about AI and innovation"
- **Multilingual Query Translation** - Automatic translation of queries (Turkish, German, French, etc.) to English for optimal RAG search
- **Universal Language Support** - Query in any language, get results in the same language (100+ languages via GPT-4o-mini)
- **Smart Filtering** - Automatic detection of language, level, and domain preferences
- **Personalized Recommendations** - LLM-powered reasoning explains why each course matches your needs
- **Action-Oriented Design** - Bot assumes you want courses (not chitchat) to save tokens and time
- **Enhanced Discovery** - Retrieve up to 50 courses with initial display of 10 (expandable by +10)

### Intent Router (Action-Oriented)
The system uses a smart intent classification system optimized for efficiency:

| Intent | Detection | LLM Cost |
|--------|-----------|----------|
| **GREETING** | Regex (hello, merhaba, selam) | 0 calls |
| **HELP** | Regex (help, ne yapabilirsin) | 0 calls |
| **FOLLOWUP** | Regex when context exists | 0 calls |
| **SEARCH** | Default for everything else | 1 call |
| **OUT_OF_SCOPE** | Only explicit non-course requests | 1 call |

**Philosophy:** *"Don't ask for permission, ask for forgiveness."* User came to find courses - that's the default assumption.

### Domain-Aware Search
The system understands two main specialization areas:

| Domain | Keywords | Course Sources |
|--------|----------|----------------|
| **Management** | innovation, entrepreneurship, marketing, strategy, business | MMT Innovation Electives, Advanced Seminars |
| **Technology** | informatics, software, programming, computer science, technical | Informatics Major Specializations |

### Metadata Filters
Automatically detected from your query:
- **Language**: German, English, German/English
- **Level**: Bachelor, Master
- **Domain**: Management, Technology

**Example queries:**
- "German language courses about entrepreneurship" → Filters: `language=German`, `domain=management`
- "Technology specialization courses" → Filters: `domain=technology`
- "Master level software engineering" → Filters: `level=Master`, `domain=technology`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                    (Chainlit Chat Application)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Intent Router                              │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐    │
│  │ Fast Regex    │ -> │ LLM Classify  │ -> │ Route to      │    │
│  │ (Greetings)   │    │ (if needed)   │    │ Handler       │    │
│  └───────────────┘    └───────────────┘    └───────────────┘    │
│  Default: SEARCH (action-oriented, token-efficient)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                               │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐    │
│  │ Query         │ -> │ Language      │ -> │ Metadata      │    │
│  │ Processing    │    │ Detection &   │    │ Filter        │    │
│  └───────────────┘    │ Translation   │    │ Detection     │    │
│                       │ (GPT-4o-mini) │    └───────────────┘    │
│                       └───────────────┘            │            │
│                                                    ▼            │
│                                           ┌───────────────┐     │
│                                           │ Vector        │     │
│                                           │ Search        │     │
│                                           │ (ChromaDB)    │     │
│                                           └───────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Reasoning Layer                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ GPT-4o-mini: Rank courses, generate explanations,         │  │
│  │ validate against hallucination rules                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Pipeline (ETL)

```
PDF Documents                 Preprocessing              Course Cards
┌──────────────┐             ┌──────────────┐           ┌──────────────┐
│ innovation   │ ──────────> │ Text         │ ───────>  │ Structured   │
│ _elective    │             │ Extraction   │           │ JSON with    │
│ _mmt_raw.pdf │             │ & Cleaning   │           │ metadata     │
├──────────────┤             └──────────────┘           ├──────────────┤
│ mmt_advanced │                                        │ • course_id  │
│ _seminars    │                                        │ • title      │
│ _raw.pdf     │                                        │ • content    │
├──────────────┤                                        │ • outcomes   │
│ informatics  │                                        │ • language   │
│ _major_spec  │                                        │ • level      │
│ _raw.pdf     │                                        │ • domain     │
└──────────────┘                                        └──────────────┘
```

---

## Project Structure

```
tum_project/
├── data/
│   ├── raw/                          # Source PDF files
│   │   ├── innovation_elective_mmt_raw.pdf
│   │   ├── mmt_advanced_seminars_raw.pdf
│   │   └── informatics_major_specialization_raw.pdf
│   ├── processed/                    # Extracted course cards
│   │   ├── course_cards.jsonl        # Merged (134 courses)
│   │   ├── course_cards_innovation.jsonl
│   │   ├── course_cards_advanced.jsonl
│   │   └── course_cards_informatics.jsonl
│   └── chroma_db/                    # Vector database
│
├── src/
│   ├── app.py                        # Chainlit chat interface (main entry)
│   ├── async_wrappers.py             # Async utilities for Chainlit
│   ├── intent_router.py              # Action-oriented intent classification
│   ├── rag_pipeline.py               # Vector search + filter detection
│   ├── llm_reasoning.py              # GPT-4o-mini ranking & explanations
│   ├── config.py                     # Centralized configuration
│   ├── preprocessing.py              # PDF → Page text (ETL Step 1)
│   ├── build_course_cards.py         # Pages → Course cards (ETL Step 2)
│   ├── merge_course_cards.py         # Combine sources + add domain
│   ├── build_index.py                # Create ChromaDB embeddings
│   ├── chainlit.md                   # Welcome message
│   ├── .chainlit/config.toml         # Chainlit configuration
│   ├── assets/
│   │   └── tum_logo.svg
│   └── public/
│       └── custom.css                # TUM branded styles
│
├── projecttum/                       # Python virtual environment
├── .env                              # API keys (not in git)
├── requirements.txt
└── README.md
```

---

## Current Dataset

| Source | Courses | Domain |
|--------|---------|--------|
| Innovation Electives (MMT) | 108 | Management |
| Advanced Seminars | 28 | Management |
| Informatics Specialization | 21 | Technology |
| **Total (unique)** | **134** | - |

### Language Distribution
- English: 87 courses
- German: 12 courses  
- German/English: 10 courses

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.12+ |
| **Chat Framework** | Chainlit |
| **Vector Database** | ChromaDB |
| **Embeddings** | OpenAI text-embedding-3-small |
| **LLM** | GPT-4o-mini |
| **PDF Processing** | PyMuPDF |
| **Async Support** | asyncio |

---

## Configuration

All settings are centralized in `src/config.py` for DRY compliance:

### Core Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | text-embedding-3-small | OpenAI embedding model |
| `LLM_MODEL_NAME` | gpt-4o-mini | LLM for reasoning |
| `CHROMA_COLLECTION_NAME` | tum_courses | ChromaDB collection |
| `CHROMA_PERSIST_DIR` | data/chroma_db | Vector database location |
| `INITIAL_VISIBLE_COUNT` | 10 | Initial courses displayed (increased from 5) |
| `MAX_RECOMMENDATIONS` | 50 | Max courses retrieved (increased from 15) |
| `DEFAULT_N_RESULTS` | 30 | Default RAG retrieval count (increased from 5) |

### LLM Parameters (Centralized)
| Setting | Default | Description |
|---------|---------|-------------|
| `API_TIMEOUT` | 30.0 | Timeout for all OpenAI API calls (seconds) |
| `LLM_TEMPERATURE_REASONING` | 0.3 | Temperature for course ranking |
| `LLM_TEMPERATURE_CONVERSATION` | 0.7 | Temperature for follow-up answers |
| `LLM_TEMPERATURE_CREATIVE` | 0.8 | Temperature for greetings/chitchat |
| `LLM_TEMPERATURE_CLASSIFICATION` | 0.1 | Temperature for intent classification |
| `EMBEDDING_BATCH_SIZE` | 32 | Batch size for index building |
| `MAX_RAW_TEXT_CHARS` | 1500 | Max chars for course description in LLM context |

### OpenAI Client Singleton
```python
from config import get_openai_client
client = get_openai_client()  # Returns singleton with timeout configured
```

### Environment Variables (`.env`)
- `OPENAI_API_KEY` - Required for embeddings and LLM
- `OPENAI_EMBEDDING_MODEL` - Override embedding model
- `OPENAI_CHAT_MODEL` - Override LLM model
- `OPENAI_API_TIMEOUT` - Override API timeout (default: 30s)

---

## Adding New Course Data

To add courses from a new PDF:

```bash
# 1. Place PDF in data/raw/
cp new_courses.pdf data/raw/

# 2. Extract pages
python -c "
from src.preprocessing import extract_pages_to_jsonl
extract_pages_to_jsonl('data/raw/new_courses.pdf', 'data/processed/pages_new.jsonl')
"

# 3. Build course cards
python -c "
from src.build_course_cards import *
pages = load_pages_from_jsonl('data/processed/pages_new.jsonl')
cards = build_course_cards(pages)
save_course_cards_to_jsonl(cards, 'data/processed/course_cards_new.jsonl')
"

# 4. Update merge_course_cards.py to include new source
# 5. Rebuild index
python src/merge_course_cards.py
python src/build_index.py
```

---

## UI Features

### Chainlit Chat Interface
- **Modern Chat UI** - Clean, minimal design with TUM branding
- **TUM Corporate Design** - Official blue color palette (#0065BD)
- **Starter Prompts** - Quick-start example queries for common searches
- **Course Cards** - Display metadata, match score, and reasoning
- **Follow-up Questions** - Ask about recommended courses
- **Load More** - Pagination for additional results
- **Multilingual** - Interface responds in user's language (EN, TR, DE)

---

## Hallucination Prevention

The system implements multiple safeguards:

1. **Source Constraint** - LLM only recommends from retrieved courses
2. **Course ID Validation** - Programmatic check against database
3. **Evidence Quotes** - LLM provides supporting text from course descriptions
4. **Structured Output** - JSON schema enforces valid responses
5. **Context-Only Answers** - Follow-up questions answered only from visible course data

---

## Intent Router Design

The intent router follows an **action-oriented** philosophy:

```
User Message
    │
    ├── Regex Match? ──────────────────────┐
    │   ├── Greeting (hi, hello, merhaba)  │ → GREETING (0 LLM calls)
    │   ├── Help (help, ne yapabilirsin)   │ → HELP (0 LLM calls)
    │   └── Followup + Context             │ → FOLLOWUP (0 LLM calls)
    │                                      │
    └── No Match ─────────────────────────┘
            │
            ▼
        LLM Classification
            │
            ├── Uncertain? → SEARCH (default)
            ├── Error? → SEARCH (fail forward)
            └── Clear intent → Route accordingly
```

**Key Principles:**
- **Default = SEARCH** - User came here for courses, assume that's their intent
- **Fail Forward** - On errors, search for courses (not chitchat)
- **Regex First** - Save tokens on obvious patterns
- **Token Budget Protection** - Minimize unnecessary LLM calls

---

## Chunking Strategy

Unlike traditional fixed-size chunking, this project uses **semantic chunking**:

> **1 chunk = 1 course card**

Each course card contains:
- Complete course description
- Extracted `Content` section (what the course covers)
- Extracted `Intended Learning Outcomes`
- Structured metadata (level, language, duration, SWS)

This ensures:
- No context fragmentation
- Each retrieved result is a complete, actionable course
- Better embedding quality using structured content

---

## Roadmap

### Completed 
- [x] Multi-source PDF processing pipeline
- [x] Domain-aware filtering (management/technology)
- [x] Language and level filters
- [x] Structured section extraction (Content, Learning Outcomes)
- [x] LLM reasoning with confidence scores
- [x] Professional Chainlit chat UI with TUM branding
- [x] Action-oriented intent router (Andrew NG style)
- [x] Fast regex pattern matching for common intents
- [x] Token-efficient conversation flow
- [x] Multi-turn conversation with follow-up support
- [x] Multilingual support (EN, TR, DE)
- [x] **Centralized OpenAI client singleton** (DRY refactor)
- [x] **API timeout handling** (30s default)
- [x] **Hardened error handling** (fail-forward to SEARCH)
- [x] **Version-pinned dependencies** (production-ready)
- [x] **Automatic query translation** (Turkish/German/etc → English for RAG)
- [x] **Enhanced retrieval coverage** (50 courses, 10 initial display, +10 load more)
- [x] **Universal multilingual support** (100+ languages via GPT-4o-mini)

---

## License

Developed for TUM course **"Programming in Python for Business and Life Science Analytics"** (MGT001437) by Edanur Öner.

---

<p align="center">
  <strong>TUM Course Assistant</strong> 
</p>
