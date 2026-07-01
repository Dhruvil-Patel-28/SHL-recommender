# SHL Assessment Recommender

A conversational AI agent exposed as a FastAPI service that helps hiring managers and recruiters find the right SHL assessments for their roles. The user describes what they are hiring for, and the agent asks clarifying questions if needed, then recommends 1–10 real SHL assessments from a pre-loaded product catalog.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Setup & Run](#setup--run)
- [Example Usage](#example-usage)
- [Deployment](#deployment)

---

## Features

- **Multi-turn conversation** — Maintains context across user messages for natural back-and-forth dialogue.
- **Clarify → Recommend → Refine → Confirm** — Follows a structured conversation flow: asks clarifying questions for vague queries, recommends assessments with enough context, supports add/drop/replace refinements, and confirms when the user is satisfied.
- **Grounded in real catalog** — Every recommended assessment name and URL is validated against the official SHL product catalog (377 entries). No hallucinated items.
- **Proactive domain defaults** — Automatically includes OPQ32r (personality), Verify G+ (cognitive ability for senior roles), DSI (safety-critical roles), and Graduate Scenarios (graduate hiring) based on role context.
- **Off-topic refusal** — Rejects salary questions, legal advice, and general HR queries. Resists prompt injection.
- **Catalog gap handling** — If a requested technology skill test doesn't exist (e.g., Rust), acknowledges the gap, pivots to alternatives, and notes the limitation.

---

## Architecture

```
┌─────────────┐     POST /chat      ┌──────────────────────────────────────────┐
│   Client     │ ──────────────────► │  FastAPI (main.py)                       │
│  (curl/UI)   │ ◄────────────────── │                                          │
└─────────────┘   ChatResponse       │  ┌──────────────────────────────────┐    │
                                     │  │  agent.py                        │    │
                                     │  │                                  │    │
                                     │  │  1. Extract facets (rule-based)  │    │
                                     │  │  2. Filter catalog (Python)      │    │
                                     │  │  3. Build system prompt          │    │
                                     │  │  4. Call Groq LLM (single call)  │    │
                                     │  │  5. Parse + validate response    │    │
                                     │  └──────────┬───────────────────────┘    │
                                     │             │                            │
                                     │  ┌──────────▼───────────────────────┐    │
                                     │  │  catalog.py                      │    │
                                     │  │  377 entries from SHL catalog    │    │
                                     │  │  Scoring: skill + level + type   │    │
                                     │  └──────────────────────────────────┘    │
                                     └──────────────────────────────────────────┘
```

### Request Flow

1. **Facet Extraction** — Rule-based keyword matching extracts skills (java, python, sql…), seniority levels, test type preferences, and language requirements from the conversation. Zero LLM cost, instant.
2. **Catalog Filtering** — Python-level scoring narrows 377 catalog items to ~30-60 relevant candidates. Uses exact substring + fuzzy matching (rapidfuzz). Always includes Personality & Ability items as fallback pool.
3. **System Prompt Construction** — Compact system prompt (~800 tokens) + filtered catalog items injected as context. Stays under Groq free-tier 12K TPM limit.
4. **Single LLM Call** — One Groq API call per request (Llama-3.3-70b-versatile). No multi-step chains.
5. **Response Parsing & Validation** — Extracts JSON from LLM output, validates every recommendation name/URL against the real catalog. Three-tier matching: exact URL → exact name → fuzzy name + substring. Two fallback layers if JSON recommendations array is empty.

---

## Design Decisions

### Why single LLM call instead of multi-step chains?

A multi-step chain (extract → filter → reason → format) would cost 3-4x the tokens and add 5-10s latency. Instead, we front-load work with zero-cost Python:
- **Facet extraction** is rule-based (keyword lists), not an LLM call.
- **Catalog filtering** is Python scoring (rapidfuzz), not an LLM call.
- Only the final reasoning + formatting is done by the LLM.

This gives us one API call per user message with ~2-3s latency.

### Why Python pre-filtering instead of sending the full catalog?

The full catalog (377 items) serializes to ~30K tokens — far exceeding Groq's free-tier 12K TPM limit and even straining paid tiers. Pre-filtering to 30-60 relevant items keeps the prompt under 8K tokens while ensuring the LLM has the right items to recommend.

### Why rule-based facet extraction instead of LLM extraction?

LLM-based extraction would add an extra API call per request. The facets we need (skill keywords, seniority signals, test type preferences) are well-suited to keyword matching — a Java developer query contains "java", not a paraphrase. The rule-based approach is instant, free, and deterministic.

### Why validate LLM recommendations against the catalog?

LLMs hallucinate URLs, modify assessment names, and invent non-existent products. Every recommendation goes through three-tier validation:
1. **Exact URL match** (with trailing-slash normalization)
2. **Exact name match** (case-insensitive)
3. **Fuzzy name match** (substring containment + partial_ratio)

Plus two fallback layers if the LLM's JSON `recommendations` array is empty but the reply text mentions assessments.

### Why `import catalog as catalog_module` instead of `from catalog import CATALOG`?

A critical bug was discovered: `from catalog import CATALOG` captures a reference to the initial empty list at import time. When `load_catalog()` later rebinds `catalog.CATALOG` to the loaded 377-item list, the reference in `agent.py` still points to `[]`. Using module-level import (`catalog_module.CATALOG`) always resolves to the current value.

### Why Groq + Llama 3.3 70B?

- **Groq**: Fastest inference for open-source models (~200 tok/s). Free tier sufficient for development.
- **Llama 3.3 70B Versatile**: Best open-source model for structured output + reasoning at this scale. Reliably produces JSON blocks and follows system prompt instructions.

### Why `rapidfuzz` for fuzzy matching?

- Pure C++ backend → 10-100x faster than `fuzzywuzzy`.
- `partial_ratio` handles short queries against long catalog names (e.g., "OPQ32r" vs "Occupational Personality Questionnaire OPQ32r").
- No Python-Levenshtein dependency issues.

### Why proactive domain defaults (OPQ32r, Verify G+, DSI)?

SHL's sample conversations (C1–C10) consistently show the agent including default assessments without being asked:
- OPQ32r for almost all hiring decisions
- Verify G+ for senior roles
- DSI for safety-critical positions

This was a deliberate design choice extracted from analyzing all 10 sample traces.

### Why `end_of_conversation` is never set proactively?

The grader specification requires `end_of_conversation: true` only when the user has **explicitly confirmed** they are satisfied. The agent never assumes the conversation is over — it waits for explicit signals like "confirmed", "that's it", "perfect", or "thanks".

---

## Project Structure

```
shl-recommender/
├── main.py              # FastAPI app — /health and /chat endpoints
├── agent.py             # LLM call logic, system prompt, response parsing
├── catalog.py           # Catalog loading, filtering, text serialization
├── models.py            # Pydantic request/response schemas
├── data/
│   └── catalog.json     # SHL product catalog (377 entries)
├── requirements.txt     # Python dependencies
├── .env                 # GROQ_API_KEY (not committed)
├── .gitignore
└── README.md
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `main.py` | FastAPI app setup, CORS, lifespan (startup loading), error handling |
| `agent.py` | System prompt, facet extraction, Groq API call, JSON parsing, catalog validation, abbreviation mapping, fallback extraction |
| `catalog.py` | Load catalog JSON, score-based filtering (skill/level/type/language), compact text serialization for LLM context |
| `models.py` | `ChatRequest`, `ChatResponse`, `Message`, `Recommendation` Pydantic models |

---

## API Reference

### `GET /health`

Health check.

**Response:**
```json
{"status": "ok"}
```

### `POST /chat`

Main conversation endpoint.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a senior Java developer"},
    {"role": "assistant", "content": "I recommend..."},
    {"role": "user", "content": "Add a SQL test too"}
  ]
}
```

**Response Body:**
```json
{
  "reply": "Here are your updated recommendations...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    },
    {
      "name": "Occupational Personality Questionnaire OPQ32r",
      "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

**Field Reference:**

| Field | Type | Description |
|-------|------|-------------|
| `reply` | string | Natural language response text |
| `recommendations` | array | 0-10 assessment objects. Always a list, never null. Empty `[]` when clarifying or refusing. |
| `end_of_conversation` | bool | `true` only when user explicitly confirms they are done |
| `test_type` | string | Code(s): `A`=Ability, `B`=Biodata/SJT, `C`=Competencies, `D`=Development, `E`=Exercises, `K`=Knowledge, `P`=Personality, `S`=Simulations. Multiple joined with comma: `"K,S"` |

### `GET /docs`

Auto-generated Swagger UI (provided by FastAPI).

---

## Setup & Run

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier works)

### Installation

```bash
# Clone and enter project
cd shl-recommender

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### Run locally

```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

---

## Example Usage

### 1. Simple recommendation
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to hire a senior Java developer with Spring and SQL experience"}
    ]
  }'
```

### 2. Vague query (triggers clarification)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need an assessment"}
    ]
  }'
```

### 3. Multi-turn refinement
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need assessments for a graduate financial analyst"},
      {"role": "assistant", "content": "I recommend Verify Numerical, OPQ32r, and Graduate Scenarios."},
      {"role": "user", "content": "Drop OPQ32r and add an SJT"}
    ]
  }'
```

### 4. Off-topic refusal
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the average salary for a software engineer?"}
    ]
  }'
```

---

## Deployment

### Render

1. Push code to GitHub (ensure `.env` is in `.gitignore`)
2. Create a new Web Service on [Render](https://render.com)
3. Configure:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable: `GROQ_API_KEY`

### Docker (optional)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... shl-recommender
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for the REST API |
| `uvicorn` | ASGI server to run FastAPI |
| `groq` | Official Groq Python client for LLM inference |
| `pydantic` | Request/response validation and serialization |
| `rapidfuzz` | Fast fuzzy string matching for catalog lookups |
| `python-dotenv` | Load `.env` file for API keys |
