# 🧠 2Human — Alpha

> A conversational self-knowledge guide grounded in Schema Therapy.

**Live demo → [2human.streamlit.app](https://2human.streamlit.app)**

---

## What is this?

2Human is an experimental AI agent that helps people explore their emotional patterns through conversation. It's not a therapist and doesn't diagnose — it's a reflective guide built on [Schema Therapy](https://en.wikipedia.org/wiki/Schema_therapy), a clinical framework developed by Jeffrey Young.

As you talk, the agent quietly maps your emotional triggers, recurring behavioral patterns, and core needs. Over time, it builds a persistent emotional profile unique to you — and uses it to maintain continuity across sessions.

**Available in English and Portuguese.**

---

## How it works

```
User message
    │
    ▼
Semantic search over indexed clinical sources (RAG)
    │
    ▼
Gemini Flash — responds using sources + user's emotional profile
    │
    ▼
Every 3 messages: silent profile update (JSON saved per user)
```

### Schema identification logic

The agent doesn't rush to label. A schema goes through three stages:

1. **Observation** — a trigger or emotional response is noted internally
2. **Candidate** — after 2+ similar patterns, the schema enters observation (shown in sidebar)
3. **Identified** — after 3+ distinct evidence entries, the schema is named in conversation

Max 5 active schemas at a time. Others become dormant — never deleted.

---

## Longitudinal memory

Most AI chatbots forget you the moment the session ends. 2Human doesn't.

Each user has a persistent emotional profile stored as a JSON file, built and refined across every conversation. The agent carries this profile into every session — so it never asks the same question twice, and patterns become clearer over time.

```json
{
  "name": "Mathaus",
  "schemas_identified": [
    {
      "name": "Emotional Deprivation",
      "intensity": 7,
      "triggers": ["not being heard", "feeling invisible"],
      "dormant": false
    }
  ],
  "schema_candidates": [
    {
      "name": "Abandonment",
      "evidence": ["fears being left out", "anxiety when ignored"]
    }
  ],
  "behavioral_patterns": [
    {
      "trigger": "conflict at work",
      "emotion": "shame",
      "behavior": "withdraws and goes silent"
    }
  ],
  "core_needs": ["emotional validation", "stable connection"],
  "last_updated": "2026-03-10"
}
```

The profile updates silently every 3 messages — the user never sees a form or questionnaire. The agent observes, infers, and saves. When something new is identified, it briefly explains what it noted and why, in italics, inline in the conversation.

This is what makes 2Human a longitudinal tool rather than a single-session chatbot: the longer someone uses it, the more accurate and personal the reflection becomes.

---

## Knowledge base

The agent's responses are grounded in indexed clinical sources via RAG (FAISS + sentence-transformers):

| Priority | Source | Type |
|----------|--------|------|
| 1 | Young & Klosko — *Reinventing Your Life* | Primary clinical |
| 1 | Young, Klosko & Weishaar — *Schema Therapy: A Practitioner's Guide* | Primary clinical |
| 2 | Bach et al. (2018) — *A New Look at the Schema Therapy Model* | Scientific article |
| 2 | Edwards (2021) — *Using Schema Modes for Case Conceptualization* | Scientific article |

---

## Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| LLM | Gemini Flash (`gemini-flash-latest`) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector store | FAISS |
| User memory | Per-user JSON (local) |

---

## Run locally

### 1. Clone and install

```bash
git clone https://github.com/mathauskruger/2human-alpha
cd 2human-alpha
pip install -r requirements.txt
```

### 2. Add your Gemini API key

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

### 3. Index the knowledge base

Place your PDF sources in a `fontes/` folder, then run:

```bash
python create_vector_db.py
```

This generates the `schema_db/` folder. Only needs to run once.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Project structure

```
2human_alpha/
├── app.py                  # Main Streamlit app
├── create_vector_db.py     # Indexes PDFs into FAISS
├── schema_db/              # Generated vector store (committed)
├── fontes/                 # PDF sources — NOT committed (copyright)
├── requirements.txt
└── .streamlit/
    └── secrets.toml        # NOT committed — add your API key here
```

---

## Roadmap

- [x] RAG over Schema Therapy sources
- [x] Persistent emotional profile per user
- [x] Schema candidate tracking (3-evidence threshold)
- [x] Bilingual (EN / PT)
- [x] Crisis / suicide ideation detection → redirect to CVV (188)
- [ ] Remote profile storage (Supabase)
- [ ] Neuropsychological screening agent (gifted assessment)

---

## Disclaimer

2Human is an experimental research tool, not a clinical service. It does not replace therapy, diagnosis, or professional mental health support. If you're in crisis, please contact a mental health professional or call **CVV: 188** (Brazil).

---

*Built with curiosity. Schema Therapy framework by Jeffrey Young.*