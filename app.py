import streamlit as st
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json
from datetime import date

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="2Human Alpha", page_icon="🧠")
st.title("🧠 2Human: Alpha Test")
st.markdown("---")

# ─────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────
api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

# ─────────────────────────────────────────
# VECTOR STORE (loads once)
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DB_PATH = os.path.join(BASE_DIR, "schema_db")

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return FAISS.load_local(
        SCHEMA_DB_PATH, embeddings, allow_dangerous_deserialization=True
    )

db = load_db() if os.path.exists(SCHEMA_DB_PATH) else None

if db is None:
    st.error("'schema_db' folder not found! Run create_vector_db.py first.")
    st.stop()

st.sidebar.success("✅ Knowledge base loaded!")

# ─────────────────────────────────────────
# EMOTIONAL MEMORY — user profile
# ─────────────────────────────────────────
PROFILE_FILE = os.path.join(BASE_DIR, "user_profile.json")

def load_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "schemas_identified": [],
        "schema_candidates": [],
        "behavioral_patterns": [],
        "core_needs": [],
        "last_updated": str(date.today())
    }

def save_profile(profile):
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

def format_profile_for_prompt(profile):
    """Formats profile compactly for the system prompt."""
    lines = []

    active = [s for s in profile.get("schemas_identified", []) if not s.get("dormant", False)]
    if active:
        lines.append("Identified schemas (active):")
        for s in active:
            triggers = ", ".join(s.get("triggers", [])) or "not yet mapped"
            lines.append(f"  - {s['name']} (intensity {s['intensity']}/10) | triggers: {triggers}")
    else:
        lines.append("Identified schemas: none yet.")

    candidates = profile.get("schema_candidates", [])
    if candidates:
        lines.append("Schema candidates (under observation):")
        for c in candidates:
            lines.append(f"  - {c['name']}: {len(c.get('evidence', []))}/3 evidence collected")

    patterns = [p for p in profile.get("behavioral_patterns", []) if isinstance(p, dict) and "trigger" in p]
    if patterns:
        lines.append("Mapped patterns:")
        for p in patterns[:3]:
            lines.append(f"  - Trigger '{p['trigger']}' → {p['emotion']} → {p['behavior']}")

    needs = ", ".join(profile.get("core_needs", [])) or "not identified yet"
    lines.append(f"Core needs: {needs}")

    return "\n".join(lines) if lines else "No data collected yet — first session."

# ─────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────
if "profile" not in st.session_state:
    st.session_state.profile = load_profile()
    save_profile(st.session_state.profile)

if "messages" not in st.session_state:
    welcome = (
        "Hello! I'm the 2Human Mentor — a conversational guide based on Schema Therapy.\n\n"
        "I'm not a therapist, and I don't diagnose. I'm an experimental tool designed to help "
        "you explore emotional patterns through reflection and conversation.\n\n"
        "As we talk, I'll quietly map your triggers, emotional responses, and recurring patterns. "
        "When I notice something worth naming, I'll share it — and explain what I saved and why.\n\n"
        "Tell me: how are you feeling right now, or what's been on your mind lately?"
    )
    st.session_state.messages = [{"role": "assistant", "content": welcome}]

# ─────────────────────────────────────────
# SIDEBAR — live profile
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂️ Your profile")
    active_schemas = [
        s for s in st.session_state.profile.get("schemas_identified", [])
        if not s.get("dormant", False)
    ]
    candidates = st.session_state.profile.get("schema_candidates", [])

    if active_schemas:
        st.markdown("**Identified schemas:**")
        for s in active_schemas:
            st.markdown(f"- {s['name']} — {s['intensity']}/10")
    else:
        st.markdown("*No schemas identified yet*")

    if candidates:
        st.markdown("**Under observation:**")
        for c in candidates:
            count = len(c.get("evidence", []))
            st.markdown(f"- {c['name']} ({count}/3 evidence)")

    st.markdown("---")
    st.caption("2Human is a guide, not a therapist.")

# ─────────────────────────────────────────
# RENDER CHAT HISTORY
# ─────────────────────────────────────────
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ─────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────
if prompt := st.chat_input("How are you feeling right now?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    MAX_HISTORY = 10
    recent = st.session_state.messages[-MAX_HISTORY:]
    history_context = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

    # Semantic search — only for substantial messages
    book_context = ""
    if len(prompt) > 15:
        docs = db.similarity_search(prompt, k=5)
        chunks = []
        for d in docs:
            source = d.metadata.get("fonte", "unknown source")
            tipo = d.metadata.get("tipo", "")
            priority = d.metadata.get("prioridade", 3)
            chunks.append(
                f"[Source: {source} | Type: {tipo} | Priority: {priority}]\n{d.page_content}"
            )
        book_context = "\n\n---\n\n".join(chunks)

    profile_context = format_profile_for_prompt(st.session_state.profile)
    user_msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])

    # ─────────────────────────────────────────
    # SYSTEM PROMPT
    # ─────────────────────────────────────────
    system_prompt = f"""You are the 2Human Mentor — a conversational self-knowledge guide based on Schema Therapy (ST).

IDENTITY:
- You are a guide, NOT a therapist. Never diagnose, prescribe, or claim clinical certainty.
- You are an experimental educational tool. Be transparent about this when relevant.
- Always respond in English, regardless of the language the user writes in.

USER PROFILE (persistent memory — never ignore):
{profile_context}

BEHAVIORAL RULES:
- Use the profile to maintain continuity. Never restart from scratch each message.
- When the user mentions something activating a known schema or candidate, name it naturally.
- When you save NEW data (new trigger, pattern, or candidate), briefly explain in italics
  what you noted and why — only when something genuinely new was identified.
- ONE open question per message. Never send questionnaires or walls of text.
- Be empathetic and direct. Depth without detours.
- If the user expresses suicidal ideation, self-harm, or crisis, 
  immediately shift to a supportive tone, provide crisis resources 
  (CVV: 188 for Brazil, or local emergency services), and do NOT 
  analyze or save the message as a schema data point.

ONBOARDING (first 3 user messages only):
- Naturally explain how you work: you observe patterns, save triggers and emotions,
  and need repeated evidence before naming a schema.
- After the 3rd message, stop explaining your process unless something new is saved.
- Current user message count: {user_msg_count}

RESPONSE FORMAT when saving new data:
  1. Warmly acknowledge the user's message (1-3 sentences).
  2. New line in italics: briefly explain what you just saved.
  Example: *I noted 'not being invited' as a trigger and 'they don't like me' as the
  associated feeling. I'll need a few more similar patterns before naming a schema.*

SOURCE RULES:
- Your knowledge base belongs to YOU, not the user.
- You MAY use clinical examples to illustrate patterns, but always frame them as yours:
  "In Schema Therapy, there's a pattern called..." — NEVER "as we studied together."

SCHEMA IDENTIFICATION:
- A schema is only IDENTIFIED after 3+ distinct evidence entries as a candidate.
- Before that: track as candidate, count evidence.
- When promoting a candidate: name it warmly in the conversation.
- Max 5 active schemas. Others become dormant (never deleted).

SOURCE HIERARCHY (when sources conflict):
1. Clinical primary [type: clinica_primaria] — Young, Klosko
2. Scientific articles [type: artigo_cientifico] — Bach, Edwards

Use the knowledge base only when the user's message relates to schemas or emotional patterns.
Ignore it for greetings and short messages."""

    # ─────────────────────────────────────────
    # GEMINI CALL
    # ─────────────────────────────────────────
    with st.chat_message("assistant"):
        try:
            contents = []
            if book_context:
                contents.append(f"Knowledge base references:\n{book_context}")
            contents.append(f"Recent conversation:\n{history_context}\n\nUser: {prompt}")

            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=contents,
                config={"system_instruction": system_prompt}
            )
            reply = response.text
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

            # ─────────────────────────────────────────
            # SILENT PROFILE UPDATE (every 3 user messages)
            # ─────────────────────────────────────────
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            if len(user_msgs) % 3 == 0:
                update_prompt = f"""You are a clinical analysis system based on Schema Therapy.
Analyze the conversation and update the user profile JSON.

Current profile:
{json.dumps(st.session_state.profile, ensure_ascii=False, indent=2)}

Last 3 user messages:
{chr(10).join([m['content'] for m in user_msgs[-3:]])}

Rules:
1. Track new patterns as CANDIDATES first — add to "schema_candidates" with evidence entries
2. A candidate is promoted to "schemas_identified" only after 3+ evidence entries
3. For identified schemas: increase intensity with new evidence, never duplicate
4. Max 5 active schemas (highest intensity). Others get "dormant": true
5. Never delete — mark dormant if inactive
6. Update triggers, behavioral_patterns, and core_needs when identified
7. Use English for all field values

Return ONLY the updated JSON. No extra text, no markdown, no backticks."""

                try:
                    profile_response = client.models.generate_content(
                        model="gemini-flash-latest",
                        contents=[update_prompt]
                    )
                    raw = profile_response.text.strip()
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    new_profile = json.loads(raw)
                    new_profile["last_updated"] = str(date.today())
                    st.session_state.profile = new_profile
                    save_profile(new_profile)
                except Exception:
                    pass  # Keep current profile if parsing fails

        except Exception as e:
            st.error("Could not get a response. Please try again in a moment.")