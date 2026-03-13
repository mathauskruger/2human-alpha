import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json
import time
from datetime import date

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="2Human Alpha", page_icon="🧠")

# ─────────────────────────────────────────
# OPENROUTER CLIENT
# ─────────────────────────────────────────
api_key = st.secrets["OPENROUTER_API_KEY"]
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
MODEL = "openai/gpt-4o-mini"

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

# ─────────────────────────────────────────
# WELCOME SCREEN — name input
# ─────────────────────────────────────────
if "username" not in st.session_state:
    st.session_state.username = None
if "lang" not in st.session_state:
    st.session_state.lang = None

if st.session_state.username is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🧠 2Human")
        st.markdown("<br>", unsafe_allow_html=True)

        # STEP 1 — choose language first
        if st.session_state.lang is None:
            st.markdown("*A self-knowledge guide based on Schema Therapy*")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("Choose your language / Escolha o idioma:")
            col_en, col_pt = st.columns(2)
            with col_en:
                if st.button("🇺🇸 English", use_container_width=True):
                    st.session_state.lang = "en"
                    st.rerun()
            with col_pt:
                if st.button("🇧🇷 Português", use_container_width=True):
                    st.session_state.lang = "pt"
                    st.rerun()

        # STEP 2 — name, already in chosen language
        elif st.session_state.lang == "pt":
            st.markdown("*Um guia de autoconhecimento baseado na Terapia do Esquema*")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("Como posso te chamar?")
            name = st.text_input("", placeholder="Seu primeiro nome", label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Começar →", use_container_width=True):
                if name.strip():
                    st.session_state.username = name.strip()
                    st.rerun()
                else:
                    st.warning("Por favor, insira seu nome para continuar.")
            if st.button("← Voltar", use_container_width=True):
                st.session_state.lang = None
                st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("2Human é um guia, não um terapeuta. Versão alpha experimental.")

        else:
            st.markdown("*A self-knowledge guide based on Schema Therapy*")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("Before we begin — how should I call you?")
            name = st.text_input("", placeholder="Your first name", label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Begin →", use_container_width=True):
                if name.strip():
                    st.session_state.username = name.strip()
                    st.rerun()
                else:
                    st.warning("Please enter your name to continue.")
            if st.button("← Back", use_container_width=True):
                st.session_state.lang = None
                st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("2Human is a guide, not a therapist. Experimental alpha.")

    st.stop()

# ─────────────────────────────────────────
# PROFILE — per user JSON
# ─────────────────────────────────────────
safe_name = "".join(c for c in st.session_state.username.lower() if c.isalnum())
PROFILE_FILE = os.path.join(BASE_DIR, f"user_profile_{safe_name}.json")

def load_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "name": st.session_state.username,
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
    lines = []

    active = [s for s in profile.get("schemas_identified", []) if isinstance(s, dict) and not s.get("dormant", False)]
    if active:
        lines.append("Identified schemas (active):")
        for s in active:
            if not s.get("name"):
                continue
            triggers = ", ".join(s.get("triggers", [])) or "not yet mapped"
            lines.append(f"  - {s['name']} (intensity {s.get('intensity','?')}/10) | triggers: {triggers}")
    else:
        lines.append("Identified schemas: none yet.")

    candidates = [c for c in profile.get("schema_candidates", []) if isinstance(c, dict) and c.get("name")]
    if candidates:
        lines.append("Schema candidates (under observation):")
        for c in candidates:
            lines.append(f"  - {c['name']}: {len(c.get('evidence', []))}/3 evidence collected")

    patterns = [p for p in profile.get("behavioral_patterns", []) if isinstance(p, dict) and "trigger" in p]
    if patterns:
        lines.append("Mapped patterns:")
        for p in patterns[:3]:
            lines.append(f"  - Trigger '{p['trigger']}' → {p['emotion']}' → {p['behavior']}")

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
    name = st.session_state.username
    lang = st.session_state.lang
    if lang == "pt":
        welcome = (
            f"Olá, {name}! Sou o Mentor 2Human — um guia conversacional baseado na Terapia do Esquema.\n\n"
            "Não sou terapeuta e não faço diagnósticos. Sou uma ferramenta experimental para te ajudar "
            "a explorar seus padrões emocionais através da reflexão e da conversa.\n\n"
            "Enquanto conversamos, vou mapeando seus gatilhos, respostas emocionais e padrões recorrentes. "
            "Quando identificar algo relevante, vou compartilhar — e explicar o que salvei e por quê.\n\n"
            "Me conta: como você está se sentindo agora, ou o que tem passado pela sua cabeça ultimamente?"
        )
    else:
        welcome = (
            f"Hello, {name}! I'm the 2Human Mentor — a conversational guide based on Schema Therapy.\n\n"
            "I'm not a therapist, and I don't diagnose. I'm an experimental tool designed to help "
            "you explore emotional patterns through reflection and conversation.\n\n"
            "As we talk, I'll quietly map your triggers, emotional responses, and recurring patterns. "
            "When I notice something worth naming, I'll share it — and explain what I saved and why.\n\n"
            "Tell me: how are you feeling right now, or what's been on your mind lately?"
        )
    st.session_state.messages = [{"role": "assistant", "content": welcome}]

# ─────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────
st.title("🧠 2Human: Alpha Test")
st.markdown("---")

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.username}")
    st.markdown("✅ " + ("Base de conhecimento carregada!" if st.session_state.get("lang") == "pt" else "Knowledge base loaded!"))
    st.markdown("---")
    _pt = st.session_state.get("lang") == "pt"

    st.markdown("### 🗂️ " + ("Seu perfil" if _pt else "Your profile"))

    active_schemas = [
        s for s in st.session_state.profile.get("schemas_identified", [])
        if not s.get("dormant", False)
    ]
    candidates = [
        c for c in st.session_state.profile.get("schema_candidates", [])
        if isinstance(c, dict) and c.get("name")
    ]

    if active_schemas:
        st.markdown("**" + ("Esquemas identificados:" if _pt else "Identified schemas:") + "**")
        for s in active_schemas:
            if isinstance(s, dict) and s.get("name"):
                st.markdown(f"- {s['name']} — {s.get('intensity', '?')}/10")
    else:
        st.markdown("*" + ("Nenhum esquema identificado ainda" if _pt else "No schemas identified yet") + "*")

    if candidates:
        st.markdown("**" + ("Em observação:" if _pt else "Under observation:") + "**")
        for c in candidates:
            count = len(c.get("evidence", []))
            label = "evidências" if _pt else "evidence"
            st.markdown(f"- {c['name']} ({count}/3 {label})")

    st.markdown("---")

    btn_fresh = "🔄 Recomeçar" if _pt else "🔄 Start fresh"
    btn_change = "🚪 Trocar usuário" if _pt else "🚪 Change user"

    if st.button(btn_fresh):
        st.session_state.profile = {
            "name": st.session_state.username,
            "schemas_identified": [],
            "schema_candidates": [],
            "behavioral_patterns": [],
            "core_needs": [],
            "last_updated": str(date.today())
        }
        save_profile(st.session_state.profile)
        del st.session_state["messages"]
        st.rerun()

    if st.button(btn_change):
        st.session_state.username = None
        st.session_state.lang = None
        st.session_state.messages = []
        if "profile" in st.session_state:
            del st.session_state.profile
        st.rerun()

    st.markdown("---")
    with st.expander("🔍 Ver perfil salvo" if _pt else "🔍 Inspect saved profile"):
        st.json(st.session_state.profile)
    st.caption("2Human é um guia, não um terapeuta." if _pt else "2Human is a guide, not a therapist.")

# ─────────────────────────────────────────
# RENDER CHAT HISTORY
# ─────────────────────────────────────────
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ─────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────
_placeholder = "How are you feeling right now?" if st.session_state.get("lang") == "en" else "Como você está se sentindo agora?"
if prompt := st.chat_input(_placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    MAX_HISTORY = 10
    recent = st.session_state.messages[-MAX_HISTORY:]
    history_context = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

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

    lang = st.session_state.get("lang", "en")
    lang_instruction = (
        "Always respond in Brazilian Portuguese, regardless of the language the user writes in."
        if lang == "pt"
        else "Always respond in English, regardless of the language the user writes in."
    )

    system_prompt = f"""You are the 2Human Mentor — a conversational self-knowledge guide based on Schema Therapy (ST).

IDENTITY:
- You are a guide, NOT a therapist. Never diagnose, prescribe, or claim clinical certainty.
- You are an experimental educational tool. Be transparent about this when relevant.
- {lang_instruction}
- The user's name is {st.session_state.username} — use it naturally and sparingly.

USER PROFILE (persistent memory — never ignore):
{profile_context}

BEHAVIORAL RULES:
- Use the profile to maintain continuity. Never restart from scratch each message.
- When the user mentions something activating a known schema or candidate, name it naturally.
- When you save NEW data (new trigger, pattern, or candidate), briefly explain in italics
  what you noted and why — only when something genuinely new was identified.
- ONE open question per message. Never send questionnaires or walls of text.
- Be empathetic and direct. Depth without detours.
- Be patient and observational. Resist the urge to name or label too early.
  Sit with ambiguity — a pattern only becomes meaningful when it repeats.
- If the user expresses suicidal ideation, self-harm, or crisis: immediately shift to a
  supportive tone, provide crisis resources (CVV: 188 for Brazil, or local emergency
  services), and do NOT analyze or save the message as a schema data point.

ONBOARDING — MANDATORY for first 3 user messages:
- This is message number {user_msg_count} from the user.
- If {user_msg_count} <= 3: you MUST include a brief explanation of how you work in your reply.
  Explain naturally, woven into your response — not as a separate paragraph.
  Tell the user: you observe emotional patterns across conversations, you save triggers and
  recurring emotions, and you need multiple instances of a pattern before naming a schema.
- If {user_msg_count} > 3: do NOT explain your process unless you are saving new data.

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

    with st.chat_message("assistant"):
        try:
            contents = []
            if book_context:
                contents.append(f"Knowledge base references:\n{book_context}")
            contents.append(f"Recent conversation:\n{history_context}\n\nUser: {prompt}")

            for _attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]],
                            {"role": "user", "content": "\n\n".join(contents)},
                        ]
                    )
                    break
                except Exception as _e:
                    if "429" in str(_e) and _attempt < 2:
                        time.sleep(10)
                    else:
                        raise
            reply = response.choices[0].message.content
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

            # Silent profile update every 3 user messages
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
7. Use {"Portuguese" if lang == "pt" else "English"} for all field values

Return ONLY the updated JSON. No extra text, no markdown, no backticks."""

                try:
                    profile_response = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": update_prompt}]
                    )
                    raw = profile_response.choices[0].message.content.strip()
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    new_profile = json.loads(raw)
                    # Validar estrutura mínima antes de salvar
                    required_keys = {"name", "schemas_identified", "schema_candidates", "behavioral_patterns", "core_needs"}
                    if required_keys.issubset(new_profile.keys()):
                        new_profile["last_updated"] = str(date.today())
                        st.session_state.profile = new_profile
                        save_profile(new_profile)
                except Exception as _profile_err:
                    st.sidebar.warning(f"⚠️ Profile update failed: `{type(_profile_err).__name__}`")

        except Exception as e:
            lang = st.session_state.get("lang", "en")
            msg = "Não foi possível obter resposta. Tente novamente." if lang == "pt" else "Could not get a response. Please try again."
            st.error(f"{msg}\n\n`{type(e).__name__}: {e}`")