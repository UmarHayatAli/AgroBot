"""
AgroBot LangGraph Agent — Multilingual Edition
================================================
Integrates pipeline.py's STT / language-detection / TTS capabilities
into the LangGraph multi-node architecture.

New capabilities vs. the original:
  • 4 languages: English · Urdu · Punjabi · Saraiki
  • Unicode-script language detection (fast, no LLM call needed)
  • Optional audio input  → Groq Whisper STT node
  • Optional audio output → Edge TTS / gTTS node
  • All specialist-node system prompts are language-aware

Architecture:
    [audio_bytes?]
        │
        ▼
    [stt_node]  ─────────────────── (optional, skipped for text input)
        │
        ▼
    [supervisor]  ← language pre-detected by Unicode script analysis
        │
        ├──► [soil_crop_node]
        ├──► [weather_node]
        ├──► [market_node]
        ├──► [rag_node]
        └──► [general_node]
                │
                ▼
        [response_formatter]
                │
                ▼
        [tts_node]  ─────────────── (optional, only when audio_output=True)
                │
                ▼
        Final Response (text + optional MP3 bytes)
"""

import os
import tempfile
import json
import pickle
import asyncio
import tempfile
from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from pathlib import Path

# ── LangGraph ──────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ── LangChain / LLM ────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Tools ───────────────────────────────────────────────────────────────────────
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Pydantic ────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field

# ── TTS / STT ──────────────────────────────────────────────────────────────────
from groq import Groq
import edge_tts
from gtts import gTTS

from dotenv import load_dotenv
load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MULTILINGUAL CONFIG  (merged from pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

LANGUAGE_CONFIG = {
    "en": {
        "label":        "English",
        "whisper_lang": "en",
        "edge_voice":   "en-US-JennyNeural",
        "gtts_lang":    "en",
        "flag":         "🇺🇸",
        "dir":          "ltr",
    },
    "ur": {
        "label":        "Urdu",
        "whisper_lang": "ur",
        "edge_voice":   "ur-PK-UzmaNeural",
        "gtts_lang":    "ur",
        "flag":         "🇵🇰",
        "dir":          "rtl",
    },
    "pa": {
        "label":        "Punjabi",
        "whisper_lang": "pa",
        "edge_voice":   "ur-PK-AsadNeural",  # closest Pakistani male voice
    "gtts_lang":    "ur",                # Urdu fallback — same script
    "flag":         "🌾",
    "dir":          "rtl",     
    },
    "skr": {
        "label":        "Saraiki",
        "whisper_lang": "ur",               # Closest Whisper model
        "edge_voice":   "ur-PK-AsadNeural", # Closest available TTS voice
        "gtts_lang":    "ur",
        "flag":         "🌻",
        "dir":          "rtl",
    },
}

# Language-specific system prompt suffixes injected into every specialist node
LANG_INSTRUCTION = {
    "en":  "Respond ONLY in English.",
    "ur":  "صرف اردو میں جواب دیں۔",
    "pa":  "صرف پنجابی زبان (شاہ مکھی رسم الخط) وچ جواب دیو۔ اردو وانگوں لکھی جانے والی پنجابی استعمال کرو۔ گورمکھی یا ہندی رسم الخط استعمال کرنا سخت منع ہے۔",
    "skr": "صرف سرائیکی زبان وچ جواب دیو۔ شاہ مکھی رسم الخط استعمال کرو۔",
}

# Specialist-node system prompts per language
# Each node picks its prompt dynamically using _lang_prompt() below.
NODE_PROMPTS = {
    "crop_interpreter": {
        "en": (
            "You are AgroBot's agronomic advisor. "
            "Convert the XGBoost prediction JSON into a clear, friendly explanation. "
            "Lead with the recommended crop and yield. Explain WHY using SHAP features. "
            "Mention top 3 alternatives. Flag any warnings. Under 200 words."
        ),
        "ur": (
            "آپ AgroBot کے زرعی مشیر ہیں۔ "
            "XGBoost کے نتائج کو کسان کے لیے آسان الفاظ میں بیان کریں۔ "
            "تجویز کردہ فصل اور پیداوار سے شروع کریں۔ SHAP فیچرز سے وجہ بتائیں۔ "
            "اوپر کی 3 متبادل فصلیں بھی بتائیں۔ 200 الفاظ سے کم۔"
        ),
        "pa": (
            "ਤੁਸੀਂ AgroBot ਦੇ ਖੇਤੀਬਾੜੀ ਸਲਾਹਕਾਰ ਹੋ। "
            "XGBoost ਦੇ ਨਤੀਜਿਆਂ ਨੂੰ ਕਿਸਾਨ ਲਈ ਸਾਦੀ ਭਾਸ਼ਾ ਵਿੱਚ ਦੱਸੋ। "
            "ਸਿਫਾਰਸ਼ੀ ਫਸਲ ਅਤੇ ਝਾੜ ਤੋਂ ਸ਼ੁਰੂ ਕਰੋ। 200 ਸ਼ਬਦਾਂ ਤੋਂ ਘੱਟ।"
        ),
        "skr": (
            "تُساں AgroBot دے زرعی مشیر ہو۔ "
            "XGBoost دے نتیجے کسان لئی سادی زبان وچ دسو۔ "
            "سفارش کردہ فصل تے پیداوار توں شروع کرو۔ 200 لفظاں توں گھٹ۔"
        ),
    },
    "weather": {
        "en": (
            "LANGUAGE: Respond ONLY in English.\n"
            "You are AgroBot's weather-advisory agent for Pakistani farmers. "
            "Given weather data (JSON), provide: (1) conditions summary, "
            "(2) agronomic implication, (3) one actionable tip. Under 150 words."
        ),
        "ur": (
            "زبان: صرف اردو میں جواب دیں۔\n"
            "آپ پاکستانی کسانوں کے لیے AgroBot کے موسمی مشیر ہیں۔ "
            "موسمی ڈیٹا کی بنیاد پر: (1) موجودہ حالات، "
            "(2) زرعی اثرات، (3) ایک عملی مشورہ دیں۔ 150 الفاظ سے کم۔"
        ),
        "pa": (
            "زبان: صرف پنجابی (شاہ مکھی) وچ جواب دیو۔\n"
            "تُساں پاکستانی کساناں لئی AgroBot دے موسمی مشیر ہو۔ "
            "موسمی ڈیٹا توں: (1) موجودہ حالات، (2) زرعی اثرات، (3) اک مشورہ۔ 150 لفظاں توں گھٹ۔"
        ),
        "skr": (
            "زبان: صرف سرائیکی وچ جواب دیو۔\n"
            "تُساں پاکستانی کساناں لئی AgroBot دے موسمی مشیر ہو۔ "
            "موسمی ڈیٹا توں: (1) موجودہ حالات، (2) زرعی اثرات، (3) اک عملی صلاح۔ 150 لفظاں توں گھٹ۔"
        ),
    },
    "market": {
        "en": (
            "You are AgroBot's market intelligence agent for Pakistani farmers. "
            "From search results: (1) price range (PKR/maund), (2) market trend, "
            "(3) sell advice, (4) one risk. Under 150 words."
        ),
        "ur": (
            "آپ پاکستانی کسانوں کے لیے AgroBot کے مارکیٹ انٹیلیجنس ایجنٹ ہیں۔ "
            "تلاش کے نتائج سے: (1) قیمت (PKR/من)، (2) مارکیٹ رجحان، "
            "(3) فروخت کا مشورہ، (4) ایک خطرہ۔ 150 الفاظ سے کم۔"
        ),
        "pa": (
            "ਤੁਸੀਂ ਪਾਕਿਸਤਾਨੀ ਕਿਸਾਨਾਂ ਲਈ AgroBot ਦੇ ਮਾਰਕਿਟ ਸਲਾਹਕਾਰ ਹੋ। "
            "ਖੋਜ ਨਤੀਜਿਆਂ ਤੋਂ: (1) ਭਾਅ (PKR/ਮਣ), (2) ਮਾਰਕਿਟ ਰੁਝਾਨ, "
            "(3) ਵੇਚਣ ਦੀ ਸਲਾਹ, (4) ਇੱਕ ਜੋਖਮ। 150 ਸ਼ਬਦਾਂ ਤੋਂ ਘੱਟ।"
        ),
        "skr": (
            "تُساں پاکستانی کساناں لئی AgroBot دے مارکیٹ ایجنٹ ہو۔ "
            "تلاش دے نتیجیاں توں: (1) قیمت (PKR/من)، (2) مارکیٹ رجحان، "
            "(3) وکرائی دی صلاح، (4) اک خطرہ۔ 150 لفظاں توں گھٹ۔"
        ),
    },
    "rag": {
        "en": (
            "You are AgroBot's agronomic knowledge expert. "
            "Use retrieved context as your primary source. "
            "If context doesn't answer, say so, then give a cautious general answer. "
            "Do not invent doses or dates. Under 200 words."
        ),
        "ur": (
            "آپ AgroBot کے زرعی علمی ماہر ہیں۔ "
            "حاصل شدہ سیاق کو بنیادی ماخذ کے طور پر استعمال کریں۔ "
            "اگر سیاق جواب نہیں دیتا تو کہیں اور پھر محتاط عام جواب دیں۔ "
            "خوراک یا تاریخیں نہ بنائیں۔ 200 الفاظ سے کم۔"
        ),
        "pa": (
            "ਤੁਸੀਂ AgroBot ਦੇ ਖੇਤੀਬਾੜੀ ਗਿਆਨ ਮਾਹਰ ਹੋ। "
            "ਪ੍ਰਾਪਤ ਸੰਦਰਭ ਨੂੰ ਮੁੱਖ ਸਰੋਤ ਵਜੋਂ ਵਰਤੋ। "
            "ਜੇ ਸੰਦਰਭ ਜਵਾਬ ਨਹੀਂ ਦਿੰਦਾ ਤਾਂ ਦੱਸੋ ਅਤੇ ਸਾਵਧਾਨੀ ਨਾਲ ਜਵਾਬ ਦਿਓ। 200 ਸ਼ਬਦਾਂ ਤੋਂ ਘੱਟ।"
        ),
        "skr": (
            "تُساں AgroBot دے زرعی علمی ماہر ہو۔ "
            "حاصل کردہ سیاق نوں بنیادی ماخذ وجوں ورتو۔ "
            "جے سیاق جواب نئیں دیندا تاں دسو تے محتاط جواب دیو۔ 200 لفظاں توں گھٹ۔"
        ),
    },
    "general": {
        "en": (
            "LANGUAGE: Respond ONLY in English.\n"
            "You are AgroBot, a friendly AI assistant for Pakistani farmers. "
            "Answer helpfully. If out of scope, redirect to farming topics. Under 120 words."
        ),
        "ur": (
            "زبان: صرف اردو میں جواب دیں۔\n"
            "آپ AgroBot ہیں، پاکستانی کسانوں کے لیے ایک دوستانہ AI مددگار۔ "
            "مددگار جواب دیں۔ اگر موضوع سے باہر ہو تو زراعت کی طرف رہنمائی کریں۔ 120 الفاظ سے کم۔"
        ),
        "pa": (
            "زبان: صرف پنجابی (شاہ مکھی) وچ جواب دیو۔\n"
            "تُساں AgroBot ہو، پاکستانی کساناں لئی اک دوستانہ AI مددگار۔ "
            "مددگار جواب دیو۔ جے موضوع توں باہر ہووے تاں زراعت ول رہنمائی کرو۔ 120 لفظاں توں گھٹ۔"
        ),
        "skr": (
            "زبان: صرف سرائیکی وچ جواب دیو۔\n"
            "تُساں AgroBot ہو، پاکستانی کساناں لئی اک دوستانہ AI مددگار۔ "
            "مددگار جواب دیو۔ جے موضوع توں باہر ہووے تاں زراعت ول رہنمائی کرو۔ 120 لفظاں توں گھٹ۔"
        ),
    },
}


def _lang_prompt(node_key: str, lang: str) -> str:
    """Return the system prompt for (node, language), defaulting to English."""
    return NODE_PROMPTS[node_key].get(lang, NODE_PROMPTS[node_key]["en"])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LANGUAGE DETECTION  (Unicode script analysis — from pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def detect_language_unicode(text: str, whisper_code: str = "") -> str:
    """
    Detect language code from Unicode script ranges.

    Returns one of: "en" | "ur" | "pa" | "skr"

    Rules:
      • Gurmukhi (U+0A00–U+0A7F) > 25%          → "pa"
      • Arabic   (U+0600–U+06FF) > 25%
          – Saraiki chars ݨ(U+0768) / ݙ(U+0759)  → "skr"
          – otherwise                             → "ur"
      • Latin ASCII alpha > 30%                  → "en"
      • Fallback on Whisper code / default "en"
    """
    if not text.strip():
        return "en"

    gurmukhi = sum(1 for c in text if '\u0A00' <= c <= '\u0A7F')
    arabic   = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin    = sum(1 for c in text if c.isascii() and c.isalpha())
    total    = max(len(text.replace(" ", "")), 1)

    if gurmukhi / total > 0.25:
        return "pa"
    if arabic / total > 0.25:
        saraiki_hits = sum(1 for c in text if c in '\u0768\u0759')
        return "skr" if saraiki_hits > 0 else "ur"
    if latin / total > 0.30:
        return "en"

    _map = {"en": "en", "ur": "ur", "pa": "pa"}
    return _map.get(whisper_code, "en")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════

class AgroBotState(TypedDict):
    messages:           Annotated[list, add_messages]
    user_query:         str
    detected_lang:      str          # "en" | "ur" | "pa" | "skr"
    intent:             str
    soil_data:          Optional[Dict[str, Any]]
    prediction_result:  Optional[Dict[str, Any]]
    weather_result:     Optional[Dict[str, Any]]
    market_result:      Optional[str]
    rag_result:         Optional[str]
    final_response:     Optional[str]
    detected_crop:      Optional[str]
    location:           Optional[str]
    # ── NEW: audio I/O ────────────────────────────────────────────────────────
    audio_input:        Optional[bytes]   # raw audio bytes (WAV/MP3) from caller
    audio_ext:          Optional[str]     # file extension: "wav" | "mp3" | "m4a"
    audio_output:       bool              # whether to run TTS at the end
    tts_audio:          Optional[bytes]   # MP3 bytes from TTS node


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SUPERVISOR SCHEMA  (expanded to 4 language codes)
# ══════════════════════════════════════════════════════════════════════════════

KNOWN_CROPS = {"cotton", "wheat", "rice", "maize", "sugarcane",
               "sunflower", "sorghum", "potato", "onion", "mango"}

class SoilData(BaseModel):
    N:           Optional[float] = None
    P:           Optional[float] = None
    K:           Optional[float] = None
    temperature: Optional[float] = None
    humidity:    Optional[float] = None
    ph:          Optional[float] = None
    rainfall:    Optional[float] = None
    EC:          Optional[float] = None
    Soil_Type:   Optional[str]   = None

class SupervisorOutput(BaseModel):
    detected_lang:  Literal["en", "ur", "pa", "skr"]   # ← 4 codes now
    intent:         Literal["soil_crop", "weather", "market", "rag_agronomy", "general"]
    soil_data:      Optional[SoilData]   = None
    location:       Optional[str]        = None
    detected_crop:  Optional[str]        = Field(
        None,
        description=f"Crop name if mentioned. One of: {KNOWN_CROPS}. None if absent.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SHARED RESOURCES
# ══════════════════════════════════════════════════════════════════════════════

def _load_xgboost_models():
    base  = Path(__file__).parent / "Models"
    paths = {
        "model":     base / "xgboost_model.pkl",
        "scaler":    base / "scaler.pkl",
        "le":        base / "label_encoder.pkl",
        "knn":       base / "knn_model.pkl",
        "explainer": base / "shap_explainer.pkl",
    }
    if not all(p.exists() for p in paths.values()):
        print("⚠️  XGBoost model files not found – crop prediction disabled.")
        return None, None, None, None, None
    return (
        pickle.load(open(paths["model"],     "rb")),
        pickle.load(open(paths["scaler"],    "rb")),
        pickle.load(open(paths["le"],        "rb")),
        pickle.load(open(paths["knn"],       "rb")),
        pickle.load(open(paths["explainer"], "rb")),
    )

def _load_vector_store():
    chroma_path = Path(__file__).parent / "chroma_db"
    if not chroma_path.exists():
        print("⚠️  ChromaDB not found – RAG disabled.")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embeddings,
            collection_name="agrobot_knowledge",
        )
    except Exception as e:
        print(f"⚠️  ChromaDB load error: {e}")
        return None

XGB_MODEL, SCALER, LE, KNN_MODEL, EXPLAINER = _load_xgboost_models()
VECTOR_STORE = _load_vector_store()
SEARCH_TOOL  = DuckDuckGoSearchRun()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_llm(temperature: float = 0.3) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return ChatGroq(model="llama-3.1-8b-instant", temperature=temperature, api_key=api_key)

def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)

def build_context_aware_query(state: AgroBotState) -> str:
    if not state["messages"]:
        return state["user_query"]
    llm = get_llm(temperature=0)
    result = llm.invoke([
        SystemMessage(content=(
            "You are a query rewriter for an agricultural chatbot. "
            "Rewrite the latest message as a fully self-contained query that includes "
            "all relevant context from the conversation. Return ONLY the rewritten query."
        )),
        *state["messages"],
        HumanMessage(content=f"Rewrite this as a self-contained query: '{state['user_query']}'")
    ]).content.strip()
    print(f"🔄 Context-aware query: {result}")
    return result

def extract_city(query: str) -> Optional[str]:
    common_cities = ["Lahore", "Karachi", "Islamabad", "Multan", "Faisalabad",
                     "Rawalpindi", "Peshawar", "Quetta", "Gujranwala", "Sialkot"]
    for city in common_cities:
        if city.lower() in query.lower():
            return city
    for token in query.split():
        if token.istitle() and len(token) > 4 and token not in ["What", "Today", "Should"]:
            return token
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 7.  NODE: STT  (NEW — from pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def stt_node(state: AgroBotState) -> AgroBotState:
    """
    Transcribe audio_input → user_query and detect language.
    Skipped automatically when audio_input is None.
    """
    audio_bytes = state.get("audio_input")
    if not audio_bytes:
        return state   # no-op for text-only input

    ext    = state.get("audio_ext") or "wav"
    client = get_groq_client()

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                response_format="verbose_json",
            )
        whisper_code = getattr(result, "language", "en") or "en"
        transcript   = result.text.strip()
    finally:
        os.unlink(path)

    # Unicode script detection is more reliable than Whisper's lang code
    lang = detect_language_unicode(transcript, whisper_code)
    print(f"🎙️  STT transcript ({lang}): {transcript}")

    return {
        **state,
        "user_query":   transcript,
        "detected_lang": lang,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  NODE: SUPERVISOR  (updated for 4-language support)
# ══════════════════════════════════════════════════════════════════════════════

SUPERVISOR_SYSTEM = """You are AgroBot's routing supervisor.

Extract structured information from the user query.

Rules:
- Detect language: "en" (English), "ur" (Urdu), "pa" (Punjabi), "skr" (Saraiki)
- Choose ONE intent from: soil_crop | weather | market | rag_agronomy | general
- 'weather' intent: Use for ALL questions about weather, rain, temperature, or forecasts.
- 'soil_crop' intent: Use ONLY for soil data analysis or asking what to plant.
- Extract soil parameters only when explicitly present (do NOT guess)
- Extract location (city/region) if present
- Extract crop name only if it matches the known list
"""

def supervisor_node(state: AgroBotState) -> AgroBotState:
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(SupervisorOutput)

    enriched_query = build_context_aware_query(state)

    # If language was already detected by STT + Unicode analysis, trust it.
    # We pass it in the prompt so the LLM doesn't override it.
    lang_hint = ""
    if state.get("audio_input") and state.get("detected_lang"):
        lang_hint = (
            f"\n[SYSTEM NOTE: Language was pre-detected from audio as '{state['detected_lang']}'. "
            "Use this value for detected_lang unless very clearly wrong.]"
        )

    try:
        result: SupervisorOutput = structured_llm.invoke([
            SystemMessage(content=SUPERVISOR_SYSTEM + lang_hint),
            *state["messages"],
            HumanMessage(content=enriched_query),
        ])
    except Exception as e:
        print(f"⚠️ Supervisor structured output failed: {e}")
        # Fallback to general intent if structured output fails
        return {
            **state,
            "detected_lang":  state.get("detected_lang") or detect_language_unicode(enriched_query),
            "user_query":     enriched_query,
            "intent":         "general",
            "messages":       state["messages"] + [HumanMessage(content=state["user_query"])],
        }

    # If audio was processed, honour the Unicode detection over LLM's guess
    final_lang = (
        state["detected_lang"]
        if state.get("audio_input") and state.get("detected_lang")
        else result.detected_lang
    )

    # Fallback: run Unicode detection on text input too (fast, free)
    if not state.get("audio_input"):
        unicode_lang = detect_language_unicode(enriched_query)
        # Trust Unicode if it's confident (Arabic/Gurmukhi heavy text)
        if unicode_lang in ("pa", "skr"):
            final_lang = unicode_lang

    detected_crop = result.detected_crop.lower().strip() if result.detected_crop else None
    soil_data     = result.soil_data.dict() if result.soil_data else None

    return {
        **state,
        "detected_lang":  final_lang,
        "user_query":     enriched_query,
        "intent":         result.intent,
        "soil_data":      soil_data,
        "detected_crop":  detected_crop,
        "location":       result.location,
        "messages":       state["messages"] + [HumanMessage(content=state["user_query"])],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9.  NODE: SOIL-TO-CROP PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

try:
    from XGBoostClassifierAndYeildPredictor import agro_predict
    _AGRO_PREDICT_AVAILABLE = True
except ImportError:
    _AGRO_PREDICT_AVAILABLE = False
    print("⚠️  agro_predict not importable – using mock prediction.")

def _mock_predict(soil_data: dict) -> dict:
    return {
        "recommended_crop": "Wheat",
        "predicted_yield_tons_per_acre": 2.65,
        "suitability": "Good 🟢",
        "confidence": 87.3,
        "top_3_crops": [("Wheat", 87.3), ("Maize", 8.1), ("Rice", 4.6)],
        "similar_cases": [],
        "feature_importance": {},
        "validation_warnings": None,
        "note": "[MOCK] Real prediction requires model files in /Models/",
    }

def soil_crop_node(state: AgroBotState) -> AgroBotState:
    soil_data = state.get("soil_data") or {}
    print(f"\n📥 SOIL DATA:\n{json.dumps(soil_data, indent=2)}\n")

    if _AGRO_PREDICT_AVAILABLE and XGB_MODEL is not None:
        try:
            result = agro_predict(soil_data)
        except Exception as e:
            result = {"error": str(e)}
    else:
        result = _mock_predict(soil_data)

    lang = state["detected_lang"]
    llm  = get_llm(temperature=0.4)
    narrative = llm.invoke([
        SystemMessage(content=_lang_prompt("crop_interpreter", lang)),
        HumanMessage(content=(
            f"Prediction result (JSON):\n{json.dumps(result, indent=2)}\n\n"
            f"User's question: {state['user_query']}"
        )),
    ]).content

    return {**state, "prediction_result": result, "final_response": narrative}


# ══════════════════════════════════════════════════════════════════════════════
# 10.  NODE: WEATHER
# ══════════════════════════════════════════════════════════════════════════════

def weather_node(state: AgroBotState) -> AgroBotState:
    import requests
    from datetime import datetime

    owm_key      = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()
    city         = state.get("location") or extract_city(state["user_query"]) or "Lahore"
    weather_data = None

    if owm_key:
        try:
            url  = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={owm_key}&units=metric"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                raw = resp.json()
                weather_data = {
                    "source":  "OpenWeatherMap",
                    "city":    raw["name"],
                    "current": {
                        "temp":       round(raw["main"]["temp"], 1),
                        "feels_like": round(raw["main"]["feels_like"], 1),
                        "humidity":   raw["main"]["humidity"],
                        "desc":       raw["weather"][0]["description"].capitalize(),
                        "wind_speed": raw["wind"].get("speed", 0),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            print(f"⚠️ OWM error: {e}")

    if not weather_data:
        try:
            snippet      = SEARCH_TOOL.run(f"current weather {city} Pakistan right now")
            weather_data = {"source": "web_search", "city": city, "search_snippet": snippet[:800]}
        except Exception as e:
            weather_data = {"error": str(e)}

    lang = state["detected_lang"]
    llm  = get_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=_lang_prompt("weather", lang)),
        HumanMessage(content=(
            f"Weather data: {json.dumps(weather_data, ensure_ascii=False, indent=2)}\n"
            f"Farmer question: {state['user_query']}\n"
            f"Date/time: {datetime.now().strftime('%Y-%m-%d %H:%M')} PKT"
        )),
    ]).content

    return {**state, "weather_result": weather_data, "final_response": response}


# ══════════════════════════════════════════════════════════════════════════════
# 11.  NODE: MARKET
# ══════════════════════════════════════════════════════════════════════════════

def market_node(state: AgroBotState) -> AgroBotState:
    query = state["user_query"]
    
    # 1. Built-in Reference Prices (Pakistan Mandi - May 2024 Estimates)
    # This ensures the LLM always has an answer even if search fails.
    REFERENCE_PRICES = """
    CURRENT MARKET ESTIMATES (May 2024 - PKR per 40kg/maund):
    - Wheat (Gandum): 3,200 - 3,600 PKR
    - Cotton (Phutti): 7,500 - 9,200 PKR
    - Rice (Basmati): 7,500 - 9,500 PKR
    - Maize (Makai): 2,200 - 2,600 PKR
    - Sugarcane (Kamad): 400 - 450 PKR
    - Onion (Piyaz): 4,000 - 6,000 PKR
    - Potato (Aloo): 2,500 - 3,500 PKR
    """

    try:
        # 2. Try web search for 5 seconds max
        search_result = SEARCH_TOOL.run(f"Pakistan {query} mandi price current month PKR")
        if not search_result or len(search_result) < 20:
             search_result = "Recent web data unavailable. Using built-in reference prices."
    except Exception as e:
        print(f"⚠️ Market Search failed: {e}")
        search_result = "Search tool offline. Using built-in reference prices."

    lang = state["detected_lang"]
    llm  = get_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=_lang_prompt("market", lang) + "\nNote: If search results are missing, use the REFERENCE_PRICES below."),
        HumanMessage(content=(
            f"{REFERENCE_PRICES}\n\n"
            f"Web Search results:\n{search_result}\n\n"
            f"Farmer question: {query}"
        )),
    ]).content

    return {**state, "market_result": search_result, "final_response": response}


# ══════════════════════════════════════════════════════════════════════════════
# 12.  NODE: RAG AGRONOMY
# ══════════════════════════════════════════════════════════════════════════════

def rag_node(state: AgroBotState) -> AgroBotState:
    query   = state["user_query"]
    topic   = state.get("detected_crop")
    context = ""
    retrieved_docs = []

    if VECTOR_STORE is not None:
        try:
            search_kwargs = {"k": 4, "fetch_k": 12}
            if topic:
                search_kwargs["filter"] = {"topic": topic}
            retriever      = VECTOR_STORE.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
            retrieved_docs = retriever.invoke(query)
        except Exception as e:
            print(f"⚠️  Retrieval error: {e}")

    if retrieved_docs:
        parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            src     = doc.metadata.get("source", f"chunk_{i}")
            section = doc.metadata.get("section", "")
            parts.append(f"[Source: {src} | Section: {section}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(parts)
    else:
        try:
            context = SEARCH_TOOL.run(f"Pakistan agriculture {query} advice")
        except Exception:
            context = "No relevant context found."

    lang = state["detected_lang"]
    llm  = get_llm(temperature=0.4)
    response = llm.invoke([
        SystemMessage(content=_lang_prompt("rag", lang)),
        *state["messages"],
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )),
    ]).content

    return {
        **state,
        "rag_result":    context[:1200] + "..." if len(context) > 1200 else context,
        "final_response": response,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 13.  NODE: GENERAL FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def general_node(state: AgroBotState) -> AgroBotState:
    lang = state["detected_lang"]
    llm  = get_llm(temperature=0.5)
    response = llm.invoke([
        SystemMessage(content=_lang_prompt("general", lang)),
        *state["messages"],
        HumanMessage(content=state["user_query"]),
    ]).content
    return {**state, "final_response": response}


# ══════════════════════════════════════════════════════════════════════════════
# 14.  NODE: RESPONSE FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def response_formatter_node(state: AgroBotState) -> AgroBotState:
    response_text = state.get("final_response", "I'm sorry, I could not generate a response.")
    intent        = state.get("intent", "general")
    lang          = state.get("detected_lang", "en")
    lang_label    = LANGUAGE_CONFIG.get(lang, {}).get("label", "English")

    badge_map = {
        "soil_crop":    "🌱 *Powered by AgroBot XGBoost Crop Engine*",
        "weather":      "🌤️ *Powered by OpenWeatherMap + AgroBot Advisory*",
        "market":       "📊 *Market data via web search – verify before trading*",
        "rag_agronomy": "📚 *AgroBot Knowledge Base*",
        "general":      "🤖 *AgroBot Assistant*",
    }
    badge         = badge_map.get(intent, "🤖 *AgroBot*")
    lang_tag      = f"_{LANGUAGE_CONFIG.get(lang, {}).get('flag', '')} {lang_label}_"
    full_response = f"{response_text}\n\n{badge}  {lang_tag}"

    return {
        **state,
        "final_response": full_response,
        "messages":       state["messages"] + [AIMessage(content=full_response)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 15.  NODE: TTS  (NEW — from pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

async def _edge_save(text: str, voice: str, path: str) -> None:
    await edge_tts.Communicate(text, voice).save(path)

def tts_node(state: AgroBotState) -> AgroBotState:
    """
    Convert final_response to MP3 audio.
    Only runs when state['audio_output'] is True.
    Uses Edge TTS (neural voices) with gTTS fallback.
    """
    if not state.get("audio_output"):
        return state

    text = state.get("final_response", "")
    lang = state.get("detected_lang", "en")
    cfg  = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG["en"])

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name

    try:
        if cfg["edge_voice"]:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_edge_save(text, cfg["edge_voice"], path))
                loop.close()
            except Exception as e:
                print(f"⚠️ Edge TTS failed ({e}), falling back to gTTS")
                gTTS(text=text, lang=cfg["gtts_lang"], slow=False).save(path)
        else:
            gTTS(text=text, lang=cfg["gtts_lang"], slow=False).save(path)

        with open(path, "rb") as f:
            mp3_bytes = f.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)

    print(f"🔊 TTS generated ({lang}, {len(mp3_bytes):,} bytes)")
    return {**state, "tts_audio": mp3_bytes}


# ══════════════════════════════════════════════════════════════════════════════
# 16.  ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def route_stt(state: AgroBotState) -> Literal["stt_node", "supervisor"]:
    """Route to STT node only when audio bytes are provided."""
    return "stt_node" if state.get("audio_input") else "supervisor"

def route_intent(
    state: AgroBotState,
) -> Literal["soil_crop", "weather", "market", "rag_agronomy", "general"]:
    return state.get("intent", "general")


# ══════════════════════════════════════════════════════════════════════════════
# 17.  BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

from langgraph.graph import StateGraph, END, START  # add START here

def build_agrobot_graph() -> StateGraph:
    graph = StateGraph(AgroBotState)

    # ── Nodes ──────────────────────────────────────────────────────
    graph.add_node("stt_node",           stt_node)
    graph.add_node("supervisor",         supervisor_node)
    graph.add_node("soil_crop",          soil_crop_node)
    graph.add_node("weather",            weather_node)
    graph.add_node("market",             market_node)
    graph.add_node("rag_agronomy",       rag_node)
    graph.add_node("general",            general_node)
    graph.add_node("response_formatter", response_formatter_node)
    graph.add_node("tts_node",           tts_node)

    # ── Entry: branch on audio vs text ─────────────────────────────
    graph.add_conditional_edges(
        START,                          # ← directly from START
        route_stt,
        {"stt_node": "stt_node", "supervisor": "supervisor"},
    )
    graph.add_edge("stt_node", "supervisor")

    # ── Routing after supervisor ────────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_intent,
        {
            "soil_crop":    "soil_crop",
            "weather":      "weather",
            "market":       "market",
            "rag_agronomy": "rag_agronomy",
            "general":      "general",
        },
    )

    # ── All specialist nodes → formatter → TTS → END ───────────────
    for node in ["soil_crop", "weather", "market", "rag_agronomy", "general"]:
        graph.add_edge(node, "response_formatter")

    graph.add_edge("response_formatter", "tts_node")
    graph.add_edge("tts_node", END)

    return graph.compile()

agrobot_graph = build_agrobot_graph()


# ══════════════════════════════════════════════════════════════════════════════
# 18.  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_agrobot(
    user_message:  str   = "",
    history:       list  = None,
    audio_bytes:   bytes = None,
    audio_ext:     str   = "wav",
    audio_output:  bool  = False,
    lang:          str   = "",   # UI-selected language — overrides auto-detection
) -> dict:
    """
    Main entry point for the AgroBot agent.

    Args:
        user_message  : Farmer's text query (English / Urdu / Punjabi / Saraiki).
                        Pass "" when providing audio_bytes.
        history       : Prior (HumanMessage | AIMessage) objects for multi-turn.
        audio_bytes   : Raw audio bytes for voice input (triggers STT node).
        audio_ext     : Audio file extension: "wav" | "mp3" | "m4a".
        audio_output  : If True, generates MP3 TTS audio in the response.

    Returns:
        {
          "text":    str,            # final response text
          "lang":    str,            # detected language code
          "audio":   bytes | None,   # MP3 bytes if audio_output=True
          "intent":  str,
        }
    """
    initial_state: AgroBotState = {
        "messages":          history or [],
        "user_query":        user_message,
        "detected_lang":     lang or detect_language_unicode(user_message) or "en",
        "intent":            "",
        "soil_data":         None,
        "prediction_result": None,
        "weather_result":    None,
        "market_result":     None,
        "rag_result":        None,
        "final_response":    None,
        "detected_crop":     None,
        "location":          None,
        # audio fields
        "audio_input":       audio_bytes,
        "audio_ext":         audio_ext,
        "audio_output":      audio_output,
        "tts_audio":         None,
    }

    result = agrobot_graph.invoke(initial_state)
    return {
        "text":   result["final_response"],
        "lang":   result["detected_lang"],
        "audio":  result.get("tts_audio"),
        "intent": result.get("intent"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 19.  CLI TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🌾 AgroBot Interactive CLI — Multilingual Edition")
    print("   Languages: English · اردو · ਪੰਜਾਬੀ · سرائیکی")
    print("   Type 'exit' to quit | '--audio' flag for TTS output")
    print("=" * 60)

    history = []

    while True:
        raw_input = input("\n👨‍🌾 You: ").strip()

        if raw_input.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting AgroBot...")
            break

        want_audio = "--audio" in raw_input
        user_text  = raw_input.replace("--audio", "").strip()

        output = run_agrobot(
            user_message=user_text,
            history=history,
            audio_output=want_audio,
        )

        print(f"\n🤖 AgroBot [{output['lang']}]: {output['text']}")

        if output["audio"]:
             audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_outputs")
             os.makedirs(audio_dir, exist_ok=True)  # creates folder if it doesn't exist
             out_path = os.path.join(audio_dir, "agrobot_response.mp3")
             with open(out_path, "wb") as f:
              f.write(output["audio"])
             print(f"🔊 Audio saved → {out_path}")

        history.append(HumanMessage(content=user_text))
        history.append(AIMessage(content=output["text"]))