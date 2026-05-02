"""
AgroBot — Centralized Configuration
=====================================
All API endpoints, model names, language config, and file paths
are defined here. Import from this module everywhere.

Architecture decision: Option A — Unified LangGraph Bot with Intent Router.
One LangGraph StateGraph routes farmer queries to specialist nodes (soil, weather,
disease, flood, general). This beats a dispatcher pattern for a hackathon demo
because the farmer sees one seamless conversational agent — no mode selection needed.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
BACKEND_DIR  = BASE_DIR / "Backend"
DATA_DIR     = BASE_DIR / "data"
PROMPTS_DIR  = BASE_DIR / "prompts"
MODELS_DIR   = BACKEND_DIR / "Models"

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
LLM_MODEL      = "llama-3.3-70b-versatile"        # Primary Groq model
VISION_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq vision model
WHISPER_MODEL  = "whisper-large-v3"               # Groq STT

# ── Optional / Third-party APIs ───────────────────────────────────────────────
KINDWISE_API_KEY       = os.getenv("KINDWISE_API_KEY", "")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")

# ── Open-Meteo (free, no key) ─────────────────────────────────────────────────
OPEN_METEO_BASE    = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEO     = "https://geocoding-api.open-meteo.com/v1/search"

# ── Groq endpoints ────────────────────────────────────────────────────────────
GROQ_CHAT_URL   = "https://api.groq.com/openai/v1/chat/completions"
KINDWISE_URL    = "https://crop.kindwise.com/api/v1/identification?details=description,treatment,cause"

# ── Language configuration ────────────────────────────────────────────────────
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
        "edge_voice":   "ur-PK-AsadNeural",   # closest Pakistani voice
        "gtts_lang":    "ur",                  # Urdu fallback — same script
        "flag":         "🌾",
        "dir":          "rtl",
    },
    "skr": {
        "label":        "Saraiki",
        "whisper_lang": "ur",                  # closest Whisper code
        "edge_voice":   "ur-PK-AsadNeural",
        "gtts_lang":    "ur",
        "flag":         "🌻",
        "dir":          "rtl",
    },
}

# ── Language enforcement instructions ─────────────────────────────────────────
LANG_INSTRUCTION = {
    "en":  "Respond ONLY in English.",
    "ur":  "صرف اردو میں جواب دیں۔ اردو رسم الخط (نستعلیق) استعمال کریں۔ ہندی رسم الخط (Devanagari) استعمال کرنا سخت منع ہے۔",
    "pa":  "صرف پنجابی زبان (شاہ مکھی رسم الخط) وچ جواب دیو۔ گورمکھی (Gurmukhi) یا ہندی رسم الخط استعمال کرنا سخت منع ہے۔ صرف اردو وانگوں فارسی عربی رسم الخط استعمال کرو۔",
    "skr": "صرف سرائیکی زبان وچ جواب دیو۔ شاہ مکھی رسم الخط استعمال کرو۔ ہندی یا گورمکھی رسم الخط استعمال کرنا سخت منع ہے۔",
}

# ── Audio settings ────────────────────────────────────────────────────────────
RECORD_SECONDS   = 5
SAMPLE_RATE      = 16000   # Hz — Whisper optimal
AUDIO_CHANNELS   = 1       # Mono

# ── ML model paths ────────────────────────────────────────────────────────────
XGB_MODEL_PATH   = MODELS_DIR / "xgboost_model.pkl"
SCALER_PATH      = MODELS_DIR / "scaler.pkl"
LE_PATH          = MODELS_DIR / "label_encoder.pkl"
KNN_PATH         = MODELS_DIR / "knn_model.pkl"
EXPLAINER_PATH   = MODELS_DIR / "shap_explainer.pkl"

# ── Data files ────────────────────────────────────────────────────────────────
DISEASE_TREATMENTS_JSON  = DATA_DIR / "disease_treatments.json"
FLOOD_RISK_JSON          = DATA_DIR / "flood_risk_crops.json"
CROP_CSV                 = BACKEND_DIR / "Crop_recommendation.csv"

# ── Weather thresholds for flood risk ─────────────────────────────────────────
FLOOD_RAIN_THRESHOLDS = {
    "low":      10,   # mm/day
    "medium":   25,
    "high":     50,
    "critical": 80,
}

# ── Rich terminal colors ──────────────────────────────────────────────────────
RISK_COLORS = {
    "Low":      "green",
    "Medium":   "yellow",
    "High":     "red",
    "Critical": "bold red",
}

# ── Validation ────────────────────────────────────────────────────────────────
def check_groq_key() -> bool:
    return bool(GROQ_API_KEY)

def get_missing_config() -> list[str]:
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    return missing
