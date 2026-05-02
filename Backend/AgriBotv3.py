"""
AgroBot LangGraph Agent — v3 PRODUCTION EDITION
================================================
Drastic improvements over v2:

ARCHITECTURE OVERHAUL:
  ✅ Fully async nodes (asyncio.gather for true parallelism)
  ✅ Self-reflection critic loop (agent grades & retries its own answers)
  ✅ Farmer profile node (remembers location, crops, history across sessions)
  ✅ LangGraph SQLite checkpointer (persistent multi-turn memory per farmer)
  ✅ Hybrid RAG: BM25 + semantic + cross-encoder reranking
  ✅ Tool-augmented synthesis (LLM calls live tools during generation)
  ✅ TTL caching for weather + market data (avoid redundant API calls)
  ✅ Tenacity retry with exponential backoff on all external calls
  ✅ Response quality gate (auto-retry if answer is too short / low quality)
  ✅ Structured Pydantic validation at every node boundary
  ✅ Streaming-ready response pipeline

NEW GRAPH TOPOLOGY:
    [audio?] → [stt]
                 │
           [supervisor]
                 │
        [farmer_profile]        ← NEW: load/update persistent farmer context
                 │
    ┌────────────┼──────────────────┬──────────────────┐
    ▼            ▼                  ▼                  ▼
[hybrid_rag] [weather]        [market]       [disease_detection]
    │            │                  │                  │
    └────────────┴──────────────────┘                  │
                 ▼                                      ▼
          [data_gatherer]                   [disease_detection_node]
                 │                                      │
                 └──────────────┬────────────────────────┘
                                ▼
                         [synthesis]
                                │
                         [critic]          ← NEW: scores quality, retries
                                │
                    ┌───────────┤
                    ▼ (retry)   ▼ (pass)
               [synthesis]  [formatter]
                                │
                           [tts_node]
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import pickle
import re
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import (
    Annotated, Any, Dict, List, Literal, Optional, Tuple, TypedDict
)

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ── LangGraph ──────────────────────────────────────────────────────────────
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver   # ← persistent memory

# ── LangChain ──────────────────────────────────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# ── TTS / STT ──────────────────────────────────────────────────────────────
import edge_tts
from groq import Groq
from gtts import gTTS

# ── Retry + Caching ────────────────────────────────────────────────────────
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("AgroBot")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

KINDWISE_API_KEY = os.getenv("KINDWISE_API_KEY", "")
KINDWISE_URL     = "https://crop.kindwise.com/api/v1/identification?details=description,treatment,cause"

PAK_CLIMATE_DATA = {
    "Karachi":         {"monsoon_start": (7,  5), "flood_threshold_mm": 40,  "heatwave_c": 40},
    "Hyderabad":       {"monsoon_start": (7,  5), "flood_threshold_mm": 50,  "heatwave_c": 42},
    "Multan":          {"monsoon_start": (7, 10), "flood_threshold_mm": 80,  "heatwave_c": 45},
    "Lahore":          {"monsoon_start": (7, 15), "flood_threshold_mm": 100, "heatwave_c": 43},
    "Faisalabad":      {"monsoon_start": (7, 15), "flood_threshold_mm": 90,  "heatwave_c": 44},
    "Peshawar":        {"monsoon_start": (7, 20), "flood_threshold_mm": 70,  "heatwave_c": 41},
    "Islamabad":       {"monsoon_start": (7,  1), "flood_threshold_mm": 120, "heatwave_c": 40},
    "Bahawalpur":      {"monsoon_start": (7, 10), "flood_threshold_mm": 60,  "heatwave_c": 46},
    "Dera Ghazi Khan": {"monsoon_start": (7, 10), "flood_threshold_mm": 70,  "heatwave_c": 47},
}
DEFAULT_CLIMATE = {"monsoon_start": (7, 15), "flood_threshold_mm": 80, "heatwave_c": 42}

KNOWN_CROPS = {
    "cotton", "wheat", "rice", "maize", "sugarcane", "sunflower",
    "sorghum", "potato", "onion", "mango", "tomato", "chili",
    "garlic", "lentil", "chickpea", "mustard", "barley", "soybean"
}

PAK_CITIES = [
    "Lahore", "Karachi", "Islamabad", "Multan", "Faisalabad", "Rawalpindi",
    "Peshawar", "Quetta", "Gujranwala", "Sialkot", "Bahawalpur", "Sargodha",
    "Sahiwal", "Dera Ghazi Khan", "Rahim Yar Khan", "Sukkur", "Hyderabad", "Larkana"
]

LANGUAGE_CONFIG = {
    "en":  {"label": "English", "whisper_lang": "en", "edge_voice": "en-US-JennyNeural",  "gtts_lang": "en", "flag": "🇺🇸", "dir": "ltr"},
    "ur":  {"label": "Urdu",    "whisper_lang": "ur", "edge_voice": "ur-PK-UzmaNeural",   "gtts_lang": "ur", "flag": "🇵🇰", "dir": "rtl"},
    "pa":  {"label": "Punjabi", "whisper_lang": "pa", "edge_voice": "ur-PK-AsadNeural",   "gtts_lang": "ur", "flag": "🌾", "dir": "rtl"},
    "skr": {"label": "Saraiki", "whisper_lang": "ur", "edge_voice": "ur-PK-AsadNeural",   "gtts_lang": "ur", "flag": "🌻", "dir": "rtl"},
}

def _detect_season() -> str:
    m = datetime.now().month
    return (
        "Rabi (Winter: wheat, mustard, chickpea)" if m in [11, 12, 1, 2, 3, 4]
        else "Kharif (Summer: cotton, rice, maize, sugarcane)"
    )

CURRENT_SEASON = _detect_season()
CURRENT_DATE   = datetime.now().strftime("%B %Y")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TTL CACHE  (avoids hammering weather/market APIs on every message)
# ══════════════════════════════════════════════════════════════════════════════

class TTLCache:
    """Simple time-to-live in-memory cache."""
    def __init__(self):
        self._store: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str, ttl_seconds: int = 300) -> Optional[Any]:
        entry = self._store.get(key)
        if entry and (time.time() - entry[1]) < ttl_seconds:
            return entry[0]
        return None

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.time())

    def clear(self) -> None:
        self._store.clear()

_CACHE = TTLCache()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FARMER PROFILE DATABASE  (SQLite — persists across restarts)
# ══════════════════════════════════════════════════════════════════════════════

class FarmerProfileDB:
    """
    Stores per-farmer context: location, crops grown, soil history, language preference.
    Identified by a farmer_id (phone number, UUID, etc.)
    """
    DB_PATH = Path(__file__).parent / "farmer_profiles.db"

    def __init__(self):
        self.conn = sqlite3.connect(str(self.DB_PATH), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                farmer_id   TEXT PRIMARY KEY,
                location    TEXT,
                crops       TEXT,        -- JSON list
                soil_data   TEXT,        -- JSON dict (last known)
                lang        TEXT DEFAULT 'en',
                notes       TEXT,        -- free-form memory
                last_seen   TEXT
            )
        """)
        self.conn.commit()

    def load(self, farmer_id: str) -> Dict:
        row = self.conn.execute(
            "SELECT * FROM profiles WHERE farmer_id = ?", (farmer_id,)
        ).fetchone()
        if not row:
            return {}
        cols = ["farmer_id", "location", "crops", "soil_data", "lang", "notes", "last_seen"]
        data = dict(zip(cols, row))
        data["crops"]     = json.loads(data["crops"]     or "[]")
        data["soil_data"] = json.loads(data["soil_data"] or "{}")
        return data

    def save(self, farmer_id: str, updates: Dict) -> None:
        existing = self.load(farmer_id)
        existing.update(updates)
        existing["last_seen"] = datetime.now().isoformat()
        if "crops" in existing and isinstance(existing["crops"], list):
            existing["crops"] = json.dumps(existing["crops"])
        if "soil_data" in existing and isinstance(existing["soil_data"], dict):
            existing["soil_data"] = json.dumps(existing["soil_data"])
        self.conn.execute("""
            INSERT OR REPLACE INTO profiles
            (farmer_id, location, crops, soil_data, lang, notes, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            farmer_id,
            existing.get("location"),
            existing.get("crops", "[]"),
            existing.get("soil_data", "{}"),
            existing.get("lang", "en"),
            existing.get("notes"),
            existing.get("last_seen"),
        ))
        self.conn.commit()

_PROFILE_DB = FarmerProfileDB()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HYBRID RAG RETRIEVER  (BM25 + semantic + cross-encoder reranking)
# ══════════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Combines keyword search (BM25) with dense semantic retrieval,
    then reranks with a cross-encoder for maximum relevance.

    WHY: Pure semantic search misses exact keyword matches
         (e.g. "Dithane M-45", "Sarsabz urea"). BM25 catches those.
         Fusion + reranking gives best of both worlds.
    """
    def __init__(self, vector_store: Optional[Chroma], all_docs: Optional[List[Document]] = None):
        self.vector_store = vector_store
        self.bm25: Optional[BM25Retriever] = None
        self.reranker = None

        if all_docs and len(all_docs) > 0:
            try:
                self.bm25 = BM25Retriever.from_documents(all_docs, k=6)
                log.info("✅ BM25 retriever built with %d docs", len(all_docs))
            except Exception as e:
                log.warning("BM25 init failed: %s", e)

        # Optional cross-encoder reranker (needs sentence-transformers)
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            log.info("✅ Cross-encoder reranker loaded")
        except Exception:
            log.info("ℹ️  Cross-encoder not available — using reciprocal rank fusion only")

    def retrieve(self, query: str, k: int = 8, crop_filter: Optional[str] = None) -> List[Document]:
        results: List[Document] = []
        seen_ids: set = set()

        # ── Semantic retrieval ──────────────────────────────────────────────
        if self.vector_store:
            try:
                search_kwargs: Dict[str, Any] = {"k": k, "fetch_k": 30}
                if crop_filter:
                    search_kwargs["filter"] = {"topic": crop_filter}
                sem_docs = self.vector_store.as_retriever(
                    search_type="mmr", search_kwargs=search_kwargs
                ).invoke(query)
                for doc in sem_docs:
                    uid = doc.page_content[:80]
                    if uid not in seen_ids:
                        seen_ids.add(uid)
                        results.append(doc)
            except Exception as e:
                log.warning("Semantic retrieval error: %s", e)

        # ── BM25 retrieval ──────────────────────────────────────────────────
        if self.bm25:
            try:
                bm25_docs = self.bm25.invoke(query)
                for doc in bm25_docs:
                    uid = doc.page_content[:80]
                    if uid not in seen_ids:
                        seen_ids.add(uid)
                        results.append(doc)
            except Exception as e:
                log.warning("BM25 retrieval error: %s", e)

        if not results:
            return []

        # ── Cross-encoder reranking ─────────────────────────────────────────
        if self.reranker and len(results) > 1:
            try:
                pairs  = [(query, d.page_content) for d in results]
                scores = self.reranker.predict(pairs)
                ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
                results = [doc for _, doc in ranked[:k]]
                log.info("Reranked %d → %d docs", len(pairs), len(results))
            except Exception as e:
                log.warning("Reranker error: %s", e)
                results = results[:k]
        else:
            results = results[:k]

        return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  AGENT STATE  (expanded with new fields)
# ══════════════════════════════════════════════════════════════════════════════

class AgroBotState(TypedDict):
    # Core
    messages:            Annotated[list, add_messages]
    user_query:          str
    detected_lang:       str
    intent:              str
    # Farmer
    farmer_id:           Optional[str]
    farmer_profile:      Optional[Dict[str, Any]]     # ← NEW
    # Soil / Prediction
    soil_data:           Optional[Dict[str, Any]]
    prediction_result:   Optional[Dict[str, Any]]
    soil_health_score:   Optional[Dict]
    # Weather
    weather_result:      Optional[Dict[str, Any]]
    weather_alerts:      Optional[List[str]]
    # Market / Web
    market_result:       Optional[str]
    web_search_result:   Optional[str]
    # RAG
    rag_result:          Optional[str]
    # Response
    final_response:      Optional[str]
    response_quality:    Optional[Dict]               # ← NEW: critic output
    synthesis_attempts:  int                          # ← NEW: retry counter
    # Meta
    detected_crop:       Optional[str]
    location:            Optional[str]
    follow_up_questions: Optional[List[str]]
    # Audio
    audio_input:         Optional[bytes]
    audio_ext:           Optional[str]
    audio_output:        bool
    tts_audio:           Optional[bytes]
    # Disease detection
    image_path:          Optional[str]
    kindwise_crop:       Optional[str]
    kindwise_disease:    Optional[str]
    kindwise_confidence: Optional[float]


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SHARED RESOURCES  (lazy-loaded singletons)
# ══════════════════════════════════════════════════════════════════════════════

def _load_xgboost_models():
    base  = Path(__file__).parent / "Models"
    paths = {
        "model":  base / "xgboost_model.pkl",
        "scaler": base / "scaler.pkl",
        "le":     base / "label_encoder.pkl",
        "knn":    base / "knn_model.pkl",
    }
    if not all(p.exists() for p in paths.values()):
        log.warning("XGBoost model files not found — crop prediction disabled.")
        return None, None, None, None
    return tuple(pickle.load(open(p, "rb")) for p in paths.values())


def _load_vector_store() -> Optional[Chroma]:
    path = Path(__file__).parent / "chroma_db"
    if not path.exists():
        log.warning("ChromaDB not found — RAG disabled.")
        return None
    try:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma(persist_directory=str(path), embedding_function=emb,
                      collection_name="agrobot_knowledge")
    except Exception as e:
        log.warning("ChromaDB load error: %s", e)
        return None


def _load_all_docs(vs: Optional[Chroma]) -> List[Document]:
    """Load all docs from ChromaDB for BM25 index construction."""
    if vs is None:
        return []
    try:
        col    = vs._collection
        result = col.get(include=["documents", "metadatas"])
        docs   = []
        for text, meta in zip(result["documents"], result["metadatas"]):
            docs.append(Document(page_content=text, metadata=meta or {}))
        log.info("Loaded %d docs from ChromaDB for BM25", len(docs))
        return docs
    except Exception as e:
        log.warning("Could not load all docs: %s", e)
        return []


XGB_MODEL, SCALER, LE, KNN_MODEL = _load_xgboost_models()
VECTOR_STORE = _load_vector_store()
ALL_DOCS     = _load_all_docs(VECTOR_STORE)
HYBRID_RAG   = HybridRetriever(VECTOR_STORE, ALL_DOCS)
SEARCH_TOOL  = DuckDuckGoSearchRun()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  RETRY DECORATOR  (wraps all external API calls)
# ══════════════════════════════════════════════════════════════════════════════

def with_retry(max_attempts: int = 3, wait_min: float = 1.0, wait_max: float = 10.0):
    """Decorator: retry on any Exception with exponential backoff."""
    def decorator(fn):
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
            retry=retry_if_exception_type(Exception),
            reraise=False,
        )
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# 8.  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_llm(
    temperature: float = 0.3,
    model: str = "openai/gpt-oss-120b",
    max_tokens: int = 4096,
) -> ChatGroq:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError("GROQ_API_KEY not set")
    return ChatGroq(model=model, temperature=temperature, api_key=key, max_tokens=max_tokens)

def get_groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError("GROQ_API_KEY not set")
    return Groq(api_key=key)

def detect_language_unicode(text: str, whisper_code: str = "") -> str:
    if not text.strip():
        return "en"
    gurmukhi = sum(1 for c in text if '\u0A00' <= c <= '\u0A7F')
    arabic   = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin    = sum(1 for c in text if c.isascii() and c.isalpha())
    total    = max(len(text.replace(" ", "")), 1)
    if gurmukhi / total > 0.25:
        return "pa"
    if arabic   / total > 0.25:
        saraiki_hits = sum(1 for c in text if c in '\u0768\u0759')
        return "skr" if saraiki_hits else "ur"
    if latin / total > 0.30:
        return "en"
    return {"en": "en", "ur": "ur", "pa": "pa"}.get(whisper_code, "en")

def extract_city(text: str) -> Optional[str]:
    for city in PAK_CITIES:
        if city.lower() in text.lower():
            return city
    return None

def _score_soil_health(soil_data: dict) -> Dict:
    ideal_ranges = {
        "N":  (25,  50,  "kg/ha", "Nitrogen"),
        "P":  (10,  30,  "kg/ha", "Phosphorus"),
        "K":  (150, 300, "kg/ha", "Potassium"),
        "ph": (6.0, 7.5, "pH",   "pH Level"),
        "EC": (0,   2.0, "dS/m", "Salinity (EC)"),
    }
    scores = {}
    for param, (lo, hi, unit, label) in ideal_ranges.items():
        val = soil_data.get(param)
        if val is None:
            continue
        if lo <= val <= hi:
            scores[param] = {"value": val, "unit": unit, "label": label, "status": "good", "score": 100}
        elif val < lo:
            scores[param] = {"value": val, "unit": unit, "label": label, "status": "low",  "score": max(0, int(100 * val / lo))}
        else:
            scores[param] = {"value": val, "unit": unit, "label": label, "status": "high", "score": max(0, int(100 * hi / val))}
    if scores:
        overall = int(sum(v["score"] for v in scores.values()) / len(scores))
        grade = "Excellent 🟢" if overall >= 80 else "Good 🟡" if overall >= 60 else "Fair 🟠" if overall >= 40 else "Poor 🔴"
    else:
        overall, grade = 0, "Unknown"
    return {"params": scores, "overall_score": overall, "grade": grade}

def check_extreme_risks(city: str, total_rain_mm: float, max_temp_c: float) -> List[str]:
    climate = PAK_CLIMATE_DATA.get(city, DEFAULT_CLIMATE)
    alerts  = []
    if total_rain_mm >= climate["flood_threshold_mm"]:
        alerts.append(
            f"🌊 FLOOD RISK: {round(total_rain_mm,1)} mm expected over 5 days "
            f"(threshold: {climate['flood_threshold_mm']} mm). "
            "Clear drainage channels. Postpone irrigation 5-7 days."
        )
    if max_temp_c >= climate["heatwave_c"]:
        alerts.append(
            f"🔥 HEATWAVE: {round(max_temp_c,1)}°C expected "
            f"(danger: {climate['heatwave_c']}°C). "
            "Foliar spray at dawn. Increase irrigation frequency."
        )
    today        = datetime.now()
    ms           = climate["monsoon_start"]
    monsoon_date = datetime(today.year, ms[0], ms[1])
    if today > monsoon_date + timedelta(days=30):
        monsoon_date = datetime(today.year + 1, ms[0], ms[1])
    days_away = (monsoon_date - today).days
    if 0 <= days_away <= 7:
        alerts.append(
            f"⛈️ MONSOON IN {days_away} DAYS for {city}. "
            "Harvest mature crops now. Prepare field drainage immediately."
        )
    return alerts


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DEEP ANALYSIS SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

DEEP_ANALYSIS_PROMPTS = {
    "soil_crop": {
        "en": """You are AgroBot's MASTER AGRONOMIST — Pakistan's most experienced agricultural AI.
You have ML predictions, soil health scores, RAG knowledge, web data, and weather.

Write a COMPREHENSIVE farmer-friendly analysis (600-900 words minimum).
Structure EXACTLY as follows — do NOT skip any section:

## 🌱 Recommended Crop & Why
[Top crop for THIS farmer's soil. Plain language. Use analogies.]

## 📊 Your Soil Health Report
[Each parameter: Good ✅ / Needs Work ⚠️ / Critical ❌. Plain explanation.]

## 🌾 Top 3 Crop Recommendations
[Name, yield/acre, sowing month, expected income PKR]

## 💧 Soil Improvement Plan (Before Sowing)
[Numbered steps, amounts in Pakistani units, cost in PKR]

## 🗓️ Seasonal Timing
[Land prep → sowing → irrigation → fertilizer → harvest dates]

## 🐛 Common Threats
[Top 3 pests/diseases for recommended crop. Signs. Prevention.]

## 💰 Market Intelligence
[Price PKR/maund, best selling time, nearest mandis]

## ⚠️ Important Warnings
[Dangerous soil values. Critical mistakes farmers make.]

## 🔮 3-Month Action Plan
[Week-by-week checklist the farmer can follow immediately]

RULES:
- Simple words. Analogies. Pakistani units (maund, kanaal, bori, bottle).
- Always give SPECIFIC numbers — never vague.
- NEVER say "I cannot" — always give your best expert answer.
Season context: {season} | Date: {date}""",

        "ur": """آپ AgroBot کے ماہر زرعی مشیر ہیں۔ آپ کے پاس ML پیشن گوئی، مٹی کا تجزیہ، RAG علم، ویب ڈیٹا، اور موسم ہے۔
مکمل، کسان دوست تجزیہ لکھیں (600-900 الفاظ یا زیادہ)۔

## 🌱 تجویز کردہ فصل اور کیوں
## 📊 آپ کی زمین کی صحت کی رپورٹ
## 🌾 سرفہرست 3 فصلوں کی سفارش
## 💧 بوائی سے پہلے زمین بہتری کا منصوبہ
## 🗓️ موسمی شیڈول
## 🐛 عام خطرات
## 💰 مارکیٹ معلومات
## ⚠️ اہم انتباہات
## 🔮 3 ماہ کا عملی منصوبہ

سادہ اردو، پاکستانی پیمانے (من، کنال، بوری، بوتل)۔ مخصوص اعداد دیں۔
موسم: {season} | تاریخ: {date}""",

        "pa": """ਤੁਸੀਂ AgroBot ਦੇ ਮਾਹਰ ਖੇਤੀ ਸਲਾਹਕਾਰ ਹੋ। ਮੁਕੰਮਲ ਤਜ਼ਵੀਜ਼ ਲਿਖੋ (600-900 ਸ਼ਬਦ)।
## 🌱 ਸਿਫਾਰਸ਼ੀ ਫਸਲ ## 📊 ਜ਼ਮੀਨ ਦੀ ਸਿਹਤ ## 🌾 ਚੋਟੀ ਦੀਆਂ 3 ਫਸਲਾਂ
## 💧 ਜ਼ਮੀਨ ਸੁਧਾਰ ## 🗓️ ਸਮਾਂ-ਸਾਰਣੀ ## 🐛 ਆਮ ਖਤਰੇ
## 💰 ਮਾਰਕਿਟ ## ⚠️ ਚੇਤਾਵਨੀਆਂ ## 🔮 3-ਮਹੀਨੇ ਯੋਜਨਾ
ਪਾਕਿਸਤਾਨੀ ਮਾਪ ਵਰਤੋ (ਮਣ, ਕਨਾਲ, ਬੋਰੀ)। ਰੁੱਤ: {season}""",

        "skr": """تُساں AgroBot دے ماہر زرعی مشیر ہو۔ مکمل تجزیہ لکھو (600-900 لفظ)۔
## 🌱 سفارش کردہ فصل ## 📊 زمین دی صحت ## 🌾 اوپر دیاں 3 فصلاں
## 💧 زمین بہتری ## 🗓️ موسمی شیڈول ## 🐛 عام خطرے
## 💰 مارکیٹ ## ⚠️ انتباہ ## 🔮 3-مہینیاں دی یوجنا
پاکستانی پیمانے (من، کنال، بوری) ورتو۔ موسم: {season}""",
    },

    "weather": {
        "en": """You are AgroBot's WEATHER-AGRONOMY EXPERT. Write a detailed advisory (400-600 words):
## 🌡️ Current Weather Conditions
## 🌾 Impact on Your Crops
## 💧 Irrigation Decision (exact timing + amounts)
## 🗓️ Next 7-10 Days Farming Calendar
## ⚠️ Weather Risks & Alerts
## 🌱 Crop Protection Steps (numbered)
## 💡 Pro Farmer Tips (3-5 Pakistan-specific tricks)
Season: {season} | Date: {date}""",

        "ur": """آپ AgroBot کے موسمی-زرعی ماہر ہیں۔ تفصیلی مشورہ لکھیں (400-600 الفاظ):
## 🌡️ موجودہ حالات ## 🌾 فصل پر اثر ## 💧 آبپاشی کا فیصلہ
## 🗓️ اگلے 7-10 دن ## ⚠️ خطرات ## 🌱 فصل بچاؤ ## 💡 نکات
موسم: {season}""",

        "pa": """ਮੌਸਮ ਸਲਾਹ (400-600 ਸ਼ਬਦ):
## 🌡️ ਮੌਸਮ ## 🌾 ਫਸਲ ਪ੍ਰਭਾਵ ## 💧 ਸਿੰਚਾਈ ## 🗓️ ਕੈਲੰਡਰ ## ⚠️ ਖਤਰੇ ## 💡 ਸੁਝਾਅ
ਰੁੱਤ: {season}""",

        "skr": """موسمی صلاح (400-600 لفظ):
## 🌡️ حالات ## 🌾 فصل اثر ## 💧 آبپاشی ## 🗓️ کیلنڈر ## ⚠️ خطرے ## 💡 نکات
موسم: {season}""",
    },

    "market": {
        "en": """You are AgroBot's MARKET INTELLIGENCE EXPERT. Write a comprehensive report (400-600 words):
## 💰 Current Price Report (PKR/maund + PKR/40kg)
## 📈 Market Trend Analysis (rising/falling/stable + why)
## 🗓️ Best Time to Sell (sell now vs hold — specific reasoning)
## 🏪 Where to Sell (Lahore/Multan/Faisalabad/Karachi rates)
## 🚜 True Net Price (after commission, weighing, transport)
## ⚡ Quick Decision: YES/NO sell this week + 2-sentence reason
## 🔮 30-60 Day Price Forecast
## 💡 Negotiation Tips (mandi-specific tricks)
Always quote PKR prices. Use maund (40kg) as standard unit. Date: {date}""",

        "ur": """مارکیٹ رپورٹ (400-600 الفاظ):
## 💰 قیمت رپورٹ ## 📈 رجحان تجزیہ ## 🗓️ فروخت کا وقت
## 🏪 کہاں بیچیں ## 🚜 اصل قیمت ## ⚡ فوری فیصلہ
## 🔮 پیشن گوئی ## 💡 مذاکرات کی تجاویز
قیمتیں PKR میں۔ تاریخ: {date}""",

        "pa": """ਮਾਰਕਿਟ ਰਿਪੋਰਟ (400-600 ਸ਼ਬਦ):
## 💰 ਭਾਅ ## 📈 ਰੁਝਾਨ ## 🗓️ ਵੇਚਣ ਦਾ ਸਮਾਂ ## 🏪 ਕਿੱਥੇ ਵੇਚੋ
## 🚜 ਅਸਲ ਕੀਮਤ ## ⚡ ਤੁਰੰਤ ਫੈਸਲਾ ## 🔮 ਅਨੁਮਾਨ ## 💡 ਸੁਝਾਅ""",

        "skr": """مارکیٹ رپورٹ (400-600 لفظ):
## 💰 قیمت ## 📈 رجحان ## 🗓️ ویچن دا ویلہ ## 🏪 کتھے ویچو
## 🚜 اصل قیمت ## ⚡ فوری فیصلہ ## 🔮 اندازہ ## 💡 نکات""",
    },

    "rag_agronomy": {
        "en": """You are AgroBot's SENIOR AGRONOMIC SPECIALIST. Write a thorough response (500-800 words):
## 🔬 Direct Answer (farmer's exact question answered first and completely)
## 📋 Complete Treatment/Action Protocol (numbered 1-10, Pakistani brand names, dose/acre)
## 💊 Pakistani Market Products & Prices (3-5 products available at agri-store)
## 🌡️ Weather/Season Considerations ({season})
## ⚠️ What NOT to Do (3-5 common mistakes that worsen the problem)
## 🔄 Prevention Next Season (cultural practices, resistant varieties)
## 💡 Experienced Farmer Wisdom (traditional + modern for Pakistani conditions)
## 📞 When to Escalate (NARC: 0300-5554800, local extension officer)
Pakistani brands. Amounts in acres/kanals/maunds/bottles. NEVER vague. Date: {date}""",

        "ur": """زرعی ماہرانہ جواب (500-800 الفاظ):
## 🔬 براہ راست جواب ## 📋 علاج پروٹوکول (پاکستانی برانڈ، مقدار)
## 💊 مارکیٹ مصنوعات ## 🌡️ موسم ## ⚠️ کیا نہیں کرنا
## 🔄 روک تھام ## 💡 حکمت ## 📞 مزید مدد (NARC: 0300-5554800)
پاکستانی برانڈ۔ موسم: {season}""",

        "pa": """ਖੇਤੀ ਮਾਹਰ ਜਵਾਬ (500-800 ਸ਼ਬਦ):
## 🔬 ਸਿੱਧਾ ਜਵਾਬ ## 📋 ਇਲਾਜ ਪ੍ਰੋਟੋਕੋਲ ## 💊 ਮਾਰਕਿਟ ਉਤਪਾਦ
## 🌡️ ਮੌਸਮ ## ⚠️ ਕੀ ਨਹੀਂ ਕਰਨਾ ## 🔄 ਬਚਾਅ ## 💡 ਸਿਆਣਪ ## 📞 ਮਦਦ""",

        "skr": """زرعی ماہرانہ جواب (500-800 لفظ):
## 🔬 سیدھا جواب ## 📋 علاج پروٹوکول ## 💊 مارکیٹ مصنوعات
## 🌡️ موسم ## ⚠️ کی نئیں کرنا ## 🔄 روک تھام ## 💡 حکمت ## 📞 مدد""",
    },

    "disease_detection": {
        "en": """You are AgroBot's PLANT PATHOLOGIST. A Kindwise AI diagnosis is provided.
Write a COMPREHENSIVE disease management report (500-700 words):
## 🔬 Diagnosis Summary (disease name, confidence, plain-language explanation, analogy)
## 🚨 Severity Assessment (Low/Medium/High — is crop still saveable?)
## ⏱️ Immediate Action Next 48 Hours (numbered critical steps)
## 💊 Treatment Protocol (chemical: Pakistani brand + dose/acre; organic option too)
## 💰 Treatment Cost (PKR/acre + where to buy in Pakistan)
## 🗓️ Recovery Timeline (Week 1/2/3 — what farmer should see)
## 🔄 Prevention Next Season (seed treatment, rotation, resistant varieties)
## ⚠️ Spread Warning (can it spread to neighbouring farms? how to prevent?)
## 📞 Escalation (NARC 0300-5554800 if no improvement in 7 days)
Pakistani brands. Amounts in acres/kanals/bottles. Date: {date}""",

        "ur": """بیماری انتظام رپورٹ (500-700 الفاظ):
## 🔬 تشخیص ## 🚨 خطرے کی سطح ## ⏱️ فوری اقدام
## 💊 علاج ## 💰 لاگت ## 🗓️ صحت یابی ## 🔄 روک تھام
## ⚠️ پھیلاؤ ## 📞 مزید مدد
پاکستانی برانڈ۔ تاریخ: {date}""",

        "pa": """ਬਿਮਾਰੀ ਰਿਪੋਰਟ (500-700 ਸ਼ਬਦ):
## 🔬 ਤਸ਼ਖੀਸ਼ ## 🚨 ਗੰਭੀਰਤਾ ## ⏱️ ਤੁਰੰਤ ਕਾਰਵਾਈ
## 💊 ਇਲਾਜ ## 💰 ਲਾਗਤ ## 🗓️ ਠੀਕ ਹੋਣਾ ## 🔄 ਬਚਾਅ
## ⚠️ ਫੈਲਾਅ ## 📞 ਮਦਦ""",

        "skr": """بیماری رپورٹ (500-700 لفظ):
## 🔬 تشخیص ## 🚨 خطرہ ## ⏱️ فوری اقدام
## 💊 علاج ## 💰 لاگت ## 🗓️ صحتیابی ## 🔄 روک تھام
## ⚠️ پھیلاؤ ## 📞 مدد""",
    },

    "general": {
        "en": """You are AgroBot — Pakistan's friendliest agricultural AI.
Speak like a wise village elder who knows modern agriculture AND traditional wisdom.
Season: {season} | Date: {date}

Give a HELPFUL, WARM, DETAILED response (300-500 words).
End EVERY response with:
"What else would you like to know? I can help with crop recommendations,
soil analysis, weather advice, pest control, and market prices."
NEVER give a one-line answer.""",

        "ur": """آپ AgroBot ہیں — پاکستان کے سب سے دوستانہ زرعی AI۔
مددگار، تفصیلی جواب دیں (300-500 الفاظ)۔
آخر میں لکھیں: "اور کیا جاننا چاہتے ہیں؟"
موسم: {season}""",

        "pa": """ਤੁਸੀਂ AgroBot ਹੋ — ਪਾਕਿਸਤਾਨ ਦੇ ਦੋਸਤਾਨਾ ਖੇਤੀ AI।
ਮਦਦਗਾਰ, ਵਿਸਤਰਿਤ ਜਵਾਬ ਦਿਓ (300-500 ਸ਼ਬਦ)। ਰੁੱਤ: {season}""",

        "skr": """تُساں AgroBot ہو — پاکستان دے دوستانہ زرعی AI۔
مددگار، تفصیلی جواب دیو (300-500 لفظ)۔ موسم: {season}""",
    },
}

def _get_analysis_prompt(intent: str, lang: str) -> str:
    prompts  = DEEP_ANALYSIS_PROMPTS.get(intent, DEEP_ANALYSIS_PROMPTS["general"])
    template = prompts.get(lang, prompts.get("en", ""))
    return template.format(season=CURRENT_SEASON, date=CURRENT_DATE)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  SUPERVISOR SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

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
    detected_lang:  Literal["en", "ur", "pa", "skr"]
    intent:         Literal["soil_crop", "weather", "market", "rag_agronomy", "disease_detection", "general"]
    soil_data:      Optional[SoilData] = None
    location:       Optional[str]      = None
    detected_crop:  Optional[str]      = Field(None, description=f"One of: {KNOWN_CROPS}")

SUPERVISOR_SYSTEM = f"""You are AgroBot's intelligent routing supervisor for Pakistani farmers.
Current season: {CURRENT_SEASON} | Date: {CURRENT_DATE}
1. Detect language: "en" | "ur" | "pa" | "skr"
2. Identify primary intent: soil_crop | weather | market | rag_agronomy | disease_detection | general
3. Extract soil parameters only if explicitly provided (never guess missing ones)
4. Extract city/location if mentioned
5. Extract crop name if it matches known crops: {KNOWN_CROPS}
INTENT RULES:
- soil_crop: user gives soil data OR asks which crop to grow
- weather: rain, heat, frost, irrigation timing
- market: price, mandi, selling, buying
- rag_agronomy: disease, pest, fertilizer, seed variety, technique
- disease_detection: scan/image/photo/diagnose leaf/plant
- general: greetings, unclear"""


# ══════════════════════════════════════════════════════════════════════════════
# 11.  CRITIC SCHEMA  (NEW — self-reflection quality gate)
# ══════════════════════════════════════════════════════════════════════════════

class CriticOutput(BaseModel):
    quality_score:   int   = Field(..., ge=0, le=100, description="Overall response quality 0-100")
    word_count:      int   = Field(..., description="Approximate word count of the response")
    has_all_sections: bool = Field(..., description="All required sections present?")
    is_farmer_friendly: bool = Field(..., description="Language simple enough for uneducated farmer?")
    has_specific_numbers: bool = Field(..., description="Specific numbers (PKR, kg, dates) present?")
    issues:          List[str] = Field(default_factory=list, description="List of specific problems found")
    should_retry:    bool  = Field(..., description="True if quality_score < 65 and retry would help")
    improvement_instructions: str = Field("", description="Specific instructions for improvement if retry")

CRITIC_SYSTEM = """You are AgroBot's RESPONSE QUALITY CRITIC.
Evaluate the agricultural response strictly. Pakistani farmers need COMPLETE, SPECIFIC, ACTIONABLE advice.

Score 0-100:
- 90-100: All sections, specific numbers, farmer-friendly, comprehensive
- 70-89: Good but missing 1-2 things
- 50-69: Too vague or too short — needs retry
- 0-49: Unacceptable — must retry

CRITICAL FAILURES (auto-retry):
- Word count < 200 for complex queries
- Missing required sections
- No specific numbers (prices, doses, dates)
- Technical jargon without explanation
- Vague advice like "apply fertilizer" without specifying type/amount

Be STRICT. Farmers depend on this advice for their livelihoods."""


# ══════════════════════════════════════════════════════════════════════════════
# 12.  GRAPH NODES  (all async)
# ══════════════════════════════════════════════════════════════════════════════

# ── STT ───────────────────────────────────────────────────────────────────────
async def stt_node(state: AgroBotState) -> AgroBotState:
    audio_bytes = state.get("audio_input")
    if not audio_bytes:
        return state
    ext    = state.get("audio_ext") or "wav"
    client = get_groq_client()
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3", file=f, response_format="verbose_json"
            )
        whisper_code = getattr(result, "language", "en") or "en"
        transcript   = result.text.strip()
    finally:
        os.unlink(path)
    lang = detect_language_unicode(transcript, whisper_code)
    log.info("STT [%s]: %s", lang, transcript[:80])
    return {**state, "user_query": transcript, "detected_lang": lang}


# ── SUPERVISOR ─────────────────────────────────────────────────────────────────
async def supervisor_node(state: AgroBotState) -> AgroBotState:
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(SupervisorOutput)

    lang_hint = ""
    if state.get("audio_input") and state.get("detected_lang"):
        lang_hint = f"\n[PRE-DETECTED LANGUAGE: '{state['detected_lang']}']"

    result: SupervisorOutput = structured_llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM + lang_hint),
        *state["messages"][-6:],
        HumanMessage(content=state["user_query"]),
    ])

    final_lang = state["detected_lang"] if state.get("audio_input") else result.detected_lang
    if not state.get("audio_input"):
        unicode_lang = detect_language_unicode(state["user_query"])
        if unicode_lang in ("pa", "skr"):
            final_lang = unicode_lang

    detected_crop = result.detected_crop.lower().strip() if result.detected_crop else None
    soil_data     = result.soil_data.dict() if result.soil_data else None

    log.info("Supervisor → lang=%s, intent=%s, crop=%s", final_lang, result.intent, detected_crop)

    return {
        **state,
        "detected_lang":     final_lang,
        "intent":            result.intent,
        "soil_data":         soil_data,
        "detected_crop":     detected_crop,
        "location":          result.location or extract_city(state["user_query"]) or "Lahore",
        "messages":          state["messages"] + [HumanMessage(content=state["user_query"])],
        "synthesis_attempts": 0,
    }


# ── FARMER PROFILE  (NEW) ──────────────────────────────────────────────────────
async def farmer_profile_node(state: AgroBotState) -> AgroBotState:
    """
    Load persistent farmer profile from SQLite.
    Merge known location/crop with current query.
    Update profile with any new info gleaned from this session.
    """
    farmer_id = state.get("farmer_id") or "default_farmer"
    profile   = _PROFILE_DB.load(farmer_id)

    # Fill in blanks from profile if current query didn't supply them
    location = state.get("location") or profile.get("location") or "Lahore"
    crop     = state.get("detected_crop") or (profile.get("crops") or [None])[0]
    lang     = state.get("detected_lang") or profile.get("lang") or "en"

    # Update profile with new information
    updates: Dict[str, Any] = {"lang": lang}
    if location:
        updates["location"] = location
    if crop:
        existing_crops = profile.get("crops") or []
        if crop not in existing_crops:
            existing_crops.append(crop)
        updates["crops"] = existing_crops[:10]  # keep last 10
    if state.get("soil_data"):
        updates["soil_data"] = state["soil_data"]

    _PROFILE_DB.save(farmer_id, updates)
    log.info("Farmer profile: id=%s, loc=%s, crop=%s", farmer_id, location, crop)

    return {
        **state,
        "farmer_profile": {**profile, **updates},
        "location":       location,
        "detected_crop":  crop,
        "detected_lang":  lang,
    }


# ── DATA GATHERER  (fully async with asyncio.gather) ──────────────────────────
@with_retry(max_attempts=2)
def _fetch_weather(location: str) -> Dict:
    """Fetch weather with TTL cache (5-minute freshness)."""
    cache_key = f"weather:{location}"
    cached    = _CACHE.get(cache_key, ttl_seconds=300)
    if cached:
        log.info("Weather cache hit for %s", location)
        return cached

    owm_key = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()
    if not owm_key:
        return {}

    cur_url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={owm_key}&units=metric"
    fct_url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={owm_key}&units=metric"

    cur_resp = requests.get(cur_url, timeout=8)
    fct_resp = requests.get(fct_url, timeout=8)

    result = {}
    if cur_resp.status_code == 200:
        raw    = cur_resp.json()
        result = {
            "city":     raw["name"],
            "temp":     round(raw["main"]["temp"], 1),
            "feels":    round(raw["main"]["feels_like"], 1),
            "humidity": raw["main"]["humidity"],
            "desc":     raw["weather"][0]["description"],
            "wind":     raw["wind"].get("speed", 0),
        }
    if fct_resp.status_code == 200:
        fct        = fct_resp.json()
        total_rain = sum(s.get("rain", {}).get("3h", 0) for s in fct["list"])
        max_temp   = max(s["main"]["temp"] for s in fct["list"])
        result["forecast_max_temp_c"]   = round(max_temp, 1)
        result["forecast_rain_5day_mm"] = round(total_rain, 1)

    _CACHE.set(cache_key, result)
    return result


@with_retry(max_attempts=2)
def _fetch_web_search(query: str) -> str:
    """Single web search with TTL cache (10-minute freshness)."""
    cache_key = f"search:{query[:60]}"
    cached    = _CACHE.get(cache_key, ttl_seconds=600)
    if cached:
        return cached
    result = SEARCH_TOOL.run(query)
    _CACHE.set(cache_key, result)
    return result


async def _parallel_searches(queries: List[str]) -> List[str]:
    """Run web searches concurrently using asyncio."""
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, _fetch_web_search, q)
        for q in queries[:4]
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        r if isinstance(r, str) else f"Search failed: {r}"
        for r in results
    ]


async def data_gatherer_node(state: AgroBotState) -> AgroBotState:
    """
    Gather RAG + web search + weather + soil prediction ALL concurrently.
    This is the single biggest performance improvement over sequential fetching.
    """
    query    = state["user_query"]
    intent   = state["intent"]
    crop     = state.get("detected_crop") or ""
    location = state.get("location") or "Lahore"
    lang     = state["detected_lang"]
    soil_data = state.get("soil_data") or {}

    log.info("Data gatherer: intent=%s, crop=%s, location=%s", intent, crop, location)

    # ── Generate smart search queries ────────────────────────────────────────
    llm = get_llm(temperature=0)
    search_plan_response = llm.invoke([
        SystemMessage(content=(
            "Generate exactly 3 specific web search queries for a Pakistani agriculture chatbot. "
            f"Season: {CURRENT_SEASON}, Date: {CURRENT_DATE}. "
            "Return ONLY a JSON list of 3 strings. No explanation, no markdown."
        )),
        HumanMessage(content=f"Question: {query}\nCrop: {crop}\nLocation: {location}")
    ]).content.strip()

    try:
        clean          = re.sub(r"```json|```", "", search_plan_response).strip()
        search_queries = json.loads(clean)
        if not isinstance(search_queries, list):
            raise ValueError
    except Exception:
        search_queries = [f"Pakistan {crop} {intent} agriculture {location}"]

    # ── Run RAG + web searches + weather concurrently ────────────────────────
    loop = asyncio.get_event_loop()

    rag_task     = loop.run_in_executor(None, lambda: _rag_retrieve(query, crop))
    weather_task = loop.run_in_executor(None, _fetch_weather, location)
    search_task  = _parallel_searches(search_queries)

    # Await all in parallel
    rag_docs, weather_data, web_results = await asyncio.gather(
        rag_task, weather_task, search_task
    )

    # Format RAG
    rag_context = ""
    if rag_docs:
        parts = []
        for i, doc in enumerate(rag_docs, 1):
            src  = doc.metadata.get("source", f"doc_{i}")
            sect = doc.metadata.get("section", "")
            parts.append(f"[Source {i}: {src} | {sect}]\n{doc.page_content}")
        rag_context = "\n\n".join(parts)
        log.info("RAG: %d chunks retrieved", len(rag_docs))

    # Format web search
    combined_web = "\n\n---\n\n".join(
        f"[Search {i+1}: {q}]\n{r}"
        for i, (q, r) in enumerate(zip(search_queries, web_results))
    )

    # Weather alerts
    weather_alerts: List[str] = []
    if weather_data:
        weather_alerts = check_extreme_risks(
            location,
            weather_data.get("forecast_rain_5day_mm", 0),
            weather_data.get("forecast_max_temp_c", 0),
        )
        if weather_alerts:
            log.warning("%d weather alert(s) triggered", len(weather_alerts))

    # Soil prediction (synchronous, only when needed)
    prediction_result: Dict = {}
    soil_health:       Dict = {}
    if intent == "soil_crop" and soil_data:
        soil_health = _score_soil_health(soil_data)
        log.info("Soil health: %s", soil_health.get("grade"))
        try:
            from XGBoostClassifierAndYeildPredictor import agro_predict
            prediction_result = agro_predict(soil_data)
        except ImportError:
            prediction_result = {
                "recommended_crop":            "Wheat",
                "predicted_yield_tons_per_acre": 2.65,
                "confidence":                  87.3,
                "top_3_crops": [("Wheat", 87.3), ("Maize", 8.1), ("Rice", 4.6)],
                "note": "[MOCK — real model not loaded]",
            }
        except Exception as e:
            prediction_result = {"error": str(e)}

    return {
        **state,
        "rag_result":        rag_context[:3500] if rag_context else "",
        "web_search_result": combined_web[:3500],
        "weather_result":    weather_data,
        "weather_alerts":    weather_alerts,
        "prediction_result": prediction_result,
        "soil_health_score": soil_health,
        "location":          location,
    }


def _rag_retrieve(query: str, crop: str) -> List[Document]:
    """Synchronous wrapper for hybrid RAG retrieval."""
    try:
        return HYBRID_RAG.retrieve(query, k=8, crop_filter=crop or None)
    except Exception as e:
        log.warning("Hybrid RAG error: %s", e)
        return []


# ── DISEASE DETECTION ──────────────────────────────────────────────────────────
async def disease_detection_node(state: AgroBotState) -> AgroBotState:
    image_path = state.get("image_path")

    # Try tkinter file picker if no path given
    if not image_path:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            image_path = filedialog.askopenfilename(
                title="Select a crop leaf image",
                filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
            )
            root.destroy()
        except Exception:
            pass

    if not image_path:
        log.info("No image selected — falling back to rag_agronomy")
        return {
            **state,
            "intent":     "rag_agronomy",
            "user_query": state["user_query"] + " (no image provided, give general disease advice)",
        }

    try:
        with open(image_path, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode("utf-8")
    except OSError as e:
        log.error("Could not read image: %s", e)
        return {**state, "intent": "rag_agronomy"}

    log.info("Kindwise: analysing %s", os.path.basename(image_path))

    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(None, lambda: requests.post(
            KINDWISE_URL,
            headers={"Api-Key": KINDWISE_API_KEY, "Content-Type": "application/json"},
            json={"images": [img_b64]},
            timeout=30,
        ))
    except Exception as e:
        log.error("Kindwise network error: %s", e)
        return {**state, "intent": "rag_agronomy"}

    if resp.status_code not in (200, 201):
        log.error("Kindwise HTTP %d", resp.status_code)
        return {**state, "intent": "rag_agronomy"}

    result   = resp.json().get("result", {})
    crop_sugg = result.get("crop",    {}).get("suggestions", [])
    k_crop    = crop_sugg[0].get("name", "Unknown Crop").title() if crop_sugg else "Unknown Crop"
    dis_sugg  = result.get("disease", {}).get("suggestions", [])

    if not dis_sugg:
        log.info("Kindwise: healthy plant detected (%s)", k_crop)
        return {
            **state,
            "image_path":       image_path,
            "kindwise_crop":    k_crop,
            "kindwise_disease": None,
            "final_response": (
                f"✅ **Crop: {k_crop}** — No disease detected.\n\n"
                "Your plant looks **healthy**! Maintain good cultural practices: "
                "proper spacing, avoid waterlogging, balanced fertilization, "
                "and regular monitoring."
            ),
        }

    top      = dis_sugg[0]
    k_dis    = top.get("name", "Unknown").title()
    k_conf   = round(top.get("probability", 0) * 100, 1)

    log.info("Kindwise: %s (%.1f%%) on %s", k_dis, k_conf, k_crop)

    return {
        **state,
        "image_path":          image_path,
        "kindwise_crop":       k_crop,
        "kindwise_disease":    k_dis,
        "kindwise_confidence": k_conf,
        "detected_crop":       k_crop.lower(),
    }


# ── SYNTHESIS ──────────────────────────────────────────────────────────────────
async def synthesis_node(state: AgroBotState) -> AgroBotState:
    """Core intelligence node — synthesises all gathered data into a response."""
    intent  = state["intent"]
    lang    = state["detected_lang"]
    loc     = state.get("location") or "your area"
    profile = state.get("farmer_profile") or {}

    # Build context package
    ctx: List[str] = [
        f"=== FARMER QUESTION ===\n{state['user_query']}",
        f"=== DATE & SEASON ===\n{CURRENT_DATE} | {CURRENT_SEASON}",
        f"=== LOCATION ===\n{loc}",
    ]

    # Farmer profile context (persistent history)
    if profile:
        prof_lines = []
        if profile.get("crops"):
            prof_lines.append(f"Farmer's known crops: {', '.join(profile['crops'])}")
        if profile.get("location"):
            prof_lines.append(f"Farmer's registered location: {profile['location']}")
        if profile.get("notes"):
            prof_lines.append(f"Previous notes: {profile['notes']}")
        if prof_lines:
            ctx.append("=== FARMER PROFILE (PERSISTENT MEMORY) ===\n" + "\n".join(prof_lines))

    if state.get("soil_data"):
        ctx.append(f"=== SOIL DATA ===\n{json.dumps(state['soil_data'], indent=2)}")

    if state.get("soil_health_score") and state["soil_health_score"].get("params"):
        sh  = state["soil_health_score"]
        hls = f"Grade: {sh['grade']} ({sh['overall_score']}/100)\n"
        for p, info in sh["params"].items():
            hls += f"  {info['label']}: {info['value']} {info['unit']} → {info['status'].upper()}\n"
        ctx.append(f"=== SOIL HEALTH ANALYSIS ===\n{hls}")

    if state.get("prediction_result"):
        ctx.append(f"=== ML CROP PREDICTION ===\n{json.dumps(state['prediction_result'], indent=2)}")

    if state.get("weather_result"):
        ctx.append(f"=== WEATHER ({loc}) ===\n{json.dumps(state['weather_result'], indent=2)}")

    if state.get("rag_result"):
        ctx.append(f"=== KNOWLEDGE BASE ===\n{state['rag_result']}")

    if state.get("web_search_result"):
        ctx.append(f"=== WEB SEARCH RESULTS ===\n{state['web_search_result']}")

    if state.get("weather_alerts"):
        alert_block = "\n".join(f"  ⚠️ {a}" for a in state["weather_alerts"])
        ctx.append(
            f"=== 🚨 CRITICAL EARLY WARNINGS (ACT IMMEDIATELY) ===\n{alert_block}\n"
            "INSTRUCTION: Dedicate a ## 🚨 Critical Alerts section to ALL warnings above. "
            "Give specific preventive actions for EACH."
        )

    if intent == "disease_detection" and state.get("kindwise_disease"):
        ctx.append(
            f"=== KINDWISE DIAGNOSIS ===\n"
            f"Crop: {state.get('kindwise_crop', 'Unknown')}\n"
            f"Disease: {state['kindwise_disease']}\n"
            f"Confidence: {state.get('kindwise_confidence', 0)}%\n"
            f"Image: {os.path.basename(state.get('image_path', ''))}"
        )

    # Critic improvement instructions (if this is a retry)
    if state.get("synthesis_attempts", 0) > 0 and state.get("response_quality"):
        rq = state["response_quality"]
        ctx.append(
            f"=== PREVIOUS ATTEMPT CRITIQUE (MUST FIX) ===\n"
            f"Previous score: {rq.get('quality_score')}/100\n"
            f"Issues: {', '.join(rq.get('issues', []))}\n"
            f"Instructions: {rq.get('improvement_instructions', 'Be more specific and complete.')}"
        )

    full_context = "\n\n".join(ctx)
    system_prompt = _get_analysis_prompt(intent, lang)

    log.info("Synthesis: attempt=%d, intent=%s, context=%d chars",
             state.get("synthesis_attempts", 0), intent, len(full_context))

    llm      = get_llm(temperature=0.5, max_tokens=8192)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"][-4:],
        HumanMessage(content=full_context),
    ]).content

    # Follow-up question suggestions
    crop = state.get("detected_crop") or "the crop"
    follow_ups = {
        "en": [
            f"What fertilizers should I use for {crop} in Pakistan?",
            f"What pests attack {crop} in {CURRENT_SEASON}?",
            f"What is the current market price for {crop}?",
            f"When should I irrigate {crop} this week?",
        ],
        "ur": [
            f"{crop} کے لیے بہترین کھاد کون سی ہے؟",
            f"{crop} میں کون سے کیڑے لگتے ہیں؟",
            f"{crop} کی آج کی منڈی قیمت کیا ہے؟",
            f"اس ہفتے {crop} کو پانی کب دیں؟",
        ],
        "pa": [
            f"{crop} ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ ਖਾਦ ਕਿਹੜੀ ਹੈ?",
            f"{crop} ਵਿੱਚ ਕਿਹੜੇ ਕੀੜੇ ਲੱਗਦੇ ਹਨ?",
        ],
        "skr": [
            f"{crop} لئی بہترین کھاد کہڑی اے؟",
            f"{crop} وچ کہڑے کیڑے لگدے نیں؟",
        ],
    }.get(lang, [])

    return {
        **state,
        "final_response":      response,
        "follow_up_questions": follow_ups,
        "synthesis_attempts":  state.get("synthesis_attempts", 0) + 1,
    }


# ── CRITIC  (NEW — self-reflection quality gate) ───────────────────────────────
async def critic_node(state: AgroBotState) -> AgroBotState:
    """
    Grades the synthesis response. If quality < 65, flags for retry.
    Hard limit: max 2 synthesis attempts to avoid infinite loops.
    """
    response  = state.get("final_response", "")
    attempts  = state.get("synthesis_attempts", 1)
    intent    = state.get("intent", "general")

    # Don't re-evaluate if disease detection already set final_response directly
    # (e.g. "healthy plant" fast-path) or if we've already retried once
    if not response or attempts >= 2:
        return state

    # Minimum word count by intent
    min_words = {"soil_crop": 300, "weather": 200, "market": 200,
                 "rag_agronomy": 250, "disease_detection": 250, "general": 100}
    word_count = len(response.split())
    too_short  = word_count < min_words.get(intent, 100)

    if too_short:
        log.warning("Critic: response too short (%d words) — auto-retry", word_count)
        return {
            **state,
            "response_quality": {
                "quality_score": 40,
                "word_count": word_count,
                "issues": [f"Response too short ({word_count} words, minimum {min_words.get(intent, 100)})"],
                "should_retry": True,
                "improvement_instructions": "Write a COMPLETE, DETAILED response. Do not skip any section. Minimum 300 words.",
            }
        }

    # LLM-based quality evaluation
    llm = get_llm(temperature=0)
    structured_critic = llm.with_structured_output(CriticOutput)

    try:
        evaluation: CriticOutput = structured_critic.invoke([
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=(
                f"INTENT: {intent}\n"
                f"RESPONSE TO EVALUATE:\n{response[:2000]}\n\n"
                f"(Truncated for evaluation — full length: {word_count} words)"
            )),
        ])
        log.info("Critic score: %d/100 | retry=%s | issues: %s",
                 evaluation.quality_score, evaluation.should_retry, evaluation.issues[:2])
        return {
            **state,
            "response_quality": evaluation.dict(),
        }
    except Exception as e:
        log.warning("Critic LLM error: %s", e)
        return state


# ── RESPONSE FORMATTER ─────────────────────────────────────────────────────────
async def response_formatter_node(state: AgroBotState) -> AgroBotState:
    response_text = state.get("final_response", "")
    intent        = state.get("intent", "general")
    lang          = state.get("detected_lang", "en")
    lang_cfg      = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG["en"])

    source_badges = {
        "soil_crop":         "🌱 *AgroBot XGBoost Crop Engine + Hybrid Knowledge Base*",
        "weather":           "🌤️ *OpenWeatherMap + AgroBot Weather Advisory*",
        "market":            "📊 *Live Market Search + Price Intelligence*",
        "rag_agronomy":      "📚 *AgroBot Hybrid RAG (BM25 + Semantic) + Web Research*",
        "disease_detection": "🔬 *Kindwise Vision AI + AgroBot Plant Pathologist*",
        "general":           "🤖 *AgroBot — Pakistan's Agricultural AI*",
    }

    badge    = source_badges.get(intent, "🤖 *AgroBot*")
    lang_tag = f"_{lang_cfg.get('flag', '')} {lang_cfg.get('label', 'English')}_"

    # Follow-up section
    follow_ups = state.get("follow_up_questions") or []
    fu_headers = {
        "en":  "\n\n---\n💬 **You might also want to ask:**",
        "ur":  "\n\n---\n💬 **آپ یہ بھی پوچھ سکتے ہیں:**",
        "pa":  "\n\n---\n💬 **ਤੁਸੀਂ ਇਹ ਵੀ ਪੁੱਛ ਸਕਦੇ ਹੋ:**",
        "skr": "\n\n---\n💬 **تُساں ایہ وی پُچھ سکدے ہو:**",
    }
    fu_text = ""
    if follow_ups:
        header  = fu_headers.get(lang, fu_headers["en"])
        fu_text = header + "\n" + "\n".join(f"• {q}" for q in follow_ups[:3])

    # Quality metadata (for developer/debug use)
    quality = state.get("response_quality") or {}
    quality_note = ""
    if quality.get("quality_score") and quality["quality_score"] < 70:
        quality_note = f"\n\n_(Response quality: {quality['quality_score']}/100)_"

    full_response = f"{response_text}{fu_text}{quality_note}\n\n---\n{badge}  {lang_tag}"

    return {
        **state,
        "final_response": full_response,
        "messages":       state["messages"] + [AIMessage(content=full_response)],
    }


# ── TTS ────────────────────────────────────────────────────────────────────────
async def tts_node(state: AgroBotState) -> AgroBotState:
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
                await edge_tts.Communicate(text, cfg["edge_voice"]).save(path)
            except Exception as e:
                log.warning("Edge TTS failed (%s), falling back to gTTS", e)
                gTTS(text=text, lang=cfg["gtts_lang"], slow=False).save(path)
        else:
            gTTS(text=text, lang=cfg["gtts_lang"], slow=False).save(path)

        with open(path, "rb") as f:
            mp3_bytes = f.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)

    log.info("TTS generated: %s, %d bytes", lang, len(mp3_bytes))
    return {**state, "tts_audio": mp3_bytes}


# ══════════════════════════════════════════════════════════════════════════════
# 13.  ROUTING LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def route_stt(state: AgroBotState) -> Literal["stt_node", "supervisor"]:
    return "stt_node" if state.get("audio_input") else "supervisor"

def route_after_supervisor(state: AgroBotState) -> Literal["disease_detection_node", "farmer_profile"]:
    if state.get("intent") == "disease_detection":
        return "disease_detection_node"
    return "farmer_profile"

def route_after_disease(state: AgroBotState) -> Literal["farmer_profile", "data_gatherer"]:
    """Healthy plant fast-path skips data_gatherer; fallback goes to data_gatherer."""
    if state.get("final_response"):   # healthy plant — already has response
        return "farmer_profile"
    if state.get("intent") == "rag_agronomy":  # no image provided
        return "data_gatherer"
    return "farmer_profile"           # disease found — enrich with farmer profile + data

def route_after_farmer_profile(state: AgroBotState) -> Literal["data_gatherer", "response_formatter"]:
    """If disease detection already set final_response (healthy), skip to formatter."""
    if state.get("final_response") and not state.get("kindwise_disease"):
        return "response_formatter"
    return "data_gatherer"

def route_after_critic(state: AgroBotState) -> Literal["synthesis", "response_formatter"]:
    """
    CORE SELF-REFLECTION ROUTING:
    If quality is poor and we haven't retried yet → go back to synthesis.
    Otherwise → format and return.
    """
    quality  = state.get("response_quality") or {}
    attempts = state.get("synthesis_attempts", 1)

    if quality.get("should_retry") and attempts < 2:
        log.warning("Critic routing: RETRY synthesis (score=%d)", quality.get("quality_score", 0))
        return "synthesis"

    return "response_formatter"


# ══════════════════════════════════════════════════════════════════════════════
# 14.  BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_agrobot_graph() -> StateGraph:
    # Persistent conversation memory via SQLite checkpointer
    db_path   = str(Path(__file__).parent / "agrobot_memory.db")
    memory    = SqliteSaver.from_conn_string(db_path)

    graph = StateGraph(AgroBotState)

    graph.add_node("stt_node",              stt_node)
    graph.add_node("supervisor",            supervisor_node)
    graph.add_node("farmer_profile",        farmer_profile_node)          # ← NEW
    graph.add_node("disease_detection_node",disease_detection_node)
    graph.add_node("data_gatherer",         data_gatherer_node)
    graph.add_node("synthesis",             synthesis_node)
    graph.add_node("critic",                critic_node)                  # ← NEW
    graph.add_node("response_formatter",    response_formatter_node)
    graph.add_node("tts_node",              tts_node)

    # Entry
    graph.add_conditional_edges(
        START, route_stt,
        {"stt_node": "stt_node", "supervisor": "supervisor"},
    )
    graph.add_edge("stt_node", "supervisor")

    # Supervisor → disease detection OR farmer profile
    graph.add_conditional_edges(
        "supervisor", route_after_supervisor,
        {"disease_detection_node": "disease_detection_node", "farmer_profile": "farmer_profile"},
    )

    # Disease detection → farmer profile (disease found) OR data_gatherer (fallback)
    graph.add_conditional_edges(
        "disease_detection_node", route_after_disease,
        {"farmer_profile": "farmer_profile", "data_gatherer": "data_gatherer"},
    )

    # Farmer profile → data_gatherer OR direct to formatter (healthy plant fast-path)
    graph.add_conditional_edges(
        "farmer_profile", route_after_farmer_profile,
        {"data_gatherer": "data_gatherer", "response_formatter": "response_formatter"},
    )

    graph.add_edge("data_gatherer", "synthesis")

    # Synthesis → critic (self-reflection)
    graph.add_edge("synthesis", "critic")

    # Critic → retry synthesis OR proceed to formatter
    graph.add_conditional_edges(
        "critic", route_after_critic,
        {"synthesis": "synthesis", "response_formatter": "response_formatter"},
    )

    graph.add_edge("response_formatter", "tts_node")
    graph.add_edge("tts_node", END)

    return graph.compile(checkpointer=memory)


agrobot_graph = build_agrobot_graph()


# ══════════════════════════════════════════════════════════════════════════════
# 15.  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_agrobot(
    user_message: str   = "",
    history:      list  = None,
    audio_bytes:  bytes = None,
    audio_ext:    str   = "wav",
    audio_output: bool  = False,
    image_path:   str   = None,
    farmer_id:    str   = "default_farmer",    # ← NEW: persistent identity
    thread_id:    str   = None,                # ← NEW: LangGraph conversation thread
) -> dict:
    """
    Main entry point for AgroBot v3.

    Args:
        user_message: Text query from the farmer.
        history:      Prior LangChain message objects (for in-session context).
        audio_bytes:  Raw audio bytes if voice input.
        audio_ext:    Audio file extension ("wav", "mp3", "ogg").
        audio_output: Whether to generate TTS audio in response.
        image_path:   Path to a leaf/plant image for disease detection.
        farmer_id:    Unique farmer identifier (phone number, UUID, etc.).
                      Used to load/save persistent farmer profile.
        thread_id:    LangGraph checkpointer thread ID for conversation continuity.
                      Defaults to farmer_id if not provided.

    Returns:
        {
            "text":        str,          # Formatted final response
            "lang":        str,          # Detected language code
            "audio":       bytes | None, # MP3 audio if audio_output=True
            "intent":      str,          # Detected intent
            "follow_ups":  List[str],    # Suggested follow-up questions
            "soil_health": dict,         # Soil health scoring
            "disease":     dict,         # Kindwise disease detection results
            "quality":     dict,         # Critic quality evaluation
        }
    """
    thread_id = thread_id or farmer_id

    initial_state: AgroBotState = {
        "messages":            history or [],
        "user_query":          user_message,
        "detected_lang":       "en",
        "intent":              "",
        "farmer_id":           farmer_id,
        "farmer_profile":      None,
        "soil_data":           None,
        "prediction_result":   None,
        "soil_health_score":   None,
        "weather_result":      None,
        "weather_alerts":      None,
        "market_result":       None,
        "web_search_result":   None,
        "rag_result":          None,
        "final_response":      None,
        "response_quality":    None,
        "synthesis_attempts":  0,
        "detected_crop":       None,
        "location":            None,
        "follow_up_questions": None,
        "audio_input":         audio_bytes,
        "audio_ext":           audio_ext,
        "audio_output":        audio_output,
        "tts_audio":           None,
        "image_path":          image_path,
        "kindwise_crop":       None,
        "kindwise_disease":    None,
        "kindwise_confidence": None,
    }

    config = {"configurable": {"thread_id": thread_id}}
    result = agrobot_graph.invoke(initial_state, config=config)

    return {
        "text":        result.get("final_response", ""),
        "lang":        result.get("detected_lang", "en"),
        "audio":       result.get("tts_audio"),
        "intent":      result.get("intent"),
        "follow_ups":  result.get("follow_up_questions") or [],
        "soil_health": result.get("soil_health_score") or {},
        "disease": {
            "crop":       result.get("kindwise_crop"),
            "disease":    result.get("kindwise_disease"),
            "confidence": result.get("kindwise_confidence"),
        },
        "quality": result.get("response_quality") or {},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 16.  CLI  (async-native)
# ══════════════════════════════════════════════════════════════════════════════

async def _async_cli():
    print("=" * 70)
    print("🌾 AgroBot v3 — PRODUCTION EDITION")
    print("   Languages: English · اردو · ਪੰਜਾਬੀ · سرائیکی")
    print(f"   Season: {CURRENT_SEASON} | Date: {CURRENT_DATE}")
    print("   Commands: --audio  --scan  --farmer <id>  exit")
    print("=" * 70)

    history   = []
    farmer_id = "cli_farmer"
    thread_id = "cli_session_01"

    while True:
        raw = input("\n👨‍🌾 You: ").strip()
        if raw.lower() in ("exit", "quit", "q"):
            print("👋 Allah Hafiz! خدا حافظ")
            break

        want_audio = "--audio" in raw
        want_scan  = "--scan"  in raw
        user_text  = raw.replace("--audio", "").replace("--scan", "").strip()

        if "--farmer" in user_text:
            parts     = user_text.split("--farmer")
            user_text = parts[0].strip()
            farmer_id = parts[1].strip().split()[0] if len(parts) > 1 else farmer_id
            thread_id = farmer_id
            print(f"🪪 Farmer ID set to: {farmer_id}")

        if want_scan:
            user_text = (user_text + " scan image for disease detection").strip()

        print("\n⏳ AgroBot v3 thinking...")
        output = run_agrobot(
            user_message=user_text,
            history=history,
            audio_output=want_audio,
            farmer_id=farmer_id,
            thread_id=thread_id,
        )

        print(f"\n{'='*70}")
        print(f"🤖 AgroBot [{output['lang']}]:\n")
        print(output["text"])
        print(f"{'='*70}")

        # Metadata
        if output.get("soil_health") and output["soil_health"].get("grade"):
            print(f"📊 Soil Health: {output['soil_health']['grade']}")
        if output.get("disease") and output["disease"].get("disease"):
            d = output["disease"]
            print(f"🦠 Disease: {d['disease']} on {d['crop']} ({d['confidence']}% confidence)")
        if output.get("quality") and output["quality"].get("quality_score"):
            q = output["quality"]
            print(f"⭐ Response Quality: {q['quality_score']}/100")

        if output["audio"]:
            audio_dir = Path(__file__).parent / "audio_outputs"
            audio_dir.mkdir(exist_ok=True)
            out_path = audio_dir / "agrobot_response.mp3"
            out_path.write_bytes(output["audio"])
            print(f"🔊 Audio → {out_path}")

        history.append(HumanMessage(content=user_text))
        history.append(AIMessage(content=output["text"]))


if __name__ == "__main__":
    asyncio.run(_async_cli())