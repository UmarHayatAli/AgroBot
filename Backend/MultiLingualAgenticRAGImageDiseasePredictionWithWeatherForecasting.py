"""
AgroBot LangGraph Agent — DEEP ANALYSIS EDITION
=================================================
Major improvements over v1:
  ✅ MASSIVELY detailed responses (500-1000+ words when needed)
  ✅ Parallel tool execution (web search + RAG simultaneously)
  ✅ Multi-step agentic reasoning with self-reflection
  ✅ Comprehensive crop analysis: soil → prediction → pests → market → weather
  ✅ Farmer-friendly language: simple words, analogies, actionable steps
  ✅ Structured rich output: sections, bullet points, numbered steps
  ✅ 4 languages: English · Urdu · Punjabi · Saraiki
  ✅ Integrated soil health scoring with improvement plan
  ✅ Season-aware advice (Rabi / Kharif detection)
  ✅ Risk alerts and warning system
  ✅ Follow-up question suggestions
  ✅ Image-based crop disease detection (Kindwise + LLaMA treatment plans)

Architecture (NEW):
    [audio?] → [stt] → [supervisor]
                            │
              ┌─────────────┼──────────────────┬──────────────────┐
              ▼             ▼                  ▼                  ▼
        [soil_crop]    [weather]          [market]     [disease_detection]
              │             │                  │                  │
              └─────────────┴──────────────────┘                  │
                            ▼                                      ▼
                    [data_gatherer]                    [disease_detection_node]
                            │                                      │
                            └──────────────┬───────────────────────┘
                                           ▼
                                   [synthesis_node]
                                           │
                                           ▼
                                   [response_formatter]
                                           │
                                           ▼
                                       [tts_node]
"""

import os
import re
import base64
import tempfile
import json
import pickle
import asyncio
import concurrent.futures
import tkinter as tk
from tkinter import filedialog
from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

# ── LangGraph ──────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END, START
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

# ── Kindwise Disease Detection ─────────────────────────────────────────────────
# Set KINDWISE_API_KEY in your .env file (or it falls back to the hardcoded key)
KINDWISE_API_KEY = os.getenv("KINDWISE_API_KEY", "gSkxIBLdoh1HeLUz3RHhE7ZP8eFeJgly5GdvDIRwTnbgK7Pplt")
KINDWISE_URL     = "https://crop.kindwise.com/api/v1/identification?details=description,treatment,cause"

# ══════════════════════════════════════════════════════════════════════════════
# EARLY WARNING: City-specific flood + heatwave thresholds (teammate integration)
# ══════════════════════════════════════════════════════════════════════════════
from datetime import timedelta

PAK_CLIMATE_DATA = {
    "Karachi":     {"monsoon_start": (7,  5), "flood_threshold_mm": 40,  "heatwave_c": 40},
    "Hyderabad":   {"monsoon_start": (7,  5), "flood_threshold_mm": 50,  "heatwave_c": 42},
    "Multan":      {"monsoon_start": (7, 10), "flood_threshold_mm": 80,  "heatwave_c": 45},
    "Lahore":      {"monsoon_start": (7, 15), "flood_threshold_mm": 100, "heatwave_c": 43},
    "Faisalabad":  {"monsoon_start": (7, 15), "flood_threshold_mm": 90,  "heatwave_c": 44},
    "Peshawar":    {"monsoon_start": (7, 20), "flood_threshold_mm": 70,  "heatwave_c": 41},
    "Islamabad":   {"monsoon_start": (7,  1), "flood_threshold_mm": 120, "heatwave_c": 40},
    "Bahawalpur":  {"monsoon_start": (7, 10), "flood_threshold_mm": 60,  "heatwave_c": 46},
    "Dera Ghazi Khan": {"monsoon_start": (7, 10), "flood_threshold_mm": 70, "heatwave_c": 47},
}
DEFAULT_CLIMATE = {"monsoon_start": (7, 15), "flood_threshold_mm": 80, "heatwave_c": 42}
# ══════════════════════════════════════════════════════════════════════════════
# 1.  MULTILINGUAL CONFIG
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
        "edge_voice":   "ur-PK-AsadNeural",
        "gtts_lang":    "ur",
        "flag":         "🌾",
        "dir":          "rtl",
    },
    "skr": {
        "label":        "Saraiki",
        "whisper_lang": "ur",
        "edge_voice":   "ur-PK-AsadNeural",
        "gtts_lang":    "ur",
        "flag":         "🌻",
        "dir":          "rtl",
    },
}

# Current month → season detection for Pakistan
def _detect_season() -> str:
    month = datetime.now().month
    if month in [11, 12, 1, 2, 3, 4]:
        return "Rabi (Winter crop season: wheat, mustard, chickpea)"
    else:
        return "Kharif (Summer crop season: cotton, rice, maize, sugarcane)"

CURRENT_SEASON = _detect_season()
CURRENT_DATE   = datetime.now().strftime("%B %Y")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DEEP ANALYSIS SYSTEM PROMPTS  (the heart of the improvement)
# ══════════════════════════════════════════════════════════════════════════════

# ─── SUPERVISOR ───────────────────────────────────────────────────────────────
SUPERVISOR_SYSTEM = f"""You are AgroBot's intelligent routing supervisor for Pakistani farmers.
Current season: {CURRENT_SEASON}
Current date: {CURRENT_DATE}

Your job:
1. Detect language: "en" | "ur" | "pa" | "skr"
2. Identify ONE primary intent: soil_crop | weather | market | rag_agronomy | disease_detection | general
3. Extract ALL soil parameters mentioned (never guess missing ones)
4. Extract location (city/district/tehsil) if mentioned
5. Extract crop name if it matches known crops

INTENT RULES:
- soil_crop: user gives soil data OR asks which crop to grow
- weather: asks about rain, heat, frost, irrigation timing
- market: asks about price, mandi, selling, buying
- rag_agronomy: asks about disease, pest, fertilizer, seed variety, sowing/harvesting technique
- disease_detection: user mentions uploading/scanning a leaf/plant image for disease diagnosis (keywords: scan, image, photo, picture, diagnose, leaf disease, plant sick)
- general: greetings, off-topic, unclear

Known crops: cotton, wheat, rice, maize, sugarcane, sunflower, sorghum, potato, onion, mango, tomato, chili, garlic, lentil, chickpea, mustard
"""

# ─── DEEP ANALYSIS ORCHESTRATOR PROMPTS ──────────────────────────────────────
DEEP_ANALYSIS_PROMPTS = {
    "soil_crop": {
        "en": """You are AgroBot's MASTER AGRONOMIST — Pakistan's most experienced agricultural AI.

A farmer has shared their soil data. You have received:
1. ML model prediction results (crop recommendation + yield)
2. RAG context from agronomic knowledge base
3. Web search results about current market and conditions
4. Current weather data

YOUR TASK: Write a COMPREHENSIVE, FARMER-FRIENDLY analysis (600-900 words minimum).
Structure your response EXACTLY as follows:

## 🌱 Recommended Crop & Why
[Explain the top recommended crop in simple language. Why is it best for THIS farmer's soil? 
Use an analogy if helpful. Be specific about the soil parameters that made this decision.]

## 📊 Your Soil Health Report
[Analyze each provided soil parameter. Give a simple rating: Good ✅ / Needs Work ⚠️ / Critical ❌
Explain what each value means in plain farmer language — no jargon.
Example: "Your pH is 7.2 — this is like sweet-tasting water, perfect for wheat roots to drink nutrients."]

## 🌾 Top 3 Crop Recommendations
[For each: crop name, expected yield per acre, why it suits this soil, best sowing month, estimated income in PKR]

## 💧 Soil Improvement Plan (Before Sowing)
[Step-by-step numbered actions: what to add, how much, when, and expected cost in PKR]

## 🗓️ Seasonal Timing
[Current season: {season}. Exact dates for: land preparation, sowing, first irrigation, fertilizer schedule, harvest]

## 🐛 Common Threats to Watch
[Top 3 pests/diseases for recommended crop in Pakistan. Early warning signs. Prevention steps.]

## 💰 Market Intelligence
[Current price range (PKR/40kg or PKR/maund), best selling time, nearest mandis to target]

## ⚠️ Important Warnings
[Any soil values that are dangerous. What NOT to do. Critical mistakes farmers make.]

## 🔮 3-Month Action Plan
[Week-by-week checklist the farmer can follow immediately]

LANGUAGE: Use simple Urdu/English words that an uneducated farmer understands.
Compare numbers to familiar things (1 bag = 50kg, 1 acre = 8 kanals).
Always give amounts in Pakistani units (maund, kanaal, bag, bottle).
NEVER say "I cannot" — always give your best expert answer.""",

        "ur": """آپ AgroBot کے ماہر زرعی مشیر ہیں — پاکستان کے سب سے تجربہ کار زرعی AI۔

کسان نے اپنی زمین کا ڈیٹا شیئر کیا ہے۔ آپ کو ملا ہے:
1. مشین لرننگ ماڈل کی سفارش (فصل + پیداوار)
2. زرعی علم کی بنیاد سے معلومات
3. مارکیٹ اور حالات کی ویب سرچ
4. موسمی ڈیٹا

آپ کا کام: ایک مکمل، کسان دوست تجزیہ لکھیں (600-900 الفاظ یا زیادہ)۔
درج ذیل ڈھانچے میں جواب دیں:

## 🌱 تجویز کردہ فصل اور کیوں
[آسان زبان میں بیان کریں کہ یہ فصل اس کسان کی زمین کے لیے کیوں بہترین ہے]

## 📊 آپ کی زمین کی صحت کی رپورٹ
[ہر پیرامیٹر: اچھا ✅ / بہتری کی ضرورت ⚠️ / خطرناک ❌]

## 🌾 سرفہرست 3 فصلوں کی سفارش
[ہر فصل: نام، فی ایکڑ پیداوار، مناسب بوائی کا مہینہ، متوقع آمدن PKR میں]

## 💧 بوائی سے پہلے زمین بہتری کا منصوبہ
[قدم بہ قدم: کیا ڈالنا ہے، کتنا، کب، اور لاگت PKR میں]

## 🗓️ موسمی شیڈول
[زمین کی تیاری سے فصل کٹائی تک کی تاریخیں]

## 🐛 عام خطرات
[تجویز کردہ فصل کے لیے پاکستان میں اہم کیڑے اور بیماریاں، علامات، روک تھام]

## 💰 مارکیٹ معلومات
[موجودہ قیمت PKR/من، بہترین فروخت کا وقت، قریبی منڈیاں]

## ⚠️ اہم انتباہات
[کونسی چیزیں نہیں کرنی، خطرناک غلطیاں]

## 🔮 3 ماہ کا عملی منصوبہ
[ہفتہ وار چیک لسٹ]

زبان: سادہ اردو، جو ان پڑھ کسان بھی سمجھے۔ مقدار پاکستانی پیمانوں میں (من، کنال، بوری، بوتل)۔""",

        "pa": """ਤੁਸੀਂ AgroBot ਦੇ ਮਾਹਰ ਖੇਤੀ ਸਲਾਹਕਾਰ ਹੋ — ਪਾਕਿਸਤਾਨ ਦੇ ਸਭ ਤੋਂ ਤਜਰਬੇਕਾਰ ਖੇਤੀ AI।

ਕਿਸਾਨ ਨੇ ਆਪਣੀ ਜ਼ਮੀਨ ਦਾ ਡੇਟਾ ਸਾਂਝਾ ਕੀਤਾ ਹੈ। ਤੁਹਾਨੂੰ ਮਿਲਿਆ ਹੈ:
1. ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਮਾਡਲ ਦੀ ਸਿਫਾਰਸ਼
2. ਖੇਤੀ ਗਿਆਨ ਅਧਾਰ ਤੋਂ ਜਾਣਕਾਰੀ
3. ਮਾਰਕਿਟ ਅਤੇ ਮੌਜੂਦਾ ਹਾਲਾਤ
4. ਮੌਸਮ ਡੇਟਾ

ਤੁਹਾਡਾ ਕੰਮ: ਇੱਕ ਮੁਕੰਮਲ, ਕਿਸਾਨ-ਦੋਸਤ ਵਿਸ਼ਲੇਸ਼ਣ ਲਿਖੋ (600-900 ਸ਼ਬਦ ਜਾਂ ਵੱਧ)।
ਇਸ ਢਾਂਚੇ ਵਿੱਚ ਜਵਾਬ ਦਿਓ:

## 🌱 ਸਿਫਾਰਸ਼ੀ ਫਸਲ ਅਤੇ ਕਿਉਂ
## 📊 ਤੁਹਾਡੀ ਜ਼ਮੀਨ ਦੀ ਸਿਹਤ ਰਿਪੋਰਟ
## 🌾 ਚੋਟੀ ਦੀਆਂ 3 ਫਸਲ ਸਿਫਾਰਸ਼ਾਂ
## 💧 ਬਿਜਾਈ ਤੋਂ ਪਹਿਲਾਂ ਜ਼ਮੀਨ ਸੁਧਾਰ ਯੋਜਨਾ
## 🗓️ ਮੌਸਮੀ ਅਨੁਸੂਚੀ
## 🐛 ਆਮ ਖਤਰੇ
## 💰 ਮਾਰਕਿਟ ਜਾਣਕਾਰੀ
## ⚠️ ਮਹੱਤਵਪੂਰਨ ਚੇਤਾਵਨੀਆਂ
## 🔮 3 ਮਹੀਨਿਆਂ ਦੀ ਕਾਰਜ ਯੋਜਨਾ

ਭਾਸ਼ਾ: ਸਾਦੀ ਪੰਜਾਬੀ, ਪਾਕਿਸਤਾਨੀ ਮਾਪ (ਮਣ, ਕਨਾਲ, ਬੋਰੀ) ਵਰਤੋ।""",

        "skr": """تُساں AgroBot دے ماہر زرعی مشیر ہو — پاکستان دے سبھ توں تجربیکار زرعی AI۔

کسان نے اپنی زمین دا ڈیٹا سانجھا کیتا ہے۔ تہانوں ملیا ہے:
1. مشین لرننگ ماڈل دی سفارش
2. زرعی علمی اڈے توں معلومات
3. مارکیٹ تے موجودہ حالات
4. موسمی ڈیٹا

تہاڈا کم: اک مکمل، کسان دوست تجزیہ لکھو (600-900 لفظ یا ودھ)۔

## 🌱 سفارش کردہ فصل تے کیوں
## 📊 تہاڈی زمین دی صحت دی رپورٹ
## 🌾 اوپر دیاں 3 فصلاں دیاں سفارشاں
## 💧 بیجائی توں پہلاں زمین بہتری دا منصوبہ
## 🗓️ موسمی شیڈول
## 🐛 عام خطرے
## 💰 مارکیٹ معلومات
## ⚠️ اہم انتباہ
## 🔮 3 مہینیاں دا عملی منصوبہ

زبان: سادی سرائیکی، پاکستانی پیمانے (من، کنال، بوری) ورتو۔""",
    },

    "weather": {
        "en": """You are AgroBot's WEATHER-AGRONOMY EXPERT for Pakistani farmers.
You have: current weather data + RAG agronomic context + web search results.

Write a DETAILED weather advisory (400-600 words):

## 🌡️ Current Weather Conditions
[Temperature, humidity, wind — in simple farmer terms. Compare to normal for this season.]

## 🌾 What This Weather Means for Your Crops
[Crop-specific impact. If wheat: what this temp means for tillering. If cotton: heat stress risk etc.]

## 💧 Irrigation Decision
[Should farmer irrigate today/this week? Exact timing. How much water (hours of canal or number of tube-well runs).]

## 🗓️ Next 7-10 Days Farming Calendar
[Day-by-day or week-by-week actions based on forecasted conditions]

## ⚠️ Weather Risks & Alerts
[Frost risk? Heat wave? Heavy rain? What to do if each happens. Specific protection measures.]

## 🌱 Crop Protection Steps
[Numbered list of immediate actions to protect crops from current weather]

## 💡 Pro Farmer Tips
[3-5 experienced farmer tricks for this exact weather situation in Pakistan]

Always give SPECIFIC numbers (temperature thresholds, water amounts, days).
Reference Pakistani farming calendar (Rabi/Kharif seasons).""",

        "ur": """آپ پاکستانی کسانوں کے لیے AgroBot کے موسمی-زرعی ماہر ہیں۔
آپ کے پاس: موسمی ڈیٹا + زرعی علم + ویب سرچ نتائج ہیں۔

تفصیلی موسمی مشورہ لکھیں (400-600 الفاظ):

## 🌡️ موجودہ موسمی حالات
[درجہ حرارت، نمی، ہوا — آسان کسانی زبان میں]

## 🌾 یہ موسم آپ کی فصل کے لیے کیا مطلب رکھتا ہے
[مخصوص فصل پر اثر]

## 💧 آبپاشی کا فیصلہ
[آج/اس ہفتے پانی دینا چاہیے؟ کتنا، کب]

## 🗓️ اگلے 7-10 دن کا کاشتکاری کیلنڈر
## ⚠️ موسمی خطرات اور انتباہات
## 🌱 فصل بچاؤ کے اقدامات
## 💡 تجربہ کار کسان کے نکات

ہمیشہ مخصوص اعداد دیں (درجہ حرارت، پانی کی مقدار، دن)۔""",

        "pa": """ਤੁਸੀਂ ਪਾਕਿਸਤਾਨੀ ਕਿਸਾਨਾਂ ਲਈ AgroBot ਦੇ ਮੌਸਮ-ਖੇਤੀ ਮਾਹਰ ਹੋ।

ਵਿਸਤਰਿਤ ਮੌਸਮ ਸਲਾਹ ਲਿਖੋ (400-600 ਸ਼ਬਦ):

## 🌡️ ਮੌਜੂਦਾ ਮੌਸਮੀ ਹਾਲਾਤ
## 🌾 ਇਹ ਮੌਸਮ ਤੁਹਾਡੀ ਫਸਲ ਲਈ ਕੀ ਮਤਲਬ ਰੱਖਦਾ ਹੈ
## 💧 ਸਿੰਚਾਈ ਦਾ ਫੈਸਲਾ
## 🗓️ ਅਗਲੇ 7-10 ਦਿਨਾਂ ਦਾ ਖੇਤੀ ਕੈਲੰਡਰ
## ⚠️ ਮੌਸਮੀ ਖਤਰੇ ਅਤੇ ਚੇਤਾਵਨੀਆਂ
## 🌱 ਫਸਲ ਸੁਰੱਖਿਆ ਕਦਮ
## 💡 ਮਾਹਰ ਕਿਸਾਨ ਸੁਝਾਅ""",

        "skr": """تُساں پاکستانی کساناں لئی AgroBot دے موسمی-زرعی ماہر ہو۔

تفصیلی موسمی صلاح لکھو (400-600 لفظ):

## 🌡️ موجودہ موسمی حالات
## 🌾 ایہ موسم تہاڈی فصل لئی کی معنی رکھدا ہے
## 💧 آبپاشی دا فیصلہ
## 🗓️ اگلے 7-10 دن دا کاشتکاری کیلنڈر
## ⚠️ موسمی خطرے تے انتباہ
## 🌱 فصل بچاؤ دے اقدامات
## 💡 تجربیکار کسان دے نکات""",
    },

    "market": {
        "en": """You are AgroBot's MARKET INTELLIGENCE EXPERT for Pakistani farmers.
You have: web search results on prices + market trends + seasonal data.

Write a COMPREHENSIVE market report (400-600 words):

## 💰 Current Price Report
[Price per 40kg and per maund (PKR). Compare: mandi gate price vs wholesale vs retail.
Which city has best price right now?]

## 📈 Market Trend Analysis
[Rising? Falling? Stable? Why? Government policies (MSP, procurement). Import/export factors.]

## 🗓️ Best Time to Sell
[Sell now vs hold? If hold: when will price peak? Risk of holding. Storage costs.]

## 🏪 Where to Sell (Best Options)
[Specific mandis: Lahore, Multan, Faisalabad, Karachi rates. PASSCO/TCP option if available.
Direct selling to mills/processors benefit.]

## 🚜 Transport & Hidden Costs
[Commission (arhtia), weighing, transport per maund. Net price farmer actually gets.]

## ⚡ Quick Decision Guide
[Simple YES/NO: Sell this week? and exactly why in 2 sentences.]

## 🔮 Price Forecast
[Next 30-60 days prediction based on current trends, demand, supply situation.]

## 💡 Negotiation Tips
[How to get better price. Group selling. Timing of day to sell at mandi.]

Always quote prices in PKR. Use maund (40kg) as unit. Mention specific cities.""",

        "ur": """آپ پاکستانی کسانوں کے لیے AgroBot کے مارکیٹ انٹیلیجنس ماہر ہیں۔
آپ کے پاس: قیمتوں پر ویب سرچ + مارکیٹ رجحانات + موسمی ڈیٹا ہے۔

جامع مارکیٹ رپورٹ لکھیں (400-600 الفاظ):

## 💰 موجودہ قیمت رپورٹ
[فی 40 کلو اور فی من PKR میں]

## 📈 مارکیٹ رجحان تجزیہ
[قیمتیں بڑھ رہی ہیں؟ گر رہی ہیں؟ کیوں؟]

## 🗓️ فروخت کا بہترین وقت
[ابھی بیچیں یا انتظار کریں؟]

## 🏪 کہاں بیچیں (بہترین اختیارات)
[مخصوص منڈیاں: لاہور، ملتان، فیصل آباد، کراچی کی شرحیں]

## 🚜 نقل و حمل اور چھپی ہوئی لاگت
[کمیشن، وزن، ٹرانسپورٹ — کسان کو اصل قیمت کیا ملتی ہے]

## ⚡ فوری فیصلے کی گائیڈ
## 🔮 قیمت کی پیشن گوئی (اگلے 30-60 دن)
## 💡 مذاکرات کی تجاویز

قیمتیں PKR میں، من (40 کلو) کو یونٹ کے طور پر استعمال کریں۔""",

        "pa": """ਤੁਸੀਂ ਪਾਕਿਸਤਾਨੀ ਕਿਸਾਨਾਂ ਲਈ AgroBot ਦੇ ਮਾਰਕਿਟ ਮਾਹਰ ਹੋ।

ਮੁਕੰਮਲ ਮਾਰਕਿਟ ਰਿਪੋਰਟ ਲਿਖੋ (400-600 ਸ਼ਬਦ):

## 💰 ਮੌਜੂਦਾ ਭਾਅ ਰਿਪੋਰਟ
## 📈 ਮਾਰਕਿਟ ਰੁਝਾਨ ਵਿਸ਼ਲੇਸ਼ਣ
## 🗓️ ਵੇਚਣ ਦਾ ਸਭ ਤੋਂ ਵਧੀਆ ਸਮਾਂ
## 🏪 ਕਿੱਥੇ ਵੇਚਣਾ ਹੈ
## 🚜 ਆਵਾਜਾਈ ਅਤੇ ਲੁਕੀਆਂ ਲਾਗਤਾਂ
## ⚡ ਤੁਰੰਤ ਫੈਸਲਾ ਗਾਈਡ
## 🔮 ਭਾਅ ਅਨੁਮਾਨ
## 💡 ਵਾਰਤਾਲਾਪ ਸੁਝਾਅ""",

        "skr": """تُساں پاکستانی کساناں لئی AgroBot دے مارکیٹ ماہر ہو۔

مکمل مارکیٹ رپورٹ لکھو (400-600 لفظ):

## 💰 موجودہ قیمت رپورٹ
## 📈 مارکیٹ رجحان تجزیہ
## 🗓️ ویچن دا بہترین ویلہ
## 🏪 کتھے ویچنا اے
## 🚜 نقل و حمل تے لکیاں لاگتاں
## ⚡ فوری فیصلے دی گائیڈ
## 🔮 قیمت دی پیشن گوئی
## 💡 گفت و شنید دے نکات""",
    },

    "rag_agronomy": {
        "en": """You are AgroBot's SENIOR AGRONOMIC SPECIALIST — Pakistan's expert on crop science, 
pest management, soil health, and modern farming techniques.

You have: RAG knowledge base context + web search results + seasonal information.

Write a THOROUGH, ACTIONABLE expert response (500-800 words):

## 🔬 Direct Answer to Your Question
[Answer the farmer's EXACT question first, clearly and completely. 
If about a pest/disease: identification, symptoms, spread mechanism.
If about fertilizer: exact NPK ratios, timing, method, Pakistani brand names.
If about variety: yield comparison, disease resistance, days to maturity.
If about technique: step-by-step numbered instructions.]

## 📋 Complete Treatment/Action Protocol
[Numbered steps 1-10. Include:
- Product names available in Pakistani market (generics + brand names)
- Exact dose per acre in Pakistani units (bottle, packet, kg, bag)
- When to apply (time of day, crop stage, soil moisture condition)
- How to apply (spray, drench, broadcast, band application)
- Safety precautions (what to wear, re-entry interval)]

## 💊 Pakistani Market: Products & Prices
[List 3-5 specific products available at kiryana/agricultural store.
Generic name + Pakistani brand + approximate price per treatment/acre.]

## 🌡️ Weather/Season Considerations
[Current season {season}. How does current weather affect this problem/solution?
Temperature and humidity effects on treatment effectiveness.]

## ⚠️ What NOT to Do (Common Mistakes)
[3-5 common farmer mistakes that make this problem worse.
What to avoid when treating this issue.]

## 🔄 Prevention for Next Season
[How to avoid this problem next year. Cultural practices, resistant varieties, crop rotation.]

## 💡 Experienced Farmer Wisdom
[Traditional Pakistani farming knowledge combined with modern science.
Tricks that work in Pakistani conditions specifically.]

## 📞 When to Call for Help
[Signs that the problem is beyond home treatment. Who to contact: NARC, agriculture extension officer, district agricultural office numbers if known.]

Use Pakistani brand names. Give amounts in acres, kanals, maunds.
Reference Pakistani conditions specifically. NEVER give vague answers.""",

        "ur": """آپ AgroBot کے سینئر زرعی ماہر ہیں — فصلوں کی سائنس، کیڑوں کی روک تھام، 
مٹی کی صحت، اور جدید کاشتکاری کے تکنیکوں کے ماہر۔

آپ کے پاس: RAG علمی اڈہ + ویب سرچ نتائج + موسمی معلومات ہیں۔

مکمل، قابل عمل ماہرانہ جواب لکھیں (500-800 الفاظ):

## 🔬 آپ کے سوال کا براہ راست جواب
[کسان کے بالکل اصل سوال کا جواب پہلے دیں]

## 📋 مکمل علاج/عمل پروٹوکول
[قدم نمبر 1 سے 10 تک۔ پاکستانی مارکیٹ میں دستیاب مصنوعات کے نام، فی ایکڑ مقدار، کب لگانا ہے]

## 💊 پاکستانی مارکیٹ: مصنوعات اور قیمتیں
[3-5 مخصوص مصنوعات جو زرعی دکان پر ملتی ہیں]

## 🌡️ موسم/سیزن کا خیال
## ⚠️ کیا نہیں کرنا (عام غلطیاں)
## 🔄 اگلے سیزن کے لیے روک تھام
## 💡 تجربہ کار کسان کی حکمت
## 📞 مدد کب مانگیں

پاکستانی برانڈ نام استعمال کریں۔ مقدار ایکڑ، کنال، من میں دیں۔""",

        "pa": """ਤੁਸੀਂ AgroBot ਦੇ ਸੀਨੀਅਰ ਖੇਤੀਬਾੜੀ ਮਾਹਰ ਹੋ।

ਮੁਕੰਮਲ, ਕਾਰਜਯੋਗ ਮਾਹਰ ਜਵਾਬ ਲਿਖੋ (500-800 ਸ਼ਬਦ):

## 🔬 ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਸਿੱਧਾ ਜਵਾਬ
## 📋 ਮੁਕੰਮਲ ਇਲਾਜ/ਕਾਰਜ ਪ੍ਰੋਟੋਕੋਲ
## 💊 ਪਾਕਿਸਤਾਨੀ ਮਾਰਕਿਟ: ਉਤਪਾਦ ਅਤੇ ਕੀਮਤਾਂ
## 🌡️ ਮੌਸਮ/ਰੁੱਤ ਸੰਬੰਧੀ ਵਿਚਾਰ
## ⚠️ ਕੀ ਨਹੀਂ ਕਰਨਾ
## 🔄 ਅਗਲੀ ਰੁੱਤ ਲਈ ਬਚਾਅ
## 💡 ਤਜਰਬੇਕਾਰ ਕਿਸਾਨ ਦੀ ਸਿਆਣਪ
## 📞 ਮਦਦ ਕਦੋਂ ਮੰਗਣੀ ਹੈ""",

        "skr": """تُساں AgroBot دے سینئر زرعی ماہر ہو۔

مکمل، قابل عمل ماہرانہ جواب لکھو (500-800 لفظ):

## 🔬 تہاڈے سوال دا سیدھا جواب
## 📋 مکمل علاج/عمل پروٹوکول
## 💊 پاکستانی مارکیٹ: مصنوعات تے قیمتاں
## 🌡️ موسم/سیزن دا خیال
## ⚠️ کی نئیں کرنا
## 🔄 اگلے سیزن لئی روک تھام
## 💡 تجربیکار کسان دی حکمت
## 📞 مدد کدوں منگنی اے""",
    },

    # ── DISEASE DETECTION: image-based Kindwise + LLaMA treatment plan ──────
    "disease_detection": {
        "en": """You are AgroBot's PLANT PATHOLOGIST and FIELD AGRONOMIST for Pakistani farmers.
A crop image has been analysed by the Kindwise AI vision system and you have the diagnosis.
You also have web search results and agronomic knowledge context.

Write a COMPREHENSIVE, FARMER-FRIENDLY disease management report (500-700 words).
Structure your response EXACTLY as follows:

## 🔬 Diagnosis Summary
[State the disease name and confidence clearly. Explain in simple language what this disease
is and how it attacks the crop. Use an analogy — e.g. "Think of it like rust eating metal."]

## 🚨 How Bad Is It? (Severity Assessment)
[Based on confidence level: Low (<50%) / Medium (50-75%) / High (>75%).
What does this mean for the farmer? Is the crop still saveable?]

## ⏱️ Immediate Action (Next 48 Hours)
[Numbered list of CRITICAL steps to take RIGHT NOW to stop the disease spreading.
Include: isolate affected plants, remove infected parts, avoid overhead watering etc.]

## 💊 Treatment Protocol
[Chemical options: active ingredient + Pakistani brand name (e.g., Score, Tilt, Dithane M-45)
+ dose per acre in Pakistani units (bottles, packets, kg).
Organic/traditional options too.
Application method: spray, drench, dust. Best time of day. Safety gear needed.]

## 💰 Estimated Treatment Cost
[Cost per acre in PKR for the recommended treatment. Where to buy in Pakistan.]

## 🗓️ Recovery Timeline
[Week 1, Week 2, Week 3: what farmer should see if treatment is working.
When to re-apply. When to know it has failed and seek further help.]

## 🔄 Prevention for Next Season
[How to avoid this disease next year. Seed treatment, crop rotation,
resistant varieties available in Pakistani market, soil amendments.]

## ⚠️ Spread Warning
[Can this disease spread to neighbouring farms? How? What precautions to take.]

## 📞 Escalation
[If no improvement in 7 days: contact NARC (0300-5554800) or local agriculture extension officer.]

Always give Pakistani brand names. Amounts in acres/kanals/bottles/packets.""",

        "ur": """آپ پاکستانی کسانوں کے لیے AgroBot کے ماہر نباتاتی امراض اور زرعی مشیر ہیں۔
Kindwise AI نے فصل کی تصویر کا تجزیہ کیا ہے اور تشخیص ہو چکی ہے۔

مکمل، کسان دوست بیماری انتظام رپورٹ لکھیں (500-700 الفاظ):

## 🔬 تشخیص کا خلاصہ
[بیماری کا نام اور یقین کی سطح واضح کریں۔ آسان زبان میں بتائیں]

## 🚨 یہ کتنا خطرناک ہے؟
[کم/درمیانہ/زیادہ خطرہ۔ کیا فصل بچائی جا سکتی ہے؟]

## ⏱️ فوری اقدام (اگلے 48 گھنٹے)
[ابھی کیا کرنا ہے — قدم بہ قدم]

## 💊 علاج کا طریقہ
[کیمیائی علاج: پاکستانی برانڈ نام، فی ایکڑ مقدار۔ قدرتی علاج بھی]

## 💰 علاج کی متوقع لاگت
[PKR میں فی ایکڑ لاگت۔ کہاں سے خریدیں]

## 🗓️ صحت یابی کی مدت
[ہفتہ 1، 2، 3 میں کیا دیکھنا ہے]

## 🔄 اگلے سیزن کے لیے روک تھام
## ⚠️ پھیلاؤ کا انتباہ
## 📞 مزید مدد

پاکستانی برانڈ نام استعمال کریں۔ مقدار ایکڑ، کنال، بوتل میں دیں۔""",

        "pa": """ਤੁਸੀਂ ਪਾਕਿਸਤਾਨੀ ਕਿਸਾਨਾਂ ਲਈ AgroBot ਦੇ ਮਾਹਰ ਪੌਦਾ ਰੋਗ ਵਿਗਿਆਨੀ ਹੋ।
Kindwise AI ਨੇ ਫਸਲ ਦੀ ਤਸਵੀਰ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਹੈ।

ਮੁਕੰਮਲ ਬਿਮਾਰੀ ਪ੍ਰਬੰਧਨ ਰਿਪੋਰਟ ਲਿਖੋ (500-700 ਸ਼ਬਦ):

## 🔬 ਤਸ਼ਖੀਸ਼ ਸੰਖੇਪ
## 🚨 ਇਹ ਕਿੰਨਾ ਖ਼ਤਰਨਾਕ ਹੈ?
## ⏱️ ਤੁਰੰਤ ਕਾਰਵਾਈ (ਅਗਲੇ 48 ਘੰਟੇ)
## 💊 ਇਲਾਜ ਪ੍ਰੋਟੋਕੋਲ
## 💰 ਇਲਾਜ ਦੀ ਅਨੁਮਾਨਿਤ ਲਾਗਤ
## 🗓️ ਠੀਕ ਹੋਣ ਦੀ ਸਮਾਂ-ਸੀਮਾ
## 🔄 ਅਗਲੀ ਰੁੱਤ ਲਈ ਬਚਾਅ
## ⚠️ ਫੈਲਾਅ ਚੇਤਾਵਨੀ
## 📞 ਹੋਰ ਮਦਦ""",

        "skr": """تُساں پاکستانی کساناں لئی AgroBot دے ماہر نباتاتی امراض تے زرعی مشیر ہو۔
Kindwise AI نے فصل دی تصویر دا تجزیہ کیتا اے۔

مکمل بیماری انتظام رپورٹ لکھو (500-700 لفظ):

## 🔬 تشخیص دا خلاصہ
## 🚨 ایہ کنا خطرناک اے؟
## ⏱️ فوری اقدام (اگلے 48 گھنٹے)
## 💊 علاج دا طریقہ
## 💰 علاج دی متوقع لاگت
## 🗓️ صحتیابی دی مدت
## 🔄 اگلے سیزن لئی روک تھام
## ⚠️ پھیلاؤ دا انتباہ
## 📞 ہور مدد""",
    },

    "general": {
        "en": """You are AgroBot — Pakistan's friendliest agricultural AI assistant.
You speak like an experienced village elder who knows both modern agriculture and traditional wisdom.

Current season: {season} | Date: {date}

Give a HELPFUL, WARM, DETAILED response (300-500 words):
- If farming question: answer fully with practical steps
- If greeting: warmly welcome and offer help with SPECIFIC examples of what you can do
- If personal issue: show empathy then redirect to agricultural help
- If unclear: ask ONE clarifying question AND give your best guess answer

Always end with: "What else would you like to know? I can help with crop recommendations, 
soil analysis, weather advice, pest control, and market prices."

NEVER give a one-line answer. Farmers need full guidance.""",

        "ur": """آپ AgroBot ہیں — پاکستان کے سب سے دوستانہ زرعی AI مددگار۔
آپ ایک تجربہ کار بزرگ کی طرح بولتے ہیں جو جدید اور روایتی زراعت دونوں جانتے ہیں۔

موسم: {season} | تاریخ: {date}

مددگار، گرمجوشی سے بھرا، تفصیلی جواب دیں (300-500 الفاظ):
- اگر زرعی سوال ہو: عملی قدموں کے ساتھ مکمل جواب دیں
- اگر سلام ہو: خوش آمدید کہیں اور بتائیں کہ آپ کیا کر سکتے ہیں
- اگر واضح نہیں: ایک سوال پوچھیں اور بہترین اندازہ بھی دیں

آخر میں ضرور لکھیں: "اور کیا جاننا چاہتے ہیں؟ میں فصل کی سفارش، مٹی کا تجزیہ، موسمی مشورہ، کیڑوں کا علاج، اور منڈی کی قیمتوں میں مدد کر سکتا ہوں۔"

ایک لائن کا جواب کبھی نہ دیں۔""",

        "pa": """ਤੁਸੀਂ AgroBot ਹੋ — ਪਾਕਿਸਤਾਨ ਦੇ ਸਭ ਤੋਂ ਦੋਸਤਾਨਾ ਖੇਤੀ AI ਸਹਾਇਕ।

ਮਦਦਗਾਰ, ਨਿੱਘਾ, ਵਿਸਤਰਿਤ ਜਵਾਬ ਦਿਓ (300-500 ਸ਼ਬਦ)।
ਕਦੇ ਵੀ ਇੱਕ ਲਾਈਨ ਦਾ ਜਵਾਬ ਨਾ ਦਿਓ।""",

        "skr": """تُساں AgroBot ہو — پاکستان دے سبھ توں دوستانہ زرعی AI مددگار۔

مددگار، گرم جوشی نال بھریا، تفصیلی جواب دیو (300-500 لفظ)۔
کدے وی اک لائن دا جواب نہ دیو۔""",
    },
}


def _get_analysis_prompt(intent: str, lang: str) -> str:
    """Get the deep analysis system prompt for (intent, language), with season/date injection."""
    prompts = DEEP_ANALYSIS_PROMPTS.get(intent, DEEP_ANALYSIS_PROMPTS["general"])
    template = prompts.get(lang, prompts.get("en", ""))
    return template.format(season=CURRENT_SEASON, date=CURRENT_DATE)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LANGUAGE DETECTION  (Unicode script analysis)
# ══════════════════════════════════════════════════════════════════════════════

def detect_language_unicode(text: str, whisper_code: str = "") -> str:
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
# 4.  AGENT STATE  (expanded with all parallel data)
# ══════════════════════════════════════════════════════════════════════════════

class AgroBotState(TypedDict):
    messages:            Annotated[list, add_messages]
    user_query:          str
    detected_lang:       str
    intent:              str
    soil_data:           Optional[Dict[str, Any]]
    prediction_result:   Optional[Dict[str, Any]]
    weather_result:      Optional[Dict[str, Any]]
    market_result:       Optional[str]
    rag_result:          Optional[str]
    web_search_result:   Optional[str]       # NEW: always search web
    pest_disease_result: Optional[str]       # NEW: dedicated pest search
    final_response:      Optional[str]
    detected_crop:       Optional[str]
    location:            Optional[str]
    audio_input:         Optional[bytes]
    audio_ext:           Optional[str]
    audio_output:        bool
    tts_audio:           Optional[bytes]
    soil_health_score:   Optional[Dict]      # NEW: soil health scoring
    follow_up_questions: Optional[List[str]] # NEW: suggested follow-ups
    # ── Disease Detection (from teammate's Kindwise integration) ──────────────
    image_path:          Optional[str]       # path to uploaded leaf image
    kindwise_crop:       Optional[str]       # crop identified by Kindwise
    kindwise_disease:    Optional[str]       # disease identified by Kindwise
    kindwise_confidence: Optional[float]     # detection confidence (0-100)
    weather_alerts: Optional[List[str]] 

# ══════════════════════════════════════════════════════════════════════════════
# 5.  SUPERVISOR SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

KNOWN_CROPS = {
    "cotton", "wheat", "rice", "maize", "sugarcane", "sunflower",
    "sorghum", "potato", "onion", "mango", "tomato", "chili",
    "garlic", "lentil", "chickpea", "mustard", "barley", "soybean"
}

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
    soil_data:      Optional[SoilData]   = None
    location:       Optional[str]        = None
    detected_crop:  Optional[str]        = Field(
        None,
        description=f"Crop name if mentioned. One of: {KNOWN_CROPS}. None if absent.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SHARED RESOURCES
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
# 7.  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_llm(temperature: float = 0.3, model: str = "openai/gpt-oss-120b", max_tokens: int = 4096) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return ChatGroq(model=model, temperature=temperature, api_key=api_key, max_tokens=max_tokens)

def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)

def build_context_aware_query(state: AgroBotState) -> str:
    if len(state["messages"]) < 2:
        return state["user_query"]
    llm = get_llm(temperature=0)
    result = llm.invoke([
        SystemMessage(content=(
            "You are a query rewriter for an agricultural chatbot. "
            "Rewrite the latest message as a fully self-contained query. "
            "Return ONLY the rewritten query, nothing else."
        )),
        *state["messages"][-6:],  # last 3 turns for context
        HumanMessage(content=f"Rewrite: '{state['user_query']}'")
    ]).content.strip()
    return result

def extract_city(query: str) -> Optional[str]:
    cities = [
        "Lahore", "Karachi", "Islamabad", "Multan", "Faisalabad",
        "Rawalpindi", "Peshawar", "Quetta", "Gujranwala", "Sialkot",
        "Bahawalpur", "Sargodha", "Sahiwal", "Dera Ghazi Khan",
        "Rahim Yar Khan", "Sukkur", "Hyderabad", "Larkana"
    ]
    for city in cities:
        if city.lower() in query.lower():
            return city
    return None

def _parallel_search(queries: List[str]) -> List[str]:
    """Execute multiple web searches in parallel."""
    results = []
    def _search(q):
        try:
            return SEARCH_TOOL.run(q)
        except Exception as e:
            return f"Search unavailable: {e}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_search, q) for q in queries]
        results = [f.result(timeout=15) for f in futures]
    return results

def _score_soil_health(soil_data: dict) -> Dict:
    """Score soil parameters against ideal ranges for Pakistan."""
    ideal_ranges = {
        "N":   (25, 50,   "kg/ha", "Nitrogen"),
        "P":   (10, 30,   "kg/ha", "Phosphorus"),
        "K":   (150, 300, "kg/ha", "Potassium"),
        "ph":  (6.0, 7.5, "pH",    "pH"),
        "EC":  (0, 2.0,   "dS/m",  "Salinity (EC)"),
    }
    scores = {}
    for param, (low, high, unit, label) in ideal_ranges.items():
        val = soil_data.get(param)
        if val is None:
            continue
        if low <= val <= high:
            scores[param] = {"value": val, "unit": unit, "label": label, "status": "good", "score": 100}
        elif val < low:
            pct = max(0, int(100 * val / low))
            scores[param] = {"value": val, "unit": unit, "label": label, "status": "low", "score": pct}
        else:
            pct = max(0, int(100 * high / val))
            scores[param] = {"value": val, "unit": unit, "label": label, "status": "high", "score": pct}
    if scores:
        overall = int(sum(v["score"] for v in scores.values()) / len(scores))
        if overall >= 80:
            grade = "Excellent 🟢"
        elif overall >= 60:
            grade = "Good 🟡"
        elif overall >= 40:
            grade = "Fair 🟠"
        else:
            grade = "Poor 🔴"
    else:
        overall, grade = 0, "Unknown"
    return {"params": scores, "overall_score": overall, "grade": grade}

def check_extreme_risks(city: str, total_rain_mm: float, max_temp_c: float) -> List[str]:
    """
    Compares 5-day accumulated rainfall and max temperature against
    city-specific thresholds. Returns a list of alert strings (empty = no alerts).
    """
    climate = PAK_CLIMATE_DATA.get(city, DEFAULT_CLIMATE)
    alerts  = []

    # ── Flood / waterlogging ─────────────────────────────────────────────
    if total_rain_mm >= climate["flood_threshold_mm"]:
        alerts.append(
            f"🌊 FLOOD/WATERLOGGING RISK: {round(total_rain_mm, 1)} mm of rain "
            f"expected over 5 days (threshold for {city}: "
            f"{climate['flood_threshold_mm']} mm). "
            "Ensure drainage channels are clear. Avoid irrigating for 5-7 days."
        )

    # ── Heatwave ─────────────────────────────────────────────────────────
    if max_temp_c >= climate["heatwave_c"]:
        alerts.append(
            f"🔥 HEATWAVE ALERT: Temperature reaching {round(max_temp_c, 1)}°C "
            f"(danger threshold for {city}: {climate['heatwave_c']}°C). "
            "Apply foliar spray in early morning. Increase irrigation frequency."
        )

    # ── Monsoon approach warning ──────────────────────────────────────────
    today        = datetime.now()
    ms           = climate["monsoon_start"]
    monsoon_date = datetime(today.year, ms[0], ms[1])
    if today > monsoon_date + timedelta(days=30):               # already passed this year
        monsoon_date = datetime(today.year + 1, ms[0], ms[1])
    days_away = (monsoon_date - today).days
    if 0 <= days_away <= 7:
        alerts.append(
            f"⛈️ MONSOON APPROACHING: Monsoon season begins in "
            f"{days_away} day(s) for {city}. "
            "Complete harvesting of mature crops. Prepare field drainage now."
        )

    return alerts
# ══════════════════════════════════════════════════════════════════════════════
# 8.  NODE: STT
# ══════════════════════════════════════════════════════════════════════════════

def stt_node(state: AgroBotState) -> AgroBotState:
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
                model="whisper-large-v3",
                file=f,
                response_format="verbose_json",
            )
        whisper_code = getattr(result, "language", "en") or "en"
        transcript   = result.text.strip()
    finally:
        os.unlink(path)
    lang = detect_language_unicode(transcript, whisper_code)
    print(f"🎙️  STT transcript ({lang}): {transcript}")
    return {**state, "user_query": transcript, "detected_lang": lang}


# ══════════════════════════════════════════════════════════════════════════════
# 9.  NODE: SUPERVISOR
# ══════════════════════════════════════════════════════════════════════════════

def supervisor_node(state: AgroBotState) -> AgroBotState:
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(SupervisorOutput)
    enriched_query = build_context_aware_query(state)

    lang_hint = ""
    if state.get("audio_input") and state.get("detected_lang"):
        lang_hint = (
            f"\n[PRE-DETECTED LANGUAGE from audio: '{state['detected_lang']}'. "
            "Use this for detected_lang unless clearly wrong.]"
        )

    result: SupervisorOutput = structured_llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM + lang_hint),
        *state["messages"][-6:],
        HumanMessage(content=enriched_query),
    ])

    final_lang = (
        state["detected_lang"]
        if state.get("audio_input") and state.get("detected_lang")
        else result.detected_lang
    )
    if not state.get("audio_input"):
        unicode_lang = detect_language_unicode(enriched_query)
        if unicode_lang in ("pa", "skr"):
            final_lang = unicode_lang

    detected_crop = result.detected_crop.lower().strip() if result.detected_crop else None
    soil_data     = result.soil_data.dict() if result.soil_data else None

    print(f"🧭 Supervisor → lang={final_lang}, intent={result.intent}, crop={detected_crop}")

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
# 10.  NODE: PARALLEL DATA GATHERER  (NEW — runs before synthesis)
# ══════════════════════════════════════════════════════════════════════════════

def data_gatherer_node(state: AgroBotState) -> AgroBotState:
    """
    Gather ALL data in parallel:
    - RAG retrieval
    - Multiple web searches (market + pest + general)
    - Weather data
    - Soil prediction (if soil_crop intent)
    This replaces per-node sequential data fetching.
    """
    import requests

    query       = state["user_query"]
    intent      = state["intent"]
    crop        = state.get("detected_crop") or ""
    location    = state.get("location") or extract_city(query) or "Lahore"
    lang        = state["detected_lang"]
    soil_data   = state.get("soil_data") or {}

    print(f"📡 Data gatherer: intent={intent}, crop={crop}, location={location}")

    # ── 1. RAG Retrieval ─────────────────────────────────────────────────────
    rag_context = ""
    if VECTOR_STORE is not None:
        try:
            search_kwargs = {"k": 6, "fetch_k": 20}
            if crop:
                search_kwargs["filter"] = {"topic": crop}
            retriever = VECTOR_STORE.as_retriever(
                search_type="mmr", search_kwargs=search_kwargs
            )
            docs = retriever.invoke(query)
            if docs:
                parts = []
                for i, doc in enumerate(docs, 1):
                    src  = doc.metadata.get("source", f"doc_{i}")
                    sect = doc.metadata.get("section", "")
                    parts.append(f"[Source {i}: {src} | {sect}]\n{doc.page_content}")
                rag_context = "\n\n".join(parts)
                print(f"📚 RAG: retrieved {len(docs)} chunks")
                print("\n" + "="*60)
                print("📄 RAG CONTEXT:")
                print("="*60)
                print(rag_context)
                print("="*60 + "\n")
        except Exception as e:
            print(f"⚠️ RAG error: {e}")

    # ── 2. Parallel Web Searches ──────────────────────────────────────────────
# ── 2. Parallel Web Searches ──────────────────────────────────────────────
    llm = get_llm(temperature=0)
    search_plan = llm.invoke([
        SystemMessage(content=(
            "You are a search query generator for a Pakistani agriculture chatbot. "
            "Given the farmer's question, generate exactly 3 specific web search queries "
            "to find: current prices, agronomic advice, and pest/weather info. "
            "Return ONLY a JSON list of 3 strings. No explanation."
            f"\nCurrent season: {CURRENT_SEASON}, Date: {CURRENT_DATE}"
        )),
        HumanMessage(content=f"Farmer's question: {query}\nCrop: {crop}\nLocation: {location}")
    ]).content.strip()

    try:
        clean = re.sub(r"```json|```", "", search_plan).strip()
        search_queries = json.loads(clean)
    except Exception:
        search_queries = [f"Pakistan {query} agriculture advice"]
    print(f"🔍 Running {len(search_queries)} web searches in parallel...")
    web_results = _parallel_search(search_queries[:4])  # cap at 4
    combined_web = "\n\n---\n\n".join(
        f"[Search {i+1}: {q}]\n{r}"
        for i, (q, r) in enumerate(zip(search_queries, web_results))
    )

    # ── 3. Weather Data ──────────────────────────────────────────────────────
# ── 3. Weather Data + Early Warning Alerts ───────────────────────────────
    weather_data = {}
    weather_alerts: List[str] = []
    owm_key = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()
    if owm_key:
        try:
            # ── Current conditions ────────────────────────────────────────
            cur_url  = (f"https://api.openweathermap.org/data/2.5/weather"
                        f"?q={location}&appid={owm_key}&units=metric")
            cur_resp = requests.get(cur_url, timeout=8)

            # ── 5-day / 3-hour forecast (for flood + heatwave check) ──────
            fct_url  = (f"https://api.openweathermap.org/data/2.5/forecast"
                        f"?q={location}&appid={owm_key}&units=metric")
            fct_resp = requests.get(fct_url, timeout=8)

            if cur_resp.status_code == 200:
                raw = cur_resp.json()
                weather_data = {
                    "city":     raw["name"],
                    "temp":     round(raw["main"]["temp"], 1),
                    "feels":    round(raw["main"]["feels_like"], 1),
                    "humidity": raw["main"]["humidity"],
                    "desc":     raw["weather"][0]["description"],
                    "wind":     raw["wind"].get("speed", 0),
                    "source":   "OpenWeatherMap",
                }
                print(f"🌤️ Weather: {weather_data['temp']}°C, "
                      f"{weather_data['desc']} in {location}")

            if fct_resp.status_code == 200:
                fct        = fct_resp.json()
                total_rain = 0.0
                max_temp   = 0.0
                for slot in fct["list"]:
                    t = slot["main"]["temp"]
                    if t > max_temp:
                        max_temp = t
                    if "rain" in slot and "3h" in slot["rain"]:
                        total_rain += slot["rain"]["3h"]

                weather_data["forecast_max_temp_c"]  = round(max_temp, 1)
                weather_data["forecast_rain_5day_mm"] = round(total_rain, 1)

                # ── Run the early warning engine ──────────────────────────
                weather_alerts = check_extreme_risks(location, total_rain, max_temp)
                if weather_alerts:
                    print(f"🚨 {len(weather_alerts)} early warning alert(s) triggered!")
                    for a in weather_alerts:
                        print(f"   ⚠️  {a[:80]}…")

        except Exception as e:
            print(f"⚠️ Weather error: {e}")

    # ── 4. Soil Prediction (if applicable) ───────────────────────────────────
    prediction_result = {}
    soil_health       = {}
    if intent == "soil_crop" and soil_data:
        soil_health = _score_soil_health(soil_data)
        print(f"🌱 Soil health: {soil_health.get('grade')}")
        try:
            from XGBoostClassifierAndYeildPredictor import agro_predict
            prediction_result = agro_predict(soil_data)
            print(f"🤖 XGBoost predicted: {prediction_result.get('recommended_crop')}")
        except ImportError:
            prediction_result = {
                "recommended_crop": "Wheat",
                "predicted_yield_tons_per_acre": 2.65,
                "confidence": 87.3,
                "top_3_crops": [("Wheat", 87.3), ("Maize", 8.1), ("Rice", 4.6)],
                "note": "[MOCK — real models not loaded]",
            }
        except Exception as e:
            prediction_result = {"error": str(e)}

    return {
        **state,
        "rag_result":         rag_context[:3000] if rag_context else "",
        "web_search_result":  combined_web[:3000],
        "weather_result":     weather_data,
        "prediction_result":  prediction_result,
        "soil_health_score":  soil_health,
        "location":           location,
        "weather_alerts":     weather_alerts
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10.5  NODE: DISEASE DETECTION  (Kindwise vision + LLaMA treatment plan)
#       Integrated from teammate's agrobot_graph.py
# ══════════════════════════════════════════════════════════════════════════════

def disease_detection_node(state: AgroBotState) -> AgroBotState:
    """
    Calls the Kindwise crop disease API on a farmer-uploaded leaf image.
    If no image_path is in state, opens an OS file picker dialog.
    Populates kindwise_crop, kindwise_disease, and kindwise_confidence in state.
    The synthesis_node then generates the full treatment plan.
    """
    import requests as _requests

    image_path = state.get("image_path")

    # ── Open file picker if no image was pre-supplied ──────────────────────
    if not image_path:
        print("\n[AgroBot] 📸 Opening image picker — select a leaf/plant photo…")
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        image_path = filedialog.askopenfilename(
            title="Select a crop leaf image for disease analysis",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")],
        )
        root.destroy()

    if not image_path:
        # Farmer cancelled — degrade gracefully to rag_agronomy
        print("[AgroBot] No image selected — falling back to text-based advice.")
        return {
            **state,
            "intent": "rag_agronomy",
            "user_query": state["user_query"] + " (no image provided, give general disease advice)",
        }

    # ── Read and base64-encode the image ──────────────────────────────────
    try:
        with open(image_path, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode("utf-8")
    except OSError as exc:
        print(f"[AgroBot] ⚠️ Could not read image: {exc}")
        return {**state, "intent": "rag_agronomy"}

    print(f"[AgroBot] 🔍 Analysing {os.path.basename(image_path)} via Kindwise…")

    # ── Call Kindwise API ──────────────────────────────────────────────────
    try:
        resp = _requests.post(
            KINDWISE_URL,
            headers={"Api-Key": KINDWISE_API_KEY, "Content-Type": "application/json"},
            json={"images": [img_b64]},
            timeout=30,
        )
    except Exception as exc:
        print(f"[AgroBot] ⚠️ Kindwise network error: {exc}")
        return {**state, "intent": "rag_agronomy"}

    if resp.status_code not in (200, 201):
        print(f"[AgroBot] ⚠️ Kindwise returned HTTP {resp.status_code}")
        return {**state, "intent": "rag_agronomy"}

    # ── Parse the Kindwise response ────────────────────────────────────────
    result = resp.json().get("result", {})

    crop_suggestions  = result.get("crop",    {}).get("suggestions", [])
    kindwise_crop     = (
        crop_suggestions[0].get("name", "Unknown Crop").title()
        if crop_suggestions else "Unknown Crop"
    )

    disease_suggestions = result.get("disease", {}).get("suggestions", [])

    if not disease_suggestions:
        print(f"[AgroBot] ✅ No disease detected on {kindwise_crop} — plant appears healthy.")
        return {
            **state,
            "image_path":          image_path,
            "kindwise_crop":       kindwise_crop,
            "kindwise_disease":    None,
            "kindwise_confidence": None,
            "final_response": (
                f"✅ **Crop identified: {kindwise_crop}**\n\n"
                "No disease detected. Your plant appears **healthy**! "
                "Keep monitoring regularly and maintain good cultural practices "
                "(proper spacing, avoid waterlogging, balanced fertilization)."
            ),
        }

    top               = disease_suggestions[0]
    kindwise_disease  = top.get("name", "Unknown Condition").title()
    kindwise_conf     = round(top.get("probability", 0) * 100, 1)

    print(f"[AgroBot] 🦠 Detected: {kindwise_disease} ({kindwise_conf}%) on {kindwise_crop}")

    return {
        **state,
        "image_path":          image_path,
        "kindwise_crop":       kindwise_crop,
        "kindwise_disease":    kindwise_disease,
        "kindwise_confidence": kindwise_conf,
        "detected_crop":       kindwise_crop.lower(),   # share with rest of pipeline
    }


# ══════════════════════════════════════════════════════════════════════════════
# 11.  NODE: SYNTHESIS  (the intelligence engine — replaces all specialist nodes)
# ══════════════════════════════════════════════════════════════════════════════

def synthesis_node(state: AgroBotState) -> AgroBotState:
    """
    Takes ALL gathered data and synthesizes ONE comprehensive, farmer-friendly response.
    This is where the magic happens — the LLM acts as a true agricultural expert.
    """
    intent  = state["intent"]
    lang    = state["detected_lang"]
    query   = state["user_query"]
    crop    = state.get("detected_crop") or "the relevant crop"
    loc     = state.get("location") or "your area"

    # Build the comprehensive context package
    context_parts = []

    context_parts.append(f"=== FARMER'S QUESTION ===\n{query}")
    context_parts.append(f"=== CURRENT DATE & SEASON ===\n{CURRENT_DATE} | {CURRENT_SEASON}")
    context_parts.append(f"=== FARMER'S LOCATION ===\n{loc}")

    if state.get("soil_data"):
        context_parts.append(
            f"=== SOIL DATA PROVIDED BY FARMER ===\n{json.dumps(state['soil_data'], indent=2)}"
        )

    if state.get("soil_health_score") and state["soil_health_score"].get("params"):
        sh = state["soil_health_score"]
        health_str = f"Overall Grade: {sh['grade']} ({sh['overall_score']}/100)\n"
        for param, info in sh["params"].items():
            health_str += f"  {info['label']}: {info['value']} {info['unit']} → {info['status'].upper()}\n"
        context_parts.append(f"=== SOIL HEALTH ANALYSIS ===\n{health_str}")

    if state.get("prediction_result"):
        context_parts.append(
            f"=== ML MODEL CROP PREDICTION ===\n{json.dumps(state['prediction_result'], indent=2)}"
        )

    if state.get("weather_result"):
        context_parts.append(
            f"=== CURRENT WEATHER ({loc}) ===\n{json.dumps(state['weather_result'], indent=2)}"
        )

    if state.get("rag_result"):
        context_parts.append(
            f"=== KNOWLEDGE BASE (RAG) CONTEXT ===\n{state['rag_result']}"
        )

    if state.get("web_search_result"):
        context_parts.append(
            f"=== WEB SEARCH RESULTS (Market/Pest/General Info) ===\n{state['web_search_result']}"
        )
    if state.get("weather_alerts"):
        alert_block = "\n".join(f"  ⚠️  {a}" for a in state["weather_alerts"])
        context_parts.append(
            f"=== 🚨 CRITICAL EARLY WARNINGS (ACT IMMEDIATELY) ===\n{alert_block}\n"
            "INSTRUCTION TO AI: You MUST address ALL of the above warnings in your "
            "response. Dedicate a prominent ## 🚨 Critical Alerts section to them. "
            "Give specific preventive actions for each alert."
        )
    # ── Disease detection results (Kindwise) ──────────────────────────────
    if intent == "disease_detection" and state.get("kindwise_disease"):
        context_parts.append(
            f"=== KINDWISE DISEASE DETECTION RESULTS ===\n"
            f"Crop Identified : {state.get('kindwise_crop', 'Unknown')}\n"
            f"Disease Detected: {state['kindwise_disease']}\n"
            f"Confidence      : {state.get('kindwise_confidence', 0)}%\n"
            f"Image Analysed  : {os.path.basename(state.get('image_path', ''))}"
        )

    full_context = "\n\n".join(context_parts)

    # Get the deep analysis system prompt
    system_prompt = _get_analysis_prompt(intent, lang)

    print(f"🧠 Synthesis: intent={intent}, lang={lang}, context_len={len(full_context)}")

    # Use higher temperature and max_tokens for rich, complete responses
    llm = get_llm(temperature=0.5, max_tokens=8192)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"][-4:],  # recent conversation history
        HumanMessage(content=full_context),
    ]).content

    # ── Generate follow-up suggestions ───────────────────────────────────────
    follow_up_prompts = {
        "en": [
            f"What are the best fertilizers for {crop} in Pakistan?",
            f"What pests should I watch for with {crop}?",
            f"What is the current market price for {crop}?",
            f"When is the best time to harvest {crop}?",
        ],
        "ur": [
            f"{crop} کے لیے پاکستان میں بہترین کھاد کون سی ہے؟",
            f"{crop} میں کون سے کیڑے لگتے ہیں؟",
            f"{crop} کی آج کی منڈی قیمت کیا ہے؟",
            f"{crop} کی کٹائی کب کرنی چاہیے؟",
        ],
        "pa": [
            f"{crop} ਲਈ ਪਾਕਿਸਤਾਨ ਵਿੱਚ ਸਭ ਤੋਂ ਵਧੀਆ ਖਾਦ ਕਿਹੜੀ ਹੈ?",
            f"{crop} ਵਿੱਚ ਕਿਹੜੇ ਕੀੜੇ ਲੱਗਦੇ ਹਨ?",
        ],
        "skr": [
            f"{crop} لئی پاکستان وچ بہترین کھاد کہڑی اے؟",
            f"{crop} وچ کہڑے کیڑے لگدے نیں؟",
        ],
    }
    # Disease-specific follow-ups override generic ones
    if intent == "disease_detection" and state.get("kindwise_disease"):
        disease = state["kindwise_disease"]
        follow_up_prompts["en"] = [
            f"How do I prevent {disease} next season?",
            f"What is the cheapest treatment for {disease} in Pakistan?",
            f"Can {disease} spread to my neighbour's farm?",
            f"What resistant {crop} varieties are available in Pakistan?",
        ]
        follow_up_prompts["ur"] = [
            f"اگلے سیزن میں {disease} سے کیسے بچوں؟",
            f"پاکستان میں {disease} کا سب سے سستا علاج کیا ہے؟",
            f"کیا {disease} پڑوسی کے کھیت میں پھیل سکتی ہے؟",
        ]
    follow_ups = follow_up_prompts.get(lang, follow_up_prompts["en"])

    return {
        **state,
        "final_response":     response,
        "follow_up_questions": follow_ups,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 12.  NODE: RESPONSE FORMATTER  (enhanced with follow-ups and rich metadata)
# ══════════════════════════════════════════════════════════════════════════════

def response_formatter_node(state: AgroBotState) -> AgroBotState:
    response_text = state.get("final_response", "")
    intent        = state.get("intent", "general")
    lang          = state.get("detected_lang", "en")
    lang_cfg      = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG["en"])

    # Source badges
    source_badges = {
        "soil_crop":         "🌱 *AgroBot XGBoost Crop Engine + Agronomic Knowledge Base*",
        "weather":           "🌤️ *OpenWeatherMap + AgroBot Weather Advisory*",
        "market":            "📊 *Live Market Search + Price Intelligence*",
        "rag_agronomy":      "📚 *AgroBot Knowledge Base + Web Research*",
        "disease_detection": "🔬 *Kindwise Vision AI + AgroBot Plant Pathologist*",
        "general":           "🤖 *AgroBot — Pakistan's Agricultural AI*",
    }
    badge    = source_badges.get(intent, "🤖 *AgroBot*")
    lang_tag = f"_{lang_cfg.get('flag', '')} {lang_cfg.get('label', 'English')}_"

    # Follow-up questions section
    follow_ups = state.get("follow_up_questions") or []
    if follow_ups:
        follow_up_headers = {
            "en":  "\n\n---\n💬 **You might also want to ask:**",
            "ur":  "\n\n---\n💬 **آپ یہ بھی پوچھ سکتے ہیں:**",
            "pa":  "\n\n---\n💬 **ਤੁਸੀਂ ਇਹ ਵੀ ਪੁੱਛ ਸਕਦੇ ਹੋ:**",
            "skr": "\n\n---\n💬 **تُساں ایہ وی پُچھ سکدے ہو:**",
        }
        header = follow_up_headers.get(lang, follow_up_headers["en"])
        fu_text = header + "\n" + "\n".join(f"• {q}" for q in follow_ups[:3])
    else:
        fu_text = ""

    full_response = f"{response_text}{fu_text}\n\n---\n{badge}  {lang_tag}"

    return {
        **state,
        "final_response": full_response,
        "messages":       state["messages"] + [AIMessage(content=full_response)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 13.  NODE: TTS
# ══════════════════════════════════════════════════════════════════════════════

async def _edge_save(text: str, voice: str, path: str) -> None:
    await edge_tts.Communicate(text, voice).save(path)

def tts_node(state: AgroBotState) -> AgroBotState:
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
# 14.  ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def route_stt(state: AgroBotState) -> Literal["stt_node", "supervisor"]:
    return "stt_node" if state.get("audio_input") else "supervisor"

def route_after_supervisor(state: AgroBotState) -> Literal["disease_detection_node", "data_gatherer"]:
    """Route disease_detection intent to the Kindwise node; everything else to data_gatherer."""
    if state.get("intent") == "disease_detection":
        return "disease_detection_node"
    return "data_gatherer"

def route_after_disease(state: AgroBotState) -> Literal["synthesis", "data_gatherer"]:
    """
    After disease detection:
    - If a disease was found (or plant is healthy with final_response set): go to synthesis.
    - If intent was downgraded to rag_agronomy (no image): go to data_gatherer.
    """
    if state.get("intent") == "rag_agronomy":
        return "data_gatherer"
    return "synthesis"


# ══════════════════════════════════════════════════════════════════════════════
# 15.  BUILD THE GRAPH  (streamlined: supervisor → data_gatherer → synthesis)
# ══════════════════════════════════════════════════════════════════════════════

def build_agrobot_graph() -> StateGraph:
    graph = StateGraph(AgroBotState)

    graph.add_node("stt_node",                stt_node)
    graph.add_node("supervisor",              supervisor_node)
    graph.add_node("disease_detection_node",  disease_detection_node)   # ← NEW
    graph.add_node("data_gatherer",           data_gatherer_node)
    graph.add_node("synthesis",               synthesis_node)
    graph.add_node("response_formatter",      response_formatter_node)
    graph.add_node("tts_node",                tts_node)

    # Entry: audio goes to STT first, text goes straight to supervisor
    graph.add_conditional_edges(
        START, route_stt,
        {"stt_node": "stt_node", "supervisor": "supervisor"},
    )
    graph.add_edge("stt_node", "supervisor")

    # Supervisor → branch on intent
    graph.add_conditional_edges(
        "supervisor", route_after_supervisor,
        {
            "disease_detection_node": "disease_detection_node",
            "data_gatherer":          "data_gatherer",
        },
    )

    # Disease node → either synthesis (image analysed) or data_gatherer (no image → fallback)
    graph.add_conditional_edges(
        "disease_detection_node", route_after_disease,
        {
            "synthesis":     "synthesis",
            "data_gatherer": "data_gatherer",
        },
    )

    graph.add_edge("data_gatherer",      "synthesis")
    graph.add_edge("synthesis",          "response_formatter")
    graph.add_edge("response_formatter", "tts_node")
    graph.add_edge("tts_node",           END)

    return graph.compile()

agrobot_graph = build_agrobot_graph()


# ══════════════════════════════════════════════════════════════════════════════
# 16.  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_agrobot(
    user_message: str   = "",
    history:      list  = None,
    audio_bytes:  bytes = None,
    audio_ext:    str   = "wav",
    audio_output: bool  = False,
    image_path:   str   = None,   # ← NEW: path to leaf image for disease detection
) -> dict:
    """
    Main entry point for the AgroBot agent.
    Returns: {"text": str, "lang": str, "audio": bytes|None, "intent": str,
              "follow_ups": list, "soil_health": dict,
              "disease": {"crop": str, "disease": str, "confidence": float}}
    """
    initial_state: AgroBotState = {
        "messages":            history or [],
        "user_query":          user_message,
        "detected_lang":       "en",
        "intent":              "",
        "soil_data":           None,
        "prediction_result":   None,
        "weather_result":      None,
        "market_result":       None,
        "rag_result":          None,
        "web_search_result":   None,
        "pest_disease_result": None,
        "final_response":      None,
        "detected_crop":       None,
        "location":            None,
        "audio_input":         audio_bytes,
        "audio_ext":           audio_ext,
        "audio_output":        audio_output,
        "tts_audio":           None,
        "soil_health_score":   None,
        "follow_up_questions": None,
        # Disease detection fields
        "image_path":          image_path,
        "kindwise_crop":       None,
        "kindwise_disease":    None,
        "kindwise_confidence": None,
        "weather_alerts":      None,
    }

    result = agrobot_graph.invoke(initial_state)
    return {
        "text":        result["final_response"],
        "lang":        result["detected_lang"],
        "audio":       result.get("tts_audio"),
        "intent":      result.get("intent"),
        "follow_ups":  result.get("follow_up_questions") or [],
        "soil_health": result.get("soil_health_score") or {},
        "disease": {
            "crop":       result.get("kindwise_crop"),
            "disease":    result.get("kindwise_disease"),
            "confidence": result.get("kindwise_confidence"),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 17.  CLI TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("🌾 AgroBot — DEEP ANALYSIS EDITION")
    print("   Languages: English · اردو · ਪੰਜਾਬੀ · سرائیکی")
    print(f"   Season: {CURRENT_SEASON}")
    print(f"   Date: {CURRENT_DATE}")
    print("   Commands:")
    print("     --audio  → enable voice output for this message")
    print("     --scan   → open image picker for leaf disease detection")
    print("     exit     → quit")
    print("=" * 70)

    history = []

    while True:
        raw_input = input("\n👨‍🌾 You: ").strip()
        if raw_input.lower() in ["exit", "quit", "q"]:
            print("👋 Allah Hafiz! خدا حافظ")
            break

        want_audio = "--audio" in raw_input
        want_scan  = "--scan"  in raw_input
        user_text  = raw_input.replace("--audio", "").replace("--scan", "").strip()

        # --scan always forces disease_detection intent, regardless of other text
        if want_scan:
            user_text = (user_text + " scan image for disease detection").strip()

        print("\n⏳ AgroBot is thinking deeply...")
        output = run_agrobot(
            user_message=user_text,
            history=history,
            audio_output=want_audio,
            image_path=None,   # disease_detection_node opens picker when intent is detected
        )

        print(f"\n{'='*70}")
        print(f"🤖 AgroBot [{output['lang']}]:\n")
        print(output["text"])
        print(f"{'='*70}")

        if output.get("soil_health") and output["soil_health"].get("grade"):
            print(f"\n📊 Soil Health: {output['soil_health']['grade']}")

        if output.get("disease") and output["disease"].get("disease"):
            d = output["disease"]
            print(f"\n🦠 Disease Detected: {d['disease']} on {d['crop']} ({d['confidence']}% confidence)")

        if output["audio"]:
            audio_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "audio_outputs"
            )
            os.makedirs(audio_dir, exist_ok=True)
            out_path = os.path.join(audio_dir, "agrobot_response.mp3")
            with open(out_path, "wb") as f:
                f.write(output["audio"])
            print(f"🔊 Audio saved → {out_path}")

        history.append(HumanMessage(content=user_text))
        history.append(AIMessage(content=output["text"]))