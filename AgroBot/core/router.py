"""
AgroBot — Intent Router
========================
Wraps the LangGraph supervisor logic as a standalone callable.
Detects intent + language from user query without running the full graph.
Used by main.py to know which feature to highlight in the UI.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from typing import Literal

from config import GROQ_API_KEY, LLM_MODEL
from core.language import detect_language_unicode

# Known Pakistan crops for structured extraction
KNOWN_CROPS = {
    "cotton", "wheat", "rice", "maize", "sugarcane",
    "sunflower", "sorghum", "potato", "onion", "mango",
    "کپاس", "گیہوں", "چاول", "مکئی", "گنا"
}

INTENT_LABELS = {
    "soil_crop":    "🌱 Soil & Crop",
    "weather":      "🌤️ Weather",
    "disease":      "🔬 Disease",
    "flood":        "🌊 Flood Alert",
    "market":       "📊 Market",
    "rag_agronomy": "📚 Knowledge",
    "general":      "💬 General",
}


class RouterOutput(BaseModel):
    detected_lang:  Literal["en", "ur", "pa", "skr"] = "en"
    intent:         Literal["soil_crop", "weather", "disease", "flood", "market", "rag_agronomy", "general"] = "general"
    location:       Optional[str] = None
    detected_crop:  Optional[str] = Field(
        None,
        description=f"Crop name if mentioned. One of: {KNOWN_CROPS}. None if absent."
    )
    has_image:      bool = False


ROUTER_SYSTEM = """You are AgroBot's intent router for a Pakistani agricultural chatbot.

Analyze the user's query and return:
- detected_lang: "en" (English), "ur" (Urdu/Arabic script), "pa" (Punjabi/Shahmukhi), "skr" (Saraiki)
- intent: ONE of these:
    soil_crop    → soil analysis, crop recommendation, EC, pH, salinity, NPK, yield
    weather      → weather forecast, rain, temperature, climate, when to plant/irrigate
    disease      → plant disease, pest, blight, rust, virus, insect attack, image analysis
    flood        → flood risk, waterlogging, heavy rain danger, crop evacuation
    market       → crop prices, mandi rates, selling advice, commodity prices
    rag_agronomy → general agronomy knowledge, fertilizer schedules, irrigation methods
    general      → greetings, off-topic, anything not fitting above
- location: city/region if mentioned (e.g., "Multan", "Punjab", "Sindh")
- detected_crop: crop name if clearly mentioned, else null
- has_image: always false (set to true only if you see [IMAGE] tag in query)

IMPORTANT: Detect language from the SCRIPT and LINGUISTIC MARKERS.
- Arabic/Urdu script with Urdu grammar/vocabulary = "ur".
- Arabic script with Punjabi markers (نیں، تے، دا، دی، کٹک) = "pa" (Shahmukhi).
- Arabic script with Saraiki markers (میڈا، تیڈا، کوں) = "skr".
- Gurmukhi script = "pa".
- Latin script = "en".
"""


def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)


def route_query(
    query: str,
    history: list = None,
    has_image: bool = False,
) -> RouterOutput:
    """
    Detect intent and language for a user query.

    Args:
        query:     User's text query.
        history:   Prior conversation messages (LangChain messages).
        has_image: Whether an image was uploaded.

    Returns:
        RouterOutput with intent + lang + location + crop.
    """
    # Fast path: Unicode detection for lang (no LLM needed)
    unicode_lang = detect_language_unicode(query)

    # If image is uploaded, route to disease detector regardless
    if has_image:
        return RouterOutput(
            detected_lang=unicode_lang,
            intent="disease",
            has_image=True,
        )

    # Simple keyword matching for very obvious intents (reduces LLM calls)
    query_lower = query.lower()

    flood_keywords   = ["flood", "سیلاب", "waterlogg", "heavy rain", "ڈوب", "پانی بھر"]
    disease_keywords = ["disease", "بیماری", "pest", "کیڑا", "blight", "rust", "زنگ", "مروڑ", "پتہ"]
    weather_keywords = ["weather", "موسم", "rain", "بارش", "temperature", "درجہ حرارت", "forecast", "irrigat", "climate"]
    soil_keywords    = ["soil", "مٹی", "ph", "ec", "فصل", "fertilizer", "کھاد", "yield", "پیداوار", "salinity", "نمکیات", "nitrogen", "potassium", "phosphorus", "npk", "nutrient", "crop", "grow", "recommend", "plant"]
    market_keywords  = ["price", "قیمت", "mandi", "منڈی", "market", "مارکیٹ", "sell", "بیچنا", "rate", "ریٹ"]

    # Fast city + crop extractors (used when keyword path bypasses LLM)
    _CITIES = [
        "lahore", "karachi", "islamabad", "rawalpindi", "multan", "faisalabad",
        "peshawar", "quetta", "gujranwala", "sialkot", "hyderabad", "bahawalpur",
        "sargodha", "sukkur", "larkana", "nawabshah", "jhang", "sheikhupura",
        "dera ghazi khan", "rahim yar khan", "sahiwal", "okara", "gujrat",
        "kasur", "hafizabad", "mianwali", "attock", "chakwal", "chiniot",
    ]

    def _fast_city(q: str):
        ql = q.lower()
        for city in _CITIES:
            if city in ql:
                return city.title()
        return None

    def _fast_crop(q: str):
        ql = q.lower()
        for crop in KNOWN_CROPS:
            if crop.lower() in ql:
                return crop
        return None

    # PRIORITY ORDER: flood > disease > soil > weather > market
    # Flood/disease are emergencies and always win.
    # Soil comes before weather (soil queries mention temperature/humidity).
    if any(kw in query_lower for kw in flood_keywords):
        return RouterOutput(detected_lang=unicode_lang, intent="flood",
                            location=_fast_city(query), detected_crop=_fast_crop(query))
    if any(kw in query_lower for kw in disease_keywords):
        return RouterOutput(detected_lang=unicode_lang, intent="disease",
                            detected_crop=_fast_crop(query))
    if any(kw in query_lower for kw in soil_keywords):
        return RouterOutput(detected_lang=unicode_lang, intent="soil_crop",
                            detected_crop=_fast_crop(query))
    if any(kw in query_lower for kw in weather_keywords):
        return RouterOutput(detected_lang=unicode_lang, intent="weather",
                            location=_fast_city(query), detected_crop=_fast_crop(query))
    if any(kw in query_lower for kw in market_keywords):
        return RouterOutput(detected_lang=unicode_lang, intent="market",
                            location=_fast_city(query))

    # Full LLM routing for ambiguous queries
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(RouterOutput)
        query_with_tag = f"[IMAGE attached] {query}" if has_image else query
        result: RouterOutput = structured_llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM),
            *(history or []),
            HumanMessage(content=query_with_tag),
        ])
        # Trust Unicode for lang if Arabic/Gurmukhi (more reliable than LLM)
        if unicode_lang in ("pa", "skr", "ur"):
            result.detected_lang = unicode_lang
        return result
    except Exception as e:
        print(f"⚠️  Router LLM failed ({e}), using keyword fallback.")
        return RouterOutput(detected_lang=unicode_lang, intent="general")


def intent_label(intent: str) -> str:
    """Return human-readable label for an intent code."""
    return INTENT_LABELS.get(intent, "💬 General")
