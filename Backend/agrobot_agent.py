"""
AgroBot LangGraph Agent
========================
A stateful, multi-node agent that routes farmer queries intelligently
across soil/crop analysis, weather, market info, and general agronomy RAG.

Architecture:
    User Input
        │
        ▼
    [supervisor] ──────────────────────────────────────┐
        │                                               │
        ├──► [soil_crop_node]   (XGBoost prediction)   │
        ├──► [weather_node]     (OpenWeatherMap)        │
        ├──► [market_node]      (DuckDuckGo search)     │
        ├──► [rag_node]         (ChromaDB + Llama3)     │
        └──► [general_node]     (LLM fallback)          │
                                                        │
    [response_formatter] ◄──────────────────────────────┘
        │
        ▼
    Final Response (EN/UR)
"""

import os
import json
import pickle
from typing import TypedDict, Annotated, Literal
from pathlib import Path

# ── LangGraph core ─────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ── LangChain / LLM ────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ── Tools ───────────────────────────────────────────────────────────────────────
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Typing ──────────────────────────────────────────────────────────────────────
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
# ══════════════════════════════════════════════════════════════════════════════
# 1.  AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════
from pydantic import BaseModel, Field
from typing import Optional, Literal

class SoilData(BaseModel):
    N: Optional[float] = Field(None)
    P: Optional[float] = Field(None)
    K: Optional[float] = Field(None)
    temperature: Optional[float] = Field(None)
    humidity: Optional[float] = Field(None)
    ph: Optional[float] = Field(None)
    rainfall: Optional[float] = Field(None)
    EC: Optional[float] = Field(None)
    Soil_Type: Optional[str] = Field(None)


class SupervisorOutput(BaseModel):
    detected_lang: Literal["en", "ur"]
    intent: Literal["soil_crop", "weather", "market", "rag_agronomy", "general"]
    soil_data: Optional[SoilData]
    location: Optional[str]  
load_dotenv()  # Load environment variables from .env file
class AgroBotState(TypedDict):
    """
    The single shared state object that flows through every node.
    LangGraph merges list fields (messages) via the add_messages reducer.
    """
    messages:        Annotated[list, add_messages]   # full conversation history
    user_query:      str                              # raw user text
    detected_lang:   str                             # "en" | "ur"
    intent:          str                             # routing decision
    soil_data:       Optional[Dict[str, Any]]        # parsed soil params (if any)
    prediction_result: Optional[Dict[str, Any]]      # XGBoost output
    weather_result:  Optional[Dict[str, Any]]        # weather data
    market_result:   Optional[str]                   # search snippet
    rag_result:      Optional[str]                   # RAG answer
    final_response:  Optional[str]                   # formatted reply
    rag_context: None | str                            # retrieved context for RAG node

# ══════════════════════════════════════════════════════════════════════════════
# 2.  SHARED RESOURCES  (loaded once at import time)
# ══════════════════════════════════════════════════════════════════════════════

def _load_xgboost_models():
    """Load XGBoost + supporting pickles. Returns None tuple if missing."""
    base = Path(__file__).parent / "Models"
    paths = {
        "model":    base / "xgboost_model.pkl",
        "scaler":   base / "scaler.pkl",
        "le":       base / "label_encoder.pkl",
        "knn":      base / "knn_model.pkl",
        "explainer":base / "shap_explainer.pkl",
    }
    if not all(p.exists() for p in paths.values()):
        print("⚠️  XGBoost model files not found – crop prediction disabled.")
        return None, None, None, None, None

    return (
        pickle.load(open(paths["model"],    "rb")),
        pickle.load(open(paths["scaler"],   "rb")),
        pickle.load(open(paths["le"],       "rb")),
        pickle.load(open(paths["knn"],      "rb")),
        pickle.load(open(paths["explainer"],"rb")),
    )


def _load_vector_store():
    """Load ChromaDB vector store. Returns None if missing."""
    chroma_path = Path(__file__).parent / "chroma_db"
    if not chroma_path.exists():
        print("⚠️  ChromaDB not found – RAG disabled (will use web search fallback).")
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


# Module-level singletons
XGB_MODEL, SCALER, LE, KNN_MODEL, EXPLAINER = _load_xgboost_models()
VECTOR_STORE = _load_vector_store()
SEARCH_TOOL  = DuckDuckGoSearchRun()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  HELPER – build the LLM
# ══════════════════════════════════════════════════════════════════════════════

def get_llm(temperature: float = 0.3) -> ChatGroq:
    """Return a ChatGroq Llama-3 instance."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=temperature,
        api_key=api_key,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  NODE: LANGUAGE DETECTION + INTENT ROUTING  (supervisor)
# ══════════════════════════════════════════════════════════════════════════════
SUPERVISOR_SYSTEM = """You are AgroBot's routing supervisor.

Extract structured information from the user query.

Rules:
- Always detect language ("en" or "ur")
- Always choose ONE intent
- If soil parameters are explicitly present, extract them as numbers
- Do NOT guess values
- Extract location if present (city or region)
- Only include soil_data if relevant to soil_crop

Return structured output using the provided schema.
"""

from langchain_core.runnables import RunnableConfig

def supervisor_node(state: AgroBotState) -> AgroBotState:
    llm = get_llm(temperature=0)

    structured_llm = llm.with_structured_output(SupervisorOutput)

    result: SupervisorOutput = structured_llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=state["user_query"])
    ])

    soil_data = result.soil_data.dict() if result.soil_data else None

    return {
        **state,
        "detected_lang": result.detected_lang,
        "intent": result.intent,
        "soil_data": soil_data,
        "messages": state["messages"] + [
            HumanMessage(content=state["user_query"])
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  NODE: SOIL-TO-CROP PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

# Import the existing prediction logic
try:
    from XGBoostClassifierAndYeildPredictor import agro_predict
    _AGRO_PREDICT_AVAILABLE = True
except ImportError:
    _AGRO_PREDICT_AVAILABLE = False
    print("⚠️  agro_predict not importable – using mock prediction.")


def _mock_predict(soil_data: dict) -> dict:
    """Fallback when model files aren't present (demo / CI)."""
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


CROP_INTERPRETER_SYSTEM = """You are AgroBot's agronomic advisor.

The XGBoost model has produced a soil-to-crop prediction. 
Convert the raw JSON result into a clear, friendly explanation for a farmer.

Guidelines:
- Lead with the recommended crop and yield estimate.
- Briefly explain WHY this crop suits the conditions (use top features from SHAP).
- Mention the top 3 alternatives.
- Flag any warnings from the model.
- If detected_lang is "ur", respond in Urdu (Roman or Nastaliq).
- Keep it under 200 words.
"""

def soil_crop_node(state: AgroBotState) -> AgroBotState:
    """Run XGBoost prediction then narrate the result with the LLM."""
    soil_data = state.get("soil_data") or {}
    print("\n" + "="*50)
    print("📥 SOIL DATA RECEIVED BY NODE:")
    print(json.dumps(soil_data, indent=2))
    print("="*50 + "\n")
    # --- Run prediction ---
    if _AGRO_PREDICT_AVAILABLE and XGB_MODEL is not None:
        try:
            result = agro_predict(soil_data)
        except Exception as e:
            result = {"error": str(e)}
    else:
        result = _mock_predict(soil_data)

    # --- Narrate result ---
    llm = get_llm(temperature=0.4)
    narration_prompt = f"""
Prediction result (JSON):
{json.dumps(result, indent=2)}

User's original question:
{state['user_query']}

Detected language: {state['detected_lang']}
"""
    messages = [
        SystemMessage(content=CROP_INTERPRETER_SYSTEM),
        HumanMessage(content=narration_prompt),
    ]
    narrative = llm.invoke(messages).content

    return {
        **state,
        "prediction_result": result,
        "final_response": narrative,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  NODE: WEATHER
# ══════════════════════════════════════════════════════════════════════════════

WEATHER_SYSTEM = """You are AgroBot's weather-advisory agent for Pakistani farmers.

Given weather data (JSON) and the farmer's question, provide:
1. Current conditions summary (temp, humidity, wind, rain)
2. Agronomic implication (is it good for planting/irrigation/harvesting?)
3. One concrete actionable tip

If weather data is missing, use the search results instead.
If detected_lang is "ur", respond in Urdu.
Keep response under 150 words.
"""

def weather_node(state: AgroBotState) -> AgroBotState:
    """Improved weather node with better fallback and direct sources."""
    import requests
    from datetime import datetime

    owm_key = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()
    city = state.get("location") or extract_city(state["user_query"]) or "Lahore"
    weather_data = None
    print(owm_key, city )
    # 1. Try OpenWeatherMap first
    if owm_key:
        try:
            # Current weather (more reliable than forecast for basic info)
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={owm_key}&units=metric"
            resp = requests.get(url, timeout=8)
            
            if resp.status_code == 200:
                raw = resp.json()
                weather_data = {
                    "source": "OpenWeatherMap",
                    "city": raw["name"],
                    "country": raw["sys"].get("country"),
                    "current": {
                        "temp": round(raw["main"]["temp"], 1),
                        "feels_like": round(raw["main"]["feels_like"], 1),
                        "humidity": raw["main"]["humidity"],
                        "desc": raw["weather"][0]["description"].capitalize(),
                        "wind_speed": raw["wind"].get("speed", 0),
                        "pressure": raw["main"].get("pressure"),
                    },
                    "timestamp": datetime.now().isoformat()
                }
            elif resp.status_code == 401:
                print("⚠️ OpenWeatherMap: Invalid API key")
            elif resp.status_code == 429:
                print("⚠️ OpenWeatherMap: Rate limit exceeded")
            else:
                print(f"⚠️ OpenWeatherMap returned {resp.status_code}")
        except Exception as e:
            print(f"⚠️ OpenWeatherMap error: {e}")

    # 2. Stronger fallback using DuckDuckGo with better query
    if not weather_data:
        search_q = f"current weather {city} Pakistan right now"
        try:
            search_result = SEARCH_TOOL.run(search_q)
            weather_data = {
                "source": "web_search",
                "city": city,
                "search_snippet": search_result[:800]  # limit size
            }
        except Exception as e:
            weather_data = {"error": f"Search failed: {str(e)}"}

    # 3. Let LLM generate the response
    llm = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=WEATHER_SYSTEM),
        HumanMessage(content=(
            f"Weather data: {json.dumps(weather_data, ensure_ascii=False, indent=2)}\n"
            f"Farmer question: {state['user_query']}\n"
            f"Detected language: {state['detected_lang']}\n"
            f"Current date/time: {datetime.now().strftime('%Y-%m-%d %H:%M')} PKT"
        )),
    ]
    response = llm.invoke(messages).content

    return {
        **state,
        "weather_result": weather_data,
        "final_response": response,
    }


def extract_city(query: str) -> Optional[str]:
    """Simple city extractor for Pakistani context."""
    common_cities = ["Lahore", "Karachi", "Islamabad", "Multan", "Faisalabad", 
                    "Rawalpindi", "Peshawar", "Quetta", "Gujranwala", "Sialkot"]
    for city in common_cities:
        if city.lower() in query.lower():
            return city
    # Fallback: look for title-case words longer than 4 letters
    for token in query.split():
        if token.istitle() and len(token) > 4 and token not in ["What", "Today", "Should"]:
            return token
    return None

# ══════════════════════════════════════════════════════════════════════════════
# 7.  NODE: MARKET INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

MARKET_SYSTEM = """You are AgroBot's market intelligence agent for Pakistani farmers.

Given web search results about crop/commodity prices, provide:
1. Current approximate price range (PKR per 40 kg / maund)
2. Market trend (rising / stable / falling) if evident
3. Best time/place to sell advice
4. One risk to watch for

If detected_lang is "ur", respond in Urdu.
Keep it concise – under 150 words.
"""

def market_node(state: AgroBotState) -> AgroBotState:
    """Search DuckDuckGo for current commodity prices then narrate."""
    query = state["user_query"]

    # Enhance query for Pakistan market context
    search_q = f"Pakistan {query} price 2026 market mandi PKR"
    try:
        search_result = SEARCH_TOOL.run(search_q)
    except Exception as e:
        search_result = f"Search unavailable: {e}"

    llm = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=MARKET_SYSTEM),
        HumanMessage(content=(
            f"Search results:\n{search_result}\n\n"
            f"Farmer question: {query}\n"
            f"Detected language: {state['detected_lang']}"
        )),
    ]
    response = llm.invoke(messages).content

    return {
        **state,
        "market_result": search_result,
        "final_response": response,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  NODE: RAG AGRONOMY
# ══════════════════════════════════════════════════════════════════════════════

RAG_SYSTEM = """You are AgroBot's agronomic knowledge expert for Pakistani farmers.

Use the retrieved context as your primary source.
If the context does not answer the question, say that clearly and then give a cautious general agronomy answer.
Do not invent exact fertilizer doses, dates, or crop recommendations unless the context supports them.
Keep response under 200 words.
If detected_lang is "ur", respond in Urdu.
"""

def rag_node(state: AgroBotState) -> AgroBotState:
    """Retrieve from ChromaDB (or fallback to DuckDuckGo) then answer."""
    query = state["user_query"]
    context = ""

    retrieved_docs = []

    if VECTOR_STORE is not None:
        try:
            retrieved_docs = VECTOR_STORE.similarity_search(query, k=4)
        except Exception as e:
            print(f"⚠️  Retrieval error: {e}")
            retrieved_docs = []

    if retrieved_docs:
        print("\n" + "="*60)
        print("📚 RAG RETRIEVAL RESULTS:")
        for i, doc in enumerate(retrieved_docs, 1):
          print(f"\n--- Doc {i} ---")
        print("Source:", doc.metadata.get("source"))
        print(doc.page_content[:200])
        print("="*60 + "\n")
        parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            src = doc.metadata.get("source", f"chunk_{i}")
            parts.append(f"[Source: {src}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(parts)
    else:
        search_q = f"Pakistan agriculture {query} advice"
        try:
            context = SEARCH_TOOL.run(search_q)
        except Exception:
            context = "No relevant context found."

    llm = get_llm(temperature=0.4)
    messages = [
        SystemMessage(content=RAG_SYSTEM),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Detected language: {state['detected_lang']}"
        )),
    ]
    response = llm.invoke(messages).content

    return {
        **state,
        "rag_result": context[:1200] + "..." if len(context) > 1200 else context,
        "final_response": response,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9.  NODE: GENERAL FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

GENERAL_SYSTEM = """You are AgroBot, a friendly AI assistant for Pakistani farmers.

Answer helpfully. If the question is out of scope (not agriculture-related),
politely redirect to farming topics.
If detected_lang is "ur", respond in Urdu.
Keep response under 120 words.
"""

def general_node(state: AgroBotState) -> AgroBotState:
    """Handle greetings, off-topic queries, and edge cases."""
    llm = get_llm(temperature=0.5)
    messages = [
        SystemMessage(content=GENERAL_SYSTEM),
        HumanMessage(content=(
            f"User message: {state['user_query']}\n"
            f"Detected language: {state['detected_lang']}"
        )),
    ]
    response = llm.invoke(messages).content

    return {
        **state,
        "final_response": response,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10.  NODE: RESPONSE FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def response_formatter_node(state: AgroBotState) -> AgroBotState:
    """
    Append the final response to the message history and optionally
    add a structured footer (source badge, disclaimer).
    """
    response_text = state.get("final_response", "I'm sorry, I could not generate a response.")
    intent        = state.get("intent", "general")

    # Add a lightweight source badge
    badge_map = {
        "soil_crop":    "🌱 *Powered by AgroBot XGBoost Crop Engine*",
        "weather":      "🌤️ *Powered by OpenWeatherMap + AgroBot Advisory*",
        "market":       "📊 *Market data via web search – verify before trading*",
        "rag_agronomy": "📚 *AgroBot Knowledge Base + Llama-3*",
        "general":      "🤖 *AgroBot Assistant*",
    }
    badge = badge_map.get(intent, "🤖 *AgroBot*")
    full_response = f"{response_text}\n\n{badge}"

    return {
        **state,
        "final_response": full_response,
        "messages": state["messages"] + [AIMessage(content=full_response)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 11.  ROUTING FUNCTION  (conditional edge from supervisor)
# ══════════════════════════════════════════════════════════════════════════════

def route_intent(
    state: AgroBotState,
) -> Literal["soil_crop", "weather", "market", "rag_agronomy", "general"]:
    return state.get("intent", "general")


# ══════════════════════════════════════════════════════════════════════════════
# 12.  BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_agrobot_graph() -> StateGraph:
    """Assemble and compile the LangGraph agent."""

    graph = StateGraph(AgroBotState)

    # --- Add nodes ---
    graph.add_node("supervisor",          supervisor_node)
    graph.add_node("soil_crop",           soil_crop_node)
    graph.add_node("weather",             weather_node)
    graph.add_node("market",              market_node)
    graph.add_node("rag_agronomy",        rag_node)
    graph.add_node("general",             general_node)
    graph.add_node("response_formatter",  response_formatter_node)

    # --- Entry point ---
    graph.set_entry_point("supervisor")

    # --- Conditional routing after supervisor ---
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

    # --- All specialist nodes → formatter → END ---
    for node in ["soil_crop", "weather", "market", "rag_agronomy", "general"]:
        graph.add_edge(node, "response_formatter")

    graph.add_edge("response_formatter", END)

    return graph.compile()


# Module-level compiled graph (import-ready)
agrobot_graph = build_agrobot_graph()


# ══════════════════════════════════════════════════════════════════════════════
# 13.  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_agrobot(user_message: str, history: list = None) -> str:
    """
    Main entry point for the AgroBot agent.

    Args:
        user_message : The farmer's query (English or Urdu).
        history      : Optional list of prior (HumanMessage | AIMessage) objects
                       for multi-turn conversation.

    Returns:
        The agent's final response as a string.
    """
    initial_state: AgroBotState = {
        "messages":          history or [],
        "user_query":        user_message,
        "detected_lang":     "en",
        "intent":            "",
        "soil_data":         None,
        "prediction_result": None,
        "weather_result":    None,
        "market_result":     None,
        "rag_result":        None,
        "final_response":    None,
    }

    result = agrobot_graph.invoke(initial_state)
    return result["final_response"]


# ══════════════════════════════════════════════════════════════════════════════
# 14.  CLI TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🌾 AgroBot Interactive CLI (type 'exit' to quit)")
    print("=" * 60)

    history = []

    while True:
        user_input = input("\n👨‍🌾 You: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting AgroBot...")
            break

        # Run agent
        response = run_agrobot(user_input, history=history)

        # Print response
        print(f"\n🤖 AgroBot: {response}")

        # Update history manually
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))