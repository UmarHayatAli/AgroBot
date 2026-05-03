"""
AgroBot — Soil & Crop Recommendation Feature
=============================================
Wraps the existing XGBoost model (Backend/XGBoostClassifierAndYeildPredictor.py)
with a LangChain NL → structured input parser and Groq LLM response narrator.

Flow:
  User NL query → LLM parser → soil_params dict → agro_predict() → LLM narrator → text response
"""

import sys
import json
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import GROQ_API_KEY, LLM_MODEL
from core.language import get_lang_instruction
from features.base_tool import BaseAgriTool
from prompts.templates import SOIL_PARSER_TEMPLATE, SOIL_INTERPRETER_TEMPLATE

# ── Add Backend to path so we can import the predictor ───────────────────────
BACKEND_DIR = Path(__file__).parent.parent / "Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Try to load real XGBoost predictor; fall back to mock
try:
    from XGBoostClassifierAndYeildPredictor import agro_predict
    _REAL_PREDICTOR = True
except Exception as e:
    _REAL_PREDICTOR = False
    print(f"⚠️  Real XGBoost predictor unavailable ({e}). Using mock.")


def _mock_predict(soil_data: dict) -> dict:
    """Realistic stub when model files are missing."""
    return {
        "recommended_crop": "Wheat",
        "predicted_yield_tons_per_acre": 2.65,
        "suitability": "Good 🟢",
        "confidence": 87.3,
        "top_3_crops": [("Wheat", 87.3), ("Maize", 8.1), ("Rice", 4.6)],
        "similar_cases": [],
        "feature_importance": {},
        "validation_warnings": None,
        "note": "[Demo mode] Real prediction requires model files in Backend/Models/",
    }


class SoilRecommendationTool(BaseAgriTool):
    """
    LangChain Tool — Precision Soil-to-Crop Recommendation.
    Uses XGBoost classifier + yield predictor with SHAP explainability.
    """

    name        = "soil_recommendation"
    description = (
        "Use this tool when the user asks about soil analysis, crop recommendations, "
        "EC (electrical conductivity), salinity, pH levels, NPK, or what crop to plant. "
        "Input: natural language describing soil conditions or explicit values."
    )

    def _get_llm(self, temperature: float = 0.3) -> ChatGroq:
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY not set.")
        return ChatGroq(model=LLM_MODEL, temperature=temperature, api_key=GROQ_API_KEY)

    def _parse_soil_params(self, query: str, lang: str) -> dict:
        """Use Groq LLM to extract structured soil parameters from NL query."""
        lang_instr = get_lang_instruction(lang)
        try:
            llm      = self._get_llm(temperature=0)
            chain    = SOIL_PARSER_TEMPLATE | llm
            response = chain.invoke({"query": query, "language": lang_instr})
            content  = response.content.strip()

            # Extract JSON from the response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            params = json.loads(content)
            # Filter out null values
            return {k: v for k, v in params.items() if v is not None}
        except Exception as e:
            print(f"⚠️  Soil param extraction failed: {e}")
            return {}

    def _narrate_result(self, prediction: dict, query: str, lang: str, **kwargs) -> str:
        """Convert raw XGBoost JSON into a farmer-friendly narrative."""
        lang_instr = get_lang_instruction(lang)
        try:
            llm   = self._get_llm(temperature=0.4)
            history = kwargs.get("history", [])
            resp  = llm.invoke([
                SystemMessage(content=SOIL_INTERPRETER_TEMPLATE.format(
                    prediction_json=json.dumps(prediction, indent=2, ensure_ascii=False),
                    query=query,
                    language=lang_instr
                )),
                *history,
                HumanMessage(content=query)
            ])
            return resp.content.strip()
        except Exception as e:
            # Graceful fallback: format the result manually
            top = prediction.get("recommended_crop", "Wheat")
            conf = prediction.get("confidence", 0)
            yield_est = prediction.get("predicted_yield_tons_per_acre", 0)
            return (
                f"Based on your soil conditions, the best crop is: **{top}** "
                f"(confidence: {conf}%). Expected yield: {yield_est} tons/acre."
            )

    def run(
        self,
        query: str,
        lang: str = "en",
        image_path: Optional[str] = None,
        history: Optional[list] = None,
        **kwargs,
    ) -> dict:
        """
        Main entry point.

        1. Parse NL query → soil params dict
        2. Run XGBoost prediction
        3. Narrate result in user's language
        """
        # Step 1: Extract soil parameters
        soil_params = self._parse_soil_params(query, lang)
        print(f"📊 Extracted soil params: {soil_params}")

        # Step 2: Run prediction
        if _REAL_PREDICTOR:
            try:
                prediction = agro_predict(soil_params)
                if "error" in prediction:
                    print(f"⚠️  Prediction error: {prediction['error']}. Using mock.")
                    prediction = _mock_predict(soil_params)
            except Exception as e:
                print(f"⚠️  XGBoost failed: {e}. Using mock.")
                prediction = _mock_predict(soil_params)
        else:
            prediction = _mock_predict(soil_params)

        # Step 3: Generate narrative response
        text = self._narrate_result(prediction, query, lang, history=history)

        return {
            "text":   text,
            "intent": self.name,
            "lang":   lang,
            "extra": {
                "soil_params": soil_params,
                "prediction":  prediction,
                "top_3_crops": prediction.get("top_3_crops", []),
                "confidence":  prediction.get("confidence", 0),
                "yield_est":   prediction.get("predicted_yield_tons_per_acre", 0),
                "suitability": prediction.get("suitability", ""),
                "warnings":    prediction.get("validation_warnings"),
            },
        }
