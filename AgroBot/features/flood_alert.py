"""
AgroBot — Predictive Flood & Risk Alert System
================================================
Rule-based risk engine cross-referencing crop vulnerability data
with Open-Meteo weather forecast flood indicators.

Flow:
  Crop + Location → weather forecast → risk calculation → LLM emergency protocol
  
Risk levels: Low | Medium | High | Critical
"""

import json
from datetime import date
from typing import Optional

from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY, LLM_MODEL, FLOOD_RISK_JSON,
    FLOOD_RAIN_THRESHOLDS, RISK_COLORS
)
from core.language import get_lang_instruction
from features.base_tool import BaseAgriTool
from features.weather_insights import WeatherInsightsTool
from prompts.templates import FLOOD_RISK_TEMPLATE


def _load_flood_db() -> dict:
    try:
        with open(FLOOD_RISK_JSON, encoding="utf-8") as f:
            return json.load(f).get("crops", {})
    except Exception:
        return {}

FLOOD_DB = _load_flood_db()


# Sensitivity score mapping
SENSITIVITY_SCORE = {
    "low":      1,
    "medium":   2,
    "high":     3,
    "very_high": 4,
}


def _calculate_risk_level(
    max_daily_rain_mm: float,
    total_7day_rain_mm: float,
    sensitivity: str,
) -> str:
    """
    Compute risk level from rainfall amounts and crop sensitivity.
    Returns: "Low" | "Medium" | "High" | "Critical"
    """
    sens_score = SENSITIVITY_SCORE.get(sensitivity, 2)

    # Base risk from rainfall
    if max_daily_rain_mm >= FLOOD_RAIN_THRESHOLDS["critical"]:
        rain_level = 4
    elif max_daily_rain_mm >= FLOOD_RAIN_THRESHOLDS["high"]:
        rain_level = 3
    elif max_daily_rain_mm >= FLOOD_RAIN_THRESHOLDS["medium"]:
        rain_level = 2
    elif max_daily_rain_mm >= FLOOD_RAIN_THRESHOLDS["low"]:
        rain_level = 1
    else:
        rain_level = 0

    # Combined risk score
    combined = rain_level + (sens_score - 1)

    if combined >= 5:
        return "Critical"
    elif combined >= 3:
        return "High"
    elif combined >= 2:
        return "Medium"
    else:
        return "Low"


def _find_threat_window(forecast_days: list) -> list[str]:
    """Return dates where rain > medium threshold (25mm)."""
    threshold = FLOOD_RAIN_THRESHOLDS["medium"]
    return [
        d["date"] for d in forecast_days
        if d.get("rain_mm", 0) >= threshold
    ]


class FloodAlertTool(BaseAgriTool):
    """
    LangChain Tool — Predictive Flood & Risk Alert System.
    Generates color-coded risk level + emergency action protocol.
    """

    name        = "flood_alert"
    description = (
        "Use this when the user asks about flood risk, waterlogging danger, "
        "heavy rain and crop safety, or needs emergency flood protocols. "
        "Input: crop type + location."
    )

    def _get_crop_info(self, crop: str) -> dict:
        """Return flood vulnerability data for a crop."""
        crop_lower = crop.lower().strip()
        for key in FLOOD_DB:
            if key in crop_lower or crop_lower in key:
                return FLOOD_DB[key]
        return FLOOD_DB.get("default", {
            "sensitivity": "medium",
            "max_waterlogging_hours": 36,
            "critical_stages": ["all stages"],
            "immediate_actions": [
                "Open all field drainage channels immediately",
                "Remove standing water as quickly as possible",
                "Contact local agriculture extension officer",
            ],
            "emergency_chemical": "Consult local agriculture department",
            "recovery_advice":    "Monitor crop carefully for 7 days after flooding.",
        })

    def _extract_crop(self, query: str) -> str:
        """Extract crop name from query."""
        crops = [
            "cotton", "wheat", "rice", "maize", "sugarcane",
            "sunflower", "potato", "onion", "mango", "sorghum",
            "کپاس", "گیہوں", "چاول", "مکئی", "گنا",
        ]
        query_lower = query.lower()
        for crop in crops:
            if crop.lower() in query_lower:
                return crop
        return "wheat"  # default Pakistan crop

    def run(
        self,
        query: str,
        lang: str = "en",
        image_path: Optional[str] = None,
        location: Optional[str] = None,
        crop: Optional[str] = None,
        **kwargs,
    ) -> dict:
        # Extract crop and location
        detected_crop = crop or self._extract_crop(query)
        crop_info     = self._get_crop_info(detected_crop)

        # Get weather data via WeatherInsightsTool
        weather_tool  = WeatherInsightsTool()
        weather_result = weather_tool.run(query=query, lang="en", location=location)

        weather_extra = weather_result.get("extra", {})
        weather_data  = weather_extra.get("weather", {})
        forecast_days = weather_data.get("7_day_forecast", [])
        total_rain    = weather_data.get("total_rain_7days_mm", 0)
        max_rain_day  = weather_extra.get("max_rain_day", {})
        max_rain_mm   = max_rain_day.get("rain_mm", 0)
        city          = weather_extra.get("city", location or "Unknown location")

        # Calculate risk
        sensitivity  = crop_info.get("sensitivity", "medium")
        risk_level   = _calculate_risk_level(max_rain_mm, total_rain, sensitivity)
        threat_days  = _find_threat_window(forecast_days)
        max_wl_hours = crop_info.get("max_waterlogging_hours", 36)
        critical_stg = crop_info.get("critical_stages", [])
        immed_actions= crop_info.get("immediate_actions", [])

        # Build weather summary for LLM
        weather_summary = {
            "city":          city,
            "max_daily_rain_mm": max_rain_mm,
            "total_7day_rain_mm": total_rain,
            "high_rain_days": threat_days,
            "7_day_forecast": forecast_days,
        }

        # Generate emergency protocol with LLM
        lang_instr = get_lang_instruction(lang)
        try:
            llm   = ChatGroq(model=LLM_MODEL, temperature=0.3, api_key=GROQ_API_KEY)
            chain = FLOOD_RISK_TEMPLATE | llm
            resp  = chain.invoke({
                "crop":           detected_crop.title(),
                "location":       city,
                "weather_json":   json.dumps(weather_summary, indent=2, ensure_ascii=False),
                "sensitivity":    sensitivity,
                "max_hours":      max_wl_hours,
                "critical_stages": ", ".join(critical_stg),
                "language":       lang_instr,
            })
            llm_text = resp.content.strip()
        except Exception as e:
            print(f"⚠️  Flood LLM failed: {e}. Using rule-based fallback.")
            # Manual fallback
            actions_text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(immed_actions))
            llm_text = (
                f"RISK LEVEL: {risk_level}\n\n"
                f"Rain forecast: {max_rain_mm}mm peak day, {total_rain}mm over 7 days.\n"
                f"Threat dates: {', '.join(threat_days) or 'None'}\n\n"
                f"Immediate Actions:\n{actions_text}"
            )

        return {
            "text":   llm_text,
            "intent": self.name,
            "lang":   lang,
            "extra": {
                "risk_level":     risk_level,
                "risk_color":     RISK_COLORS.get(risk_level, "white"),
                "crop":           detected_crop.title(),
                "city":           city,
                "max_rain_mm":    max_rain_mm,
                "total_rain_7d":  total_rain,
                "threat_days":    threat_days,
                "sensitivity":    sensitivity,
                "immediate_actions": immed_actions,
                "recovery_advice":   crop_info.get("recovery_advice", ""),
                "emergency_chemical": crop_info.get("emergency_chemical", ""),
            },
        }
