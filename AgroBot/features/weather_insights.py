"""
AgroBot — Weather Insights Feature
=====================================
Uses Open-Meteo API (free, no API key) for 7-day weather forecast.
Geocodes city name via Open-Meteo geocoding API (also free).
Groq LLM converts weather data into actionable farming advice.

Flow:
  City name → geocode → lat/lon → Open-Meteo forecast → LLM advisor → response
"""

import json
import requests
from datetime import datetime, date, timedelta
from typing import Optional

from langchain_groq import ChatGroq

from config import GROQ_API_KEY, LLM_MODEL, OPEN_METEO_BASE, OPEN_METEO_GEO
from core.language import get_lang_instruction
from features.base_tool import BaseAgriTool
from prompts.templates import WEATHER_ADVISOR_TEMPLATE

# Pakistani city coordinates fallback (when geocoding fails)
CITY_COORDS = {
    "lahore":      (31.5497, 74.3436),
    "karachi":     (24.8607, 67.0011),
    "islamabad":   (33.6844, 73.0479),
    "rawalpindi":  (33.5651, 73.0169),
    "multan":      (30.1978, 71.4711),
    "faisalabad":  (31.4180, 73.0790),
    "peshawar":    (34.0150, 71.5249),
    "quetta":      (30.1798, 66.9750),
    "gujranwala":  (32.1877, 74.1945),
    "sialkot":     (32.4945, 74.5229),
    "hyderabad":   (25.3792, 68.3683),
    "bahawalpur":  (29.3956, 71.6836),
    "sargodha":    (32.0836, 72.6711),
    "dera ghazi khan": (30.0461, 70.6339),
    "sukkur":      (27.7052, 68.8574),
    "larkana":     (27.5570, 68.2122),
    "nawabshah":   (26.2442, 68.4100),
    "rahim yar khan": (28.4202, 70.2952),
    "sheikhupura": (31.7167, 73.9850),
    "jhang":       (31.2681, 72.3181),
}

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

# CLIMATE & EXTREME WEATHER THRESHOLDS
PAK_CLIMATE_DATA = {
    "Karachi": {"monsoon_start": (7, 5), "flood_threshold_mm": 40, "heatwave_c": 40},
    "Hyderabad": {"monsoon_start": (7, 5), "flood_threshold_mm": 50, "heatwave_c": 42},
    "Multan": {"monsoon_start": (7, 10), "flood_threshold_mm": 80, "heatwave_c": 45},
    "Lahore": {"monsoon_start": (7, 15), "flood_threshold_mm": 90, "heatwave_c": 43},
    "Faisalabad": {"monsoon_start": (7, 15), "flood_threshold_mm": 90, "heatwave_c": 45},
    "Peshawar": {"monsoon_start": (7, 20), "flood_threshold_mm": 70, "heatwave_c": 41},
    "Islamabad": {"monsoon_start": (7, 1), "flood_threshold_mm": 120, "heatwave_c": 40},
}
DEFAULT_CLIMATE = {"monsoon_start": (7, 15), "flood_threshold_mm": 80, "heatwave_c": 42}


class WeatherInsightsTool(BaseAgriTool):
    """
    LangChain Tool — Dynamic Weather Insight Integration.
    Provides 7-day forecast + farming action advice.
    """

    name        = "weather_insights"
    description = (
        "Use this when the user asks about weather, rainfall, temperature, "
        "when to plant or irrigate, or flood risk from rain. "
        "Input: location (city name) + optional date range."
    )

    def _geocode(self, city: str) -> tuple[float, float] | None:
        """Geocode a city name to (lat, lon). Returns None on failure."""
        city_lower = city.lower().strip()

        # Check hardcoded fallback first
        for key, coords in CITY_COORDS.items():
            if key in city_lower or city_lower in key:
                return coords

        import os
        owm_key = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()
        if not owm_key:
            return None

        # Try OpenWeatherMap geocoding API
        try:
            url = f"http://api.openweathermap.org/geo/1.0/direct?q={city},PK&limit=1&appid={owm_key}"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    return (data[0]["lat"], data[0]["lon"])
        except Exception as e:
            print(f"⚠️  OWM Geocoding failed: {e}")

        return None

    def _fetch_forecast(self, lat: float, lon: float) -> dict | None:
        """Fetch 5-day forecast from OpenWeatherMap."""
        import os
        owm_key = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()
        if not owm_key:
            print("⚠️ OPENWEATHERMAP_API_KEY is not set.")
            return None

        try:
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={owm_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"⚠️  OpenWeatherMap API returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"⚠️  OpenWeatherMap API exception: {e}")
        return None

    def _format_forecast(self, raw: dict, city: str) -> dict:
        """Convert OpenWeatherMap raw JSON into a clean farming-friendly dict."""
        from collections import defaultdict, Counter

        list_data = raw.get("list", [])
        
        daily_agg = defaultdict(lambda: {
            "max_temp_c": -999.0,
            "min_temp_c": 999.0,
            "rain_mm": 0.0,
            "conditions": [],
            "rain_probs": []
        })

        for item in list_data:
            dt_txt = item.get("dt_txt", "").split(" ")[0]
            if not dt_txt: continue
            
            temp = item["main"]["temp"]
            rain = item.get("rain", {}).get("3h", 0)
            cond = item["weather"][0]["description"].capitalize() if item.get("weather") else "Unknown"
            pop  = item.get("pop", 0) * 100
            
            day = daily_agg[dt_txt]
            day["max_temp_c"] = max(day["max_temp_c"], temp)
            day["min_temp_c"] = min(day["min_temp_c"], temp)
            day["rain_mm"] += rain
            day["conditions"].append(cond)
            day["rain_probs"].append(pop)

        days = []
        for dt_txt, day in sorted(daily_agg.items()):
            most_common_cond = Counter(day["conditions"]).most_common(1)[0][0] if day["conditions"] else "Unknown"
            days.append({
                "date": dt_txt,
                "max_temp_c": round(day["max_temp_c"], 1),
                "min_temp_c": round(day["min_temp_c"], 1),
                "rain_mm": round(day["rain_mm"], 1),
                "condition": most_common_cond,
                "rain_prob_%": round(max(day["rain_probs"] or [0])),
            })

        current = list_data[0] if list_data else {}
        cur_main = current.get("main", {})
        cur_weather = current.get("weather", [{}])[0]
        
        return {
            "city": city,
            "source": "OpenWeatherMap",
            "current": {
                "temp_c": round(cur_main.get("temp", 0), 1),
                "rain_mm": round(current.get("rain", {}).get("3h", 0), 1),
                "humidity": cur_main.get("humidity", 0),
                "condition": cur_weather.get("description", "Unknown").capitalize(),
            },
            "7_day_forecast": days, # Actually 5-day but we keep key for downstream compatibility
            "total_rain_7days_mm": round(sum(d["rain_mm"] for d in days), 1),
            "max_rain_day": max(days, key=lambda d: d["rain_mm"]) if days else {},
            "max_temp_7days_c": max((d["max_temp_c"] for d in days), default=0),
        }

    def _check_extreme_risks(self, city: str, total_rain_mm: float, max_temp_c: float) -> list[str]:
        """Check for flood and heatwave risks based on climate thresholds."""
        city_key = city.title() if city.title() in PAK_CLIMATE_DATA else "Lahore"
        climate = PAK_CLIMATE_DATA.get(city_key, DEFAULT_CLIMATE)
        alerts = []

        if total_rain_mm >= climate["flood_threshold_mm"]:
            alerts.append(f"🌊 SEVERE FLOOD/WATERLOGGING RISK: {total_rain_mm}mm of rain expected over 7 days.")

        if max_temp_c >= climate["heatwave_c"]:
            alerts.append(f"🔥 HEATWAVE ALERT: Temperatures reaching {max_temp_c}°C. High risk of crop burning and rapid evaporation.")

        today = datetime.now()
        monsoon_date = datetime(today.year, climate["monsoon_start"][0], climate["monsoon_start"][1])
        if today > monsoon_date + timedelta(days=30):
            monsoon_date = datetime(today.year + 1, climate["monsoon_start"][0], climate["monsoon_start"][1])

        days_to_monsoon = (monsoon_date - today).days
        if 0 <= days_to_monsoon <= 5:
            alerts.append(f"⛈️ MONSOON WARNING: Monsoon season begins in {days_to_monsoon} days.")

        return alerts

    def _get_llm(self) -> ChatGroq:
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY not set.")
        return ChatGroq(model=LLM_MODEL, temperature=0.3, api_key=GROQ_API_KEY)

    def _extract_city(self, query: str) -> str:
        """Extract city name from query using known city list, or return default."""
        query_lower = query.lower()
        for city in CITY_COORDS:
            if city in query_lower:
                return city.title()
        return "Lahore"

    def run(
        self,
        query: str,
        lang: str = "en",
        image_path: Optional[str] = None,
        location: Optional[str] = None,
        history: Optional[list] = None,
        **kwargs,
    ) -> dict:
        city = location or self._extract_city(query)

        # Geocode
        coords = self._geocode(city)
        if not coords:
            # Graceful fallback — use Lahore
            print(f"⚠️  Could not geocode '{city}', defaulting to Lahore.")
            city   = "Lahore"
            coords = CITY_COORDS["lahore"]

        lat, lon = coords

        # Fetch forecast
        raw_forecast = self._fetch_forecast(lat, lon)
        if not raw_forecast:
            error_msgs = {
                "en": f"Sorry, I couldn't fetch weather data for {city} right now. Please check your internet connection and try again.",
                "ur": f"معذرت، {city} کا موسمی ڈیٹا ابھی دستیاب نہیں۔ انٹرنیٹ چیک کریں اور دوبارہ کوشش کریں۔",
                "pa": f"معاف کرنا، {city} دا موسم دا ڈیٹا نہیں ملیا۔ اپنا انٹرنیٹ چیک کرو۔",
                "skr": f"معافی، {city} دا موسم دا ڈیٹا کائنی ملیا۔ اپنا انٹرنیٹ چیک کرو۔",
            }
            return self._error_response(lang, error_msgs.get(lang, error_msgs["en"]))

        weather_dict = self._format_forecast(raw_forecast, city)

        # Check extreme weather risks
        total_rain = weather_dict.get("total_rain_7days_mm", 0)
        max_temp = weather_dict.get("max_temp_7days_c", 0)
        alerts = self._check_extreme_risks(city, total_rain, max_temp)
        risk_alerts_str = " | ".join(alerts) if alerts else "No critical disaster alerts."

        # Generate farming advice with LLM
        lang_instr = get_lang_instruction(lang)
        try:
            llm   = self._get_llm()
            prompt_msgs = WEATHER_ADVISOR_TEMPLATE.invoke({
                "location":    city,
                "weather_json": json.dumps(weather_dict, indent=2, ensure_ascii=False),
                "query":       query,
                "date":        date.today().strftime("%A, %d %B %Y"),
                "language":    lang_instr,
                "risk_alerts": risk_alerts_str,
            })
            
            messages = [
                prompt_msgs.to_messages()[0],
                *(history or []),
                prompt_msgs.to_messages()[1]
            ]
            
            print("Invoking LLM...")
            resp  = llm.invoke(messages)
            print("LLM returned!")
            text = resp.content.strip()
        except Exception as e:
            print(f"⚠️  LLM weather narration failed: {e}")
            # Manual fallback
            cur = weather_dict.get("current", {})
            fc  = weather_dict.get("7_day_forecast", [])
            text = (
                f"{city} weather: {cur.get('temp_c', '?')}°C, "
                f"{cur.get('condition', '')}. "
                f"Total rain next 7 days: {weather_dict.get('total_rain_7days_mm', 0)}mm."
            )

        return {
            "text":   text,
            "intent": self.name,
            "lang":   lang,
            "extra": {
                "city":          city,
                "lat":           lat,
                "lon":           lon,
                "weather":       weather_dict,
                "total_rain_7d": weather_dict.get("total_rain_7days_mm", 0),
                "max_rain_day":  weather_dict.get("max_rain_day", {}),
            },
        }
