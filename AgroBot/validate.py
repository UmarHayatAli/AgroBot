"""Quick import validation script — run before demo."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

errors = []

def test(label, fn):
    try:
        result = fn()
        print(f"  OK  {label}" + (f" — {result}" if result else ""))
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        errors.append(label)

print("=" * 55)
print("AgroBot Import Validation")
print("=" * 55)

# Config
test("config.py", lambda: (
    __import__("config"),
    f"LLM={__import__('config').LLM_MODEL}"
)[1])

# Core
test("core.language", lambda: (
    __import__("core.language", fromlist=["detect_language_unicode"]),
    f"ur={__import__('core.language', fromlist=['detect_language_unicode']).detect_language_unicode('میری زمین کا pH')} | en={__import__('core.language', fromlist=['detect_language_unicode']).detect_language_unicode('what crop to plant')}"
)[1])

test("core.memory", lambda: (
    __import__("core.memory", fromlist=["ConversationMemory"]),
    "ConversationMemory OK"
)[1])

test("core.router", lambda: (
    __import__("core.router", fromlist=["route_query"]),
    "route_query OK"
)[1])

test("core.voice", lambda: (
    __import__("core.voice", fromlist=["speak"]),
    "voice module OK"
)[1])

# Features
test("features.base_tool", lambda: (
    __import__("features.base_tool", fromlist=["BaseAgriTool"]),
    "BaseAgriTool OK"
)[1])

test("features.soil_recommendation", lambda: (
    __import__("features.soil_recommendation", fromlist=["SoilRecommendationTool"]),
    "SoilRecommendationTool OK"
)[1])

test("features.weather_insights", lambda: (
    __import__("features.weather_insights", fromlist=["WeatherInsightsTool", "CITY_COORDS"]),
    f"{len(__import__('features.weather_insights', fromlist=['CITY_COORDS']).CITY_COORDS)} cities loaded"
)[1])

test("features.disease_detector", lambda: (
    __import__("features.disease_detector", fromlist=["DiseaseDetectorTool", "DISEASE_DB"]),
    f"{len(__import__('features.disease_detector', fromlist=['DISEASE_DB']).DISEASE_DB)} diseases in DB"
)[1])

test("features.flood_alert", lambda: (
    __import__("features.flood_alert", fromlist=["FloodAlertTool", "FLOOD_DB"]),
    f"{len(__import__('features.flood_alert', fromlist=['FLOOD_DB']).FLOOD_DB)} crops in flood DB"
)[1])

# Prompts
test("prompts.templates", lambda: (
    __import__("prompts.templates", fromlist=["SOIL_INTERPRETER_TEMPLATE"]),
    "templates OK"
)[1])

# Data files
import json, pathlib
def check_data():
    d1 = json.loads(pathlib.Path("data/disease_treatments.json").read_text(encoding="utf-8"))
    d2 = json.loads(pathlib.Path("data/flood_risk_crops.json").read_text(encoding="utf-8"))
    return f"diseases={len(d1['diseases'])} crops_flood={len(d2['crops'])}"
test("data files", check_data)

# XGBoost models
def check_models():
    import pathlib
    models_dir = pathlib.Path("Backend/Models")
    files = ["xgboost_model.pkl", "scaler.pkl", "label_encoder.pkl", "knn_model.pkl", "shap_explainer.pkl"]
    found = [f for f in files if (models_dir / f).exists()]
    return f"{len(found)}/5 model files present"
test("Backend/Models", check_models)

print()
print("=" * 55)
from config import GROQ_API_KEY
if errors:
    print(f"FAILED: {len(errors)} import(s) failed: {errors}")
else:
    print("ALL CHECKS PASSED!")
print(f"GROQ_API_KEY configured: {bool(GROQ_API_KEY)}")
if not GROQ_API_KEY:
    print("  >> Create .env file with GROQ_API_KEY=your_key")
print("=" * 55)
