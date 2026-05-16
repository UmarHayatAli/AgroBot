# AgroBot 🌾 — Multilingual Agricultural AI Assistant

> **Smart agricultural intelligence for Pakistani smallholder farmers.**  
> Supports English · اردو · ਪੰਜਾਬੀ · سرائیکی — voice in, voice out.

---

## 🟢 Current Status: PRODUCTION-READY

All core features are implemented, tested, and verified. No active blockers.

| Module | Status | Notes |
|--------|--------|-------|
| Intent Router | ✅ Complete | Priority: Flood > Disease > Soil > Weather > General |
| Soil & Crop Prediction | ✅ Complete | XGBoost + LLM narration in 4 languages |
| Weather Insights | ✅ Complete | **New:** Flood & Heatwave alerts + Zero-Hindi Urdu output |
| Disease Detection | ✅ Complete | Groq Vision → Kindwise → keyword fallback pipeline |
| Flood Alert | ✅ Complete | Emergency routing with priority override |
| Voice Input (STT) | ✅ Complete | **New:** Dynamic push-to-talk (no time limit, press Enter to stop) |
| Voice Output (TTS) | ✅ Complete | Edge TTS with regional Pakistani voices |
| Multilingual (4 langs) | ✅ Complete | Urdu/Punjabi/Saraiki/English — auto-detected from script |

---

## Architecture

```
User Input (Text / Voice / Image)
        │
        ▼
  [Intent Router] ── Unicode lang detect + keyword match + Groq LLM
        │   Priority: Flood > Disease > Soil > Weather > General
        │
        ├──► [Soil Recommendation]   XGBoost + SHAP + Groq narrator
        ├──► [Weather Insights]      Open-Meteo API + Extreme Risk Engine + Groq advisor
        ├──► [Disease Detection]     Groq Vision + LLM treatment plan
        ├──► [Flood Alert]           Risk engine + emergency protocol
        └──► [General / RAG]         LangGraph MultiLingualChatBot
                │
                ▼
       [Rich CLI Output]  +  [Edge TTS / gTTS Audio]
```

**Architecture Choice: Option A — Unified Bot with Intent Router**  
One conversational agent automatically routes to the right feature. Farmers don't navigate menus — they just ask in their language.

---

## Setup

### 1. Prerequisites
- Python **3.11** (required for compatibility)
- Windows / Linux / macOS

### 2. Clone & install
```bash
cd AgroBot
pip install -r requirements.txt
```

### 3. Configure API keys
```bash
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get your **free** Groq API key at: https://console.groq.com/

### 4. Run AgroBot
```bash
python main.py
```

### 5. Validate all modules
```bash
python validate.py
```

---

## Project Structure

```
AgroBot/
├── main.py                    ← CLI entry point (run this)
├── config.py                  ← All constants & API config
├── validate.py                ← Module health checker
├── requirements.txt
├── .env.example               ← Copy to .env
│
├── core/
│   ├── language.py            ← Unicode language detection
│   ├── voice.py               ← STT (Groq Whisper) + TTS (Edge TTS) [Dynamic Push-to-Talk]
│   ├── router.py              ← Intent detection (priority-ordered)
│   └── memory.py              ← Conversation history
│
├── features/
│   ├── base_tool.py           ← Abstract base class
│   ├── soil_recommendation.py ← XGBoost crop predictor
│   ├── weather_insights.py    ← Open-Meteo + Extreme Risk Engine (Flood/Heatwave/Monsoon)
│   ├── disease_detector.py    ← Groq Vision + treatment plan
│   └── flood_alert.py         ← Flood risk engine
│
├── data/
│   ├── disease_treatments.json ← 14 common disease lookup table
│   └── flood_risk_crops.json   ← Crop flood vulnerability data
│
├── prompts/
│   ├── system_prompt_en.txt
│   ├── system_prompt_ur.txt
│   └── templates.py            ← All LLM prompt templates (strict Zero-Hindi rules)
│
└── Backend/                   ← Existing ML models (untouched)
    ├── MultiLingualChatBot.py ← LangGraph core agent
    ├── XGBoostClassifierAndYeildPredictor.py
    ├── DiseasePredictorModel.py
    ├── weather.py              ← Collaborator's Gradio version (reference only)
    └── Models/                ← pkl files (XGBoost, scaler, etc.)
```

---

## Features

| Feature | Technology | Input | Output |
|---------|-----------|-------|--------|
| 🌱 Soil & Crop | XGBoost + SHAP | Soil params (pH, EC, NPK) | Top 3 crops + yield |
| 🌤️ Weather | Open-Meteo + Risk Engine | City name | 7-day forecast + Heatwave/Flood alerts |
| 🔬 Disease | Groq Vision (llama-4-scout) | Image + description | Disease + treatment plan |
| 🌊 Flood Alert | Rule engine + Open-Meteo | Crop + location | Risk level + emergency actions |
| 💬 General | LangGraph + Llama 3.3 | Any question | Agricultural advice |

---

## Key Technical Decisions

### 1. Intent Priority Order
`router.py` uses a strict detection hierarchy to prevent cross-contamination:
```
Flood → Disease → Soil → Weather → Market → General
```

### 2. Extreme Weather Detection (`weather_insights.py`)
Before every LLM call, `_check_extreme_risks()` silently checks:
- **Flood risk:** 7-day total rainfall vs city-specific threshold (e.g., 90mm for Lahore)
- **Heatwave:** Peak weekly temp vs city-specific limit (e.g., 45°C for Multan)
- **Monsoon countdown:** Warns if monsoon is within 5 days of historical start date

### 3. Zero-Hindi Urdu Rendering
Prompts in `templates.py` explicitly forbid Devanagari characters and mandate pure Perso-Arabic Urdu transliteration (e.g., "این پی کے" not "NPK").

### 4. Dynamic Voice Recording
`core/voice.py` uses `sounddevice.InputStream` with a background queue. Recording stops when the user presses **ENTER** — no time limit.

---

## Sample Interactions

### Test 1 — Urdu Soil Query
```
Input:  میری زمین کا pH 7.2 ہے اور EC 2.5 ہے، کون سی فصل لگاؤں؟
Output: [Urdu] گیہوں کی فصل سب سے موزوں ہے... (+ TTS in Urdu)
```

### Test 2 — English Weather + Heatwave Alert
```
Input:  What's the weather forecast for Multan this week?
Output: HEATWAVE ALERT: Temperatures reaching 42.1°C. Ensure proper irrigation...
```

### Test 3 — Image + Urdu (Disease Detection)
```
Input:  [image: tomato_leaf.jpg] "Is yeh patta kharab kyun ho raha hai?"
Output: [Urdu] آپ کی فصل میں Late Blight ہے... (treatment steps in Urdu)
```

### Test 4 — Flood Alert
```
Input:  Islamabad mein cotton hai, kya flood ka khatrah hai?
Output: RISK: Medium — Cotton is highly flood-sensitive...
```

### Test 5 — Punjabi (Shahmukhi script)
```
Input:  ملتان دا موسم کیہو جیہا اے؟
Output: [Punjabi/Urdu] لاہور میں اس ہفتے کا موسم گرم اور خشک...
```

---

## Voice Support

| Language | STT | TTS Voice |
|----------|-----|-----------|
| English | Groq Whisper | Edge TTS: en-US-JennyNeural |
| Urdu | Groq Whisper | Edge TTS: ur-PK-UzmaNeural |
| Punjabi | Groq Whisper (pa) | Edge TTS: ur-PK-AsadNeural |
| Saraiki | Groq Whisper (ur fallback) | Edge TTS: ur-PK-AsadNeural |

---

## APIs Used

| API | Purpose | Cost |
|-----|---------|------|
| [Groq](https://groq.com) | LLM (Llama 3.3) + Whisper STT + Vision (Llama 4 Scout) | Free tier |
| [Open-Meteo](https://open-meteo.com) | Weather forecast + geocoding | Free, no key |
| [Edge TTS](https://github.com/rany2/edge-tts) | Neural voice TTS | Free |

---

## Error Handling

- **No mic**: Falls back to text input automatically
- **No image**: Clear error message, asks for correct path
- **Groq API down**: Friendly error message with retry suggestion
- **Unknown language**: Defaults to English
- **XGBoost models missing**: Uses realistic mock prediction with note
- **Weather API down**: Graceful error, suggests internet check

---

## Future Improvements (Not Yet Implemented)

- **Gradio/Streamlit Frontend**: Backend `dispatch()` is fully decoupled — a web UI can be added with ~50 lines.
- **Historical Logging**: `ConversationMemory` class is ready to extend with crop yield logs.
- **More Languages**: Architecture supports any language Whisper supports.

---

## License

MIT — Free for educational and research use.

---

*Built for Pakistani farmers. اپنی زمین، اپنی فصل، اپنا مشیر۔*
Ha