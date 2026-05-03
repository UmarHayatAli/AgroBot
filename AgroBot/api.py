import os
import sys
from pathlib import Path
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, Form, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure the correct path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import dispatch, FEATURES
from core.memory import ConversationMemory

app = FastAPI(title="AgroBot API")

# Setup CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (mapping session_id -> ConversationMemory)
sessions: dict[str, ConversationMemory] = {}

@app.get("/")
def health_check():
    return {"status": "ok", "message": "AgroBot API is running"}

from fastapi.responses import StreamingResponse
from core.voice import async_speak_stream

@app.get("/api/tts")
async def tts_endpoint(text: str, lang: str = "en"):
    """
    Generate TTS audio as a stream for instant playback (TTFB optimization).
    """
    print(f"[API] TTS Request: text='{text[:20]}...', lang='{lang}'")
    try:
        return StreamingResponse(
            async_speak_stream(text, lang),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline"}
        )
    except Exception as e:
        print(f"[API] TTS Error: {e}")
        return {"error": str(e)}, 500

@app.post("/api/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    if not file or not file.filename:
        return {"error": "No file uploaded"}, 400
        
    ext = os.path.splitext(file.filename)[1]
    
    try:
        audio_bytes = await file.read()
            
        from core.voice import transcribe_audio
        transcript, whisper_lang = transcribe_audio(audio_bytes, audio_ext=ext.strip("."))
        
        # Determine strict language or let frontend handle it
        from core.language import detect_language_unicode
        lang = detect_language_unicode(transcript, whisper_lang)
        
        return {
            "text": transcript,
            "lang": lang
        }
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/api/chat")
async def chat_endpoint(
    query: str = Form(default=""),
    lang: str = Form(default="en"),
    session_id: str = Form(default=""),
    file: Optional[UploadFile] = File(None)
):
    if not session_id:
        session_id = str(uuid.uuid4())
        
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory(max_turns=10)
        
    memory = sessions[session_id]
    
    image_path = None
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            tf.write(await file.read())
            image_path = tf.name
            
    # Default queries if missing
    if not query:
        if image_path:
            query = "Please analyze this image."
        else:
            query = "Hello"
        
    try:
        # dispatch returns a dictionary like: 
        # {"text": str, "intent": str, "lang": str, "extra": dict}
        result = dispatch(query, lang, image_path, memory)
        
        response_text = result.get("text", "")
        if response_text:
            memory.add_exchange(query, response_text)
            
        # Update persistent context with newly detected location/crop
        ctx = result.get("context", {})
        if ctx:
            memory.update_context(location=ctx.get("location"), crop=ctx.get("crop"))
            
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            
        return {
            "session_id": session_id,
            "result": result
        }
    except Exception as e:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        return {"error": str(e)}

@app.get("/api/weather")
async def get_weather(city: str = "Multan", lang: str = "en"):
    try:
        from features.weather_insights import WeatherInsightsTool
        tool = WeatherInsightsTool()
        result = tool.run(query=f"weather in {city}", lang=lang, location=city)
        return result
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
