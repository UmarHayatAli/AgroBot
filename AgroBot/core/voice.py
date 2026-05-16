"""
AgroBot — Voice I/O (STT + TTS)
=================================
STT:  sounddevice mic recording → Groq Whisper API transcription
TTS:  Edge TTS (neural, primary) → gTTS (Urdu-optimized fallback)

NOTE: Requires Python 3.11 (faster-whisper compatibility).
      On Windows, audio playback uses os.startfile for MP3 files.
"""

import os
import sys
import asyncio
import tempfile
import subprocess
import queue
from pathlib import Path

from config import (
    GROQ_API_KEY, WHISPER_MODEL, RECORD_SECONDS, SAMPLE_RATE,
    AUDIO_CHANNELS, LANGUAGE_CONFIG
)
from core.language import detect_language_unicode, get_tts_config


# ── STT: Mic recording ────────────────────────────────────────────────────────

def record_audio(sample_rate: int = SAMPLE_RATE) -> bytes | None:
    """
    Record audio from the default microphone dynamically until the user presses Enter.
    Returns WAV bytes, or None if mic is unavailable.
    """
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wav
        import numpy as np
    except ImportError:
        print("⚠️  sounddevice/scipy/numpy not installed. Falling back to text input.")
        return None

    try:
        q = queue.Queue()
        recording = True

        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            if recording:
                q.put(indata.copy())

        # Start background stream
        stream = sd.InputStream(samplerate=sample_rate, channels=AUDIO_CHANNELS, dtype='int16', callback=callback)
        with stream:
            input()  # Blocks here until user presses Enter
            recording = False

        # Drain the queue
        data = []
        while not q.empty():
            data.append(q.get())

        if not data:
            return None

        # Stitch all chunks together
        audio = np.concatenate(data, axis=0)

        # Write to in-memory WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav.write(tmp.name, sample_rate, audio)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()

        os.unlink(tmp_path)
        return wav_bytes

    except Exception as e:
        print(f"⚠️  Mic recording failed: {e}")
        return None


# ── STT: Groq Whisper transcription ──────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, audio_ext: str = "wav") -> tuple[str, str]:
    """
    Transcribe audio bytes using Groq Whisper API.

    Returns:
        (transcript, whisper_lang_code)
        Falls back to ("", "en") on failure.
    """
    if not audio_bytes:
        return "", "en"

    if not GROQ_API_KEY:
        print("⚠️  GROQ_API_KEY not set — cannot transcribe audio.")
        return "", "en"

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        # Pass bytes directly with a filename tuple to avoid disk I/O
        file_tuple = (f"audio.{audio_ext}", audio_bytes)
        
        result = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=file_tuple,
            response_format="verbose_json",
        )
        whisper_lang = getattr(result, "language", "en") or "en"
        transcript   = (result.text or "").strip()

        return transcript, whisper_lang

    except Exception as e:
        print(f"⚠️  Whisper transcription failed: {e}")
        return "", "en"


def record_and_transcribe() -> tuple[str, str]:
    """
    High-level: record mic → transcribe → return (text, lang_code).
    Returns ("", "en") if any step fails (caller falls back to text input).
    """
    audio_bytes = record_audio()
    if not audio_bytes:
        return "", "en"

    transcript, whisper_lang = transcribe_audio(audio_bytes)
    if not transcript:
        return "", "en"

    # Unicode detection is more reliable than Whisper's lang code
    lang = detect_language_unicode(transcript, whisper_lang)
    return transcript, lang



async def async_speak_stream(text: str, lang: str = "en"):
    """
    Generate TTS audio as an async stream of bytes.
    This provides 'Time To First Byte' (TTFB) optimization.
    """
    if not text.strip():
        return

    cfg = get_tts_config(lang)
    voice = cfg["edge_voice"]

    # Trim text if absolutely necessary, but edge-tts handles long text well
    tts_text = text[:1000] if len(text) > 1000 else text

    try:
        import edge_tts
        communicate = edge_tts.Communicate(tts_text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    except Exception as e:
        print(f"⚠️  Streaming Edge TTS failed: {e}")
        # Fallback to a non-streaming byte response if needed (handled by caller)
        return


def speak(text: str, lang: str = "en", play: bool = True) -> bytes | None:
    """
    Convert text to speech and optionally play it.

    Priority:
      1. Edge TTS (neural voices, high quality)
      2. gTTS (Google TTS — good Urdu support)
      3. pyttsx3 (offline fallback)

    Args:
        text:  Text to speak.
        lang:  Language code ("en", "ur", "pa", "skr").
        play:  Whether to play the audio immediately.

    Returns:
        MP3 bytes if audio was generated, None otherwise.
    """
    if not text.strip():
        return None

    cfg     = get_tts_config(lang)
    voice   = cfg["edge_voice"]
    gtts_lg = cfg["gtts_lang"]

    # Trim text for voice (under 500 chars works best)
    tts_text = text[:500] if len(text) > 500 else text

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        out_path = tmp.name

    mp3_bytes = None

    try:
        # ── Attempt 1: Edge TTS ──────────────────────────────────────────
        try:
            asyncio.run(_edge_tts_to_file(tts_text, voice, out_path))
        except Exception as edge_err:
            print(f"⚠️  Edge TTS failed ({edge_err}), trying gTTS...")


            # ── Attempt 2: gTTS ──────────────────────────────────────────
            try:
                from gtts import gTTS
                gTTS(text=tts_text, lang=gtts_lg, slow=False).save(out_path)
            except Exception as gtts_err:
                print(f"⚠️  gTTS failed ({gtts_err}), trying pyttsx3...")

                # ── Attempt 3: pyttsx3 (offline) ─────────────────────────
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.save_to_file(tts_text, out_path.replace(".mp3", ".wav"))
                    engine.runAndWait()
                    out_path = out_path.replace(".mp3", ".wav")
                except Exception as pyttsx3_err:
                    print(f"⚠️  All TTS methods failed: {pyttsx3_err}")
                    return None

        # Read generated audio bytes
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "rb") as f:
                mp3_bytes = f.read()

            # Play audio
            if play and mp3_bytes:
                _play_audio(out_path)

    finally:
        # Cleanup temp file after playback (small delay)
        try:
            if os.path.exists(out_path) and not play:
                os.unlink(out_path)
        except Exception:
            pass

    return mp3_bytes


def _play_audio(file_path: str) -> None:
    """Play audio file using the best available method for Windows."""
    try:
        if sys.platform == "win32":
            os.startfile(file_path)
        elif sys.platform == "darwin":
            subprocess.run(["afplay", file_path], check=False)
        else:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", file_path],
                           check=False, capture_output=True)
    except Exception as e:
        print(f"⚠️  Audio playback failed: {e} (audio was generated but not played)")


def save_audio(mp3_bytes: bytes, directory: str = ".", filename: str = "response.mp3") -> str | None:
    """Save MP3 bytes to a file. Returns the saved path or None."""
    if not mp3_bytes:
        return None
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        with open(path, "wb") as f:
            f.write(mp3_bytes)
        return path
    except Exception as e:
        print(f"⚠️  Failed to save audio: {e}")
        return None
