"""
AgroBot — Language Detection & Config
=======================================
Unicode script-based language detection (fast, no API call).
Extracted from MultiLingualChatBot.py and made standalone.
"""

from config import LANGUAGE_CONFIG, LANG_INSTRUCTION


def detect_language_unicode(text: str, whisper_code: str = "") -> str:
    """
    Detect language code from Unicode script ranges.

    Returns one of: "en" | "ur" | "pa" | "skr"

    Rules:
      • Gurmukhi (U+0A00–U+0A7F) > 25%       → "pa"
      • Arabic   (U+0600–U+06FF) > 25%
          – Saraiki chars ݨ(U+0768) / ݙ(U+0759)  → "skr"
          – otherwise                              → "ur"
      • Latin ASCII alpha > 30%                 → "en"
      • Fallback on Whisper language code / default "en"
    """
    if not text.strip():
        return whisper_code if whisper_code in LANGUAGE_CONFIG else "en"

    gurmukhi = sum(1 for c in text if '\u0A00' <= c <= '\u0A7F')
    arabic   = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin    = sum(1 for c in text if c.isascii() and c.isalpha())
    total    = max(len(text.replace(" ", "")), 1)

    if gurmukhi / total > 0.25:
        return "pa"
    if arabic / total > 0.25:
        # Check for Saraiki-specific Unicode characters first
        saraiki_chars = '\u0768\u0759\u0671\u0681\u0691\u06BA\u06BB\u06BC\u06BD\u06BE\u06BF'
        saraiki_char_hits = sum(1 for c in text if c in saraiki_chars)
        
        # Keyword-based detection for Shahmukhi (Punjabi) and Saraiki
        punjabi_keywords = [
            "نیں", "تے", "دی", "دا", "دے", "نوں", "سی", "اے", "وچ", "کٹک", 
            "کد", "کتھے", "ساتھوں", "تہاڈا", "تواڈا", "ساڈا", "تسی", "اسی",
            "کنک", "چٹے", "بوٹا", "بوٹے", "کدوں", "ایتھے", "اوتھے", "کدر", "ہویا"
        ]
        saraiki_keywords = ["ہن", "ہئی", "کوں", "توں", "وی", "اچ", "تہاڈا", "میڈا", "تیڈا", "ساڈا"]
        
        text_lower = " " + text.lower() + " "
        punjabi_hits = sum(1 for kw in punjabi_keywords if f" {kw} " in text_lower)
        saraiki_hits = sum(1 for kw in saraiki_keywords if f" {kw} " in text_lower)

        if saraiki_char_hits > 0 or saraiki_hits > punjabi_hits + 1:
            return "skr"
        if punjabi_hits > 0:
            return "pa"
            
        return "ur"
    if latin / total > 0.30:
        return "en"

    # Fallback to Whisper's detected language code
    _map = {"en": "en", "ur": "ur", "pa": "pa"}
    return _map.get(whisper_code, "en")


def get_lang_label(lang: str) -> str:
    """Return human-readable label for a language code."""
    return LANGUAGE_CONFIG.get(lang, {}).get("label", "English")


def get_lang_flag(lang: str) -> str:
    """Return flag emoji for a language code."""
    return LANGUAGE_CONFIG.get(lang, {}).get("flag", "🌐")


def get_lang_instruction(lang: str) -> str:
    """Return the LLM language enforcement instruction string."""
    return LANG_INSTRUCTION.get(lang, LANG_INSTRUCTION["en"])


def get_tts_config(lang: str) -> dict:
    """Return TTS config (edge_voice, gtts_lang) for a language code."""
    cfg = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG["en"])
    return {
        "edge_voice": cfg.get("edge_voice", "en-US-JennyNeural"),
        "gtts_lang":  cfg.get("gtts_lang", "en"),
        "label":      cfg.get("label", "English"),
    }


def is_rtl(lang: str) -> bool:
    """Returns True if the language is right-to-left."""
    return LANGUAGE_CONFIG.get(lang, {}).get("dir", "ltr") == "rtl"
