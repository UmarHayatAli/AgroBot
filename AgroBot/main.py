"""
AgroBot — Main CLI Entry Point
================================
Architecture: Option A — Unified Bot with LangGraph Intent Router.

WHY OPTION A: Pakistani smallholder farmers should never need to navigate menus
or choose a "section." A single conversational agent that understands Urdu, English,
Punjabi, and Saraiki and automatically routes to the right feature (soil, weather,
disease, flood, general) creates a natural, phone-conversation-like experience.
LangGraph's StateGraph pattern keeps all features modular and UI-ready — a Gradio
or Streamlit frontend can be dropped on top with zero logic changes.

Run:   python main.py
"""

import os
import sys
import time
from pathlib import Path

# ── Windows UTF-8 fix (emoji support in terminal) ─────────────────────────────
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ── Add project root to sys.path for clean imports ───────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Dependency check ─────────────────────────────────────────────────────────
def _check_deps():
    missing = []
    for pkg, import_name in [("rich", "rich"), ("langchain_groq", "langchain_groq"),
                              ("groq", "groq"), ("dotenv", "dotenv")]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

_check_deps()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns
from rich.rule import Rule
from rich import box
from rich.prompt import Prompt, Confirm
from rich.markup import escape

from config import check_groq_key, get_missing_config, RISK_COLORS
from core.language import (
    detect_language_unicode, get_lang_label, get_lang_flag, get_tts_config
)
from core.memory import ConversationMemory
from core.router import route_query, intent_label

# ── Feature imports ───────────────────────────────────────────────────────────
from features.soil_recommendation import SoilRecommendationTool
from features.weather_insights    import WeatherInsightsTool
from features.disease_detector    import DiseaseDetectorTool
from features.flood_alert         import FloodAlertTool

# ── Rich console ──────────────────────────────────────────────────────────────
console = Console()

# ── Feature instances (singletons) ───────────────────────────────────────────
FEATURES = {
    "soil_crop":    SoilRecommendationTool(),
    "weather":      WeatherInsightsTool(),
    "disease":      DiseaseDetectorTool(),
    "flood":        FloodAlertTool(),
}


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def show_welcome_banner():
    """Display the animated AgroBot welcome banner using rich."""
    console.print()
    banner = Text()
    banner.append("  █████╗  ██████╗ ██████╗  ██████╗ ██████╗  ██████╗ ████████╗\n", style="bold green")
    banner.append(" ██╔══██╗██╔════╝ ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝\n", style="bold green")
    banner.append(" ███████║██║  ███╗██████╔╝██║   ██║██████╔╝██║   ██║   ██║   \n", style="bold bright_green")
    banner.append(" ██╔══██║██║   ██║██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║   \n", style="bold bright_green")
    banner.append(" ██║  ██║╚██████╔╝██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║   \n", style="bold green")
    banner.append(" ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝   ╚═╝   \n", style="bold green")

    subtitle = Text()
    subtitle.append("          🌾  ", style="bold yellow")
    subtitle.append("آپ کا زرعی مشیر  —  Your Agricultural Advisor  🌾", style="bold cyan")

    console.print(Panel(
        Text.assemble(banner, "\n", subtitle),
        border_style="bold green",
        padding=(0, 2),
    ))
    console.print()

    # Info row
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("", style="bold yellow", no_wrap=True)
    info_table.add_column("", style="white")
    info_table.add_row("🌍 Languages:", "English · اردو · ਪੰਜਾਬੀ · سرائیکی")
    info_table.add_row("🤖 Powered by:", "Groq Llama 3.3 · XGBoost · Open-Meteo")
    info_table.add_row("🎯 Features:", "Soil · Weather · Disease · Flood · General")
    console.print(Panel(info_table, border_style="dim green", title="[bold green]System Ready[/]"))
    console.print()


def show_mode_menu() -> str:
    """Ask the user to choose input mode. Returns '1', '2', or '3'."""
    console.print(Panel(
        "[bold cyan]Choose Input Mode:[/]\n\n"
        "  [bold green]\\[1][/]  ⌨️   Type a question  (fastest)\n"
        "  [bold green]\\[2][/]  🎤  Speak into mic    (Urdu/English/Punjabi/Saraiki)\n"
        "  [bold green]\\[3][/]  📷  Upload image + question  (disease detection)\n"
        "  [bold green]\\[q][/]  ❌  Quit",
        title="[bold green]🌾 AgroBot[/]",
        border_style="green",
        padding=(1, 2),
    ))
    choice = Prompt.ask(
        "[bold green]Select[/]",
        choices=["1", "2", "3", "q"],
        default="1",
    )
    return choice


def show_spinner(message: str = "AgroBot soch raha hai... 🌱"):
    """Return a rich Progress spinner context manager."""
    return Progress(
        SpinnerColumn(style="bold green"),
        TextColumn(f"[bold green]{message}[/]"),
        transient=True,
        console=console,
    )


def render_soil_response(result: dict):
    """Rich-formatted soil/crop recommendation response."""
    extra = result.get("extra", {}) or {}
    top3  = extra.get("top_3_crops", [])
    warn  = extra.get("warnings")
    suit  = extra.get("suitability", "")
    yield_est = extra.get("yield_est", 0)

    # Main response panel
    console.print(Panel(
        escape(result["text"]),
        title="[bold green]🌱 Crop Recommendation[/]",
        border_style="green",
        padding=(1, 2),
    ))

    # Top 3 crops table
    if top3:
        table = Table(title="Top Crop Recommendations", box=box.ROUNDED,
                      header_style="bold green", border_style="dim green")
        table.add_column("Rank", style="bold yellow", width=6, justify="center")
        table.add_column("Crop", style="bold white", width=15)
        table.add_column("Confidence", style="cyan", width=15, justify="right")
        table.add_column("Suitability", style="white", width=20)

        for i, (crop_name, conf) in enumerate(top3[:3], 1):
            rank_emoji = ["🥇", "🥈", "🥉"][i - 1]
            suit_label = suit if i == 1 else ""
            table.add_row(rank_emoji, str(crop_name), f"{conf:.1f}%", suit_label)

        console.print(table)

    # Warnings
    if warn:
        for w in (warn if isinstance(warn, list) else [warn]):
            console.print(f"  [bold yellow]⚠️  {escape(str(w))}[/]")
    console.print()


def render_weather_response(result: dict):
    """Rich-formatted weather response."""
    extra    = result.get("extra", {}) or {}
    city     = extra.get("city", "")
    weather  = extra.get("weather", {})
    forecast = weather.get("7_day_forecast", [])
    total_r  = extra.get("total_rain_7d", 0)

    console.print(Panel(
        escape(result["text"]),
        title=f"[bold cyan]🌤️ Weather Insights — {city}[/]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # 7-day forecast table
    if forecast:
        table = Table(title=f"7-Day Forecast — {city}", box=box.SIMPLE_HEAD,
                      header_style="bold cyan", border_style="dim cyan")
        table.add_column("Date", style="bold white", width=12)
        table.add_column("Max °C", style="bold red", width=8, justify="right")
        table.add_column("Min °C", style="bold blue", width=8, justify="right")
        table.add_column("Rain mm", style="bold cyan", width=10, justify="right")
        table.add_column("Condition", style="white", width=22)

        for day in forecast[:7]:
            rain_mm = day.get("rain_mm", 0)
            rain_style = "bold red" if rain_mm >= 50 else ("yellow" if rain_mm >= 25 else "white")
            table.add_row(
                day.get("date", ""),
                str(day.get("max_temp_c", "?")),
                str(day.get("min_temp_c", "?")),
                Text(f"{rain_mm:.0f}", style=rain_style),
                day.get("condition", ""),
            )

        console.print(table)
        console.print(f"  [cyan]Total rain in 7 days: {total_r:.0f}mm[/]\n")
    console.print()


def render_disease_response(result: dict):
    """Rich-formatted disease detection response."""
    extra    = result.get("extra", {}) or {}
    disease  = extra.get("disease", "")
    crop     = extra.get("crop", "")
    conf     = extra.get("confidence", "")
    is_healthy = extra.get("is_healthy", False)
    symptoms = extra.get("symptoms", "")

    if is_healthy:
        console.print(Panel(
            escape(result["text"]),
            title="[bold green]✅ Plant Health Check[/]",
            border_style="bright_green",
            padding=(1, 2),
        ))
    else:
        # Disease header box
        if disease:
            conf_color = {"High": "bold red", "Medium": "yellow", "Low": "dim yellow"}.get(conf, "white")
            disease_panel = (
                f"[bold white]Crop:[/] [cyan]{escape(crop)}[/]   "
                f"[bold white]Disease:[/] [bold red]{escape(disease)}[/]   "
                f"[bold white]Confidence:[/] [{conf_color}]{escape(conf)}[/{conf_color}]"
            )
            if symptoms:
                disease_panel += f"\n[dim]Symptoms: {escape(symptoms[:100])}[/]"

            console.print(Panel(
                disease_panel,
                title="[bold red]🔬 Disease Diagnosis[/]",
                border_style="bold red",
                padding=(1, 2),
            ))

        # Treatment plan
        console.print(Panel(
            escape(result["text"]),
            title="[bold yellow]💊 Treatment Plan[/]",
            border_style="yellow",
            padding=(1, 2),
        ))
    console.print()


def render_flood_response(result: dict):
    """Rich-formatted flood alert response with color-coded risk."""
    extra      = result.get("extra", {}) or {}
    risk_level = extra.get("risk_level", "Low")
    crop       = extra.get("crop", "")
    city       = extra.get("city", "")
    max_rain   = extra.get("max_rain_mm", 0)
    total_rain = extra.get("total_rain_7d", 0)
    threat_days = extra.get("threat_days", [])
    immed_actions = extra.get("immediate_actions", [])

    risk_color = RISK_COLORS.get(risk_level, "white")
    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Critical": "🚨"}.get(risk_level, "⚪")

    # Risk level badge
    console.print(Panel(
        f"[{risk_color}] {risk_emoji}  FLOOD RISK: {risk_level.upper()}  {risk_emoji} [/{risk_color}]\n\n"
        f"[white]Crop:[/] [cyan]{escape(crop)}[/]  "
        f"[white]Location:[/] [cyan]{escape(city)}[/]\n"
        f"[white]Peak rain:[/] [{'bold red' if max_rain >= 50 else 'yellow'}]{max_rain:.0f}mm/day[/]  "
        f"[white]7-day total:[/] [yellow]{total_rain:.0f}mm[/]",
        title="[bold red]🌊 Flood Risk Assessment[/]",
        border_style=risk_color,
        padding=(1, 2),
    ))

    # Threat window
    if threat_days:
        console.print(f"  [bold red]⚠️  High-risk dates: {', '.join(threat_days[:5])}[/]")

    # Response text
    console.print(Panel(
        escape(result["text"]),
        title="[bold yellow]🚒 Emergency Protocol[/]",
        border_style="yellow",
        padding=(1, 2),
    ))

    # Immediate actions (if extra data available)
    if immed_actions:
        console.print(Rule("[bold yellow]Action Checklist[/]", style="yellow"))
        for i, action in enumerate(immed_actions, 1):
            console.print(f"  [bold yellow]{i}.[/] {escape(action)}")
    console.print()


def render_general_response(result: dict):
    """Default response panel for general queries."""
    console.print(Panel(
        escape(result["text"]),
        title="[bold green]🤖 AgroBot[/]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()


def render_response(result: dict):
    """Route rendering to the correct formatter based on intent."""
    intent = result.get("intent", "general")
    lang   = result.get("lang", "en")
    flag   = get_lang_flag(lang)
    label  = get_lang_label(lang)

    # Language badge
    console.print(f"  [dim]{flag} Responding in {label}[/]")

    renderer_map = {
        "soil_recommendation": render_soil_response,
        "weather_insights":    render_weather_response,
        "disease_detector":    render_disease_response,
        "flood_alert":         render_flood_response,
    }
    renderer = renderer_map.get(intent, render_general_response)
    renderer(result)


# ══════════════════════════════════════════════════════════════════════════════
# INPUT HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def handle_text_input() -> tuple[str, str, str | None]:
    """Get text input from user. Returns (text, lang, image_path)."""
    query = Prompt.ask("\n[bold green]👨‍🌾 You[/]").strip()
    if not query:
        return "", "en", None
    lang = detect_language_unicode(query)
    return query, lang, None


def handle_voice_input() -> tuple[str, str, str | None]:
    """Record mic audio and transcribe. Returns (text, lang, image_path)."""
    try:
        from core.voice import record_and_transcribe
    except ImportError as e:
        console.print(f"[yellow]⚠️  Voice module unavailable: {e}. Switching to text.[/]")
        return handle_text_input()

    console.print(Panel(
        "[bold cyan]🎤 Mic ready! Speak now...[/]\n"
        "[dim]Press ENTER to stop recording — speak in Urdu, English, Punjabi, or Saraiki[/]",
        border_style="cyan",
        padding=(0, 2),
    ))

    transcript, lang = record_and_transcribe()

    if not transcript:
        console.print("[yellow]⚠️  Could not transcribe audio. Switching to text input.[/]")
        return handle_text_input()

    flag = get_lang_flag(lang)
    label = get_lang_label(lang)
    console.print(Panel(
        f"[bold white]{escape(transcript)}[/]\n[dim]{flag} {label} detected[/]",
        title="[bold cyan]🎤 Transcribed[/]",
        border_style="cyan",
        padding=(0, 2),
    ))
    return transcript, lang, None


def handle_image_input() -> tuple[str, str, str | None]:
    """Get image path + optional text/voice query. Returns (text, lang, image_path)."""
    console.print(Panel(
        "[bold yellow]📷 Image Upload Mode[/]\n"
        "[dim]Enter the full path to your crop image (JPG, PNG, BMP)[/]",
        border_style="yellow",
        padding=(0, 2),
    ))
    image_path = Prompt.ask("[bold yellow]Image path[/]").strip().strip('"').strip("'")

    if not image_path or not Path(image_path).exists():
        console.print(f"[red]❌ Image not found: {image_path}[/]")
        image_path = None

    # Sub-mode: text or voice description
    console.print("\n[bold yellow]Add a description (text or voice)?[/]")
    desc_mode = Prompt.ask("Mode", choices=["text", "voice", "skip"], default="text")

    if desc_mode == "text":
        query = Prompt.ask("[bold green]Describe the problem[/]").strip()
        lang  = detect_language_unicode(query)
    elif desc_mode == "voice":
        query, lang, _ = handle_voice_input()
    else:
        query = "Please analyze this crop image for any disease or problem."
        lang  = "en"

    return query, lang, image_path


# ══════════════════════════════════════════════════════════════════════════════
# CORE DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

def dispatch(
    query: str,
    lang: str,
    image_path: str | None,
    memory: ConversationMemory,
) -> dict:
    """
    Route query to the correct feature and return a result dict.
    Uses the intent router (fast keyword matching + LLM fallback).
    """
    history   = memory.get_messages()
    has_image = bool(image_path)

    # Detect intent & language from query script
    route = route_query(query, history, has_image=has_image)
    intent = route.intent
    
    # Priority: Always respect the explicitly passed UI language parameter.
    # Auto-detection is only used as a fallback if lang is missing.
    lang = lang or route.detected_lang or "en"
    
    location = route.location
    crop     = route.detected_crop

    console.print(f"  [dim]Intent: {intent_label(intent)} | Lang: {lang} | Location: {location or '?'}[/]")

    feature = FEATURES.get(intent)

    if feature:
        res = feature.run(
            query=query,
            lang=lang,
            image_path=image_path,
            location=location,
            crop=crop,
        )
        res["context"] = {"location": location, "crop": crop}
        return res

    # General / RAG fallback — use LLM directly via MultiLingualChatBot
    try:
        backend_dir = PROJECT_ROOT / "Backend"
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        from MultiLingualChatBot import run_agrobot
        result = run_agrobot(user_message=query, history=history)
        return {
            "text":   result.get("text", ""),
            "intent": result.get("intent", "general"),
            "lang":   result.get("lang", lang),
            "extra":  None,
            "context": {"location": location, "crop": crop}
        }
    except Exception as e:
        print(f"⚠️  General fallback LLM failed: {e}")
        return {
            "text":   "I'm sorry, I couldn't process your request. Please try again.",
            "intent": "general",
            "lang":   lang,
            "extra":  None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def play_response_audio(text: str, lang: str):
    """Convert response to speech and play it."""
    try:
        from core.voice import speak
        # Strip markdown formatting for TTS
        clean_text = (text
            .replace("**", "").replace("*", "").replace("__", "")
            .replace("##", "").replace("#", "").replace("---", "")
            .replace("```", ""))
        # Keep it short for voice
        if len(clean_text) > 400:
            clean_text = clean_text[:400] + "..."
        speak(clean_text, lang=lang, play=True)
    except Exception as e:
        console.print(f"[dim yellow]⚠️  TTS failed: {e}[/]")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """AgroBot main CLI loop."""
    show_welcome_banner()

    # Check configuration
    missing = get_missing_config()
    if missing:
        console.print(Panel(
            f"[bold red]⚠️  Missing required configuration:[/]\n"
            f"  {', '.join(missing)}\n\n"
            "[yellow]Create a .env file with your GROQ_API_KEY.\n"
            "See .env.example for the template.[/]",
            border_style="red",
            title="[bold red]Configuration Error[/]",
        ))
        sys.exit(1)

    console.print("[bold green]✅ All systems ready![/]\n")

    memory     = ConversationMemory(max_turns=10)
    turn_count = 0

    while True:
        turn_count += 1

        # ── Input mode selection ──────────────────────────────────────────
        if turn_count == 1 or Confirm.ask("\n[dim]Change input mode?[/]", default=False):
            mode = show_mode_menu()
        if mode == "q":
            break

        # ── Get input ─────────────────────────────────────────────────────
        query, lang, image_path = "", "en", None

        if mode == "1":
            query, lang, image_path = handle_text_input()
        elif mode == "2":
            query, lang, image_path = handle_voice_input()
        elif mode == "3":
            query, lang, image_path = handle_image_input()

        if not query:
            console.print("[yellow]No input received. Try again.[/]")
            continue

        if query.lower() in ["exit", "quit", "q", "bye", "خداحافظ", "فی امان اللہ"]:
            break

        # ── Process query ─────────────────────────────────────────────────
        with show_spinner("AgroBot soch raha hai... 🌱") as progress:
            task = progress.add_task("thinking", total=None)
            try:
                result = dispatch(query, lang, image_path, memory)
            except Exception as e:
                progress.stop()
                console.print(f"[bold red]❌ Error: {escape(str(e))}[/]")
                continue

        # ── Render response ───────────────────────────────────────────────
        render_response(result)

        # ── Auto-play TTS ─────────────────────────────────────────────────
        response_text = result.get("text", "")
        response_lang = result.get("lang", lang)

        if response_text:
            tts_choice = Confirm.ask(
                f"[dim]🔊 Play audio response in {get_lang_label(response_lang)}?[/]",
                default=False,
            )
            if tts_choice:
                with show_spinner("Generating audio... 🔊"):
                    play_response_audio(response_text, response_lang)

        # ── Update memory ─────────────────────────────────────────────────
        memory.add_exchange(query, response_text)

        # ── Continue? ─────────────────────────────────────────────────────
        continue_labels = {
            "en":  "Do you have another question?",
            "ur":  "کیا آپ کا کوئی اور سوال ہے؟",
            "pa":  "ਕੀ ਤੁਹਾਡਾ ਕੋਈ ਹੋਰ ਸਵਾਲ ਹੈ?",
            "skr": "کیا تہاڈا کوئی ہور سوال ہے؟",
        }
        label = continue_labels.get(response_lang, continue_labels["en"])
        if not Confirm.ask(f"\n[bold green]{label}[/]", default=True):
            break

    # ── Goodbye ───────────────────────────────────────────────────────────
    goodbyes = {
        "en":  "Thank you for using AgroBot. Good luck with your harvest! 🌾",
        "ur":  "AgroBot استعمال کرنے کا شکریہ۔ آپ کی فصل خوب ہو! 🌾",
        "pa":  "AgroBot ਵਰਤਣ ਦਾ ਧੰਨਵਾਦ। ਤੁਹਾਡੀ ਫਸਲ ਚੰਗੀ ਹੋਵੇ! 🌾",
        "skr": "AgroBot ورتن دا شکریہ۔ تہاڈی فصل چنگی ہووے! 🌾",
    }
    final_lang = result.get("lang", "en") if "result" in dir() else "en"
    console.print(Panel(
        f"[bold green]{goodbyes.get(final_lang, goodbyes['en'])}[/]",
        border_style="green",
        padding=(1, 2),
    ))


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]👋 AgroBot shutting down...[/]")
