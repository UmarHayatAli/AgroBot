# ── AGROBOT: PROFESSIONAL FIELD DIAGNOSTICS ──────────────────
import requests
import base64
import io
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

# ── CONFIGURATION ──────────────────────────────────────────────
# Keys loaded from .env file — never hardcode API keys in source code
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
KINDWISE_API_KEY = os.getenv("KINDWISE_API_KEY", "")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
KINDWISE_URL = "https://crop.kindwise.com/api/v1/identification?details=description,treatment,cause"

# ── UTILITIES ──────────────────────────────────────────────────
def encode_image(image_bytes):
    """Converts image bytes to base64 string for API transmission."""
    return base64.b64encode(image_bytes).decode("utf-8")

# ── AGROBOT BRAIN: LLAMA 3.3 (GROQ) ───────────────────────────
def generate_treatment_plan(crop_name, disease_name):
    """Generates a professional field report for agricultural management."""
    
    # SYSTEM PROMPT: Strictly forbids emojis and sets professional tone
    system_msg = (
        "You are a professional Agronomist providing a technical field report. "
        "Do not use emojis. Use clear, formal language and structured sections."
    )
    
    # USER PROMPT: Focused on clarity and immediate action
    user_prompt = f"""
    TECHNICAL FIELD REPORT
    Crop Subject: {crop_name}
    Diagnosed Condition: {disease_name}

    Please provide a concise management strategy using the following structure:
    1. CONDITION OVERVIEW: A brief technical summary of how this affects the crop.
    2. IMMEDIATE INTERVENTION: Critical steps to take within 48 hours to stop spread.
    3. CHEMICAL MANAGEMENT: Recommended active ingredients, dosage guidance, and application timing.
    4. LONG-TERM PREVENTION: Cultural practices and soil health strategies to prevent recurrence.

    Formatting: Use bold headers and bullet points. No conversational filler.
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2, # Lowered for maximum factual reliability
        "max_tokens": 1024
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        report = response.json()["choices"][0]["message"]["content"]
        
        print("\n" + "─" * 60)
        print(" AGROBOT: FIELD TREATMENT STRATEGY")
        print("─" * 60)
        print(report)
        print("─" * 60 + "\n")

    except Exception as e:
        print(f"Error: Failed to generate agronomy report. {e}")

# ── AGROBOT EYES: KINDWISE DETECTION ─────────────────────────
def analyze_crop_image(file_path):
    """Reads a local file, identifies disease, and initiates reporting."""
    
    # Read the image file into bytes
    try:
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Display preview for confirmation using the OS default image viewer
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.show()

    img_b64 = encode_image(image_bytes)

    print(f"Analyzing {os.path.basename(file_path)}...")

    try:
        response = requests.post(
            KINDWISE_URL,
            headers={
                "Api-Key": KINDWISE_API_KEY,
                "Content-Type": "application/json"
            },
            json={"images": [img_b64]},
            timeout=30
        )
        
        if response.status_code not in [200, 201]:
            print(f"Network Error: {response.status_code}")
            return

        data = response.json()
        result = data.get("result", {})
        
        # Internal extraction of crop for LLM context (not printed)
        crop_suggestions = result.get("crop", {}).get("suggestions", [])
        crop_name = crop_suggestions[0].get("name", "Crop").title() if crop_suggestions else "Crop"

        # Disease Diagnosis extraction
        disease_suggestions = result.get("disease", {}).get("suggestions", [])
        
        if not disease_suggestions:
            print("Status: No specific disease detected. Crop appears healthy.")
            return

        top_disease = disease_suggestions[0]
        disease_name = top_disease.get("name", "Unknown Condition").title()
        confidence = round(top_disease.get("probability", 0) * 100, 1)

        print(f"Diagnosis: {disease_name}")
        print(f"Accuracy: {confidence}%")

        # Triggering the LLM Brain
        generate_treatment_plan(crop_name, disease_name)

    except Exception as e:
        print(f"Error: Analysis failed. {e}")

# ── MAIN EXECUTION ───────────────────────────────────────────
if __name__ == "__main__":
    print("AgroBot Diagnostics System Online.")
    print("Please select a leaf image in the pop-up window...")

    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Force tkinter dialog to stay on top
    root.attributes('-topmost', True)

    try:
        # Open standard OS file dialog
        file_path = filedialog.askopenfilename(
            title="Select Crop Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )

        if file_path:
            analyze_crop_image(file_path)
        else:
            print("Upload Cancelled: No file selected.")
            
    except Exception as e:
        print(f"System Error: {e}")