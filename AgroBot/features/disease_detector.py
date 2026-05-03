"""
AgroBot — Intelligent Disease Identification Feature
=====================================================
Multi-modal tool: accepts image + optional text description.

Detection pipeline:
  1. PRIMARY: Groq Vision (llama-4-scout) → image analysis JSON
  2. FALLBACK: Kindwise API (if API key set)
  3. FALLBACK: Keyword matching against disease_treatments.json
  
Treatment:
  Groq LLM (llama-3.3-70b-versatile) → detailed treatment plan in user's language.
"""

import os
import json
import base64
import requests
from pathlib import Path
from typing import Optional
from PIL import Image

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    GROQ_API_KEY, LLM_MODEL, VISION_MODEL, DISEASE_TREATMENTS_JSON,
    GROQ_CHAT_URL, KINDWISE_API_KEY, KINDWISE_URL
)
from core.language import get_lang_instruction
from features.base_tool import BaseAgriTool
from prompts.templates import DISEASE_TREATMENT_TEMPLATE


def _load_disease_db() -> list:
    """Load the disease treatments lookup from JSON."""
    try:
        with open(DISEASE_TREATMENTS_JSON, encoding="utf-8") as f:
            return json.load(f).get("diseases", [])
    except Exception:
        return []

DISEASE_DB = _load_disease_db()


class DiseaseDetectorTool(BaseAgriTool):
    """
    LangChain Tool — Intelligent Disease Identification.
    Accepts image + text description. Returns disease name + treatment plan.
    """

    name        = "disease_detector"
    description = (
        "Use this when the user uploads a crop image, asks about plant disease, "
        "pest identification, blight, rust, virus, or shows symptoms. "
        "Input: image path + optional text description of symptoms."
    )

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """
        Encode image to base64 for API transmission.
        Returns (base64_string, mime_type).
        """
        path = Path(image_path)
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png",  ".bmp":  "image/bmp",
                    ".webp": "image/webp"}
        mime = mime_map.get(suffix, "image/jpeg")

        # Resize large images to avoid API limits
        img = Image.open(image_path).convert("RGB")
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)

        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64, "image/jpeg"

    def _analyze_with_groq_vision(
        self,
        image_path: str,
        description: str,
        lang: str,
    ) -> dict | None:
        """Use Groq Vision (llama-4-scout) to identify crop disease from image."""
        if not GROQ_API_KEY or not image_path:
            return None

        try:
            b64_img, mime = self._encode_image(image_path)
            lang_instr    = get_lang_instruction(lang)

            system_prompt = (
                "You are AgroBot's plant disease expert. Analyze this crop image carefully.\n"
                "Return ONLY valid JSON with this exact structure:\n"
                '{"crop": "crop name or Unknown", "disease": "disease name or Healthy", '
                '"confidence": "High|Medium|Low", "symptoms": "visible symptoms description", '
                '"is_healthy": true/false}\n'
                "If the plant looks healthy, set is_healthy=true and disease='Healthy'."
            )

            payload = {
                "model": VISION_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64_img}"},
                            },
                            {
                                "type": "text",
                                "text": f"Farmer's description: {description or 'No additional description.'}\nPlease identify any disease.",
                            },
                        ],
                    },
                ],
                "temperature": 0.1,
                "max_tokens":  512,
            }

            resp = requests.post(
                GROQ_CHAT_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                # Parse JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                return json.loads(content)
        except Exception as e:
            print(f"⚠️  Groq Vision analysis failed: {e}")
        return None

    def _analyze_with_kindwise(self, image_path: str) -> dict | None:
        """Use Kindwise API if key is available."""
        if not KINDWISE_API_KEY or not image_path:
            return None
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            resp = requests.post(
                KINDWISE_URL,
                headers={"Api-Key": KINDWISE_API_KEY, "Content-Type": "application/json"},
                json={"images": [b64]},
                timeout=30,
            )
            if resp.status_code in (200, 201):
                data = resp.json().get("result", {})
                crops = data.get("crop", {}).get("suggestions", [])
                diseases = data.get("disease", {}).get("suggestions", [])

                crop_name = crops[0]["name"].title() if crops else "Unknown"
                if not diseases:
                    return {"crop": crop_name, "disease": "Healthy", "confidence": "High",
                            "symptoms": "No disease detected", "is_healthy": True}

                top = diseases[0]
                return {
                    "crop":       crop_name,
                    "disease":    top.get("name", "Unknown Disease").title(),
                    "confidence": "High" if top.get("probability", 0) > 0.7 else "Medium",
                    "symptoms":   top.get("details", {}).get("description", "")[:200],
                    "is_healthy": False,
                }
        except Exception as e:
            print(f"⚠️  Kindwise analysis failed: {e}")
        return None

    def _keyword_disease_lookup(self, description: str) -> dict | None:
        """Fallback: keyword matching against disease_treatments.json."""
        if not description or not DISEASE_DB:
            return None
        desc_lower = description.lower()
        for disease in DISEASE_DB:
            for kw in disease.get("keywords", []):
                if kw.lower() in desc_lower:
                    return {
                        "crop":       disease["crops"][0].title() if disease["crops"] else "Unknown",
                        "disease":    disease["name"],
                        "confidence": "Medium",
                        "symptoms":   disease["symptoms"],
                        "is_healthy": False,
                        "_db_record": disease,
                    }
        return None

    def _generate_treatment(self, diagnosis: dict, description: str, lang: str, **kwargs) -> str:
        """Generate full treatment plan using Groq LLM."""
        lang_instr = get_lang_instruction(lang)

        # If we have a DB record, use it to enrich the prompt
        db_record = diagnosis.pop("_db_record", None)

        # Check if pre-built treatment is available
        if db_record:
            chem    = db_record.get("chemical_treatment", "")
            organic = db_record.get("organic_treatment", "")
        else:
            chem = organic = ""

        try:
            llm   = ChatGroq(model=LLM_MODEL, temperature=0.3, api_key=GROQ_API_KEY)
            history = kwargs.get("history", [])
            messages = [
                SystemMessage(content=DISEASE_TREATMENT_TEMPLATE.format(
                    crop=diagnosis.get("crop", "Unknown"),
                    disease=diagnosis.get("disease", "Unknown"),
                    symptoms=diagnosis.get("symptoms", description),
                    description=(
                        f"{description or 'No description provided.'}\n"
                        f"Chemical: {chem}\nOrganic: {organic}"
                        if (chem or organic) else description
                    ),
                    language=lang_instr
                )),
                *history,
                HumanMessage(content=description)
            ]
            resp  = llm.invoke(messages)
            return resp.content.strip()
        except Exception as e:
            print(f"⚠️  Treatment LLM failed: {e}")
            if db_record:
                return (
                    f"**Disease:** {db_record['name']}\n\n"
                    f"**Chemical Treatment:** {db_record['chemical_treatment']}\n\n"
                    f"**Organic Treatment:** {db_record['organic_treatment']}"
                )
            return f"Disease identified: {diagnosis.get('disease', 'Unknown')}. Please consult a local agriculture expert for treatment."

    def run(
        self,
        query: str,
        lang: str = "en",
        image_path: Optional[str] = None,
        history: Optional[list] = None,
        **kwargs,
    ) -> dict:
        # Validate image path
        if image_path and not Path(image_path).exists():
            return self._error_response(
                lang,
                f"Image not found: {image_path}. Please provide a valid file path."
            )

        diagnosis = None

        # Attempt detection in priority order
        if image_path:
            # 1. Try Groq Vision first
            diagnosis = self._analyze_with_groq_vision(image_path, query, lang)

            # 2. Try Kindwise if Groq Vision failed
            if not diagnosis:
                diagnosis = self._analyze_with_kindwise(image_path)

        # 3. Keyword fallback from description
        if not diagnosis:
            diagnosis = self._keyword_disease_lookup(query)

        # 4. If nothing found, ask LLM directly
        if not diagnosis:
            try:
                llm  = ChatGroq(model=LLM_MODEL, temperature=0.4, api_key=GROQ_API_KEY)
                lang_instr = get_lang_instruction(lang)
                resp = llm.invoke([
                    SystemMessage(content=(
                        f"You are a plant disease expert for Pakistani farmers. {lang_instr}\n"
                        "The farmer has described some symptoms. Identify the likely disease and provide treatment.\n"
                        "Be specific about chemical names and doses."
                    )),
                    *(history or []),
                    HumanMessage(content=query),
                ])
                return {
                    "text":   resp.content.strip(),
                    "intent": self.name,
                    "lang":   lang,
                    "extra": {"diagnosis": None, "image_analyzed": False},
                }
            except Exception as e:
                return self._error_response(lang, f"Disease analysis failed: {e}")

        is_healthy = diagnosis.get("is_healthy", False)

        if is_healthy:
            healthy_msgs = {
                "en": f"✅ Your {diagnosis.get('crop', 'crop')} looks **healthy**! No disease detected. Keep up good field management practices.",
                "ur": f"✅ آپ کی {diagnosis.get('crop', 'فصل')} **صحت مند** نظر آتی ہے! کوئی بیماری نہیں ملی۔ اچھے کاشتکاری عمل جاری رکھیں۔",
            }
            return {
                "text":   healthy_msgs.get(lang, healthy_msgs["en"]),
                "intent": self.name,
                "lang":   lang,
                "extra": {"diagnosis": diagnosis, "is_healthy": True, "image_analyzed": bool(image_path)},
            }

        # Generate treatment plan
        treatment_text = self._generate_treatment(diagnosis, query, lang, history=history)

        return {
            "text":   treatment_text,
            "intent": self.name,
            "lang":   lang,
            "extra": {
                "diagnosis":      diagnosis,
                "crop":           diagnosis.get("crop", "Unknown"),
                "disease":        diagnosis.get("disease", "Unknown"),
                "confidence":     diagnosis.get("confidence", "Low"),
                "symptoms":       diagnosis.get("symptoms", ""),
                "is_healthy":     False,
                "image_analyzed": bool(image_path),
            },
        }
