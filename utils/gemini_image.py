"""
Gemini native image generation (Google AI Studio / Gemini API free tier).
"""
from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"


class GeminiImageError(Exception):
    """Image generation failure; `status_code` is returned to the HTTP client."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def get_gemini_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def generate_gemini_illustration_image(prompt: str) -> bytes:
    if not prompt or not prompt.strip():
        raise GeminiImageError("Image prompt cannot be empty", status_code=400)

    api_key = get_gemini_api_key()
    if not api_key:
        raise GeminiImageError(
            "Gemini is not configured. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env. "
            "Create a free key at https://aistudio.google.com/apikey",
            status_code=500,
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise GeminiImageError(
            "google-genai is not installed. Run: pip install google-genai",
            status_code=500,
        ) from exc

    model = os.getenv("GEMINI_IMAGE_MODEL", "").strip() or DEFAULT_GEMINI_IMAGE_MODEL
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt.strip()[:500],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            ),
        )
    except Exception as exc:
        logger.warning("Gemini image request failed: %s", exc)
        msg = str(exc).strip()[:500] or "Gemini image generation failed"
        lowered = msg.lower()
        if "api key" in lowered or ("invalid" in lowered and "key" in lowered):
            raise GeminiImageError(
                "Gemini API key is invalid or not allowed for this model. Check GEMINI_API_KEY in .env.",
                status_code=502,
            ) from exc
        if "429" in msg or "resource exhausted" in lowered or "quota" in lowered:
            raise GeminiImageError(
                "Gemini rate limit reached. Try again later or check quota in AI Studio.",
                status_code=429,
            ) from exc
        raise GeminiImageError(f"Image generation failed: {msg}", status_code=502) from exc
    finally:
        try:
            client.close()
        except Exception:
            pass

    try:
        return _image_bytes_from_gemini_response(response)
    except GeminiImageError:
        raise
    except Exception as exc:
        logger.warning("Could not parse Gemini image response: %s", exc)
        raise GeminiImageError(
            "Gemini returned a response without usable image data.",
            status_code=502,
        ) from exc


def _image_bytes_from_gemini_response(response: Any) -> bytes:
    parts: list[Any] = []
    if getattr(response, "parts", None):
        parts = list(response.parts)
    elif getattr(response, "candidates", None) and response.candidates:
        content = getattr(response.candidates[0], "content", None)
        if content and getattr(content, "parts", None):
            parts = list(content.parts)

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None:
            raw = getattr(inline, "data", None)
            if isinstance(raw, bytes) and raw:
                return raw
            if isinstance(raw, str) and raw:
                return base64.b64decode(raw)
        if hasattr(part, "as_image"):
            try:
                img = part.as_image()
                if img is not None:
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    out = buf.getvalue()
                    if out:
                        return out
            except Exception:
                continue

    raise GeminiImageError(
        "Gemini did not return an image (safety filter or empty response). "
        "Try rephrasing the story or use image_provider=huggingface.",
        status_code=502,
    )
