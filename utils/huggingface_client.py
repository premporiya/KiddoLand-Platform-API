"""
Hugging Face API Client
Handles communication with Hugging Face Inference API
"""
from __future__ import annotations

import logging
import os
import random
from io import BytesIO
from dataclasses import dataclass
from typing import Optional, Tuple

import requests
from gtts import gTTS
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

from utils.config import get_huggingface_config

# Timeout configuration
REQUEST_TIMEOUT = 60  # seconds
IMAGE_GEN_TIMEOUT = 120  # seconds (Stable Diffusion can be slow)

# Models must have Hub inferenceProviderMapping (Inference Providers). Older IDs like
# runwayml/stable-diffusion-v1-5 have empty mapping and 404 on hf-inference.
DEFAULT_SD_MODEL = "ByteDance/SDXL-Lightning"
DEFAULT_IMAGE_MODEL_FALLBACKS: Tuple[str, ...] = (
    "ByteDance/SDXL-Lightning",
    "black-forest-labs/FLUX.1-schnell",
)

logger = logging.getLogger(__name__)

DEFAULT_TTS_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_TTS_API_URL_TEMPLATE = "https://router.huggingface.co/hf-inference/models/{model}"


def _normalize_tts_url_template(url_template: str) -> str:
    normalized = url_template.strip()
    if not normalized:
        return DEFAULT_TTS_API_URL_TEMPLATE

    deprecated_prefix = "https://api-inference.huggingface.co/models"
    router_prefix = "https://router.huggingface.co/hf-inference/models"
    if normalized.startswith(deprecated_prefix):
        return normalized.replace(deprecated_prefix, router_prefix, 1)

    return normalized


@dataclass
class HuggingFaceError(Exception):
    message: str
    status_code: int = 502

    def __str__(self) -> str:
        return self.message


class HuggingFaceConfigError(HuggingFaceError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=500)


class HuggingFaceAuthError(HuggingFaceError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=401)


class HuggingFaceTimeoutError(HuggingFaceError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=504)


class HuggingFaceNetworkError(HuggingFaceError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=503)


class HuggingFaceResponseError(HuggingFaceError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=502)


def _image_model_candidates() -> list[str]:
    primary = os.getenv("HUGGINGFACE_IMAGE_MODEL", "").strip() or DEFAULT_SD_MODEL
    raw = os.getenv("HUGGINGFACE_IMAGE_MODEL_FALLBACKS", "").strip()
    if raw:
        fallbacks = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        fallbacks = list(DEFAULT_IMAGE_MODEL_FALLBACKS)
    out: list[str] = []
    seen: set[str] = set()
    for m in (primary, *fallbacks):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _pil_image_to_png_bytes(image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    raw = buf.getvalue()
    if not raw:
        raise HuggingFaceResponseError(
            "Empty image response from Hugging Face."
        )
    return raw


def _call_huggingface_api(
    messages: list,
    max_length: int = 1000,
    *,
    temperature: float | None = None,
) -> str:
    """
    Internal function to call Hugging Face Inference API.

    Args:
        messages: List of chat messages in OpenAI-compatible format
        max_length: Maximum length of generated text
        temperature: Sampling temperature. When None, uses 0.7 (legacy default).
            For activity generation, pass an explicit value (never 0) so each
            request can vary. No seed is ever sent to the API.

    Returns:
        Generated text from the model

    Raises:
        Exception: If API call fails
    """
    try:
        config = get_huggingface_config()
    except RuntimeError as exc:
        logger.error("Hugging Face config error: %s", exc)
        raise HuggingFaceConfigError(
            "Hugging Face is not configured on the server."
        )
    
    headers = {
        "Authorization": f"Bearer {config.api_token}",
        "Content-Type": "application/json",
    }

    # Stochastic sampling: never use temperature=0 or a fixed seed here.
    eff_temperature = 0.7 if temperature is None else float(temperature)
    if eff_temperature <= 0.0:
        eff_temperature = 0.7

    payload = {
        "model": config.model_id,
        "messages": messages,
        "temperature": eff_temperature,
        "top_p": 0.9,
        "max_tokens": max_length,
        "stream": False,
    }
    
    try:
        response = requests.post(
            config.api_url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code in {401, 403}:
            logger.warning(
                "Hugging Face auth failed with status %s (%s)",
                response.status_code,
                config.safe_summary(),
            )
            raise HuggingFaceAuthError(
                "Hugging Face token is invalid or expired."
            )

        if response.status_code == 503:
            logger.warning(
                "Hugging Face model unavailable (503) (%s)",
                config.safe_summary(),
            )
            raise HuggingFaceResponseError(
                "The AI model is currently loading. Please try again in a few moments."
            )
        
        if response.status_code != 200:
            try:
                error_detail = response.json().get("error", "Unknown error")
            except ValueError:
                error_detail = response.text.strip() or "Unknown error"
            logger.warning(
                "Hugging Face API error status=%s detail=%s (%s)",
                response.status_code,
                str(error_detail)[:200],
                config.safe_summary(),
            )
            raise HuggingFaceResponseError(
                f"Hugging Face API error: {error_detail}"
            )
        
        try:
            result = response.json()
        except ValueError:
            logger.warning(
                "Hugging Face returned non-JSON response (%s)",
                config.safe_summary(),
            )
            raise HuggingFaceResponseError(
                "Hugging Face API returned a non-JSON response"
            )
        
        # OpenAI-compatible response format
        if isinstance(result, dict):
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                generated_text = message.get("content", "")
            else:
                generated_text = ""
        else:
            generated_text = ""
        
        if not generated_text or generated_text.strip() == "":
            logger.warning(
                "Hugging Face returned empty response (%s)",
                config.safe_summary(),
            )
            raise HuggingFaceResponseError("Model returned empty response")
        
        return generated_text.strip()
    
    except requests.exceptions.Timeout:
        logger.warning(
            "Hugging Face request timed out (%s)",
            config.safe_summary(),
        )
        raise HuggingFaceTimeoutError(
            "Request to Hugging Face API timed out. Please try again."
        )
    except requests.exceptions.RequestException as exc:
        logger.warning(
            "Hugging Face network error: %s (%s)",
            str(exc),
            config.safe_summary(),
        )
        raise HuggingFaceNetworkError(
            "Network error while calling Hugging Face API. Please try again."
        )


def generate_stable_diffusion_image(prompt: str) -> bytes:
    """
    Generate a single image from a text prompt via Hugging Face Inference Providers.

    Uses huggingface_hub.InferenceClient (same token as chat/TTS).
    - HUGGINGFACE_IMAGE_MODEL: primary model (default: fast SDXL-Lightning with provider mapping).
    - HUGGINGFACE_IMAGE_MODEL_FALLBACKS: comma-separated alternates if the primary fails.
    - HUGGINGFACE_IMAGE_PROVIDER: default "auto" (routed providers). Use "hf-inference" only
      for models still served on the HF inference router.
    """
    if not prompt or not prompt.strip():
        raise HuggingFaceResponseError("Image prompt cannot be empty")

    try:
        config = get_huggingface_config()
    except RuntimeError as exc:
        logger.error("Hugging Face config error: %s", exc)
        raise HuggingFaceConfigError(
            "Hugging Face is not configured on the server."
        )

    models = _image_model_candidates()
    image_provider = os.getenv("HUGGINGFACE_IMAGE_PROVIDER", "").strip() or "auto"
    prompt_text = prompt.strip()[:500]
    last_detail = "Unknown error"
    last_exc: Optional[BaseException] = None

    for model in models:
        client = InferenceClient(
            token=config.api_token,
            timeout=IMAGE_GEN_TIMEOUT,
            provider=image_provider,
        )
        try:
            image = client.text_to_image(prompt_text, model=model)
        except StopIteration as exc:
            last_exc = exc
            logger.warning(
                "No inference provider mapping for image model %s (provider=%s)",
                model,
                image_provider,
            )
            last_detail = "No inference provider mapping for this model"
            continue
        except ValueError as exc:
            last_exc = exc
            lowered = str(exc).lower()
            if (
                "provider mapping" in lowered
                or "doesn't support task" in lowered
            ):
                logger.warning("Skipping image model %s: %s", model, exc)
                last_detail = str(exc)[:500]
                continue
            raise
        except InferenceTimeoutError as exc:
            raise HuggingFaceTimeoutError(
                "Image generation timed out. Please try again."
            ) from exc
        except HfHubHTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            detail = (exc.server_message or str(exc)).strip()[:500] or "Unknown error"
            last_detail = detail
            logger.warning(
                "Hugging Face image HTTP error model=%s status=%s: %s",
                model,
                status,
                detail[:200],
            )
            if status in {401, 403}:
                raise HuggingFaceAuthError(
                    "Hugging Face token is invalid or expired."
                ) from exc
            if status in {404, 410, 503}:
                continue
            raise HuggingFaceResponseError(
                f"Image generation failed: {detail}"
            ) from exc
        except requests.exceptions.Timeout:
            raise HuggingFaceTimeoutError(
                "Image generation timed out. Please try again."
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("Hugging Face image network error: %s", str(exc))
            raise HuggingFaceNetworkError(
                "Network error while generating the image. Please try again."
            ) from exc

        if image is None:
            last_detail = "Empty image response"
            continue
        return _pil_image_to_png_bytes(image)

    raise HuggingFaceResponseError(
        f"Image generation failed after trying {len(models)} model(s). Last error: {last_detail}"
    ) from last_exc


def generate_story(prompt: str, age: int) -> str:
    """
    Generate a story based on user prompt and age.
    
    Args:
        prompt: User's story prompt (free-form)
        age: Child's age (1-18)
        
    Returns:
        Generated story text
    """
    # Build age-appropriate system instruction
    age_guidance = _get_age_guidance(age)
    
    system_msg = (
        "You are a creative storyteller for children. "
        f"{age_guidance}"
    )
    user_msg = (
        "Story request: "
        f"{prompt}\n\n"
        "Write a complete, engaging story based on this request. "
        "Make it appropriate and enjoyable for the target age group."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

     # Call Hugging Face API (OpenAI-compatible)
    story = _call_huggingface_api(messages, max_length=8000)
    
    return story


def rewrite_story(original_story: str, instruction: str, age: int) -> str:
    """
    Rewrite a story based on user instruction.
    
    Args:
        original_story: The original story text
        instruction: User's rewrite instruction (free-form)
        age: Child's age (1-18)
        
    Returns:
        Rewritten story text
    """
    # Build age-appropriate system instruction
    age_guidance = _get_age_guidance(age)
    
    system_msg = (
        "You are a creative storyteller for children. "
        f"{age_guidance}"
    )
    user_msg = (
    "Original Story:\n"
    f"{original_story}\n\n"
    f"Rewrite Instruction: {instruction}\n\n"
    "Rewrite the story according to the instruction above. "
    "Keep the main characters and setting the same. "
    "Ensure the rewritten story has a proper and natural ending. "
    "Do NOT stop mid-sentence."
)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Call Hugging Face API (OpenAI-compatible)
    rewritten_story = _call_huggingface_api(messages, max_length=3000)
    
    return rewritten_story


def sample_completion(prompt: str) -> str:
    """
    Generate a short completion for a sample AI endpoint.

    Args:
        prompt: User prompt to send to the model

    Returns:
        Generated response text
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant for kids."},
        {"role": "user", "content": prompt},
    ]
    return _call_huggingface_api(messages, max_length=800)


def sample_completion_activity(prompt: str) -> str:
    """
    Text generation for learning activities / quizzes.

    Uses randomized temperature in (0.65, 0.95) per request so outputs vary
    across calls. Does not use temperature=0 or any fixed seed.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant for kids."},
        {"role": "user", "content": prompt},
    ]
    sampling_temp = random.uniform(0.65, 0.95)
    return _call_huggingface_api(messages, max_length=1200, temperature=sampling_temp)


def generate_rhyme(prompt: str, age: int) -> str:
    """
    Generate a short rhyme or nursery rhyme based on user prompt and age.

    Args:
        prompt: User's rhyme prompt (free-form)
        age: Child's age (1-18)

    Returns:
        Generated rhyme text
    """
    # Build age-appropriate system instruction using existing guidance
    age_guidance = _get_age_guidance(age)

    system_msg = (
        "You are a creative children's poet. "
        f"{age_guidance}"
    )
    user_msg = (
        "Rhyme request: "
        f"{prompt}\n\n"
        "Write a short, rhyming poem or nursery rhyme based on this request. "
        "Use simple vocabulary and short lines suitable for the target age. "
        "Keep the rhyme playful and end with a clear, positive ending. "
        "If a child's name is included in the prompt, incorporate it naturally into the rhyme."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Call Hugging Face API (OpenAI-compatible)
    rhyme = _call_huggingface_api(messages, max_length=1200)

    return rhyme


def generate_tts_audio(text: str) -> Tuple[bytes, str]:
    """
    Generate speech audio from plain text using Hugging Face Inference API.

    Args:
        text: Text to convert into speech

    Returns:
        Tuple (audio_bytes, media_type)
    """
    if not text or not text.strip():
        raise HuggingFaceResponseError("TTS text cannot be empty")

    try:
        config = get_huggingface_config()
    except RuntimeError as exc:
        logger.error("Hugging Face config error: %s", exc)
        raise HuggingFaceConfigError(
            "Hugging Face is not configured on the server."
        )

    tts_model = os.getenv("HUGGINGFACE_TTS_MODEL", "").strip() or DEFAULT_TTS_MODEL
    tts_url_template = _normalize_tts_url_template(
        os.getenv("HUGGINGFACE_TTS_API_URL", "")
    )
    tts_url = tts_url_template.format(model=tts_model)

    headers = {
        "Authorization": f"Bearer {config.api_token}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg, audio/wav, audio/flac, audio/ogg, application/octet-stream",
    }
    payload = {"inputs": text.strip()}

    try:
        response = requests.post(
            tts_url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            error_detail = response.text.strip() or "Unknown error"
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    error_detail = parsed.get("error") or parsed.get("message") or error_detail
            except ValueError:
                pass
            return _generate_gtts_audio(text.strip(), error_detail)

        media_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if not media_type:
            media_type = "application/octet-stream"

        # Some hosted endpoints return octet-stream for binary audio; accept it.
        if not (media_type.startswith("audio/") or media_type == "application/octet-stream"):
            logger.warning("Unexpected TTS content-type: %s", media_type)
            raise HuggingFaceResponseError(
                "TTS endpoint returned a non-audio response."
            )

        if not response.content:
            raise HuggingFaceResponseError("TTS endpoint returned empty audio data.")

        return response.content, media_type

    except requests.exceptions.Timeout:
        return _generate_gtts_audio(text.strip(), "Request to Hugging Face TTS API timed out")
    except requests.exceptions.RequestException as exc:
        logger.warning("Hugging Face TTS network error: %s", str(exc))
        return _generate_gtts_audio(text.strip(), str(exc))


def _generate_gtts_audio(text: str, original_error: str) -> Tuple[bytes, str]:
    """
    Fallback TTS using Google TTS when Hugging Face TTS is unavailable.
    """
    try:
        logger.warning("Falling back to gTTS because Hugging Face TTS failed: %s", original_error)
        buffer = BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buffer)
        audio_bytes = buffer.getvalue()
        if not audio_bytes:
            raise HuggingFaceResponseError("Fallback TTS returned empty audio data.")
        return audio_bytes, "audio/mpeg"
    except Exception as exc:
        logger.warning("Fallback gTTS failed: %s", str(exc))
        raise HuggingFaceNetworkError(
            f"TTS is currently unavailable. Hugging Face error: {original_error}"
        )


def _get_age_guidance(age: int) -> str:
    """
    Get age-appropriate guidance for story generation.
    
    Args:
        age: Child's age
        
    Returns:
        Age-appropriate guidance string
    """
    if age <= 5:
        return "Create simple, colorful stories with clear lessons. Use basic vocabulary and short sentences."
    elif age <= 8:
        return "Create engaging stories with simple adventures. Use age-appropriate vocabulary and moderate complexity."
    elif age <= 12:
        return "Create interesting stories with meaningful themes. Use varied vocabulary and good narrative structure."
    else:
        return "Create compelling stories with deeper themes. Use rich vocabulary and sophisticated storytelling."
