"""
Hugging Face API Client
Handles communication with Hugging Face Inference API
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from utils.config import get_huggingface_config

# Timeout configuration
REQUEST_TIMEOUT = 60  # seconds

logger = logging.getLogger(__name__)


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


def _call_huggingface_api(messages: list, max_length: int = 1000) -> str:
    """
    Internal function to call Hugging Face Inference API.
    
    Args:
        messages: List of chat messages in OpenAI-compatible format
        max_length: Maximum length of generated text
        
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

    payload = {
        "model": config.model_id,
        "messages": messages,
        "temperature": 0.7,
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
