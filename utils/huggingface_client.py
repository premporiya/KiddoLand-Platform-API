"""
Hugging Face API Client
Handles communication with Hugging Face Inference API
"""
import os
import requests
from typing import Optional
from dotenv import load_dotenv

# Load environment variables (in case app wasn't started from project root)
load_dotenv()

# Hugging Face API configuration (Inference Providers, OpenAI-compatible)
HUGGINGFACE_API_URL = os.getenv(
    "HUGGINGFACE_API_URL",
    "https://router.huggingface.co/v1/chat/completions"
)
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HUGGINGFACE_MODEL = os.getenv(
    "HUGGINGFACE_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.2"
)

# Timeout configuration
REQUEST_TIMEOUT = 60  # seconds


def _call_huggingface_api(messages: list, max_length: int = 1000) -> str:
    """
    Internal function to call Hugging Face Inference API.
    
    Args:
        prompt: The prompt to send to the model
        max_length: Maximum length of generated text
        
    Returns:
        Generated text from the model
        
    Raises:
        Exception: If API call fails
    """
    if not HUGGINGFACE_API_TOKEN:
        raise Exception("HUGGINGFACE_API_TOKEN not found in environment variables")
    
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": HUGGINGFACE_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": max_length,
        "stream": False,
    }
    
    try:
        response = requests.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 503:
            # Model is loading
            raise Exception("The AI model is currently loading. Please try again in a few moments.")
        
        if response.status_code != 200:
            try:
                error_detail = response.json().get("error", "Unknown error")
            except ValueError:
                error_detail = response.text.strip() or "Unknown error"
            raise Exception(f"Hugging Face API error: {error_detail}")
        
        try:
            result = response.json()
        except ValueError:
            raise Exception("Hugging Face API returned a non-JSON response")
        
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
            raise Exception("Model returned empty response")
        
        return generated_text.strip()
    
    except requests.exceptions.Timeout:
        raise Exception("Request to Hugging Face API timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")


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
    story = _call_huggingface_api(messages, max_length=1500)
    
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
        "Please rewrite the story according to the instruction above. "
        "Keep it appropriate and enjoyable for the target age group."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Call Hugging Face API (OpenAI-compatible)
    rewritten_story = _call_huggingface_api(messages, max_length=1500)
    
    return rewritten_story


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
