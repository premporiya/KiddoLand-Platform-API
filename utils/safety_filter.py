"""
Content Safety Filter
Checks generated content for child-inappropriate material
"""
import re
from typing import List, Optional

# Unsafe content patterns (basic filtering)
UNSAFE_KEYWORDS = [
    # Explicit violence
    r"\b(murder|kill|blood|gore|torture|weapon|gun|knife|stab|shoot)\b",
    # Sexual content
    r"\b(sex|sexual|naked|nude|porn|explicit)\b",
    # Strong profanity
    r"\b(fuck|shit|bitch|damn|hell|ass|crap)\b",
    # Harmful content
    r"\b(suicide|drug|alcohol|cigarette|abuse)\b",
]


def is_content_safe(text: str) -> bool:
    """
    Check if content is safe for children.
    
    This is a SOFT safety filter that catches only extreme violations.
    It does NOT over-restrict normal storytelling elements.
    
    Args:
        text: The text content to check
        
    Returns:
        True if content is safe, False if unsafe
    """
    if not text or len(text.strip()) == 0:
        return True
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check for unsafe patterns
    for pattern in UNSAFE_KEYWORDS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False
    
    return True


def clean_text_for_model(text: str) -> str:
    """
    Lightweight text normalization before sending user input to AI.

    - Removes control characters
    - Collapses repeated whitespace
    - Trims leading/trailing spaces
    """
    if not isinstance(text, str):
        return ""

    cleaned = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def extract_child_name(text: str) -> Optional[str]:
    """
    Try to extract a child name from free-form prompt/instruction text.

    Returns a title-cased name when found, otherwise None.
    """
    if not isinstance(text, str):
        return None

    cleaned = clean_text_for_model(text)
    if not cleaned:
        return None

    patterns = [
        r"\bnamed\s+([A-Za-z][A-Za-z'\-]{1,30})\b",
        r"\bcalled\s+([A-Za-z][A-Za-z'\-]{1,30})\b",
        r"\bname\s+is\s+([A-Za-z][A-Za-z'\-]{1,30})\b",
        r"\b(?:son|daughter|kid|child)\s+named\s+([A-Za-z][A-Za-z'\-]{1,30})\b",
        r"\b(?:son|daughter|kid|child)\s+([A-Za-z][A-Za-z'\-]{1,30})\b",
        r"\bfor\s+([A-Za-z][A-Za-z'\-]{1,30})\b",
    ]

    disallowed = {
        "a", "an", "the", "my", "our", "your", "their",
        "kid", "child", "son", "daughter", "boy", "girl",
        "story", "age", "years", "year", "old",
    }

    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if not match:
            continue

        candidate = match.group(1).strip("-'")
        if len(candidate) < 2:
            continue

        if candidate.lower() in disallowed:
            continue

        return candidate.capitalize()

    return None


def get_unsafe_content_reasons(text: str) -> List[str]:
    """
    Get detailed reasons why content is unsafe (for debugging/logging).
    
    Args:
        text: The text content to check
        
    Returns:
        List of reasons why content is flagged as unsafe
    """
    if not text or len(text.strip()) == 0:
        return []
    
    text_lower = text.lower()
    reasons = []
    
    for pattern in UNSAFE_KEYWORDS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            reasons.append(f"Contains inappropriate keyword: {', '.join(set(matches))}")
    
    return reasons
