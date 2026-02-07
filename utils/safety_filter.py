"""
Content Safety Filter
Checks generated content for child-inappropriate material
"""
import re
from typing import List

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
