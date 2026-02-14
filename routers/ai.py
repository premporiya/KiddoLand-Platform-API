"""
AI Router
Sample endpoint for testing Hugging Face integration
"""
import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from schemas.ai import (
    AiSampleRequest,
    AiSampleResponse,
    AiSaveFavoriteRequest,
    AiSaveFavoriteResponse,
    AiStoryHistoryResponse,
)
from schemas.auth import AuthUser
from utils.auth_service import get_current_user
from utils.huggingface_client import HuggingFaceError, sample_completion
from utils.safety_filter import clean_text_for_model, extract_child_name, is_content_safe
from utils.story_favorites_service import save_favorite_record
from utils.story_history_service import list_story_records, save_story_record

logger = logging.getLogger(__name__)

router = APIRouter()


def _extract_age_from_prompt(prompt: str) -> Optional[int]:
    patterns = [
        r"\b(\d{1,2})\s*[- ]\s*year\s*[- ]\s*old\b",
        r"\b(\d{1,2})\s*(?:years?\s*old|year\s*old|yr\s*old|y/o)\b",
        r"\bage\s*(\d{1,2})\b",
        r"\bfor\s+(\d{1,2})\s*(?:years?\s*old|year\s*old)?\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if not match:
            continue

        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            continue

        if 1 <= value <= 10:
            return value

    return None


@router.post("/sample", response_model=AiSampleResponse)
def sample_ai_endpoint(
    request: AiSampleRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> AiSampleResponse:
    """
    Sample AI endpoint that calls Hugging Face using configured values.
    """
    cleaned_prompt = clean_text_for_model(request.prompt)
    if not cleaned_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if not is_content_safe(cleaned_prompt):
        raise HTTPException(
            status_code=400,
            detail="Prompt contains unsafe content and cannot be processed.",
        )

    extracted_age = _extract_age_from_prompt(cleaned_prompt)
    if extracted_age is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Child age is required in the prompt. "
                "Please include an age between 1 and 10, for example: 'for a 7-year-old'."
            ),
        )

    child_name = extract_child_name(cleaned_prompt)
    if child_name is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Child name is required in the prompt. "
                "Please include at least one child name, for example: 'for Emma, age 7'."
            ),
        )

    try:
        output = sample_completion(cleaned_prompt)
        try:
            save_story_record(
                user_id=current_user.user_id,
                child_name=child_name,
                prompt=cleaned_prompt,
                story=output,
                age=extracted_age,
                mode=current_user.mode,
                record_type="generate",
            )
        except ValueError as exc:
            logger.warning("Auto-save story history validation failed: %s", str(exc))

        return AiSampleResponse(output=output)
    except HuggingFaceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=f"AI sample failed: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AI sample failed: {str(exc)}",
        )


@router.post("/save-favorite", response_model=AiSaveFavoriteResponse)
def save_ai_favorite_endpoint(
    request: AiSaveFavoriteRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> AiSaveFavoriteResponse:
    """
    Persist a story as favorite after explicit user action.
    """
    try:
        saved = save_favorite_record(
            user_id=current_user.user_id,
            prompt=request.prompt,
            story=request.story,
            age=request.age,
            mode=current_user.mode,
            record_type=request.type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if saved:
        return AiSaveFavoriteResponse(saved=True, message="Story saved to favorites.")

    return AiSaveFavoriteResponse(
        saved=False,
        message="Favorite save is currently unavailable.",
    )


@router.get("/history", response_model=AiStoryHistoryResponse)
def get_story_history_endpoint(
    current_user: AuthUser = Depends(get_current_user),
) -> AiStoryHistoryResponse:
    items = list_story_records(user_id=current_user.user_id, limit=100)
    return AiStoryHistoryResponse(items=items)
