"""
Learning activity generation (quiz) — POST /ai/activity
"""
import logging

from fastapi import APIRouter, Depends, HTTPException

from schemas.activity import (
    ActivityErrorEnvelope,
    ActivityGenerateRequest,
    ActivitySuccessEnvelope,
)
from schemas.auth import AuthUser
from services.learning_activity_service import generate_learning_activity
from utils.auth_service import get_current_user
from utils.safety_filter import clean_text_for_model, is_content_safe

logger = logging.getLogger(__name__)

router = APIRouter()


def _sanitize_and_check_inputs(body: ActivityGenerateRequest) -> ActivityGenerateRequest:
    """Normalize and reject unsafe user-controlled strings."""
    age = clean_text_for_model(body.age_band)
    theme = clean_text_for_model(body.theme)
    goal = clean_text_for_model(body.learning_goal)
    if not age or not theme or not goal:
        raise HTTPException(
            status_code=400,
            detail="age_band, theme, and learning_goal must be non-empty after cleaning.",
        )
    combined = f"{age} {theme} {goal}"
    if not is_content_safe(combined):
        raise HTTPException(
            status_code=400,
            detail="Request contains content that cannot be processed.",
        )
    return ActivityGenerateRequest(
        age_band=age,
        theme=theme,
        learning_goal=goal,
        difficulty=body.difficulty,
    )


@router.post(
    "/activity",
    response_model=ActivitySuccessEnvelope | ActivityErrorEnvelope,
    summary="Generate a kid-friendly quiz activity (JSON)",
)
async def create_learning_activity(
    request: ActivityGenerateRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> ActivitySuccessEnvelope | ActivityErrorEnvelope:
    """
    Generates exactly 5 multiple-choice questions via the configured text model.
    Requires authentication (same as other /ai routes).
    """
    _ = current_user
    safe_request = _sanitize_and_check_inputs(request)
    result = await generate_learning_activity(safe_request)
    if result.success and result.data is not None:
        return ActivitySuccessEnvelope(data=result.data)
    logger.warning("Learning activity generation failed after retries")
    return ActivityErrorEnvelope(error=result.error or "Failed to generate activity")
