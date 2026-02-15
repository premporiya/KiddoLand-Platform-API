"""
Story Router
Handles story generation and rewriting endpoints
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from schemas.story import (
    StoryRewriteRequest,
    StoryRewriteResponse,
    ErrorResponse
)
from utils.huggingface_client import (
    HuggingFaceError,
    rewrite_story,
)
from utils.safety_filter import clean_text_for_model, extract_child_name, is_content_safe
from utils.auth_service import get_current_user
from schemas.auth import AuthUser
from utils.story_history_service import save_story_record

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/rewrite", response_model=StoryRewriteResponse)
def rewrite_story_endpoint(
    request: StoryRewriteRequest,
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Rewrite an existing story based on user instructions.
    
    - **age**: Child's age (1-18)
    - **original_story**: The original story text
    - **instruction**: Free-form rewrite instruction (e.g., "make it happier", "change the ending")
    
    Returns rewritten story text.
    """
    # Validate age range
    if request.age < 1 or request.age > 10:
        raise HTTPException(
            status_code=400,
            detail="Age must be between 1 and 10"
        )
    
    # Validate inputs
    if not request.original_story or len(request.original_story.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Original story cannot be empty"
        )

    cleaned_original_story = clean_text_for_model(request.original_story)
    if not cleaned_original_story:
        raise HTTPException(
            status_code=400,
            detail="Original story cannot be empty"
        )

    if not is_content_safe(cleaned_original_story):
        raise HTTPException(
            status_code=400,
            detail="Original story contains unsafe content and cannot be processed.",
        )
    
    if not request.instruction or len(request.instruction.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Rewrite instruction cannot be empty"
        )

    cleaned_instruction = clean_text_for_model(request.instruction)
    if not cleaned_instruction:
        raise HTTPException(
            status_code=400,
            detail="Rewrite instruction cannot be empty"
        )

    if not is_content_safe(cleaned_instruction):
        raise HTTPException(
            status_code=400,
            detail="Rewrite instruction contains unsafe content and cannot be processed.",
        )

    child_name = extract_child_name(cleaned_instruction) or extract_child_name(cleaned_original_story)
    if child_name is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Child name is required. Please include at least one child name "
                "in the instruction or original story."
            ),
        )
    
    try:
        # Rewrite story using Hugging Face
        rewritten_story = rewrite_story(
            original_story=cleaned_original_story,
            instruction=cleaned_instruction,
            age=request.age
        )
        
        # Safety check
        if not is_content_safe(rewritten_story):
            return StoryRewriteResponse(
                story="I'm sorry, but the rewritten story contains inappropriate content for children. Please try a different instruction."
            )
        
        try:
            save_story_record(
                user_id=current_user.user_id,
                child_name=child_name,
                prompt=request.instruction,
                story=rewritten_story,
                age=request.age,
                mode=current_user.mode,
                record_type="rewrite",
            )
        except ValueError as exc:
            logger.warning("Story history validation failed: %s", str(exc))

        return StoryRewriteResponse(story=rewritten_story)
    
    except HuggingFaceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=f"Story rewriting failed: {str(exc)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Story rewriting failed: {str(e)}"
        )
