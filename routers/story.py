"""
Story Router
Handles story generation and rewriting endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from schemas.story import (
    StoryGenerateRequest,
    StoryGenerateResponse,
    StoryRewriteRequest,
    StoryRewriteResponse,
    ErrorResponse
)
from utils.huggingface_client import generate_story, rewrite_story
from utils.safety_filter import is_content_safe
from utils.auth_service import get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.post("/generate", response_model=StoryGenerateResponse)
def generate_story_endpoint(request: StoryGenerateRequest):
    """
    Generate a new story based on user prompt.
    
    - **age**: Child's age (1-18)
    - **prompt**: Free-form story prompt (any text)
    
    Returns generated story text.
    """
    # Validate age range
    if request.age < 1 or request.age > 18:
        raise HTTPException(
            status_code=400,
            detail="Age must be between 1 and 18"
        )
    
    # Validate prompt
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty"
        )
    
    try:
        # Generate story using Hugging Face
        generated_story = generate_story(
            prompt=request.prompt,
            age=request.age
        )
        
        # Safety check
        if not is_content_safe(generated_story):
            return StoryGenerateResponse(
                story="I'm sorry, but I cannot generate this story as it contains inappropriate content for children. Please try a different prompt."
            )
        
        return StoryGenerateResponse(story=generated_story)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Story generation failed: {str(e)}"
        )


@router.post("/rewrite", response_model=StoryRewriteResponse)
def rewrite_story_endpoint(request: StoryRewriteRequest):
    """
    Rewrite an existing story based on user instructions.
    
    - **age**: Child's age (1-18)
    - **original_story**: The original story text
    - **instruction**: Free-form rewrite instruction (e.g., "make it happier", "change the ending")
    
    Returns rewritten story text.
    """
    # Validate age range
    if request.age < 1 or request.age > 18:
        raise HTTPException(
            status_code=400,
            detail="Age must be between 1 and 18"
        )
    
    # Validate inputs
    if not request.original_story or len(request.original_story.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Original story cannot be empty"
        )
    
    if not request.instruction or len(request.instruction.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Rewrite instruction cannot be empty"
        )
    
    try:
        # Rewrite story using Hugging Face
        rewritten_story = rewrite_story(
            original_story=request.original_story,
            instruction=request.instruction,
            age=request.age
        )
        
        # Safety check
        if not is_content_safe(rewritten_story):
            return StoryRewriteResponse(
                story="I'm sorry, but the rewritten story contains inappropriate content for children. Please try a different instruction."
            )
        
        return StoryRewriteResponse(story=rewritten_story)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Story rewriting failed: {str(e)}"
        )
