"""
AI Router
Sample endpoint for testing Hugging Face integration
"""
from fastapi import APIRouter, Depends, HTTPException

from schemas.ai import AiSampleRequest, AiSampleResponse
from utils.auth_service import get_current_user
from utils.huggingface_client import HuggingFaceError, sample_completion

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.post("/sample", response_model=AiSampleResponse)
def sample_ai_endpoint(request: AiSampleRequest) -> AiSampleResponse:
    """
    Sample AI endpoint that calls Hugging Face using configured values.
    """
    try:
        output = sample_completion(request.prompt)
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
