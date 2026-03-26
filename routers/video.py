"""
Story-to-video: scenes → illustrations (Gemini or Hugging Face) → moviepy slideshow (+ optional gTTS).
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from schemas.auth import AuthUser
from schemas.video import GenerateVideoRequest
from utils.auth_service import get_current_user
from utils.gemini_image import GeminiImageError
from utils.huggingface_client import HuggingFaceError
from utils.safety_filter import clean_text_for_model, is_content_safe
from utils.story_video import build_story_video_file

logger = logging.getLogger(__name__)

router = APIRouter()


def _delete_file(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


@router.post("/generate-video")
def generate_story_video(
    request: GenerateVideoRequest,
    current_user: AuthUser = Depends(get_current_user),
):
    cleaned = clean_text_for_model(request.story)
    if not cleaned or len(cleaned) < 10:
        raise HTTPException(
            status_code=400,
            detail="Story text is too short.",
        )
    if not is_content_safe(cleaned):
        raise HTTPException(
            status_code=400,
            detail="Story contains content that cannot be used for video.",
        )

    try:
        path = build_story_video_file(
            cleaned,
            request.include_voice,
            request.image_provider,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GeminiImageError as exc:
        logger.warning("Story video Gemini error: %s", exc)
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc
    except HuggingFaceError as exc:
        logger.warning("Story video Hugging Face error: %s", exc)
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.warning("Story video runtime error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Story video failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate video. Please try again later.",
        ) from exc

    return FileResponse(
        path,
        media_type="video/mp4",
        filename="kiddoland-story-video.mp4",
        background=BackgroundTask(_delete_file, path),
    )
