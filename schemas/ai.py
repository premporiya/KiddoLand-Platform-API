"""
Pydantic Schemas for AI sample endpoint
"""
from typing import Literal
from datetime import datetime

from pydantic import BaseModel, Field


class AiSampleRequest(BaseModel):
    """Request model for sample AI call"""
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Prompt for the sample AI response",
    )
    include_tts: bool = Field(
        default=False,
        description="When true, also return TTS audio data for the generated output",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Say hello to a curious 7-year-old who loves space.",
                "include_tts": True,
            }
        }


class AiSampleResponse(BaseModel):
    """Response model for sample AI call"""
    output: str = Field(
        ...,
        description="Generated response text",
    )
    tts_audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded audio payload when include_tts=true",
    )
    tts_media_type: str | None = Field(
        default=None,
        description="Media type of synthesized audio, for example audio/mpeg",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "output": "Hi there, space explorer! Ready to zoom past the stars today?",
                "tts_audio_base64": "UklGRhQAAABXQVZFZm10IBAAAAABAAEA...",
                "tts_media_type": "audio/mpeg",
            }
        }


class AiSaveFavoriteRequest(BaseModel):
    """Request model for saving a story to favorites"""
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Prompt used for generation",
    )
    story: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Story text to save as favorite",
    )
    age: int = Field(
        ...,
        ge=1,
        le=10,
        description="Child's age (1-10)",
    )
    type: Literal["generate", "rewrite"] = Field(
        "generate",
        description="Record type: generate or rewrite",
    )
    content_kind: Literal["story", "rhyme"] = Field(
        "story",
        description="Whether this favorite is from story creation or rhyme creation",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Tell a story about a tiny astronaut cat.",
                "story": "Luna the cat put on her silver helmet...",
                "age": 8,
                "type": "generate",
                "content_kind": "story",
            }
        }


class AiSaveFavoriteResponse(BaseModel):
    """Response model for save-favorite endpoint"""
    saved: bool = Field(
        ...,
        description="True when favorite was successfully stored",
    )
    message: str = Field(
        ...,
        description="Favorite save operation status message",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "saved": True,
                "message": "Story saved to favorites.",
            }
        }


class AiStoryHistoryItem(BaseModel):
    id: str
    user_id: str
    child_name: str
    prompt: str
    story: str
    age: int | None
    is_favorite: bool = False
    mode: str
    type: Literal["generate", "rewrite"]
    content_kind: Literal["story", "rhyme"] = "story"
    created_at: datetime | None
    updated_at: datetime | None
    tts_audio_base64: str | None = None
    tts_media_type: str | None = None


class AiStoryHistoryResponse(BaseModel):
    items: list[AiStoryHistoryItem]


class DownloadAttemptRequest(BaseModel):
    download_type: Literal["audio", "pdf"] = Field(
        ...,
        description="Download intent type.",
    )


class DownloadAttemptResponse(BaseModel):
    allowed: bool = Field(..., description="Whether this download is allowed.")
    plan: Literal["free", "paid"] = Field(..., description="Current user plan.")
    used_downloads: int = Field(..., description="Downloads used in current month.")
    monthly_limit: int | None = Field(
        default=None,
        description="Monthly limit for this user plan; null means unlimited.",
    )
    remaining_downloads: int | None = Field(
        default=None,
        description="Remaining downloads in current month; null means unlimited.",
    )
    message: str = Field(..., description="User-friendly status message.")
