"""
Pydantic Schemas for Story API
"""
from pydantic import BaseModel, Field


class StoryGenerateRequest(BaseModel):
    """Request model for story generation"""
    age: int = Field(
        ...,
        ge=1,
        le=10,
        description="Child's age (1-10)"
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Free-form story prompt"
    )
    include_tts: bool = Field(
        default=False,
        description="When true, also return TTS audio data for the generated story"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 10,
                "prompt": "Write a story about a shy dragon who learns to make friends",
                "include_tts": True,
            }
        }


class StoryGenerateResponse(BaseModel):
    """Response model for story generation"""
    story: str = Field(
        ...,
        description="Generated story text"
    )
    tts_audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded audio payload when include_tts=true"
    )
    tts_media_type: str | None = Field(
        default=None,
        description="Media type of synthesized audio, for example audio/mpeg"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "story": "Once upon a time, there was a shy dragon named Ember...",
                "tts_audio_base64": "UklGRhQAAABXQVZFZm10IBAAAAABAAEA...",
                "tts_media_type": "audio/mpeg",
            }
        }


class StoryRewriteRequest(BaseModel):
    """Request model for story rewriting"""
    age: int = Field(
        ...,
        ge=1,
        le=10,
        description="Child's age (1-10)"
    )
    original_story: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Original story text"
    )
    instruction: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Free-form rewrite instruction (e.g., 'make it happier', 'change the ending')"
    )
    include_tts: bool = Field(
        default=False,
        description="When true, also return TTS audio data for the rewritten story"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 10,
                "original_story": "Once upon a time, there was a shy dragon...",
                "instruction": "Change the middle part to make it funnier",
                "include_tts": True,
            }
        }


class StoryRewriteResponse(BaseModel):
    """Response model for story rewriting"""
    story: str = Field(
        ...,
        description="Rewritten story text"
    )
    tts_audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded audio payload when include_tts=true"
    )
    tts_media_type: str | None = Field(
        default=None,
        description="Media type of synthesized audio, for example audio/mpeg"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "story": "Once upon a time, there was a shy dragon who was terrified of sneezing...",
                "tts_audio_base64": "UklGRhQAAABXQVZFZm10IBAAAAABAAEA...",
                "tts_media_type": "audio/mpeg",
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(
        ...,
        description="Error message"
    )
