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

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Say hello to a curious 7-year-old who loves space.",
            }
        }


class AiSampleResponse(BaseModel):
    """Response model for sample AI call"""
    output: str = Field(
        ...,
        description="Generated response text",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "output": "Hi there, space explorer! Ready to zoom past the stars today?"
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

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Tell a story about a tiny astronaut cat.",
                "story": "Luna the cat put on her silver helmet...",
                "age": 8,
                "type": "generate",
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
    created_at: datetime | None
    updated_at: datetime | None


class AiStoryHistoryResponse(BaseModel):
    items: list[AiStoryHistoryItem]
