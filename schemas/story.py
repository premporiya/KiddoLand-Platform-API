"""
Pydantic Schemas for Story API
"""
from pydantic import BaseModel, Field


class StoryGenerateRequest(BaseModel):
    """Request model for story generation"""
    age: int = Field(
        ...,
        ge=1,
        le=18,
        description="Child's age (1-18)"
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Free-form story prompt"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 10,
                "prompt": "Write a story about a shy dragon who learns to make friends"
            }
        }


class StoryGenerateResponse(BaseModel):
    """Response model for story generation"""
    story: str = Field(
        ...,
        description="Generated story text"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "story": "Once upon a time, there was a shy dragon named Ember..."
            }
        }


class StoryRewriteRequest(BaseModel):
    """Request model for story rewriting"""
    age: int = Field(
        ...,
        ge=1,
        le=18,
        description="Child's age (1-18)"
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 10,
                "original_story": "Once upon a time, there was a shy dragon...",
                "instruction": "Change the middle part to make it funnier"
            }
        }


class StoryRewriteResponse(BaseModel):
    """Response model for story rewriting"""
    story: str = Field(
        ...,
        description="Rewritten story text"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "story": "Once upon a time, there was a shy dragon who was terrified of sneezing..."
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(
        ...,
        description="Error message"
    )
