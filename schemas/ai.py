"""
Pydantic Schemas for AI sample endpoint
"""
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
                "prompt": "Say hello to a curious 7-year-old who loves space."
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
