"""Pydantic schemas for story-to-video generation."""
from typing import Literal

from pydantic import BaseModel, Field


class GenerateVideoRequest(BaseModel):
    story: str = Field(..., min_length=10, max_length=50000)
    include_voice: bool = True
    image_provider: Literal["gemini", "huggingface"] = Field(
        default="gemini",
        description="Illustrations: gemini (AI Studio free tier) or huggingface (Inference Providers).",
    )
