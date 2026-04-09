"""
Pydantic schemas for AI-powered learning activity generation (quiz).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ActivityGenerateRequest(BaseModel):
    """POST /ai/activity request body."""

    age_band: str = Field(
        ...,
        min_length=1,
        description='e.g. "5-7"',
    )
    theme: str = Field(..., min_length=1, description="Topic theme, e.g. Animals")
    learning_goal: str = Field(
        ...,
        min_length=1,
        description="What the child should practice, e.g. Vocabulary",
    )
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(
        default=None,
        description="Optional difficulty; defaults to medium in the prompt if omitted",
    )

    @field_validator("age_band", "theme", "learning_goal", mode="before")
    @classmethod
    def strip_strings(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("age_band", "theme", "learning_goal")
    @classmethod
    def not_empty_after_strip(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v


class ActivityQuestion(BaseModel):
    """One quiz question in the activity payload."""

    prompt: str = Field(..., min_length=1)
    options: list[str] = Field(..., min_length=2, max_length=3)
    correct_index: int = Field(..., ge=0)
    feedback_correct: str = Field(..., min_length=1)
    feedback_incorrect: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def correct_index_in_range(self) -> ActivityQuestion:
        if self.correct_index >= len(self.options):
            raise ValueError("correct_index must be within options range")
        return self


class ActivityQuizData(BaseModel):
    """Validated activity returned to clients."""

    title: str = Field(..., min_length=1)
    questions: list[ActivityQuestion] = Field(..., min_length=5, max_length=5)


class ActivitySuccessEnvelope(BaseModel):
    success: Literal[True] = True
    data: ActivityQuizData


class ActivityErrorEnvelope(BaseModel):
    success: Literal[False] = False
    error: str = Field(..., min_length=1)
