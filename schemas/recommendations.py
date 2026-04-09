from typing import Optional

from pydantic import BaseModel, Field


class BookRecommendation(BaseModel):
    title: str
    author: str
    cover: Optional[str] = Field(None, description="Cover image URL when available")
    link: str
    reason: str
    score: Optional[float] = Field(
        None,
        description="Semantic similarity score (0–1 scale) when from embedding ranker",
    )
