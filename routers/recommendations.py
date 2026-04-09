from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query

from schemas.recommendations import BookRecommendation
from utils.semantic_book_recommendations import fetch_semantic_recommendations

router = APIRouter()


@router.get("/recommend-books", response_model=list[BookRecommendation])
async def recommend_books(
    topic: str = Query(..., min_length=1, description="Subject or theme for recommendations"),
    age: Optional[int] = Query(None, ge=0, le=18, description="Optional child age for tailoring reasons"),
) -> list[BookRecommendation]:
    """Book recommendations from Open Library + semantic ranker (GET + query params)."""
    print("RECEIVED QUERY:", topic)
    try:
        items = await fetch_semantic_recommendations(topic, age)
        return [BookRecommendation(**item) for item in items]
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Open Library returned an unexpected response.",
        ) from None
    except httpx.HTTPStatusError:
        raise HTTPException(
            status_code=500,
            detail="Open Library returned an error. Please try again later.",
        ) from None
    except httpx.RequestError:
        raise HTTPException(
            status_code=500,
            detail="Unable to reach Open Library. Please try again later.",
        ) from None

