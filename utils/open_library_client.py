"""
Async client for Open Library search API.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx

OPEN_LIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
RECOMMENDATION_LIMIT = 5
REQUEST_TIMEOUT = 30.0


def _normalize_title(raw: Any) -> str:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return "Unknown Title"


def _first_author(doc: dict) -> str:
    authors = doc.get("author_name")
    if isinstance(authors, list) and authors:
        name = authors[0]
        if isinstance(name, str) and name.strip():
            return name.strip()
    return "Unknown Author"


def _build_link(doc: dict) -> str:
    key = doc.get("key")
    if isinstance(key, str) and key.startswith("/"):
        return f"https://openlibrary.org{key}"
    return "https://openlibrary.org"


def _build_cover(doc: dict) -> Optional[str]:
    cover_i = doc.get("cover_i")
    if cover_i is None:
        return None
    try:
        return f"https://covers.openlibrary.org/b/id/{int(cover_i)}-M.jpg"
    except (TypeError, ValueError):
        return None


def _reason_for(topic: str, age: Optional[int]) -> str:
    if age is not None:
        if age <= 3:
            return "Perfect for toddlers"
        if age <= 7:
            return "Great for young learners"
        return "Recommended for growing readers"
    return f"Because you like {topic} stories"


def _doc_to_item(doc: dict, topic: str, age: Optional[int]) -> dict[str, Any]:
    cover = _build_cover(doc)
    return {
        "title": _normalize_title(doc.get("title")),
        "author": _first_author(doc),
        "cover": cover,
        "link": _build_link(doc),
        "reason": _reason_for(topic, age),
    }


async def fetch_recommendations(topic: str, age: Optional[int] = None) -> list[dict[str, Any]]:
    cleaned = topic.strip()
    if not cleaned:
        return []
    query = f"{cleaned}+kids"

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(
            OPEN_LIBRARY_SEARCH_URL,
            params={"q": query},
        )
        response.raise_for_status()
        data = response.json()

    docs = data.get("docs")
    if not isinstance(docs, list) or not docs:
        return []

    # Up to five titles from the first page (typical use: 3–5 recommendations).
    selected = docs[:RECOMMENDATION_LIMIT]
    return [_doc_to_item(d, topic.strip(), age) for d in selected if isinstance(d, dict)]
