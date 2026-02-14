"""
Persistence service for story favorite records.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Literal

from pymongo import errors

from utils.mongo import get_collection

logger = logging.getLogger(__name__)
_indexes_initialized = False


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _validate_required_text(value: str, field_name: str) -> str:
    cleaned = value.strip() if isinstance(value, str) else ""
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def save_favorite_record(
    *,
    user_id: str,
    prompt: str,
    story: str,
    age: int,
    mode: str,
    record_type: Literal["generate", "rewrite"],
) -> bool:
    """
    Save a story favorite record.

    Returns True when persisted, False when persistence is unavailable.
    Raises ValueError for invalid required field values.
    """
    cleaned_user_id = _validate_required_text(user_id, "user_id")
    cleaned_prompt = _validate_required_text(prompt, "prompt")
    cleaned_story = _validate_required_text(story, "story")

    if record_type not in {"generate", "rewrite"}:
        raise ValueError("type must be either 'generate' or 'rewrite'")

    collection_name = os.getenv("MONGODB_STORY_FAVORITES_COLLECTION", "story_favorites")
    collection = get_collection(collection_name)
    if collection is None:
        return False

    global _indexes_initialized
    if not _indexes_initialized:
        try:
            collection.create_index("user_id")
            collection.create_index("created_at")
            _indexes_initialized = True
        except errors.PyMongoError as exc:
            logger.warning("Failed to initialize story favorites indexes: %s", str(exc))

    timestamp_utc = _now_utc()
    document = {
        "user_id": cleaned_user_id,
        "prompt": cleaned_prompt,
        "story": cleaned_story,
        "age": age,
        "mode": mode,
        "type": record_type,
        "created_at": timestamp_utc,
        "updated_at": timestamp_utc,
    }

    try:
        collection.insert_one(document)
        return True
    except errors.PyMongoError as exc:
        logger.warning("Failed to persist story favorite record: %s", str(exc))
        return False
