"""
Persistence service for AI story history records.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Literal, Optional

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


def save_story_record(
    *,
    user_id: str,
    child_name: str,
    prompt: str,
    story: str,
    age: Optional[int],
    is_favorite: Optional[bool] = False,
    mode: str,
    record_type: Literal["generate", "rewrite"],
) -> bool:
    """
    Save a story history record.

    Returns True when persisted, False when persistence is unavailable.
    Raises ValueError for invalid required field values.
    """
    cleaned_user_id = _validate_required_text(user_id, "user_id")
    cleaned_child_name = _validate_required_text(child_name, "child_name")
    cleaned_prompt = _validate_required_text(prompt, "prompt")
    cleaned_story = _validate_required_text(story, "story")

    if record_type not in {"generate", "rewrite"}:
        raise ValueError("type must be either 'generate' or 'rewrite'")

    collection_name = os.getenv("MONGODB_STORY_HISTORY_COLLECTION", "story_history")
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
            logger.warning("Failed to initialize story history indexes: %s", str(exc))

    timestamp_utc = _now_utc()
    document = {
        "user_id": cleaned_user_id,
        "child_name": cleaned_child_name,
        "prompt": cleaned_prompt,
        "story": cleaned_story,
        "age": age,
        "is_favorite": bool(is_favorite),
        "mode": mode,
        "type": record_type,
        "created_at": timestamp_utc,
        "updated_at": timestamp_utc,
    }

    try:
        collection.insert_one(document)
        return True
    except errors.PyMongoError as exc:
        logger.warning("Failed to persist story history record: %s", str(exc))
        return False


def list_story_records(*, user_id: str, limit: int = 50) -> list[dict]:
    """
    List story history records for a user sorted newest first.
    """
    cleaned_user_id = _validate_required_text(user_id, "user_id")
    collection_name = os.getenv("MONGODB_STORY_HISTORY_COLLECTION", "story_history")
    collection = get_collection(collection_name)
    if collection is None:
        return []

    safe_limit = max(1, min(limit, 200))

    try:
        cursor = (
            collection.find({"user_id": cleaned_user_id})
            .sort("created_at", -1)
            .limit(safe_limit)
        )

        records: list[dict] = []
        for doc in cursor:
            records.append(
                {
                    "id": str(doc.get("_id")),
                    "user_id": str(doc.get("user_id", "")),
                    "child_name": str(doc.get("child_name", "")),
                    "prompt": str(doc.get("prompt", "")),
                    "story": str(doc.get("story", "")),
                    "age": doc.get("age"),
                    "is_favorite": bool(doc.get("is_favorite", False)),
                    "mode": str(doc.get("mode", "")),
                    "type": str(doc.get("type", "")),
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                }
            )

        return records
    except errors.PyMongoError as exc:
        logger.warning("Failed to list story history records: %s", str(exc))
        return []


def list_favorite_records(*, user_id: str, limit: int = 50) -> list[dict]:
    """
    List favorite story history records for a user sorted newest first.
    """
    cleaned_user_id = _validate_required_text(user_id, "user_id")
    collection_name = os.getenv("MONGODB_STORY_HISTORY_COLLECTION", "story_history")
    collection = get_collection(collection_name)
    if collection is None:
        return []

    safe_limit = max(1, min(limit, 200))

    try:
        cursor = (
            collection.find({"user_id": cleaned_user_id, "is_favorite": True})
            .sort("created_at", -1)
            .limit(safe_limit)
        )

        records: list[dict] = []
        for doc in cursor:
            records.append(
                {
                    "id": str(doc.get("_id")),
                    "user_id": str(doc.get("user_id", "")),
                    "child_name": str(doc.get("child_name", "")),
                    "prompt": str(doc.get("prompt", "")),
                    "story": str(doc.get("story", "")),
                    "age": doc.get("age"),
                    "is_favorite": True,
                    "mode": str(doc.get("mode", "")),
                    "type": str(doc.get("type", "")),
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                }
            )

        return records
    except errors.PyMongoError as exc:
        logger.warning("Failed to list favorite story history records: %s", str(exc))
        return []


def mark_story_favorite(
    *,
    user_id: str,
    prompt: str,
    story: str,
    age: Optional[int],
    mode: str,
    record_type: Literal["generate", "rewrite"],
    is_favorite: bool = True,
) -> bool:
    """
    Mark a story history record as favorite (or not). If a matching history
    record does not exist it will be upserted with the provided fields.
    """
    cleaned_user_id = _validate_required_text(user_id, "user_id")
    cleaned_prompt = _validate_required_text(prompt, "prompt")
    cleaned_story = _validate_required_text(story, "story")

    if record_type not in {"generate", "rewrite"}:
        raise ValueError("type must be either 'generate' or 'rewrite'")

    collection_name = os.getenv("MONGODB_STORY_HISTORY_COLLECTION", "story_history")
    collection = get_collection(collection_name)
    if collection is None:
        return False

    timestamp_utc = _now_utc()

    filter_doc = {"user_id": cleaned_user_id, "prompt": cleaned_prompt, "story": cleaned_story}
    update_doc = {
        "$set": {"is_favorite": bool(is_favorite), "updated_at": timestamp_utc},
        "$setOnInsert": {
            "user_id": cleaned_user_id,
            "child_name": "",
            "prompt": cleaned_prompt,
            "story": cleaned_story,
            "age": age,
            "mode": mode,
            "type": record_type,
            "created_at": timestamp_utc,
        },
    }

    try:
        collection.update_one(filter_doc, update_doc, upsert=True)
        return True
    except errors.PyMongoError as exc:
        logger.warning("Failed to mark story favorite: %s", str(exc))
        return False
