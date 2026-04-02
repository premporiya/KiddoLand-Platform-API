"""
Persistence service for AI story history records.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

ContentKind = Literal["story", "rhyme"]

from pymongo import errors

from utils.mongo import get_collection

logger = logging.getLogger(__name__)
_indexes_initialized = False

# Collection in MONGODB_DB_NAME (default kiddoland in mongo.py); not configurable via .env.
STORY_HISTORY_COLLECTION = "story_history"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _validate_required_text(value: str, field_name: str) -> str:
    cleaned = value.strip() if isinstance(value, str) else ""
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def _optional_tts_fields_from_doc(doc: dict) -> tuple[Optional[str], Optional[str]]:
    """Return (base64, media_type) for API responses; None when missing."""
    raw_b64 = doc.get("tts_audio_base64")
    raw_type = doc.get("tts_media_type")
    b64 = raw_b64.strip() if isinstance(raw_b64, str) else ""
    media = raw_type.strip() if isinstance(raw_type, str) else ""
    if not b64:
        return None, None
    if not media:
        media = "audio/mpeg"
    return b64, media


def _story_record_from_doc(doc: dict) -> dict:
    """Shape one MongoDB story_history document for API / frontend."""
    ck = _effective_content_kind(doc)
    tts_b64, tts_type = _optional_tts_fields_from_doc(doc)
    return {
        "id": str(doc.get("_id")),
        "user_id": str(doc.get("user_id", "")),
        "child_name": str(doc.get("child_name", "")),
        "prompt": str(doc.get("prompt", "")),
        "story": str(doc.get("story", "")),
        "age": doc.get("age"),
        "is_favorite": bool(doc.get("is_favorite", False)),
        "mode": str(doc.get("mode", "")),
        "type": str(doc.get("type", "")),
        "content_kind": ck,
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "tts_audio_base64": tts_b64,
        "tts_media_type": tts_type,
    }


def _effective_content_kind(doc: dict) -> ContentKind:
    """
    Resolve content_kind for filtering and UI. Honor explicit DB values; for older
    documents created before content_kind existed, detect rhyme flow via prompt
    text from buildRhymePrompt (Create Rhyme page).
    """
    raw = doc.get("content_kind")
    if isinstance(raw, str) and raw.strip().lower() == "rhyme":
        return "rhyme"
    if isinstance(raw, str) and raw.strip().lower() == "story":
        return "story"
    prompt = str(doc.get("prompt", "")).lower()
    if "this rhyme is for" in prompt or "write a short, playful rhyme" in prompt:
        return "rhyme"
    return "story"


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
    content_kind: ContentKind = "story",
    tts_audio_base64: Optional[str] = None,
    tts_media_type: Optional[str] = None,
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

    collection_name = STORY_HISTORY_COLLECTION
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
        "content_kind": content_kind,
        "created_at": timestamp_utc,
        "updated_at": timestamp_utc,
    }

    tts_b64 = tts_audio_base64.strip() if isinstance(tts_audio_base64, str) else ""
    tts_type = tts_media_type.strip() if isinstance(tts_media_type, str) else ""
    if tts_b64:
        document["tts_audio_base64"] = tts_b64
        document["tts_media_type"] = tts_type or "audio/mpeg"

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
    collection_name = STORY_HISTORY_COLLECTION
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
            records.append(_story_record_from_doc(doc))

        return records
    except errors.PyMongoError as exc:
        logger.warning("Failed to list story history records: %s", str(exc))
        return []


def list_favorite_records(*, user_id: str, limit: int = 50) -> list[dict]:
    """
    List favorite story history records for a user sorted newest first.
    """
    cleaned_user_id = _validate_required_text(user_id, "user_id")
    collection_name = STORY_HISTORY_COLLECTION
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
            row = _story_record_from_doc(doc)
            row["is_favorite"] = True
            records.append(row)

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
    content_kind: ContentKind = "story",
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

    collection_name = STORY_HISTORY_COLLECTION
    collection = get_collection(collection_name)
    if collection is None:
        return False

    timestamp_utc = _now_utc()
    # Try to find an existing history record by exact story match first.
    try:
        existing = collection.find_one({"user_id": cleaned_user_id, "story": cleaned_story})
        if existing:
            try:
                collection.update_one(
                    {"_id": existing.get("_id")},
                    {
                        "$set": {
                            "is_favorite": bool(is_favorite),
                            "updated_at": timestamp_utc,
                            "content_kind": content_kind,
                        }
                    },
                )
                return True
            except errors.PyMongoError as exc:
                logger.warning("Failed to update existing story_history favorite flag: %s", str(exc))
                return False

        # No exact-story match found — fall back to upsert by prompt+story
        filter_doc = {"user_id": cleaned_user_id, "prompt": cleaned_prompt, "story": cleaned_story}
        update_doc = {
            "$set": {
                "is_favorite": bool(is_favorite),
                "updated_at": timestamp_utc,
                "content_kind": content_kind,
            },
            "$setOnInsert": {
                "user_id": cleaned_user_id,
                "child_name": "",
                "prompt": cleaned_prompt,
                "story": cleaned_story,
                "age": age,
                "mode": mode,
                "type": record_type,
                "content_kind": content_kind,
                "created_at": timestamp_utc,
            },
        }

        collection.update_one(filter_doc, update_doc, upsert=True)
        return True
    except errors.PyMongoError as exc:
        logger.warning("Failed to mark story favorite: %s", str(exc))
        return False


def delete_story_record(*, user_id: str, story_id: str) -> bool:
    """
    Delete a story history record by ID for a specific user.
    
    Returns True if deleted successfully, False otherwise.
    Raises ValueError for invalid required field values.
    """
    from bson import ObjectId
    from bson.errors import InvalidId
    
    cleaned_user_id = _validate_required_text(user_id, "user_id")
    cleaned_story_id = _validate_required_text(story_id, "story_id")
    
    # Validate ObjectId format
    try:
        object_id = ObjectId(cleaned_story_id)
    except InvalidId:
        raise ValueError(f"Invalid story_id format: {cleaned_story_id}")
    
    collection_name = STORY_HISTORY_COLLECTION
    collection = get_collection(collection_name)
    if collection is None:
        return False
    
    try:
        result = collection.delete_one({
            "_id": object_id,
            "user_id": cleaned_user_id  # Ensure user can only delete their own stories
        })
        return result.deleted_count > 0
    except errors.PyMongoError as exc:
        logger.warning("Failed to delete story history record: %s", str(exc))
        return False


def toggle_story_favorite(*, user_id: str, story_id: str) -> Optional[bool]:
    """
    Toggle the favorite status of a story by ID.
    
    Returns the new favorite status (True/False) if successful, None if story not found.
    Raises ValueError for invalid required field values.
    """
    from bson import ObjectId
    from bson.errors import InvalidId
    
    cleaned_user_id = _validate_required_text(user_id, "user_id")
    cleaned_story_id = _validate_required_text(story_id, "story_id")
    
    # Validate ObjectId format
    try:
        object_id = ObjectId(cleaned_story_id)
    except InvalidId:
        raise ValueError(f"Invalid story_id format: {cleaned_story_id}")
    
    collection_name = STORY_HISTORY_COLLECTION
    collection = get_collection(collection_name)
    if collection is None:
        return None
    
    timestamp_utc = _now_utc()
    
    try:
        # First, find the story
        story = collection.find_one({
            "_id": object_id,
            "user_id": cleaned_user_id
        })
        
        if not story:
            return None
        
        # Toggle the is_favorite status
        current_favorite = bool(story.get("is_favorite", False))
        new_favorite = not current_favorite
        
        # Update the record
        result = collection.update_one(
            {"_id": object_id, "user_id": cleaned_user_id},
            {"$set": {"is_favorite": new_favorite, "updated_at": timestamp_utc}}
        )
        
        if result.modified_count > 0:
            return new_favorite
        
        return None
    except errors.PyMongoError as exc:
        logger.warning("Failed to toggle favorite status: %s", str(exc))
        return None