from __future__ import annotations

from datetime import datetime, timezone

from pymongo import errors

from utils.mongo import get_collection

DOWNLOAD_LIMIT_COLLECTION = "download_limits"
FREE_MONTHLY_DOWNLOAD_LIMIT = 3
_download_indexes_initialized = False


def _month_key(now_utc: datetime | None = None) -> str:
    stamp = now_utc or datetime.now(timezone.utc)
    return f"{stamp.year:04d}-{stamp.month:02d}"


def _ensure_indexes(collection) -> None:
    global _download_indexes_initialized
    if _download_indexes_initialized:
        return
    try:
        collection.create_index([("user_id", 1), ("month", 1)], unique=True)
        _download_indexes_initialized = True
    except errors.PyMongoError:
        pass


def get_monthly_download_usage(user_id: str) -> dict:
    collection = get_collection(DOWNLOAD_LIMIT_COLLECTION)
    if collection is None:
        return {"used": 0, "limit": FREE_MONTHLY_DOWNLOAD_LIMIT, "remaining": FREE_MONTHLY_DOWNLOAD_LIMIT}
    _ensure_indexes(collection)
    month = _month_key()
    doc = collection.find_one({"user_id": user_id, "month": month}) or {}
    used = int(doc.get("download_count", 0) or 0)
    remaining = max(0, FREE_MONTHLY_DOWNLOAD_LIMIT - used)
    return {"used": used, "limit": FREE_MONTHLY_DOWNLOAD_LIMIT, "remaining": remaining}


def consume_download_slot(user_id: str) -> dict:
    collection = get_collection(DOWNLOAD_LIMIT_COLLECTION)
    if collection is None:
        usage = {"used": 0, "limit": FREE_MONTHLY_DOWNLOAD_LIMIT, "remaining": FREE_MONTHLY_DOWNLOAD_LIMIT}
        return {**usage, "allowed": True}

    _ensure_indexes(collection)
    month = _month_key()
    now = datetime.now(timezone.utc)
    doc = collection.find_one({"user_id": user_id, "month": month}) or {}
    used = int(doc.get("download_count", 0) or 0)
    if used >= FREE_MONTHLY_DOWNLOAD_LIMIT:
        return {
            "allowed": False,
            "used": used,
            "limit": FREE_MONTHLY_DOWNLOAD_LIMIT,
            "remaining": 0,
        }

    next_used = used + 1
    try:
        collection.update_one(
            {"user_id": user_id, "month": month},
            {
                "$set": {"updated_at": now},
                "$setOnInsert": {"created_at": now},
                "$inc": {"download_count": 1},
            },
            upsert=True,
        )
    except errors.PyMongoError:
        # Fail open to avoid blocking legitimate downloads on transient DB issues.
        return {
            "allowed": True,
            "used": used,
            "limit": FREE_MONTHLY_DOWNLOAD_LIMIT,
            "remaining": max(0, FREE_MONTHLY_DOWNLOAD_LIMIT - used),
        }

    return {
        "allowed": True,
        "used": next_used,
        "limit": FREE_MONTHLY_DOWNLOAD_LIMIT,
        "remaining": max(0, FREE_MONTHLY_DOWNLOAD_LIMIT - next_used),
    }


def reset_monthly_download_usage(user_id: str) -> bool:
    collection = get_collection(DOWNLOAD_LIMIT_COLLECTION)
    if collection is None:
        return False
    _ensure_indexes(collection)
    month = _month_key()
    try:
        collection.delete_one({"user_id": user_id, "month": month})
        return True
    except errors.PyMongoError:
        return False
