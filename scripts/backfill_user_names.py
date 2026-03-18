"""
Backfill missing user `name` fields in MongoDB.

Default behavior:
- For users without a `name` (or empty/whitespace), set name to the email prefix.

Environment:
- MONGODB_URI
- MONGODB_DB_NAME (default: kiddoland)
- MONGODB_USERS_COLLECTION (default: users)
"""
import argparse
import os
from typing import Optional

from pymongo import MongoClient


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _fallback_name_from_email(email: str) -> str:
    trimmed = email.strip()
    at_index = trimmed.find("@")
    return trimmed[:at_index] if at_index > 0 else trimmed


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill missing user name fields.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without writing to the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of updates (0 = no limit).",
    )
    args = parser.parse_args()

    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise SystemExit("MONGODB_URI is not set.")

    db_name = os.getenv("MONGODB_DB_NAME", "kiddoland")
    collection_name = os.getenv("MONGODB_USERS_COLLECTION", "users")

    client = MongoClient(mongo_uri)
    collection = client[db_name][collection_name]

    query = {
        "$or": [
            {"name": {"$exists": False}},
            {"name": None},
            {"name": ""},
            {"name": {"$regex": r"^\s*$"}},
        ]
    }
    cursor = collection.find(query)

    updated = 0
    scanned = 0
    for doc in cursor:
        scanned += 1
        if args.limit and updated >= args.limit:
            break

        email = _optional_str(doc.get("email"))
        if not email:
            continue

        fallback_name = _fallback_name_from_email(email)
        if not fallback_name:
            continue

        if args.dry_run:
            print(f"[DRY RUN] would set name for {email} -> {fallback_name}")
            updated += 1
            continue

        result = collection.update_one({"_id": doc["_id"]}, {"$set": {"name": fallback_name}})
        if result.modified_count:
            print(f"Updated {email} -> {fallback_name}")
            updated += 1

    print(f"Scanned {scanned} user(s). Updated {updated} user(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
