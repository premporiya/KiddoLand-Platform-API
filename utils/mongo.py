"""
MongoDB utilities shared across backend services.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from pymongo import MongoClient, errors
from pymongo.collection import Collection

_mongo_client: Optional[MongoClient] = None
_collection_cache: Dict[str, Collection] = {}


def get_collection(collection_name: str) -> Optional[Collection]:
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        return None

    global _mongo_client
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            _mongo_client.admin.command("ping")
        except errors.PyMongoError:
            _mongo_client = None
            return None

    if collection_name in _collection_cache:
        return _collection_cache[collection_name]

    db_name = os.getenv("MONGODB_DB_NAME", "kiddoland")
    collection = _mongo_client[db_name][collection_name]
    _collection_cache[collection_name] = collection
    return collection
