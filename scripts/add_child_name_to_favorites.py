"""
Script to add 'child_name' field to all documents in the `story_history` collection
if missing. The project now stores favorites inside `story_history` (field
`is_favorite`) so this script was updated to operate on that collection.
"""
import os
from pymongo import MongoClient

# Read the project's Mongo env vars
MONGO_URI = os.getenv("MONGODB_URI", "mongodb+srv://<username>:<password>@<cluster-url>/")
DB_NAME = os.getenv("MONGODB_DB_NAME", os.getenv("MONGODB_DB_NAME", "kiddoland"))
COLLECTION_NAME = os.getenv("MONGODB_STORY_HISTORY_COLLECTION", "story_history")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

result = collection.update_many(
    {"child_name": {"$exists": False}},
    {"$set": {"child_name": "Unknown"}}
)

print(f"Updated {result.modified_count} documents to add 'child_name' in {COLLECTION_NAME}.")
