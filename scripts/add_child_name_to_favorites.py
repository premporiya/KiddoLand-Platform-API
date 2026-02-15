# Script to add 'child_name' field to all documents in the 'story_favorites' collection if missing
import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://<username>:<password>@<cluster-url>/kiddoland?retryWrites=true&w=majority")
DB_NAME = os.getenv("MONGO_DB_NAME", "kiddoland")
COLLECTION_NAME = os.getenv("MONGODB_STORY_FAVORITES_COLLECTION", "story_favorites")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

result = collection.update_many(
    {"child_name": {"$exists": False}},
    {"$set": {"child_name": "Unknown"}}
)

print(f"Updated {result.modified_count} documents to add 'child_name'.")
