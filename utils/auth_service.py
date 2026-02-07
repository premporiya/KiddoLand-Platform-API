"""
Authentication utilities: password hashing, token creation, and validation.
"""
import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pymongo import MongoClient, errors

from schemas.auth import AuthUser

load_dotenv()

_bearer_scheme = HTTPBearer(auto_error=False)
_user_cache: Optional[List[Dict[str, object]]] = None
_mongo_client: Optional[MongoClient] = None
_mongo_collection = None


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _get_auth_secret() -> bytes:
    secret = os.getenv("KIDDOLAND_AUTH_SECRET")
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="KIDDOLAND_AUTH_SECRET is not configured on the server.",
        )
    return secret.encode("utf-8")


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)


def _verify_password(password: str, salt: bytes, expected_hash: bytes) -> bool:
    candidate = _hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


def _get_mongo_collection():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return None

    global _mongo_client, _mongo_collection
    if _mongo_collection is not None:
        return _mongo_collection

    try:
        # Jaturaput Jongsubcharoen: connect to MongoDB for auth persistence.
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        _mongo_client.admin.command("ping")
        db_name = os.getenv("MONGODB_DB_NAME", "kiddoland")
        collection_name = os.getenv("MONGODB_USERS_COLLECTION", "users")
        _mongo_collection = _mongo_client[db_name][collection_name]
        _mongo_collection.create_index("email", unique=True)
    except errors.PyMongoError:
        _mongo_collection = None

    return _mongo_collection


def _deserialize_db_user(doc: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": str(doc.get("_id")),
        "email": str(doc.get("email", "")),
        "password_hash": base64.b64decode(doc.get("password_hash", "")),
        "password_salt": base64.b64decode(doc.get("password_salt", "")),
        "role": str(doc.get("role", "")),
        "modes": list(doc.get("modes") or []),
    }


def _load_user_from_db(email: str) -> Optional[Dict[str, object]]:
    collection = _get_mongo_collection()
    if collection is None:
        return None

    doc = collection.find_one({"email": email})
    if not doc:
        return None

    try:
        return _deserialize_db_user(doc)
    except (TypeError, binascii.Error, ValueError):
        return None


def _load_users_from_env() -> Optional[List[Dict[str, object]]]:
    raw = os.getenv("KIDDOLAND_AUTH_USERS")
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"KIDDOLAND_AUTH_USERS is not valid JSON: {exc}",
        )

    users: List[Dict[str, object]] = []
    for entry in data:
        email = str(entry.get("email", "")).strip().lower()
        role = str(entry.get("role", "")).strip()
        modes = entry.get("modes") or []

        if not email or not role:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Each user entry must include email and role.",
            )

        if "password_hash" in entry and "password_salt" in entry:
            password_hash = base64.b64decode(entry["password_hash"])
            password_salt = base64.b64decode(entry["password_salt"])
        elif "password" in entry:
            password_salt = secrets.token_bytes(16)
            password_hash = _hash_password(str(entry["password"]), password_salt)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Each user entry must include password or password_hash/password_salt.",
            )

        users.append(
            {
                "id": entry.get("id") or str(uuid.uuid4()),
                "email": email,
                "password_hash": password_hash,
                "password_salt": password_salt,
                "role": role,
                "modes": [str(mode) for mode in modes] if modes else [],
            }
        )

    return users


def _load_demo_users() -> List[Dict[str, object]]:
    demo_specs = [
        {
            "email": "parent@kiddoland.local",
            "password": "Parent123!",
            "role": "Parent",
            "modes": ["home"],
        },
        {
            "email": "teacher@kiddoland.local",
            "password": "Teacher123!",
            "role": "Teacher",
            "modes": ["institution"],
        },
        {
            "email": "admin@kiddoland.local",
            "password": "Admin123!",
            "role": "Admin",
            "modes": ["institution"],
        },
    ]

    users: List[Dict[str, object]] = []
    for entry in demo_specs:
        salt = secrets.token_bytes(16)
        password_hash = _hash_password(entry["password"], salt)
        users.append(
            {
                "id": str(uuid.uuid4()),
                "email": entry["email"],
                "password_hash": password_hash,
                "password_salt": salt,
                "role": entry["role"],
                "modes": entry["modes"],
            }
        )

    return users


def _get_user_store() -> List[Dict[str, object]]:
    global _user_cache
    if _user_cache is not None:
        return _user_cache

    users = _load_users_from_env()
    _user_cache = users if users is not None else _load_demo_users()
    return _user_cache


def authenticate_user(email: str, password: str, mode: str) -> Dict[str, object]:
    normalized_email = email.strip().lower()

    user = _load_user_from_db(normalized_email)
    if user is None:
        users = _get_user_store()
        user = next((item for item in users if item["email"] == normalized_email), None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not _verify_password(password, user["password_salt"], user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    modes = user.get("modes") or []
    if modes and mode not in modes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not permitted to access this mode.",
        )

    return user


def register_user(email: str, password: str, mode: str, role: str) -> Dict[str, object]:
    collection = _get_mongo_collection()
    if collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MongoDB is not configured for registration.",
        )

    normalized_email = email.strip().lower()
    existing = collection.find_one({"email": normalized_email})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    password_salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, password_salt)
    modes = [mode]
    doc = {
        "email": normalized_email,
        "password_hash": base64.b64encode(password_hash).decode("ascii"),
        "password_salt": base64.b64encode(password_salt).decode("ascii"),
        "role": role,
        "modes": modes,
        "created_at": int(time.time()),
    }

    result = collection.insert_one(doc)
    return {
        "id": str(result.inserted_id),
        "email": normalized_email,
        "role": role,
        "modes": modes,
    }


def create_access_token(user: Dict[str, object], mode: str) -> Dict[str, object]:
    now = int(time.time())
    ttl_seconds = int(os.getenv("KIDDOLAND_AUTH_TTL_SECONDS", "3600"))
    payload = {
        "sub": user["id"],
        "role": user["role"],
        "mode": mode,
        "iat": now,
        "exp": now + ttl_seconds,
    }

    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload_bytes)

    signature = hmac.new(_get_auth_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest()
    token = f"{payload_b64}.{_b64url_encode(signature)}"

    return {"token": token, "expires_in": ttl_seconds}


def verify_access_token(token: str) -> AuthUser:
    parts = token.split(".")
    if len(parts) != 2:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format.",
        )

    payload_b64, signature_b64 = parts
    expected_signature = hmac.new(
        _get_auth_secret(), payload_b64.encode("ascii"), hashlib.sha256
    ).digest()

    try:
        provided_signature = _b64url_decode(signature_b64)
    except (ValueError, binascii.Error):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature.",
        )

    if not hmac.compare_digest(expected_signature, provided_signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature.",
        )

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
        )

    exp = payload.get("exp")
    if not isinstance(exp, int) or exp < int(time.time()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
        )

    role = payload.get("role")
    mode = payload.get("mode")
    user_id = payload.get("sub")

    if not role or not mode or not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing required claims.",
        )

    if role not in {"Parent", "Teacher", "Admin", "Librarian"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has an invalid role.",
        )

    if mode not in {"home", "institution"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has an invalid mode.",
        )

    return AuthUser(user_id=user_id, role=role, mode=mode)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> AuthUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return verify_access_token(credentials.credentials)


def require_roles(roles: List[str]):
    def _role_guard(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User does not have access to this resource.",
            )
        return user

    return _role_guard
