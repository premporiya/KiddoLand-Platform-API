"""Verify Hugging Face integration via the running API.

Run the backend first, then execute this script.
"""
import json
import os
import sys
from typing import Any, Dict

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_EMAIL = "parent@kiddoland.local"
DEFAULT_PASSWORD = "Parent123!"
DEFAULT_MODE = "home"
DEFAULT_PROMPT = "Write a short story about a friendly robot"
DEFAULT_AGE = 8


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Invalid integer for {name}: {value}")
        sys.exit(2)


def _request_json(method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
    try:
        response = requests.request(method, url, timeout=60, **kwargs)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        sys.exit(2)

    if response.status_code >= 400:
        detail = response.text.strip()
        print(f"HTTP {response.status_code} from {url}: {detail}")
        sys.exit(1)

    try:
        return response.json()
    except ValueError:
        print(f"Non-JSON response from {url}: {response.text[:200]}")
        sys.exit(1)


def main() -> int:
    base_url = _get_env("KIDDOLAND_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    email = _get_env("KIDDOLAND_TEST_EMAIL", DEFAULT_EMAIL)
    password = _get_env("KIDDOLAND_TEST_PASSWORD", DEFAULT_PASSWORD)
    mode = _get_env("KIDDOLAND_TEST_MODE", DEFAULT_MODE)
    prompt = _get_env("KIDDOLAND_TEST_PROMPT", DEFAULT_PROMPT)
    age = _get_env_int("KIDDOLAND_TEST_AGE", DEFAULT_AGE)

    login_payload = {"email": email, "password": password, "mode": mode}
    login_url = f"{base_url}/auth/login"
    login_data = _request_json("POST", login_url, json=login_payload)

    access_token = login_data.get("access_token")
    if not access_token:
        print("Login succeeded but no access token was returned.")
        return 1

    headers = {"Authorization": f"Bearer {access_token}"}
    story_payload = {"age": age, "prompt": prompt}
    story_url = f"{base_url}/story/generate"
    story_data = _request_json("POST", story_url, json=story_payload, headers=headers)

    story = story_data.get("story", "").strip()
    if not story:
        print("Story generation succeeded but returned empty content.")
        return 1

    print("Hugging Face integration OK.")
    print(f"Story length: {len(story)} chars")
    print(json.dumps({"preview": story[:200]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
