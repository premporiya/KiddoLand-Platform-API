"""
Application configuration and validation helpers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables early.
load_dotenv()


@dataclass(frozen=True)
class HuggingFaceConfig:
    api_token: str
    api_url: str
    model_id: str

    def safe_summary(self) -> dict:
        """Return a redacted view safe for logs/debugging."""
        return {
            "api_url": self.api_url,
            "model_id": self.model_id,
            "api_token": "redacted",
        }


_HF_CONFIG: Optional[HuggingFaceConfig] = None


def _read_env(name: str) -> str:
    return os.getenv(name, "").strip()


def _load_huggingface_config() -> HuggingFaceConfig:
    missing = []

    api_token = _read_env("HUGGINGFACE_API_TOKEN")
    api_url = _read_env("HUGGINGFACE_API_URL")
    model_id = _read_env("HUGGINGFACE_MODEL")

    if not api_token:
        missing.append("HUGGINGFACE_API_TOKEN")
    if not api_url:
        missing.append("HUGGINGFACE_API_URL")
    if not model_id:
        missing.append("HUGGINGFACE_MODEL")

    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            "Missing required Hugging Face configuration: "
            f"{missing_list}. Set these as environment variables before starting the API."
        )

    return HuggingFaceConfig(
        api_token=api_token,
        api_url=api_url,
        model_id=model_id,
    )


def get_huggingface_config() -> HuggingFaceConfig:
    global _HF_CONFIG
    if _HF_CONFIG is None:
        _HF_CONFIG = _load_huggingface_config()
    return _HF_CONFIG


def validate_huggingface_config() -> HuggingFaceConfig:
    """Validate Hugging Face configuration and return the cached config."""
    return get_huggingface_config()
