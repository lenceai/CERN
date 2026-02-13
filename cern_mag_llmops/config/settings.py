"""
Global settings and configuration for the CERN Magazine LLMOps framework.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_bool_env(name: str, default: bool) -> bool:
    """Parse common truthy/falsy env values into a bool."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    """Get an integer env var with fallback and clear error messages."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got '{value}'") from exc


def _get_float_env(name: str, default: float) -> float:
    """Get a float env var with fallback and clear error messages."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got '{value}'") from exc


def _get_csv_env(name: str, default: List[str]) -> List[str]:
    """Get a comma-separated env var as a list of strings."""
    value = os.getenv(name)
    if value is None:
        return default
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def require_openai_api_key() -> str:
    """
    Return OPENAI_API_KEY or raise with an actionable error.

    Components that actually call OpenAI should use this helper instead of
    validating at module import time.
    """
    if OPENAI_API_KEY:
        return OPENAI_API_KEY
    raise ValueError(
        "OPENAI_API_KEY is required for this operation. "
        "Set it in your environment or .env file."
    )


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
PDFS_DIR = os.path.join(DATA_DIR, "pdfs")
VECTORDB_DIR = os.path.join(DATA_DIR, "vectordb")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PDFS_DIR, exist_ok=True)
os.makedirs(VECTORDB_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# API and model credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL", "")

# OpenAI call behavior
OPENAI_TIMEOUT_SECONDS = _get_float_env("OPENAI_TIMEOUT_SECONDS", 60.0)
OPENAI_MAX_RETRIES = _get_int_env("OPENAI_MAX_RETRIES", 3)

# Model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")

# CERN Courier crawler settings
CERN_BASE_URL = os.getenv("CERN_BASE_URL", "https://home.cern/resources")
CRAWL_START_PAGE = _get_int_env("CRAWL_START_PAGE", 0)
CRAWL_END_PAGE = _get_int_env("CRAWL_END_PAGE", 7)
CRAWL_DELAY = _get_float_env("CRAWL_DELAY", 1.0)  # seconds between requests
CRAWL_MAX_RETRIES = _get_int_env("CRAWL_MAX_RETRIES", 3)
CRAWL_REQUEST_TIMEOUT_SECONDS = _get_float_env("CRAWL_REQUEST_TIMEOUT_SECONDS", 20.0)

# Data processing settings
CHUNK_SIZE = _get_int_env("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _get_int_env("CHUNK_OVERLAP", 200)
MAX_TOKENS_PER_DOC = _get_int_env("MAX_TOKENS_PER_DOC", 2000)
MAX_EMBEDDING_BATCH_SIZE = _get_int_env("MAX_EMBEDDING_BATCH_SIZE", 100)

# RAG settings
TOP_K_DOCUMENTS = _get_int_env("TOP_K_DOCUMENTS", 4)
SIMILARITY_THRESHOLD = _get_float_env("SIMILARITY_THRESHOLD", 0.7)

# API server settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _get_int_env("API_PORT", 8000)
DEBUG_MODE = _get_bool_env("DEBUG", False)

# API security settings
API_AUTH_ENABLED = _get_bool_env("API_AUTH_ENABLED", False)
API_AUTH_KEY = os.getenv("API_AUTH_KEY", "")

# CORS settings
CORS_ALLOW_ORIGINS = _get_csv_env(
    "CORS_ALLOW_ORIGINS",
    [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
)
CORS_ALLOW_CREDENTIALS = _get_bool_env("CORS_ALLOW_CREDENTIALS", False)
CORS_ALLOW_METHODS = _get_csv_env("CORS_ALLOW_METHODS", ["GET", "POST", "OPTIONS"])
CORS_ALLOW_HEADERS = _get_csv_env(
    "CORS_ALLOW_HEADERS",
    ["Content-Type", "Authorization", "X-API-Key"],
)

# Fine-tuning job monitoring behavior
FINE_TUNING_POLL_INTERVAL_SECONDS = _get_int_env("FINE_TUNING_POLL_INTERVAL_SECONDS", 60)
FINE_TUNING_POLL_TIMEOUT_SECONDS = _get_int_env("FINE_TUNING_POLL_TIMEOUT_SECONDS", 7200)