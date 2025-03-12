"""
Global settings and configuration for the CERN Magazine LLMOps framework
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# API configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Model settings
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL", "")

# CERN Courier crawler settings
CERN_BASE_URL = "https://home.cern/resources"
CRAWL_START_PAGE = 0
CRAWL_END_PAGE = 7
CRAWL_DELAY = 1.0  # seconds between requests

# Data processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS_PER_DOC = 2000
MAX_EMBEDDING_BATCH_SIZE = 100

# RAG settings
TOP_K_DOCUMENTS = 4
SIMILARITY_THRESHOLD = 0.7

# API server settings
API_HOST = "0.0.0.0"
API_PORT = 8000
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true" 