# config.py
# Configuration loader and constants for Yojana Sahayak (Standard RAG with ChromaDB).

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR = BASE_DIR / "chroma_db"
TXT_DIR = DATA_DIR / "schemes_txt"
TXT_DIR.mkdir(exist_ok=True)

# File paths
RAW_SCHEMES_PATH      = DATA_DIR / "schemes.json"
ENRICHED_SCHEMES_PATH = DATA_DIR / "schemes_with_metadata.json"
SAMPLE_SCHEMES_PATH   = DATA_DIR / "sample_schemes.json"

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY         = os.getenv("GOOGLE_API_KEY", "")

# ── Model Configuration ───────────────────────────────────────────────────────
CLAUDE_EXTRACTION_MODEL = os.getenv("CLAUDE_EXTRACTION_MODEL", "claude-3-5-haiku-20241022")
CLAUDE_CHAT_MODEL       = os.getenv("CLAUDE_CHAT_MODEL",       "claude-3-5-sonnet-20241022")

GEMINI_EXTRACTION_MODEL = os.getenv("GEMINI_EXTRACTION_MODEL", "gemini-1.5-flash")
GEMINI_CHAT_MODEL       = os.getenv("GEMINI_CHAT_MODEL",       "gemini-1.5-flash")

# ── Embeddings & ChromaDB ──────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME    = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME  = "yojana_schemes"

VERBOSE = True

def print_config_summary():
    print("=== YOJANA SAHAYAK CONFIGURATION ===")
    print(f"Anthropic Key Set : {'Yes ✓' if ANTHROPIC_API_KEY else 'No ✗'}")
    print(f"Google Key Set    : {'Yes ✓' if GOOGLE_API_KEY else 'No ✗'}")
    print(f"Claude Chat Model : {CLAUDE_CHAT_MODEL}")
    print(f"Claude Extract    : {CLAUDE_EXTRACTION_MODEL}")
    print(f"Embedding Model   : {EMBEDDING_MODEL_NAME} (local sentence-transformers)")
    print(f"Chroma DB Path    : {CHROMA_DIR}")
    print("=====================================")

if __name__ == "__main__":
    print_config_summary()
