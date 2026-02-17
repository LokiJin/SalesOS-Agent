"""
Configuration for Agentic KB
All paths, URLs, and settings in one place
"""
import os
from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

# Base directory (where this config file is)
BASE_DIR = Path(__file__).parent

# Database paths
SALES_DB_PATH = BASE_DIR / "sales_db" / "sales_data.db"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
DOCS_PATH = BASE_DIR / "kb"

# ============================================================================
# LLM Configuration
# ============================================================================

# OpenAI API key (required for some models/libraries even if not actually used)
os.environ['OPENAI_API_KEY'] = "not_a_real_key"

# Llama.cpp server configuration
LLAMA_SERVER_URL = "http://localhost:8080/v1/"
MODEL_NAME = "gpt-oss-20b-Q4_K_M.gguf"

# LLM parameters
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 5000

# ============================================================================
# RAG Configuration
# ============================================================================

RAG_AVAILABLE = True
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Supported file types for document ingestion
SUPPORTED_FILE_TYPES = {
    '.txt': 'TextLoader',
    '.pdf': 'PyPDFLoader',
    '.docx': 'Docx2txtLoader',
    '.md': 'UnstructuredMarkdownLoader',
    '.csv': 'CSVLoader',
    '.json': 'JSONLoader',
    '.html': 'UnstructuredHTMLLoader',
    '.htm': 'UnstructuredHTMLLoader',
}

# Text splitting parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 6  # Number of relevant chunks to retrieve for RAG
MIN_RAG_SCORE = 0.4 # Keep chunks with distance <= 0.3 (lower = better)

# ============================================================================
# Agent Configuration
# ============================================================================

# Debug mode (shows detailed logs) 
DEBUG_MODE = False

#Print SQL queries to console (can be noisy, so separate flag)
SQL_PRINTING_ENABLED = True

# Conversation settings
DEFAULT_THREAD_ID = "default_user"

# ============================================================================
# Display Settings
# ============================================================================

BANNER = """
╔══════════════════════════════════════════════════════════╗                          
║                 SalesOS Agent with Tools                 ║
╚══════════════════════════════════════════════════════════╝
"""
