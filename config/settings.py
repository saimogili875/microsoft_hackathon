"""
Configuration settings for Deal Intelligence Agent pipeline.
Loads environment variables from .env securely.
"""

import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")


class SystemConfig(BaseModel):
    app_name: str = "Deal Intelligence Agent"
    version: str = "2.0.0"

    # Data directory paths
    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    normalized_dir: Path = BASE_DIR / "data" / "normalized"
    episodes_dir: Path = BASE_DIR / "data" / "episodes"
    pending_review_dir: Path = BASE_DIR / "data" / "pending_review"
    confirmed_dir: Path = BASE_DIR / "data" / "confirmed"

    # Hindsight Configuration
    hindsight_api_url: str = os.getenv("HINDSIGHT_API_URL", "https://api.hindsight.ai")
    hindsight_api_key: str = os.getenv("HINDSIGHT_API_KEY", "")
    use_local_hindsight_mock: bool = os.getenv("USE_LOCAL_HINDSIGHT_MOCK", "true").lower() in ("true", "1", "yes")

    # Groq Configuration
    groq_api_url: str = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Pipeline Settings
    default_anonymize: bool = True
    min_extraction_confidence: float = 0.5
    min_causal_confidence: float = 0.3


settings = SystemConfig()
