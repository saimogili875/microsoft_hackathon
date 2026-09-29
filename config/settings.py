"""
Configuration settings for Deal Intelligence Agent pipeline.
"""

from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

class SystemConfig(BaseModel):
    app_name: str = "Deal Intelligence Agent"
    version: str = "1.0.0"
    
    # Data directory paths
    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    normalized_dir: Path = BASE_DIR / "data" / "normalized"
    episodes_dir: Path = BASE_DIR / "data" / "episodes"
    pending_review_dir: Path = BASE_DIR / "data" / "pending_review"
    confirmed_dir: Path = BASE_DIR / "data" / "confirmed"
    
    # Hindsight Configuration
    hindsight_api_url: str = "http://localhost:8000"
    hindsight_api_key: str = "hackathon-secret-key"
    use_local_hindsight_mock: bool = True
    
    # Pipeline Settings
    default_anonymize: bool = True
    min_extraction_confidence: float = 0.5
    min_causal_confidence: float = 0.3

settings = SystemConfig()
