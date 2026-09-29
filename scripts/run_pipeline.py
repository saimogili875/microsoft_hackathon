"""
Script to execute the complete Deal Intelligence Pipeline end-to-end.
"""

import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.pipeline import DealIntelligencePipeline

def main():
    print("=== DEAL INTELLIGENCE PIPELINE INITIALIZATION ===")
    pipeline = DealIntelligencePipeline()
    print("Pipeline ready.")
    print(f"Hindsight Status: {pipeline.hindsight_client.health_check()}")

if __name__ == "__main__":
    main()
