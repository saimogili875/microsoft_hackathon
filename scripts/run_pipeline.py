"""
Script to execute the complete Deal Intelligence Pipeline end-to-end.
"""

from pathlib import Path
import json
from app.services.pipeline import DealIntelligencePipeline

def main():
    print("=== DEAL INTELLIGENCE PIPELINE INITIALIZATION ===")
    pipeline = DealIntelligencePipeline()
    print("Pipeline ready.")
    print(f"Hindsight Status: {pipeline.hindsight_client.health_check()}")

if __name__ == "__main__":
    main()
