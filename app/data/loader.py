"""
Data Loader module for raw CRM / CPQ dataset ingestion.
Supports CSV and JSON files, creating structured data reports without hallucinating records.
"""

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import csv
import json
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class DataLoadReport:
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    win_count: int = 0
    loss_count: int = 0
    source_file: str = ""
    columns: List[str] = field(default_factory=list)


class DataLoader:
    """
    Ingests CRM / CPQ dataset files and returns normalized DataFrames & load reports.
    """

    def load_file(self, file_path: Union[str, Path]) -> tuple[pd.DataFrame, DataLoadReport]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        file_ext = path.suffix.lower()
        if file_ext == ".csv":
            df = pd.read_csv(path)
        elif file_ext in [".json", ".jsonl"]:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
            df = pd.DataFrame(content)
        else:
            raise ValueError(f"Unsupported dataset format '{file_ext}' for file: {path.name}")

        report = DataLoadReport(
            total_records=len(df),
            valid_records=len(df),
            invalid_records=0,
            win_count=int(df["win_loss_outcome"].sum()) if "win_loss_outcome" in df.columns else 0,
            loss_count=int((df["win_loss_outcome"] == 0).sum()) if "win_loss_outcome" in df.columns else 0,
            source_file=str(path.resolve()),
            columns=list(df.columns),
        )

        return df, report

    def load_records(self, records: List[Dict[str, Any]]) -> tuple[pd.DataFrame, DataLoadReport]:
        df = pd.DataFrame(records)
        report = DataLoadReport(
            total_records=len(df),
            valid_records=len(df),
            invalid_records=0,
            win_count=int(df["win_loss_outcome"].sum()) if "win_loss_outcome" in df.columns else 0,
            loss_count=int((df["win_loss_outcome"] == 0).sum()) if "win_loss_outcome" in df.columns else 0,
            source_file="in_memory_records",
            columns=list(df.columns),
        )
        return df, report
