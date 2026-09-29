"""
CSV loader implementation.
"""

import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Union
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord


class CSVLoader(BaseLoader):
    """
    Loader for CSV files, parsing rows into raw dictionary records.
    """

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        records: List[RawRecord] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"Empty or malformed CSV file: {path}")

            for row_idx, row in enumerate(reader, 1):
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k}
                st = source_type if source_type != "auto" else clean_row.get("source_type", "csv_row")
                sid = clean_row.get("source_id", clean_row.get("deal_id", f"{path.stem}_row_{row_idx}"))
                rec = RawRecord(
                    source_type=st,
                    source_id=str(sid),
                    timestamp=clean_row.get("timestamp") or RawRecord.model_fields["timestamp"].default_factory(),
                    file_format="csv",
                    raw_content=clean_row,
                    metadata={"file_path": str(path), "row_number": row_idx},
                )
                records.append(rec)

        return records

    def load_raw_content(
        self, content: Union[str, Dict[str, Any], List[Any]], source_type: str, source_id: str, metadata: Dict[str, Any] = None
    ) -> List[RawRecord]:
        metadata = metadata or {}
        records: List[RawRecord] = []

        if isinstance(content, str):
            f = io.StringIO(content)
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, 1):
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k}
                sid = clean_row.get("source_id", f"{source_id}_row_{row_idx}")
                rec = RawRecord(
                    source_type=source_type,
                    source_id=str(sid),
                    file_format="csv",
                    raw_content=clean_row,
                    metadata=metadata,
                )
                records.append(rec)
        elif isinstance(content, list):
            for idx, item in enumerate(content):
                sid = item.get("source_id", f"{source_id}_{idx}") if isinstance(item, dict) else f"{source_id}_{idx}"
                records.append(
                    RawRecord(
                        source_type=source_type,
                        source_id=str(sid),
                        file_format="csv",
                        raw_content=item,
                        metadata=metadata,
                    )
                )

        return records
