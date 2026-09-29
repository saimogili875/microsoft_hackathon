"""
CSV Loader for ingestion of CSV files, extracting rows into structured RawRecords.
"""

import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from datetime import datetime, timezone
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector


class CSVLoader(BaseLoader):
    """
    Loader for CSV files, parsing rows into raw dictionary records.
    Extracts deal_id, customer_id, salesperson_id, and timestamps when available.
    """

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        records: List[RawRecord] = []
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty or malformed CSV file: {path}")

            stype = FileDetector.detect_source_type(path, user_source_type=source_type)

            for row_idx, row in enumerate(reader, 1):
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k and v is not None}
                row_source_type = clean_row.get("source_type") or stype
                sid = clean_row.get("source_id") or clean_row.get("deal_id") or f"{path.stem}_row_{row_idx}"

                deal_id = clean_row.get("deal_id") or clean_row.get("deal")
                customer_id = clean_row.get("customer_id") or clean_row.get("client_id") or clean_row.get("customer") or clean_row.get("client")
                salesperson_id = clean_row.get("salesperson_id") or clean_row.get("rep_id") or clean_row.get("salesperson")

                ts = clean_row.get("timestamp") or clean_row.get("date") or datetime.now(timezone.utc).isoformat()

                rec = RawRecord(
                    source_id=str(sid),
                    source_type=row_source_type,
                    file_name=path.name,
                    timestamp=ts,
                    file_format="csv",
                    raw_content=clean_row,
                    metadata={"file_path": str(path), "row_number": row_idx},
                    deal_id=deal_id,
                    customer_id=customer_id,
                    salesperson_id=salesperson_id,
                    source_location=str(path.resolve()),
                    extraction_method="direct",
                    extraction_confidence=1.0,
                )
                records.append(rec)

        return records

    def load_raw_content(
        self,
        content: Union[str, Dict[str, Any], List[Any]],
        source_type: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawRecord]:
        meta = metadata or {}
        records: List[RawRecord] = []

        if isinstance(content, str):
            f = io.StringIO(content)
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, 1):
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k}
                sid = clean_row.get("source_id", f"{source_id}_row_{row_idx}")
                rec = RawRecord(
                    source_id=str(sid),
                    source_type=source_type or "crm_record",
                    file_name=meta.get("file_name"),
                    timestamp=clean_row.get("timestamp") or meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                    file_format="csv",
                    raw_content=clean_row,
                    metadata=meta,
                    deal_id=clean_row.get("deal_id") or meta.get("deal_id"),
                    customer_id=clean_row.get("customer_id") or clean_row.get("client_id") or meta.get("customer_id"),
                    salesperson_id=clean_row.get("salesperson_id") or meta.get("salesperson_id"),
                    source_location=meta.get("source_location"),
                    extraction_method="direct",
                    extraction_confidence=1.0,
                )
                records.append(rec)
        elif isinstance(content, list):
            for idx, item in enumerate(content):
                if isinstance(item, dict):
                    sid = item.get("source_id", f"{source_id}_{idx}")
                    rec = RawRecord(
                        source_id=str(sid),
                        source_type=source_type or "crm_record",
                        file_name=meta.get("file_name"),
                        timestamp=item.get("timestamp") or meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        file_format="csv",
                        raw_content=item,
                        metadata=meta,
                        deal_id=item.get("deal_id") or meta.get("deal_id"),
                        customer_id=item.get("customer_id") or item.get("client_id") or meta.get("customer_id"),
                        salesperson_id=item.get("salesperson_id") or meta.get("salesperson_id"),
                        source_location=meta.get("source_location"),
                        extraction_method="direct",
                        extraction_confidence=1.0,
                    )
                else:
                    rec = RawRecord(
                        source_id=f"{source_id}_{idx}",
                        source_type=source_type,
                        raw_content=str(item),
                        metadata=meta,
                    )
                records.append(rec)

        return records
