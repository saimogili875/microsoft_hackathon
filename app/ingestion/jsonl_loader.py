"""
JSONL (JSON Lines / NDJSON) Loader for ingestion of line-delimited JSON records.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from datetime import datetime, timezone
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector


class JsonlLoader(BaseLoader):
    """
    Loader for .jsonl / .ndjson files containing line-by-line JSON records.
    """

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stype = FileDetector.detect_source_type(path, user_source_type=source_type)
        records: List[RawRecord] = []

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue

                try:
                    data = json.loads(line_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSONL on line {line_num} of {path}: {str(e)}")

                sid = data.get("source_id") or data.get("deal_id") or f"{path.stem}_line_{line_num}" if isinstance(data, dict) else f"{path.stem}_line_{line_num}"
                row_stype = data.get("source_type") or stype if isinstance(data, dict) else stype
                deal_id = data.get("deal_id") if isinstance(data, dict) else None
                customer_id = (data.get("customer_id") or data.get("client_id")) if isinstance(data, dict) else None
                salesperson_id = (data.get("salesperson_id") or data.get("rep_id")) if isinstance(data, dict) else None
                ts = (data.get("timestamp") or data.get("created_at")) if isinstance(data, dict) else None

                rec = RawRecord(
                    source_id=str(sid),
                    source_type=row_stype,
                    file_name=path.name,
                    timestamp=ts or datetime.now(timezone.utc).isoformat(),
                    file_format="jsonl",
                    raw_content=data,
                    metadata={"file_path": str(path.resolve()), "line_number": line_num},
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
            lines = content.splitlines()
            for line_num, line in enumerate(lines, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                data = json.loads(line_str)
                sid = data.get("source_id") or f"{source_id}_line_{line_num}" if isinstance(data, dict) else f"{source_id}_line_{line_num}"
                rec = RawRecord(
                    source_id=str(sid),
                    source_type=source_type or "crm_record",
                    file_name=meta.get("file_name"),
                    timestamp=data.get("timestamp") if isinstance(data, dict) else datetime.now(timezone.utc).isoformat(),
                    file_format="jsonl",
                    raw_content=data,
                    metadata={"line_number": line_num, **meta},
                    deal_id=data.get("deal_id") if isinstance(data, dict) else None,
                    customer_id=(data.get("customer_id") or data.get("client_id")) if isinstance(data, dict) else None,
                    salesperson_id=data.get("salesperson_id") if isinstance(data, dict) else None,
                    source_location=meta.get("source_location"),
                    extraction_method="direct",
                    extraction_confidence=1.0,
                )
                records.append(rec)
        elif isinstance(content, list):
            for idx, item in enumerate(content):
                sid = item.get("source_id", f"{source_id}_{idx}") if isinstance(item, dict) else f"{source_id}_{idx}"
                rec = RawRecord(
                    source_id=str(sid),
                    source_type=source_type,
                    file_format="jsonl",
                    raw_content=item,
                    metadata=meta,
                )
                records.append(rec)

        return records
