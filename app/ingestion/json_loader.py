"""
JSON Loader for ingestion of JSON files containing objects or arrays.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from datetime import datetime, timezone
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector


class JsonLoader(BaseLoader):
    """
    Loader for .json files containing single records or arrays of records.
    Extracts deal_id, customer_id, salesperson_id, timestamps.
    """

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.loads(f.read())
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON in {path}: {str(e)}")

        stype = FileDetector.detect_source_type(path, user_source_type=source_type)
        records: List[RawRecord] = []

        if isinstance(data, list):
            for idx, item in enumerate(data):
                rec = self._build_record(
                    item=item,
                    default_source_type=stype,
                    default_source_id=f"{path.stem}_{idx}",
                    file_name=path.name,
                    file_path=str(path.resolve()),
                    index=idx,
                )
                records.append(rec)
        elif isinstance(data, dict):
            rec = self._build_record(
                item=data,
                default_source_type=stype,
                default_source_id=path.stem,
                file_name=path.name,
                file_path=str(path.resolve()),
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
        if isinstance(content, str):
            data = json.loads(content)
        else:
            data = content

        if isinstance(data, list):
            res = []
            for idx, item in enumerate(data):
                sid = item.get("source_id", f"{source_id}_{idx}") if isinstance(item, dict) else f"{source_id}_{idx}"
                res.append(self._build_record(item, source_type, sid, meta.get("file_name"), meta.get("source_location"), idx, meta))
            return res
        else:
            return [self._build_record(data, source_type, source_id, meta.get("file_name"), meta.get("source_location"), metadata=meta)]

    def _build_record(
        self,
        item: Any,
        default_source_type: str,
        default_source_id: str,
        file_name: Optional[str] = None,
        file_path: Optional[str] = None,
        index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RawRecord:
        meta = metadata.copy() if metadata else {}
        if file_path:
            meta["file_path"] = file_path
        if index is not None:
            meta["index"] = index

        if isinstance(item, dict):
            stype = item.get("source_type") or default_source_type
            sid = item.get("source_id") or item.get("deal_id") or default_source_id
            deal_id = item.get("deal_id")
            customer_id = item.get("customer_id") or item.get("client_id")
            salesperson_id = item.get("salesperson_id") or item.get("rep_id")
            ts = item.get("timestamp") or item.get("created_at") or datetime.now(timezone.utc).isoformat()
        else:
            stype = default_source_type
            sid = default_source_id
            deal_id = None
            customer_id = None
            salesperson_id = None
            ts = datetime.now(timezone.utc).isoformat()

        return RawRecord(
            source_id=str(sid),
            source_type=stype,
            file_name=file_name,
            timestamp=ts,
            file_format="json",
            raw_content=item,
            metadata=meta,
            deal_id=deal_id,
            customer_id=customer_id,
            salesperson_id=salesperson_id,
            source_location=file_path,
            extraction_method="direct",
            extraction_confidence=1.0,
        )


# Alias for backward compatibility
JSONLoader = JsonLoader
