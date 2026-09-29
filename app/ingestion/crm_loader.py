"""
CRM Loader for ingestion of structured CRM records, deal histories, contact maps, and database entities.
"""

from typing import List, Union, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import json
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector
from app.ingestion.csv_loader import CSVLoader
from app.ingestion.json_loader import JsonLoader


class CrmLoader(BaseLoader):
    """
    Loader for CRM Data & Deal History (CSV, JSON, JSONL, DB dicts).
    Normalizes CRM entities (deals, customers, contacts, activities, stages).
    """

    def __init__(self):
        self._csv_loader = CSVLoader()
        self._json_loader = JsonLoader()

    def load_file(self, file_path: Union[str, Path], source_type: str = "crm_record") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        fmt = FileDetector.detect_format(path)
        if fmt == "csv":
            raw_recs = self._csv_loader.load_file(path, source_type=source_type)
        else:
            raw_recs = self._json_loader.load_file(path, source_type=source_type)

        # Enforce CRM entity extraction
        for rec in raw_recs:
            rec.source_type = source_type if source_type != "auto" else rec.source_type
            if isinstance(rec.raw_content, dict):
                content = rec.raw_content
                rec.deal_id = rec.deal_id or content.get("deal_id") or content.get("deal")
                rec.customer_id = rec.customer_id or content.get("customer_id") or content.get("client_id") or content.get("account_id")
                rec.salesperson_id = rec.salesperson_id or content.get("salesperson_id") or content.get("owner_id")

        return raw_recs

    def load_raw_content(
        self,
        content: Union[str, Dict[str, Any], List[Any]],
        source_type: str = "crm_record",
        source_id: str = "crm_db",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawRecord]:
        meta = metadata or {}
        records: List[RawRecord] = []

        if isinstance(content, list):
            for idx, item in enumerate(content):
                sid = item.get("source_id") or item.get("deal_id") or f"{source_id}_{idx}" if isinstance(item, dict) else f"{source_id}_{idx}"
                deal_id = item.get("deal_id") if isinstance(item, dict) else meta.get("deal_id")
                customer_id = (item.get("customer_id") or item.get("client_id") or item.get("account_id")) if isinstance(item, dict) else meta.get("customer_id")
                salesperson_id = (item.get("salesperson_id") or item.get("owner_id")) if isinstance(item, dict) else meta.get("salesperson_id")
                ts = item.get("timestamp") or item.get("created_at") or meta.get("timestamp") or datetime.now(timezone.utc).isoformat() if isinstance(item, dict) else datetime.now(timezone.utc).isoformat()

                rec = RawRecord(
                    source_id=str(sid),
                    source_type=source_type,
                    file_name=meta.get("file_name"),
                    timestamp=ts,
                    file_format="dict",
                    raw_content=item,
                    metadata=meta,
                    deal_id=deal_id,
                    customer_id=customer_id,
                    salesperson_id=salesperson_id,
                    source_location=meta.get("source_location", "database"),
                    extraction_method="direct",
                    extraction_confidence=1.0,
                )
                records.append(rec)
        elif isinstance(content, dict):
            sid = content.get("source_id") or content.get("deal_id") or source_id
            rec = RawRecord(
                source_id=str(sid),
                source_type=source_type,
                file_name=meta.get("file_name"),
                timestamp=content.get("timestamp") or content.get("created_at") or meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                file_format="dict",
                raw_content=content,
                metadata=meta,
                deal_id=content.get("deal_id") or meta.get("deal_id"),
                customer_id=content.get("customer_id") or content.get("client_id") or content.get("account_id") or meta.get("customer_id"),
                salesperson_id=content.get("salesperson_id") or content.get("owner_id") or meta.get("salesperson_id"),
                source_location=meta.get("source_location", "database"),
                extraction_method="direct",
                extraction_confidence=1.0,
            )
            records.append(rec)
        else:
            rec = RawRecord(
                source_id=source_id,
                source_type=source_type,
                raw_content=str(content),
                metadata=meta,
            )
            records.append(rec)

        return records
