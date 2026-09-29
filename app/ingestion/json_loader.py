"""
JSON and JSONL loader implementation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord


class JSONLoader(BaseLoader):
    """
    Loader for .json and .jsonl files containing single records or arrays of records.
    """

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_ext = path.suffix.lower()
        records: List[RawRecord] = []

        if file_ext == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        rec = self._build_record(
                            content=data,
                            source_type=source_type if source_type != "auto" else data.get("source_type", "json_record"),
                            source_id=data.get("source_id", data.get("deal_id", f"{path.stem}_{line_num}")),
                            file_format="jsonl",
                            metadata={"file_path": str(path), "line_number": line_num},
                        )
                        records.append(rec)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Malformed JSONL on line {line_num} of {path}: {str(e)}")
        else:  # .json
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.loads(f.read())
                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSON in {path}: {str(e)}")

            if isinstance(data, list):
                for idx, item in enumerate(data):
                    st = source_type if source_type != "auto" else item.get("source_type", "json_record") if isinstance(item, dict) else "json_record"
                    sid = item.get("source_id", item.get("deal_id", f"{path.stem}_{idx}")) if isinstance(item, dict) else f"{path.stem}_{idx}"
                    rec = self._build_record(
                        content=item,
                        source_type=st,
                        source_id=sid,
                        file_format="json",
                        metadata={"file_path": str(path), "index": idx},
                    )
                    records.append(rec)
            elif isinstance(data, dict):
                st = source_type if source_type != "auto" else data.get("source_type", "json_record")
                sid = data.get("source_id", data.get("deal_id", path.stem))
                rec = self._build_record(
                    content=data,
                    source_type=st,
                    source_id=sid,
                    file_format="json",
                    metadata={"file_path": str(path)},
                )
                records.append(rec)

        return records

    def load_raw_content(
        self, content: Union[str, Dict[str, Any], List[Any]], source_type: str, source_id: str, metadata: Dict[str, Any] = None
    ) -> List[RawRecord]:
        metadata = metadata or {}
        if isinstance(content, str):
            data = json.loads(content)
        else:
            data = content

        if isinstance(data, list):
            res = []
            for idx, item in enumerate(data):
                sid = item.get("source_id", f"{source_id}_{idx}") if isinstance(item, dict) else f"{source_id}_{idx}"
                res.append(self._build_record(item, source_type, sid, "json", metadata))
            return res
        else:
            return [self._build_record(data, source_type, source_id, "json", metadata)]

    def _build_record(
        self, content: Any, source_type: str, source_id: str, file_format: str, metadata: Dict[str, Any]
    ) -> RawRecord:
        timestamp = content.get("timestamp") if isinstance(content, dict) else None
        return RawRecord(
            source_type=source_type,
            source_id=str(source_id),
            timestamp=timestamp or RawRecord.model_fields["timestamp"].default_factory(),
            file_format=file_format,
            raw_content=content,
            metadata=metadata,
        )
