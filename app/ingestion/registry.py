"""
Registry for dynamic loader discovery and resolution.
"""

from pathlib import Path
from typing import Dict, Type, Union, List
from app.ingestion.base import BaseLoader
from app.ingestion.json_loader import JSONLoader
from app.ingestion.csv_loader import CSVLoader
from app.ingestion.text_loader import TextLoader
from app.models.raw_data import RawRecord


class LoaderRegistry:
    """
    Registry that dispatches files to appropriate ingestion loaders.
    """

    def __init__(self):
        self._extension_map: Dict[str, BaseLoader] = {
            ".json": JSONLoader(),
            ".jsonl": JSONLoader(),
            ".csv": CSVLoader(),
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".log": TextLoader(),
        }

    def register_loader(self, extension: str, loader: BaseLoader):
        ext = extension if extension.startswith(".") else f".{extension}"
        self._extension_map[ext.lower()] = loader

    def get_loader(self, file_path_or_ext: str) -> BaseLoader:
        path_obj = Path(file_path_or_ext)
        ext = path_obj.suffix.lower()
        if not ext and file_path_or_ext:
            ext = file_path_or_ext.lower() if file_path_or_ext.startswith(".") else f".{file_path_or_ext.lower()}"

        if ext not in self._extension_map:
            raise ValueError(f"Unsupported file extension/type: '{ext}'. Supported: {list(self._extension_map.keys())}")
        return self._extension_map[ext]

    def load(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        loader = self.get_loader(str(path))
        return loader.load_file(path, source_type=source_type)


default_loader_registry = LoaderRegistry()
