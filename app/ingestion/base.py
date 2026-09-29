"""
Base Loader Interface for Data Ingestion.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
from pathlib import Path
from app.models.raw_data import RawRecord


class BaseLoader(ABC):
    """
    Abstract base class for all file and source loaders.
    """

    @abstractmethod
    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        """
        Load records from a file path.
        """
        pass

    @abstractmethod
    def load_raw_content(
        self, content: Union[str, Dict[str, Any], List[Any]], source_type: str, source_id: str, metadata: Dict[str, Any] = None
    ) -> List[RawRecord]:
        """
        Load records directly from raw in-memory content.
        """
        pass
