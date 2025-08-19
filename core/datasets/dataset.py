from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import uuid


@dataclass
class Dataset:
    path: Path
    name: str
    settings: Dict[str, Any] = None
    delimiter: str = ","
    encoding: str = 'utf-8'
    target_column: list = None
    unused_columns: list = None
    numeric_columns: list = None
    category_columns: list = None
    target_column_type: str = None
    time_columns: list = None
    features: list = None

    def __post_init__(self):
        self.id = str(uuid.uuid4())