from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import uuid


@dataclass
class Dataset:
    """
        A class whose object will contain information about the dataset and its settings.

        Attributes:
            path: system path to dataset
            name: dataset name (without format file)
            delimiter: separating value for .csv and .txt datasets
            encoding: dataset encoding
            target_column: list of target columns. When training the model, the first element from this list will be taken.
            unused_columns: list of columns, that are not used in training
            numeric_columns: for StandartScaler()
            category_columns: for OneHotEncoding()
            target_column_type: if this parameter takes 'category' value, LabelEncoder() will be applied to the target column.
            time_columns: list of columns that will be converted to numeric format
            features: list of feature columns
    """
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