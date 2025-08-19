from dataclasses import dataclass
from typing import Dict, Any
from flet import Colors
import uuid



@dataclass
class Model:
    name: str = "no name model"
    settings: Dict[str, Any] = None
    description: str = "its a model bruh"
    task: str = "Classification"
    model_name: str = 'Nothing'
    color: Colors = Colors.BLACK
    dataset_id: str = None
    metrics: dict = None
    ml_model = None

    def __post_init__(self):
        self.id = str(uuid.uuid4())