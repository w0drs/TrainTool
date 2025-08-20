from dataclasses import dataclass
from typing import Dict, Any
from flet import Colors
import uuid



@dataclass
class Model:
    """
    A class that stores information about a block with a machine learning model.
        name: name of the block.
        description: description of the block.
        color: color of the block.
        task: 'Classification' or 'Regression'.
        model_name: name of the machine learning model.
        dataset_id: id of the dataset to be used for training the model.
        metrics: metrics of the model after training.
        ml_model: it contains a trained model.
    """
    name: str = ""
    settings: Dict[str, Any] = None
    description: str = ""
    task: str = "Classification"
    model_name: str = 'Nothing'
    color: Colors = Colors.BLACK
    dataset_id: str = None
    metrics: dict = None
    ml_model = None

    def __post_init__(self):
        self.id = str(uuid.uuid4())