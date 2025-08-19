from typing import Dict
from core.models.model import Model


class ModelManager:
    def __init__(self):
        self._models: Dict[str, Model] = {}

    def add_model(self) -> str:
        curr_model = Model(
        )
        self._models[curr_model.id] = curr_model
        return curr_model.id

    def get_model(self, dataset_id: str) -> Model:
        return self._models.get(dataset_id)

    def get_all(self) -> Dict[str, Model]:
        return self._models.copy()

    def remove_model(self, dataset_id: str):
        self._models.pop(dataset_id, None)