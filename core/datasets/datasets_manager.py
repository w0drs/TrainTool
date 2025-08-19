from typing import Dict
from core.datasets.dataset import Dataset
from pathlib import Path


class DatasetManager:
    def __init__(self):
        self._datasets: Dict[str, Dataset] = {}

    def add_dataset(self, file_path: Path) -> str:
        dataset = Dataset(
            path=file_path,
            name=Path(file_path).stem
        )
        self._datasets[dataset.id] = dataset
        return dataset.id

    def get_dataset(self, dataset_id: str) -> Dataset:
        return self._datasets.get(dataset_id, None)

    def get_all(self) -> Dict[str, Dataset]:
        return self._datasets.copy()

    def remove_dataset(self, dataset_id: str):
        self._datasets.pop(dataset_id, None)