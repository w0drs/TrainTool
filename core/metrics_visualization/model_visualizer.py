import os
import tempfile
import matplotlib
import matplotlib.pyplot as plt
from numpy import ndarray
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

matplotlib.use('Agg')


class ModelVisualizer:
    @staticmethod
    def _get_temp_dir() -> str:
        """Возвращает системную временную директорию (temp в Windows, /tmp в Linux)"""
        return tempfile.gettempdir()

    @staticmethod
    def _get_full_path(filename: str) -> str:
        """Генерирует полный путь во временной директории"""
        temp_dir = ModelVisualizer._get_temp_dir()
        return os.path.join(temp_dir, filename)

    @staticmethod
    def regression_line_compare(y_true: ndarray | None, y_pred: ndarray | None) -> str:
        if y_true is None or y_pred is None:
            return ""
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, 'o-', label="Истинные значения (y)", markersize=8, linewidth=2)
        plt.plot(y_pred, 's--', label="Предсказанные (y_pred)", markersize=6, linewidth=2)
        plt.plot(y_true - y_pred, 's--', label="Остатки", markersize=6, linewidth=2)
        plt.xlabel("Номер точки (или время)")
        plt.ylabel("Значение")
        plt.legend()
        plt.grid(True)
        plt.title("Сравнение истинных и предсказанных значений")

        file_name = "regression_lines_compare.png"
        file_path = ModelVisualizer._get_full_path(file_name)

        plt.savefig(file_path, bbox_inches="tight", dpi=120)
        plt.close()
        return file_path

    @staticmethod
    def confusion_matrix(y_true: ndarray | None, y_pred: ndarray | None) -> str:
        if y_true is None or y_pred is None:
            return ""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title("Confusion Matrix")

        file_name = "confusion_matrix.png"
        file_path = ModelVisualizer._get_full_path(file_name)

        plt.savefig(file_path, bbox_inches="tight", dpi=120)
        plt.close()
        return file_path

    @staticmethod
    def roc_curve(y_true: ndarray | None, y_probs: ndarray | None) -> str:
        if y_true is None or y_probs is None:
            return ""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-кривая")
        plt.legend()
        plt.grid(True)

        file_name = "classification_roc_curve.png"
        file_path = ModelVisualizer._get_full_path(file_name)

        plt.savefig(file_path, bbox_inches="tight", dpi=120)
        plt.close()
        return file_path