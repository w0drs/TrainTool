from fpdf import FPDF

from core.datasets.dataset import Dataset
from core.models.model import Model


class PDFReportGenerator:
    @staticmethod
    def generate(
            images: list,
            metrics: dict,
            output_path: str,
            model: Model,
            dataset: Dataset
    ) -> None:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Заголовок
        pdf.cell(200, 10, txt="Model Evaluation Report", ln=True, align='C')
        pdf.ln(10)

        # О модели
        pdf.set_font("", "B")
        pdf.cell(200, 10, txt="Model:", ln=True)
        pdf.set_font("")
        pdf.cell(200, 10, txt=f"Name: {model.model_name}", ln=True)
        pdf.cell(200, 10, txt=f"Task: {model.task}", ln=True)

        # dataset path
        pdf.set_font("", "B")
        pdf.cell(200, 10, txt=f"Dataset path: {dataset.path}", ln=True)

        # Метрики
        pdf.set_font("", "B")
        pdf.cell(200, 10, txt="Metrics:", ln=True)
        pdf.set_font("")
        for k, v in metrics.items():
            pdf.cell(200, 10, txt=f"- {k}: {v}", ln=True)
        pdf.ln(10)

        # Изображения
        pdf.set_font("", "B")
        pdf.cell(200, 10, txt="Visualizations:", ln=True)
        pdf.set_font("")
        for img in images:
            pdf.image(img, x=10, w=180)
            pdf.ln(5)

        pdf.output(output_path)