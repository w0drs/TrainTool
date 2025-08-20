from fpdf import FPDF

from core.datasets.dataset import Dataset
from core.models.model import Model


class PDFReportGenerator:
    """
    Class that generate .pdf report of model. It includes information about dataset and model, graphs, that created using metrics
    """
    @staticmethod
    def generate(
            images: list,
            metrics: dict,
            output_path: str,
            model: Model,
            dataset: Dataset
    ) -> None:
        """
            Generate .pdf report.
            Attributes:
                images: list of paths to images for graphs
                metrics: dictionary of metrics of trained model
                output_path: path to folder where the report will be saved
                model: current model, which will be used in report
                dataset: model dataset, which will be used in report
        """
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Head
        pdf.cell(200, 10, txt="Model Evaluation Report", ln=True, align='C')
        pdf.ln(10)

        # About model
        pdf.set_font("", "B")
        pdf.cell(200, 10, txt="Model:", ln=True)
        pdf.set_font("")
        pdf.cell(200, 10, txt=f"Name: {model.model_name}", ln=True)
        pdf.cell(200, 10, txt=f"Task: {model.task}", ln=True)

        # Dataset path
        pdf.set_font("", "B")
        pdf.cell(200, 10, txt=f"Dataset path: {dataset.path}", ln=True)

        # Metrics
        pdf.set_font("", "B")
        pdf.cell(200, 10, txt="Metrics:", ln=True)
        pdf.set_font("")
        for k, v in metrics.items():
            pdf.cell(200, 10, txt=f"- {k}: {v}", ln=True)
        pdf.ln(10)

        # Images
        pdf.set_font("", "B")
        pdf.cell(200, 10, txt="Visualizations:", ln=True)
        pdf.set_font("")
        for img in images:
            pdf.image(img, x=10, w=180)
            pdf.ln(5)

        pdf.output(output_path)