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

        # model settings
        pdf.set_font("","B")
        if model.settings:
            best_params: dict | None = model.settings.get("best_params", None)
            print(best_params)
            if best_params:
                pdf.cell(200, 10, txt="Best parameters of model:", ln=True)
                pdf.set_font("")
                for setting, value in best_params.items():
                    print(f"{setting}: {value}")
                    pdf.cell(200, 10, txt=f"'{setting}': {value}", ln=True)
            else:
                pdf.cell(200, 10, txt="Using default model settings", ln=True)
        else:
            pdf.cell(200, 10, txt="Using default model settings", ln=True)

        # Dataset path
        pdf.set_font("", "B")
        pdf.cell(200, 10, txt=f"Dataset path: {dataset.path}", ln=True)

        # Dataset settings
        if dataset.settings:
            pdf.set_font("", "B")
            pdf.cell(200, 10, txt="Dataset transforms:", ln=True)
            pdf.set_font("")

            for column, list_of_methods in dataset.settings.items():
                pdf.cell(200, 10, txt=f"{column}:", ln=True)
                for i, method in enumerate(list_of_methods):
                    pdf.cell(200, 10, txt=f"{i+1}) {method}", ln=True)
            pdf.ln(10)


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