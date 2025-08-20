import flet as ft
from core.models.model import Model
from joblib import dump as jb_dump
from pickle import dump as pkl_dump
import os

class ExportModelDialog:
    """
        Dialog, which exporting trained model
        Attributes:
            page: current ft.Page (from main.py)
            model: trained model
            method: export model method (joblib or pickle)
            selected_folder: selected folder path for export model
    """
    def __init__(
            self,
            page:ft.Page,
            model: Model,
    ) -> None:
        self.page = page
        self.model = model

        self.method = "joblib"
        self.selected_folder = None
        self._initialize_widgets()
        self._initialize_dialog()

    def _initialize_widgets(self):
        self.file_picker = ft.FilePicker(on_result=self._on_dialog_result)
        self.page.overlay.append(self.file_picker)

        self.file_name_textfield = ft.TextField(label="name for exported file",value=self.model.name)

        self.buttons_action = ft.Row(
            controls=[
                ft.ElevatedButton("Export", on_click=self._export_model),
                ft.ElevatedButton("Cancel", on_click=self._close)
            ],
            alignment=ft.MainAxisAlignment.END
        )

        self.content = [
            self.file_name_textfield,
            ft.DropdownM2(
                label="method",
                value="joblib",
                options=[
                    ft.dropdownm2.Option("joblib"),
                    ft.dropdownm2.Option("pickle"),
                    ft.dropdownm2.Option("ONNX"),
                ],
                on_change=self._on_method_change,
                width=200,
            ),
        ]

    def _initialize_dialog(self):
        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Export {self.model.name}"),
            content=ft.Container(
                content=ft.Column(
                    controls=self.content,
                    scroll=ft.ScrollMode.AUTO
                ),
                width=400,
                height=300,
                padding=20,
            ),
            actions=[self.buttons_action],
            actions_alignment=ft.MainAxisAlignment.END
        )

    def show(self):
        self.page.dialog = self.dialog
        self.page.open(self.dialog)
        self.page.update()

    def _close(self, e):
        self.page.close(self.dialog)
        self.page.update()

    def _on_method_change(self, e):
        self.method = e.control.value

    def _get_full_path(self, extension: str) -> str:
        filename = f"{self.file_name_textfield.value}.{extension}"
        return os.path.join(self.selected_folder, filename)

    def _export_model(self, e):
        if self.model.ml_model is None:
            snack_bar = ft.SnackBar(
                ft.Text(f"Error: model doesn't trained!")
            )
            self.page.open(snack_bar)
            self.page.update()
            return

        self.file_picker.get_directory_path("Choose the folder for export")

    def _on_dialog_result(self, e: ft.FilePickerResultEvent):
        if e.path:
            self.selected_folder = e.path
            try:
                if self.method == "joblib":
                    self._export_to_joblib()
                elif self.method == "pickle":
                    self._export_to_pickle()
                elif self.method == "ONNX":
                    self._export_to_onnx()

                snack_bar = ft.SnackBar(
                    ft.Text(f"Model was exported to {e.path}")
                )
                self.page.open(snack_bar)
            except Exception as ex:
                print(ex)
            finally:
                self._close(e=None)
                self.page.update()
        else:
            snack_bar = ft.SnackBar(
                ft.Text(f"Error: folder doesn't choose!")
            )
            self.page.open(snack_bar)
            self.page.update()

    def _export_to_joblib(self):
        try:
            jb_dump(self.model.ml_model, self._get_full_path("joblib"))
            return True
        except Exception as e:
            raise Exception(f"Joblib export failed: {str(e)}")

    def _export_to_pickle(self):
        try:
            with open(self._get_full_path("pkl"), 'wb') as f:
                pkl_dump(self.model.ml_model, f)
            return True
        except Exception as e:
            raise Exception(f"Pickle export failed: {str(e)}")

    def _export_to_onnx(self):
        pass
