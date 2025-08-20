import flet as ft
from core.datasets.datasets_manager import DatasetManager

class DatasetInfoDialog:
    """Dialog, which contains information about current dataset
        Attributes:
            page: current ft.Page (from main.py)
            dataset_manager: current dataset manager. It controls all datasets
            dataset_id: id of current dataset
    """
    def __init__(
            self,
            page: ft.Page,
            dataset_manager: DatasetManager,
            dataset_id: str
    ):
        self.page = page
        self.dataset_manager = dataset_manager
        self.dataset_id = dataset_id

        self.current_dataset = self.dataset_manager.get_dataset(self.dataset_id)
        self._initialize_content()
        self._initialize_dialog()

    def _initialize_content(self) -> None:
        self.content = ft.Column(
                controls=[
                    self._info_row("Name: ", self.current_dataset.name),
                    self._info_row("Encoding: ", self.current_dataset.encoding),
                    self._info_row("Delimiter: ", self.current_dataset.delimiter)
                ]
        )
        if self.current_dataset.target_column:
            self.content.controls.append(ft.Divider(height=1, thickness=2, color=ft.Colors.GREY_300))
            self.content.controls.append(
                ft.Text("Target column: ",size=20)
            )
            self.content.controls.append(
                ft.Text(f"{self.current_dataset.target_column[0]}", size=15, color=ft.Colors.LIGHT_BLUE)
            )
        if self.current_dataset.features:
            self.content.controls.append(ft.Divider(height=1, thickness=2, color=ft.Colors.GREY_300))
            self.content.controls.append(ft.Text("Features:",size=20))
            self.content.controls.extend(
                [
                    ft.Text(f"{column}",size=15, color=ft.Colors.LIGHT_BLUE)
                    for column in self.current_dataset.features
                ]
            )

    def _initialize_dialog(self):
        self.dialog = ft.AlertDialog(
            content=ft.Container(
                content=self.content,
                width=350,
                height=500,
                padding=20,
            ),
            actions=[
                ft.ElevatedButton("Cancel", on_click=self._close)
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            shape=ft.RoundedRectangleBorder(radius=1),
            inset_padding=20,
        )

    def _info_row(self, label: str, value: any) -> ft.Row:
        """returns text widget (private method)"""
        return ft.Row(
            controls=[
                ft.Text(f"{label}", size=20),
                ft.Text(f"{value}", size=20, color=ft.Colors.LIGHT_BLUE)
            ]
        )

    def show(self):
        self.page.dialog = self.dialog
        self.page.open(self.dialog)
        self.page.update()

    def _close(self, e):
        self.page.close(self.dialog)
        self.page.update()
