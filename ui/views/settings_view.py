import flet as ft
from core.datasets.datasets_manager import DatasetManager
from pathlib import Path

from ui.dialogs.dataset_edit_dialog import DatasetEditDialog


def create_settings_content(
        page:ft.Page,
        on_back_click: callable,
        state: DatasetManager,
) -> ft.ListView:
    """
        Creating settings content with dataset management
        Args:
        page: ft.Page
        on_back_click: button for back to main content
        state: DatasetManager - object, that contain datasets (class Dataset)
    """
    divider = ft.Divider(height=1, thickness=2, color=ft.Colors.WHITE)
    file_picker = ft.FilePicker()
    page.overlay.append(file_picker)
    page.update()

    datasets_grid = ft.GridView(
        max_extent=250,
        child_aspect_ratio=1.1,
        spacing=20,
        run_spacing=20,
        expand=True,
    )

    def update_datasets_list():
        if not datasets_grid.page:
            return

        datasets_grid.controls.clear()

        for dataset_id, dataset in state.get_all().items():
            dataset_card = ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.ListTile(
                            leading=ft.Icon(ft.Icons.DATASET),
                            title=ft.Text(dataset.name, weight=ft.FontWeight.BOLD),
                            subtitle=ft.Text(
                                f"Encoding: {dataset.encoding}\n"
                                f"Delimiter: {dataset.delimiter}",
                                size=12
                            ),
                            expand=True,
                        ),
                        ft.Row([
                            ft.IconButton(
                                icon=ft.Icons.EDIT,
                                on_click=lambda e, ds_id=dataset_id: _edit_dataset(ds_id)
                            ),
                            ft.IconButton(
                                icon=ft.Icons.DELETE,
                                icon_color=ft.Colors.RED,
                                on_click=lambda e, ds_id=dataset_id: _delete_dataset(ds_id)
                            )
                        ],
                            alignment=ft.MainAxisAlignment.END,
                        )
                    ], spacing=0, expand=True,

                    ),
                    width=250,
                    height=250,
                    padding=10
                ),
                elevation=5,
                margin=2
            )
            datasets_grid.controls.append(dataset_card)

        datasets_grid.update()

    def _edit_dataset(dataset_id: str):
        """Redactor"""
        dataset = state.get_dataset(dataset_id)
        print(f"Editing dataset: {dataset.name}")
        DatasetEditDialog(
            page=page,
            dataset_manager=state,
            dataset_id=dataset_id,
            update_dataset_list=update_datasets_list
        ).show()

    def _delete_dataset(dataset_id: str):
        """Deleting dataset"""
        state.remove_dataset(dataset_id)
        update_datasets_list()

    def _handle_file_pick(e: ft.FilePickerResultEvent):
        """Changing files for datasets (csv,txt...)"""
        allowed_extensions = [".csv", ".json", ".xlsx", ".txt"]
        if e.files:
            for file in e.files:
                if not any(file.name.lower().endswith(fr) for fr in allowed_extensions):
                    print(f"Файл '{file.name}' не поддерживается!")
                    continue
                state.add_dataset(Path(file.path))
            update_datasets_list()

    file_picker.on_result = _handle_file_pick

    update_datasets_list()

    content_column = ft.Column(
        controls=[
            ft.Text("Settings", size=35),
            divider,
            ft.Text("Datasets", size=25),
            ft.ElevatedButton(
                "Import Dataset",
                icon=ft.Icons.UPLOAD,
                on_click=lambda _: file_picker.pick_files()
            ),
            ft.Text("Available datasets:"),
            ft.Container(
                content=datasets_grid,
                height=200,
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=5
            ),
            ft.ElevatedButton("Back", on_click=on_back_click)
        ],
        spacing=20,
        expand=True
    )

    return ft.ListView(
        controls=[content_column],
        expand=True,
        spacing=10,
        padding=20,
        visible=False
    )