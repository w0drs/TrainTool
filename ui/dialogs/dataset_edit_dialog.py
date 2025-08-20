import flet as ft
from core.datasets.datasets_manager import DatasetManager
from ui.components.dataset_columns_widget import ColumnsWidget
from ui.dialogs.information_windows import show_error_dialog


class DatasetEditDialog:
    """A dialog for editing dataset properties and columns configuration.

        This dialog provides an interface to modify dataset encoding, delimiter, and column
        configurations. It displays current dataset properties and allows saving changes.

        Attributes:
            page (ft.Page): The Flet page where the dialog will be displayed.
            dsm (DatasetManager): The dataset manager instance for dataset operations.
            ds_id (str): The ID of the dataset being edited.
            dialog (ft.AlertDialog | None): The dialog instance (None until shown).
            current_dataset: The dataset object being edited.
            update_dataset_list (callable): Callback function to refresh dataset list after changes.
            dataset_widget (ColumnsWidget): Widget for managing dataset columns.
            divider (ft.Divider): Visual divider element.
            encoding_dropdown (ft.Dropdown): Dropdown for selecting text encoding.
            delimiter_textfield (ft.TextField): Input field for delimiter character.
    """
    def __init__(
            self,
            page:ft.Page,
            dataset_manager:DatasetManager,
            dataset_id:str,
            update_dataset_list: callable
    ) -> None:
        """Initializes the dataset edit dialog.

            Args:
                page: The Flet page where the dialog will be displayed.
                dataset_manager: Manager instance for dataset operations.
                dataset_id: ID of the dataset to edit.
                update_dataset_list: Callback function to refresh the dataset list
                                    after saving changes.
        """
        self.page = page
        self.dsm = dataset_manager
        self.ds_id = dataset_id
        self.dialog = None
        self.current_dataset = self.dsm.get_dataset(dataset_id)
        self.update_dataset_list = update_dataset_list
        self.dataset_widget = ColumnsWidget(
            page=page,
            dataset_manager=dataset_manager,
            dataset_id=dataset_id,
        )

        self.divider = ft.Divider(height=1, thickness=2, color=ft.Colors.WHITE)
        self.encoding_dropdown = ft.Dropdown(
            label="Encoding",
            value=self.current_dataset.encoding,
            options=[
                ft.dropdown.Option('utf-8'),
                ft.dropdown.Option('utf-16'),
                ft.dropdown.Option('utf-32'),
                ft.dropdown.Option('latin1'),
                ft.dropdown.Option('cp1251'),
                ft.dropdown.Option('cp1252'),
                ft.dropdown.Option('ascii'),
                ft.dropdown.Option('koi8-r'),
                ft.dropdown.Option('mac_cyrillic'),
            ]
        )

        self.delimiter_textfield = ft.TextField(
            label="Delimiter",
            value=self.current_dataset.delimiter,
            max_length=10,
            width=150,
        )

    def _get_content(self) -> ft.ListView:
        """Builds and returns the dialog's content components(private method).

            Returns:
            A ListView containing:
            - Dataset path display
            - Encoding and delimiter controls
            - Dataset columns configuration widget
        """
        return ft.ListView(
            controls=[
                ft.Text(f"Dataset path: {self.current_dataset.path}", size=25),
                self.divider,
                ft.Row(
                    controls=[
                        self.encoding_dropdown,
                        self.delimiter_textfield
                    ]
                ),
                self.divider,
                self.dataset_widget.build_all_widgets()
            ],
            spacing=40,
            padding=20,
            auto_scroll=False
        )

    def show(self) -> None:
        """Displays the edit dialog on the page.

            Shows an alert dialog with dataset editing controls. If the dataset has too few
            columns, displays an error message about delimiter configuration.
        """
        self.dialog = ft.AlertDialog(
            title=ft.Text(f"Edit Dataset {self.current_dataset.name}"),
            content=ft.Container(
                content=self._get_content(),
                width=800,
                height=600,
                padding=20,
            ),
            actions=[
                ft.ElevatedButton("Save", on_click=self._save),
                ft.ElevatedButton("Cancel", on_click=self._close)
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            shape=ft.RoundedRectangleBorder(radius=1),
            inset_padding=20,
        )
        self.page.dialog = self.dialog
        self.page.open(self.dialog)
        self.page.update()

        if self.dataset_widget.few_columns_error:
            show_error_dialog(
                page=self.page,
                title="Error while upload dataset",
                message="Few columns, try to change another delimiter in setting"
            )

    def _close(self, e) -> None:
        """Closes the dialog (private method).
        """
        self.page.close(self.dialog)
        self.page.update()

    def _save(self, e) -> None:
        """Saves all changes made in the dialog (private method).

            Updates dataset encoding and delimiter, saves column configurations,
            closes the dialog, and refreshes the dataset list.

            Args:
                e: The event that triggered the save action.
        """
        self.dataset_widget.save_state()
        self.current_dataset.encoding = self.encoding_dropdown.value
        self.current_dataset.delimiter = self.delimiter_textfield.value

        self._close(e)

        self.update_dataset_list()