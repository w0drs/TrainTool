import flet as ft

from core.datasets.datasets_manager import DatasetManager
from core.models.model_manager import ModelManager
from ui.dialogs.dataset_info_dialog import DatasetInfoDialog


class EditDialog:
    """
        Dialog for adding current block with model
        Attributes:
            page: current ft.Page (from main.py)
            model_manager: current model manager. It controls all model
            dataset_manager: current dataset manager. It controls all datasets
            refresh_grid: this is functional, which refreshing grid of blocks after editing current block
            changed_model: Ml-model name
            task: 'Classification' or 'Regression'
            selected_dataset: id of dataset, which contained in current model
    """
    color_map: dict[str, ft.Colors] = {
        "black": ft.Colors.BLACK,
        "blue-grey": ft.Colors.BLUE_GREY_900,
        "brown": ft.Colors.BROWN_900,
        "deep-orange": ft.Colors.DEEP_ORANGE_900,
        "cyan": ft.Colors.CYAN_900
    }
    reversed_color_map = {v: k for k, v in color_map.items()}

    def __init__(
            self,
            page: ft.Page,
            block_data: dict,
            model_manager: ModelManager,
            dataset_manager: DatasetManager,
            update_func: callable
    ) -> None:
        self.page = page
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
        self.current_model_id = block_data.get('id', 'error!')
        self.current_model = self.model_manager.get_model(self.current_model_id)
        self.dialog = None
        self.refresh_grid = update_func

        # params
        self.changed_model = self.current_model.model_name
        self.task = self.current_model.task
        self.selected_dataset: str | None = None

        # all edit dialog widgets:
        self._initialize_widgets()

    def _initialize_widgets(self):
        # Text for dataset
        self.dataset_name = ft.Text(size=25)
        self.dataset_label = ft.Text(size=25)

        # dropdown for change classification models
        self.classification_model_dropdown = ft.DropdownM2(
            label="Classification model",
            value=self.changed_model,
            options=[
                ft.dropdownm2.Option("Nothing"),
                ft.dropdownm2.Option("Logistic regression"),
                ft.dropdownm2.Option("SVM"),
                ft.dropdownm2.Option("Random forest"),
                ft.dropdownm2.Option("Decision tree"),
                ft.dropdownm2.Option("K-nearest neighbors"),

            ],
            on_change=self._on_model_change,
            width=250,
        )

        # dropdown for change regression models
        self.regression_model_dropdown = ft.DropdownM2(
            label="Regression model",
            value=self.changed_model,
            options=[
                ft.dropdownm2.Option("Nothing"),
                ft.dropdownm2.Option("Linear regression"),
                ft.dropdownm2.Option("SVM"),
                ft.dropdownm2.Option("Random forest"),
                ft.dropdownm2.Option("Decision tree"),
                ft.dropdownm2.Option("K-nearest neighbors"),

            ],
            on_change=self._on_model_change,
            width=250,
        )

        # task dropdown
        self.task_dropdown = ft.DropdownM2(
            label="Task",
            value=self.task,
            options=[
                ft.dropdownm2.Option("Classification"),
                ft.dropdownm2.Option("Regression"),

            ],
            on_change=self._on_task_change,
            width=250,
        )

        self.name_field = ft.TextField(
            label="Name",
            value=self.current_model.name
        )

        self.description_field = ft.TextField(
            label="Description",
            value=self.current_model.description
        )

        self.color_dropdown = ft.Dropdown(
            label="Color",
            value=self._get_color_key(self.current_model.color),
            options=[
                ft.dropdown.Option("black"),
                ft.dropdown.Option("blue-grey"),
                ft.dropdown.Option("brown"),
                ft.dropdown.Option("deep-orange"),
                ft.dropdown.Option("cyan"),
            ]
        )

        self.dropdown_row = ft.Row(
            controls=[
                self.task_dropdown,
                self.classification_model_dropdown,
                self.regression_model_dropdown
            ]
        )

        self.divider = ft.Divider(height=1, thickness=2, color=ft.Colors.WHITE)
        self.content = ft.ListView(
            controls=[
                ft.Text("Visual", size=25),
                self.name_field,
                self.description_field,
                self.color_dropdown,
                self.divider,
                ft.Text("Model", size=25),
                self._task_state_after_window_open(),
                self.divider,
                self._select_dataset_setting(),
                self.divider,
                ft.Row(
                    [
                        ft.Text("Delete this model"),
                        ft.ElevatedButton("delete", on_click=self._delete_model)
                    ]
                )
            ],
            spacing=10,
            padding=20,
            auto_scroll=False
        )

    def _get_color_key(self, color: ft.Colors) -> str:
        """Converts color back to key for Dropdown (private method)"""
        return self.reversed_color_map.get(color, "black")

    def show(self):
        self.dialog = ft.AlertDialog(
            title=ft.Text("Edit Model"),
            content=ft.Container(
                content=self.content,
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

    def _close(self, e=None):
        self.page.close(self.dialog)
        self.page.update()

    def _save(self, e):
        """Saving parameters for current block (private method)"""
        self.current_model.color = self.color_map.get(self.color_dropdown.value, ft.Colors.BLACK)
        self.current_model.model_name = self.changed_model
        self.current_model.name = self.name_field.value
        self.current_model.task = self.task
        self.current_model.description = self.description_field.value
        if self.selected_dataset is not None:
            self.current_model.dataset_id = self.selected_dataset

        self._close(e)
        self.refresh_grid()

    def _on_model_change(self, e):
        self.changed_model = e.control.value

    def _on_task_change(self, e):
        self.task = e.control.value

        self._check_task_value()
        self.page.update()

    def _delete_model(self, e=None):
        self.model_manager.remove_model(self.current_model_id)
        self.refresh_grid()

        self._close()

    def _task_state_after_window_open(self) -> ft.Row:
        self._check_task_value()
        return self.dropdown_row

    def _check_task_value(self):
        if self.task == "Classification":
            self.regression_model_dropdown.visible = False
            self.classification_model_dropdown.visible = True
        else:
            self.classification_model_dropdown.visible = False
            self.regression_model_dropdown.visible = True
        self.page.update()

    def _select_dataset_setting(self) -> ft.Row:
        """Creates a widget with datasets if there are datasets loaded (private method)"""
        if self._have_manager_datasets():
            if bool(self.current_model.dataset_id):
                ds_id = self.current_model.dataset_id
                self.dataset_name.value = self.dataset_manager.get_dataset(ds_id).name
                self.dataset_label.value = "Dataset: "
            else:
                self.dataset_name.scale = 0.0
                self.dataset_label.value = "Dataset"
        else:
            self.dataset_name.scale = 0.0
            self.dataset_label.value = "Dataset: there are not datasets"

        return ft.Row(
                    controls=[
                        self._create_dataset_menu(),
                        self.dataset_label,
                        self.dataset_name
                    ]
        )

    def _have_manager_datasets(self) -> bool:
        datasets = self.dataset_manager.get_all()
        return bool(datasets)

    def _create_dataset_menu(self):
        """Widget, which contains all loaded datasets (private method)"""
        if not self._have_manager_datasets():
            return ft.Text(visible=False)

        datasets = self.dataset_manager.get_all()
        return ft.PopupMenuButton(
            items=[
                ft.PopupMenuItem(
                    content=ft.Row([
                        ft.Icon(ft.Icons.DATASET),
                        # V current dataset name V
                        ft.Text(datasets.get(dataset_id).name),
                        ft.IconButton(
                            icon=ft.Icons.INFO,
                            on_click=lambda e, ds=dataset_id: self._show_dataset_info(ds)
                        )
                    ]),
                    on_click=lambda e, ds=dataset_id: self._select_dataset(ds)
                )
                for dataset_id in datasets
            ],
            width=50,
            height=50,
        )

    def _show_dataset_info(self, dataset_id: str):
        """Shows a window with information about the dataset (private method)
        """
        DatasetInfoDialog(
            page=self.page,
            dataset_manager=self.dataset_manager,
            dataset_id=dataset_id
        ).show()

    def _select_dataset(self, dataset_id: str):
        """Selecting a dataset for a block (private method)"""
        self.dataset_name.value = self.dataset_manager.get_dataset(dataset_id).name
        self.dataset_name.scale = 1.0
        self.dataset_label.value = "Dataset: "
        self.selected_dataset = dataset_id
        self.page.update()
