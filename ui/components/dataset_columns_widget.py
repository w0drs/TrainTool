import flet as ft
import pandas as pd

from core.datasets.dataset import Dataset
from core.datasets.datasets_manager import DatasetManager
from ui.dialogs.information_windows import show_snackbar


class ColumnsWidget:
    """
        Class, that is used in the dataset settings tab. This widget consists of two columns: features and target.
        If the dataset is loaded successfully, the column names from the dataset will be added to the feature column.
        These columns can be moved to the target column. This is how the features and target columns are formed for training model.
        You can disable some columns so that they are not used during training.
        Attributes:
                page: current ft.Page (from main.py)
                ds_id: id of dataset, which will be loaded
                dsm: current dataset manger. It controls all datasets
                current_dataset: current dataset
                all_columns: all loaded columns from dataset
                target_columns: target columns that are loaded from the current dataset. First value will be taken during training
                features: feature columns in dataset
    """
    def __init__(
            self,
            page: ft.Page,
            dataset_manager: DatasetManager,
            dataset_id: str,
    ) -> None:
        self.page = page
        self.ds_id = dataset_id
        self.dsm = dataset_manager
        self.current_dataset = self.dsm.get_dataset(self.ds_id)

        self.all_columns = []
        self.target_columns = []
        self.features = []

        self._initialize_types_of_columns()

        # need for uploading settings
        self.unselected_features = []

        # need to adding feature to dataset
        self.unused_features = []

        # error for open error dialog
        self.few_columns_error = False

        self._initialize_ui()

        if not self.current_dataset.target_column and not self.current_dataset.features:
            self._load_initial_state()
        else:
            self.target_columns = self.current_dataset.target_column
            self.features = self.current_dataset.features
            self.unused_features = self.current_dataset.unused_columns
            self._update_lists()

    def _initialize_ui(self):
        """
            Attributes:
                self.features_list: features column for ColumnsWidget
                self.target_list: target column for ColumnsWidget
        """
        self.features_list = ft.ListView(expand=True, spacing=5)
        self.target_list = ft.ListView(expand=True, spacing=5)

    def _initialize_types_of_columns(self):
        if self.current_dataset.category_columns is None and self.current_dataset.numeric_columns is None:
            self.categorical_features = []
            self.numerical_features = []
            self.time_features = []
        else:
            self.categorical_features = self.current_dataset.category_columns
            self.numerical_features = self.current_dataset.numeric_columns
            self.time_features = self.current_dataset.time_columns


    def _handle_checkbox(self, name: str, is_checked: bool):
        if not is_checked:
            self.unused_features.append(name)
        else:
            self.unused_features.remove(name)

    def _build_item(self, column_name: str, is_target: bool, is_select:bool):
        column_type = "category"
        if column_name in self.numerical_features:
            column_type = "numeric"
        elif column_name in self.categorical_features:
            column_type = "category"
        elif column_name in self.time_features:
            column_type = "time"
        elif column_name in self.target_columns:
            column_type = self.current_dataset.target_column_type

        checkbox = ft.Checkbox(
            value=is_select,
            on_change=lambda e: self._handle_checkbox(column_name, e.control.value)
        )
        column_type_dropdown = ft.DropdownM2(
            label="type",
            value=column_type,
            options=[
                ft.dropdownm2.Option("numeric"),
                ft.dropdownm2.Option("category"),
                ft.dropdownm2.Option("time"),
            ],
            on_change=self._change_column_type,
            data=column_name,
            width=120,
        )
        return ft.Container(
            content=ft.Row([
                ft.Text(column_name, expand=True),
                checkbox,
                column_type_dropdown,
                ft.IconButton(
                    icon=ft.Icons.ARROW_BACK if is_target else ft.Icons.ARROW_FORWARD,
                    on_click=lambda e: (
                        self._move_to_features(column_name) if is_target
                        else self._move_to_target(column_name)
                    ),
                )
            ]),
            padding=5,
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=5,
            alignment=ft.alignment.top_left
        )

    def _move_to_target(self, column_name: str):
        """Перемещает колонку в target"""
        if column_name in self.features:
            self.features.remove(column_name)
            self.target_columns.append(column_name)
            self._update_lists()

    def _move_to_features(self, column_name: str):
        """Перемещает колонку в features"""
        if column_name in self.target_columns:
            self.target_columns.remove(column_name)
            self.features.append(column_name)
            self._update_lists()

    def _update_lists(self):
        """Refreshes both lists on the screen"""
        # Adding a used feature
        self.features_list.controls = [
            self._build_item(col, False, True)
            if col not in self.unused_features
            else self._build_item(col, False, False)
            for col in self.features + self.unused_features
        ]

        # Adding a target columns
        self.target_list.controls = [
            self._build_item(col, True, True)
            if col not in self.unused_features
            else self._build_item(col, True, False)
            for col in self.target_columns
        ]
        self.page.update()

    def _clear_features(self, e):
        """Clears all selected columns"""
        self._load_initial_state()
        print(self.unused_features)
        print(self.features)

    def _build_featured_widget(self) -> ft.Column:
        return ft.Column(
            controls=[
                ft.Text("Features"),
                ft.Container(
                    content=self.features_list,
                    expand=True,
                )
            ],
            expand=True,
        )

    def _build_target_widget(self) -> ft.Column:
        return ft.Column(
            controls=[
                ft.Text("Target"),
                ft.Container(
                    content=self.target_list,
                    expand=True,
                )
            ],
            expand=True,
        )

    def build_all_widgets(self) -> ft.ListView:
        return ft.ListView(
            controls=[
                ft.Row([
                    self._build_featured_widget(),
                    ft.Column([
                        ft.IconButton(
                                icon=ft.Icons.CLEAR_ALL,
                                on_click=self._clear_features,
                                tooltip="Clear all",
                                alignment=ft.alignment.top_left
                        ),
                    ],
                    ),
                    self._build_target_widget(),
                ],
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    expand=True,
                ),
            ],
            expand=True
        )

    def _load_initial_state(self):
        """Loading dataset and adding its columns to the widget"""
        self.features = []
        self.target_columns = []
        self.unused_features = []
        if not self.current_dataset:
            return
        try:
            df = self._load_dataset()
            if df is None:
                show_snackbar(self.page,ft.Text("Error to upload dataset"), time=6)
                return

            else: self.all_columns = df.columns.tolist()

            if len(self.all_columns) < 2:
                self.few_columns_error = True
                self.all_columns = []

            features = self.current_dataset.features
            target = self.current_dataset.target_column
            if features and target:
                self.target_columns = target
                self.features = features

                # Checking for an unused feature
                for col in self.all_columns:
                    if (col not in self.target_columns) and (col not in self.features):
                        self.unused_features.append(col)
                        self.features.append(col)

            else:
                self.features = self.all_columns.copy()
                if "target" in self.features:
                    self.target_columns = ["target"]
                    self.features.remove("target")
                self.unused_features = []
            self._update_lists()

        except Exception as e:
            print(f"Error loading dataset: {e}")
            show_snackbar(self.page, ft.Text(f"Error loading dataset"), time=6)

    def _load_dataset(self) -> pd.DataFrame | None:
        df = None
        if self.current_dataset.path.name.endswith(".csv") or self.current_dataset.path.name.endswith(".txt"):
            df = self._load_csv(self.current_dataset)
        elif self.current_dataset.path.name.endswith(".xlsx"):
            df = self._load_xlsx(self.current_dataset)
        elif self.current_dataset.path.name.endswith(".json"):
            df = self._load_json(self.current_dataset)
        if df is not None:
            self._set_feature_types(df)
        return df

    def _load_csv(self, current_dataset: Dataset) -> pd.DataFrame | None:
        df = None
        try:
            df = pd.read_csv(
                current_dataset.path,
                delimiter=current_dataset.delimiter,
                encoding=current_dataset.encoding,
            )
        except Exception as e:
            print(f"Error to upload dataset: {e}")
            show_snackbar(self.page, ft.Text(f"Error to upload dataset"), time=6)
        finally:
            return df

    def _load_xlsx(self, current_dataset: Dataset) -> pd.DataFrame | None:
        df = None
        try:
            df = pd.read_excel(
                current_dataset.path,
                sheet_name=0
            )
        except Exception as e:
            print(f"Error to upload xlsx dataset: {e}")
            show_snackbar(self.page, ft.Text(f"Error to upload xlsx dataset"), time=6)
        finally:
            return df

    def _load_json(self, current_dataset: Dataset) -> pd.DataFrame | None:
        df = None
        try:
            df = pd.read_json(
                str(current_dataset),
                encoding=current_dataset.encoding
            )
        except Exception as e:
            print(f"Error to upload json dataset: {e}")
            show_snackbar(self.page, ft.Text(f"Error to upload json dataset"), time=6)
        finally:
                return df

    def _set_feature_types(self, df: pd.DataFrame, max_unique_cat: int = 10):
        """Automatic column type detection"""
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.time_features.append(col)
            elif df[col].dtype in ["object", "bool", "category"]:
                self.categorical_features.append(col)
            elif df[col].nunique() <= max_unique_cat:
                self.categorical_features.append(col)
            else:
                self.numerical_features.append(col)

    def _change_column_type(self, e):
        selected_value = e.control.value
        current_column = e.control.data

        if current_column in self.numerical_features:
            self.numerical_features.remove(current_column)
        if current_column in self.categorical_features:
            self.categorical_features.remove(current_column)
        if current_column in self.time_features:
            self.time_features.remove(current_column)

        if selected_value == "numeric":
            self.numerical_features.append(current_column)
        elif selected_value == "time":
            self.time_features.append(current_column)
        else:
            self.categorical_features.append(current_column)
        self.page.update()

    def save_state(self):
        """Saving target, feature, category, numeric, time columns in dataset"""
        if self.current_dataset:
            final_feature = [col for col in self.features if col not in self.unused_features]
            final_target = [col for col in self.target_columns if col not in self.unused_features]
            self.current_dataset.target_column = final_target
            self.current_dataset.features = final_feature
            self.current_dataset.unused_columns = self.unused_features

            # removing target column from type columns
            if final_target[0] in self.categorical_features:
                self.categorical_features.remove(final_target[0])
                self.current_dataset.target_column_type = "category"
            elif final_target[0] in self.numerical_features:
                self.numerical_features.remove(final_target[0])
                self.current_dataset.target_column_type = "numeric"
            elif final_target[0] in self.time_features:
                self.time_features.remove(final_target[0])
                self.current_dataset.target_column_type = "time"

            final_categories = [col for col in self.categorical_features if col not in self.unused_features]
            final_numerics = [col for col in self.numerical_features if col not in self.unused_features]
            final_time = [col for col in self.time_features if col not in self.unused_features]
            self.current_dataset.category_columns = final_categories if final_categories else []
            self.current_dataset.numeric_columns = final_numerics if final_numerics else []
            self.current_dataset.time_columns = final_time if final_time else []
