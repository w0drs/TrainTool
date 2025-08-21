import pandas as pd
import flet as ft
import os

from core.datasets.dataset import Dataset
from core.datasets.datasets_manager import DatasetManager
from core.models.model import Model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer, f1_score, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
import numpy as np
from datetime import datetime

from core.reports.report_generator import PDFReportGenerator as pdf
from ui.dialogs.export_model_dialog import ExportModelDialog
from ui.dialogs.information_windows import show_snackbar
from core.metrics_visualization.model_visualizer import ModelVisualizer as mv


class TrainEditDialog:
    """
    This class is a model training window.
    This is where the dataset of the block with the model is loaded,
    the model itself is selected and trained.

    Attributes:
        page: ft.Page from main.py
        ds_manager: Dataset manager for control all datasets
        model: object of MODEL class
        features: feature columns in dataset
        target: target column in dataset
    """
    def __init__(
            self,
            page:ft.Page,
            dataset_manager:DatasetManager,
            model: Model
    ) -> None:
        self.divider = ft.Divider(height=1, thickness=2, color=ft.Colors.GREY_300)

        self.page = page
        self.ds_manager = dataset_manager
        self.model = model

        self.features = []
        self.target = ""

        self._initialize_widgets()
        self._initialize_dialog()

        # for export to pdf
        self.y_true: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None
        self.file_picker = ft.FilePicker(on_result=self._on_export_report_button_clicked)
        self.page.overlay.append(self.file_picker)

    def show(self):
        self.page.dialog = self.dialog
        self.page.open(self.dialog)
        self.page.update()

    def _initialize_dialog(self) -> None:
        """
        initialization window for training model
        """
        self.dialog = ft.AlertDialog(
            title=ft.Text(f"Train model"),
            content=ft.Container(
                content=self._content_for_dialog(),
                width=900,
                height=650,
                padding=20,
            ),
            actions=[
                self.buttons_action
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            shape=ft.RoundedRectangleBorder(radius=1),
            inset_padding=20,
            modal= True
        )

    def _content_for_dialog(self) -> ft.ListView:
        """
        returns widget, that contains all other widgets for this window
        """
        m = self.model
        dsm = self.ds_manager

        if m.dataset_id:
            dataset_name = dsm.get_dataset(m.dataset_id).name
        else:
            dataset_name = "this model hasn't dataset"

        return ft.ListView(
            controls=[
                ft.Column(
                    controls=[
                        ft.Text(value="Current stats",size=35),
                        self._info_row("Model: ", f"{m.model_name} ({m.task.lower()})"),
                        self._info_row("Dataset: ", f"{dataset_name}"),
                        self.divider,
                        self.list_widget_of_metrics,
                        ft.Text(value="Process of training", size=35),
                        self.list_bar,
                    ]
                )
            ],
        )

    def _initialize_widgets(self):
        self.dataset_bar = ft.Text(size=25, color=ft.Colors.RED)
        self.train_bar = ft.Text(size=25, color=ft.Colors.RED)
        self.save_metric_bar = ft.Text(size=25, color=ft.Colors.RED)
        self.draw_charts_bar = ft.Text(size=25, color=ft.Colors.RED)

        # Widget, that contains text widgets for checking process of training
        self.list_bar = ft.ListView(
            controls=[
                self.dataset_bar,
                self.train_bar,
                self.save_metric_bar,
                self.draw_charts_bar
            ]
        )
        # button for export ML-model
        self.export_model_button = ft.ElevatedButton(
            text="Export model",
            on_click=self._export_model_dialog,
            disabled=self._export_button_state()
        )

        # button for export pdf report
        self.export_report_button = ft.ElevatedButton(
            text="Export report",
            on_click=self._export_report,
            disabled=self._export_button_state()
        )

        # Lower interface buttons (to exit the window or train the model)
        self.buttons_action = ft.Row(
            controls=[
                ft.ElevatedButton("Train", on_click=self._train_model),
                self.export_model_button,
                self.export_report_button,
                ft.ElevatedButton("Cancel", on_click=self._close)
            ],
            alignment=ft.MainAxisAlignment.END
        )

        # Field of widgets for checking last metrics of model
        self.list_widget_of_metrics = ft.ListView(
            controls=self._last_metrics_list()
        )

    def _get_dataset(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        try:
            df = None
            dataset_id = self.model.dataset_id
            current_dataset = self.ds_manager.get_dataset(dataset_id)

            if not current_dataset or current_dataset is None:
                print("Error: Dataset not found")
                return None, None

            usecols = current_dataset.features + [current_dataset.target_column[0]]

            # Используем менеджер контекста для файлов
            if current_dataset.path.name.endswith((".csv", ".txt")):
                with pd.option_context('mode.chained_assignment', None):
                    df = self._load_csv(current_dataset, usecols)
            elif current_dataset.path.name.endswith(".xlsx"):
                df = self._load_xlsx(current_dataset, usecols)
            elif current_dataset.path.name.endswith(".json"):
                df = self._load_json(current_dataset, usecols)

            if df is not None:
                features = df[current_dataset.features]
                transformed_features = self._transform_feature(features, current_dataset)

                target = df[[current_dataset.target_column[0]]]
                transformed_target = self._transform_target(target, current_dataset)

                del df, features, target
                return transformed_features, transformed_target

        except MemoryError:
            print("Error: Not enough memory to load dataset")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")

        return None, None

    def _transform_feature(self, df: pd.DataFrame, dataset: Dataset):
        transformers = []
        if not dataset.can_transform:
            return df
        if dataset.numeric_columns:
            transformers.append(
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")), # fill NaN
                    ("scaler", StandardScaler())
                ]), dataset.numeric_columns)
            )
        if dataset.category_columns:
            transformers.append(
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")), # fill NaN
                    ("encoder", OneHotEncoder())
                ]), dataset.category_columns)
            )
        if dataset.time_columns:
            date_processor = FunctionTransformer(
                self._full_date_processing,
                kw_args={'date_columns': dataset.time_columns}
            )

            transformers.append(
                ("date", date_processor, dataset.time_columns)
            )
        if transformers:
            try:
                preprocessor = ColumnTransformer(
                    transformers=transformers,
                    remainder="passthrough"
                )
                processed_data = preprocessor.fit_transform(df)
                return processed_data
            except Exception as e:
                print(f"Error to transform features: {e}")
        return df

    def _extract_date_features(self, df, date_columns):
        features = {}
        for col in date_columns:
            dt_series = pd.to_datetime(df[col], errors='coerce')
            features.update({
                f'{col}_year': dt_series.dt.year.fillna(-1).astype(int),
                f'{col}_month': dt_series.dt.month.fillna(-1).astype(int),
                f'{col}_day': dt_series.dt.day.fillna(-1).astype(int),
                f'{col}_dayofweek': dt_series.dt.dayofweek.fillna(-1).astype(int),
                f'{col}_hour': dt_series.dt.hour.fillna(-1).astype(int) if hasattr(dt_series.dt, 'hour') else -1
            })
        return pd.DataFrame(features)

    def _full_date_processing(self, X, date_columns):
        df = pd.DataFrame(X, columns=date_columns)
        features = self._extract_date_features(df, date_columns)

        imputer = SimpleImputer(strategy="most_frequent")
        scaler = StandardScaler()

        return scaler.fit_transform(imputer.fit_transform(features))

    def _transform_target(self, df: pd.DataFrame, dataset: Dataset):
        target_col = dataset.target_column[0]

        if dataset.target_column_type == "numeric":
            return df[target_col].values

        elif dataset.target_column_type == "category":
            if self.model.task == "Classification":

                imputer = SimpleImputer(strategy="most_frequent")
                le = LabelEncoder()

                imputed = imputer.fit_transform(df[[target_col]]).ravel()

                return le.fit_transform(imputed)

            elif self.model.task == "Regression":
                imputer = SimpleImputer(strategy="most_frequent")
                return imputer.fit_transform(df[[target_col]]).ravel()

        return df[target_col].values

    def _load_csv(self, current_dataset: Dataset, usecols: list[str]) -> pd.DataFrame | None:
        df = None
        try:
            df = pd.read_csv(
                current_dataset.path,
                delimiter=current_dataset.delimiter,
                encoding=current_dataset.encoding,
                usecols=usecols
            )
        except Exception as e:
            error = "Error to upload dataset"
            print(f"{error}: {e}")
            show_snackbar(self.page, ft.Text(error), time=6)
        finally:
            return df

    def _load_xlsx(self, current_dataset: Dataset, usecols: list[str] | list[int] | int) -> pd.DataFrame | None:
        df = None
        try:
            df = pd.read_excel(
                current_dataset.path,
                sheet_name=0,
                usecols=usecols,
            )
        except Exception as e:
            error = "Error to upload xlsx dataset"
            print(f"{error}: {e}")
            show_snackbar(self.page, ft.Text(error), time=6)
        finally:
            return df

    def _load_json(self, current_dataset: Dataset, usecols: list[str]) -> pd.DataFrame | None:
        df = None
        try:
            df = pd.read_json(
                str(current_dataset),
                encoding=current_dataset.encoding
            )
        except Exception as e:
            error = "Error to upload json dataset"
            print(f"{error}: {e}")
            show_snackbar(self.page, ft.Text(error), time=6)
        finally:
            if df is not None:
                return df[usecols]
            else:
                return df

    def _upload_dataset(self) -> tuple[np.ndarray, np.ndarray, None] | tuple[None, None, str]:
        X, y = self._get_dataset()
        if X is None or y is None:
            error = "Failed to load dataset..."
            return None, None, error

        upload_dataset = "Dataset was uploaded"
        self._progress_text_info(bar=self.dataset_bar, text=upload_dataset, color=ft.Colors.GREEN)
        return X, y, None

    def _select_model(self):
        """
        Selecting model for training
        Return machine learning model or None type
        """
        model_name = self.model.model_name
        if not model_name or model_name == "nothing":
            return None


        # GridSearchCV parameters
        params = {}
        model = None

        if model_name == "Linear regression":
            model = LinearRegression(n_jobs=-1)

        elif model_name == "Logistic regression":
            model = LogisticRegression(n_jobs=-1)
            params = {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            }

        elif model_name == "SVM":
            model = SVC() if self.model.task == "Classification" else SVR()
            params = {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        elif model_name == "Random forest":
            if self.model.task == "Regression":
                model = RandomForestRegressor(n_jobs=-1)
            else:
                model = RandomForestClassifier(n_jobs=-1)
            params = {
                "criterion": ["entropy", "gini"],
                'max_depth': range(1, 5),
                'min_samples_split': range(2, 4),
                "min_samples_leaf": range(1, 3)
            }

        elif model_name == "Decision tree":
            if self.model.task == "Regression":
                model = DecisionTreeRegressor()
            else:
                model = DecisionTreeClassifier()
            params = {
                "criterion": ["entropy", "gini"],
                'max_depth': range(1,10),
                'min_samples_split': range(2,7),
                "min_samples_leaf": range(1, 7)
            }

        elif model_name == "K-nearest neighbors":
            if self.model.task == "Regression":
                model = KNeighborsRegressor(n_jobs=-1)
            else:
                model = KNeighborsClassifier(n_jobs=-1)
            params = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        # for metrics
        if self.model.task == "Classification":
            scoring = {'accuracy': make_scorer(accuracy_score), 'f1': make_scorer(f1_score, average='weighted')}
            refit = 'f1'
        else:
            scoring = {'r2': make_scorer(r2_score), 'mse': make_scorer(mean_squared_error, greater_is_better=False)}
            refit = 'r2'

        if model is not None and params:
            return GridSearchCV(model, params, cv=3, scoring=scoring, refit=refit)
        return model

    def _train_model(self, e) -> None:
        """Trains the selected model on the uploaded data"""
        current_step = "uploading dataset"
        if not self.model.dataset_id:
            error = "Error: Dataset doesnt choose"
            self._progress_text_info(bar=self.dataset_bar,text=error, color=ft.Colors.RED)
            return

        try:
            # ds = self.ds_manager.get_dataset(self.model.dataset_id)
            # print(self.model.task)
            # print(ds.target_column_type)
            # print(ds.target_column[0])
            # print(ds.features)
            # print(ds.numeric_columns)
            # print(ds.category_columns)
            # print(ds.time_columns)
            # print(ds.unused_columns)
            # Blocking current interface
            self._set_ui_state(training=True)
            info = "Uploading dataset..."
            self._progress_text_info(bar=self.dataset_bar, text=info, color=ft.Colors.ORANGE)

            # Uploading dataset
            X, y, error = self._upload_dataset()
            if not self._validate_data(X, y):
                self._progress_text_info(bar=self.dataset_bar, text=error, color=ft.Colors.RED)
                return

            upload_dataset = "Dataset was uploaded"
            self._progress_text_info(bar=self.dataset_bar, text=upload_dataset, color=ft.Colors.GREEN)

            model_selecting = "Selecting and training model..."
            self._progress_text_info(bar=self.train_bar, text=model_selecting, color=ft.Colors.ORANGE)

            # Selecting model for training
            current_step = "selecting and training model"

            model = self._select_model()
            if not self._validate_model(model):
                error = "Model wasn't selected"
                self._progress_text_info(bar=self.train_bar, text=error, color=ft.Colors.RED)
                return

            # training
            try:
                model.fit(X, y)
            except MemoryError:
                print("Error: Not enough memory for training")
                return
            except Exception as e:
                print(f"Error: {e}")
                return

            self.model.ml_model = model

            model_train = "Model was trained"
            self._progress_text_info(bar=self.train_bar, text=model_train, color=ft.Colors.GREEN)

            saving_metrics = "Saving metrics..."
            self._progress_text_info(bar=self.save_metric_bar, text=saving_metrics, color=ft.Colors.ORANGE)
            current_step = "saving metrics"

            # save metrics (accuracy, f1 or r2, mse)
            self.y_pred = model.predict(X)
            self.y_true = y
            if not self._save_metrics(model, self.y_true, self.y_pred):
                print("Error: Failed to save metrics")

            saving_metrics = "Metrics was saved"
            self._progress_text_info(bar=self.save_metric_bar, text=saving_metrics, color=ft.Colors.GREEN)

            print("Training was ended!")
            show_snackbar(
                self.page,
                ft.Text("Training was ended!", color=ft.Colors.WHITE),
                ft.Colors.GREEN_100
            )
            print(self.model.metrics)

        except Exception as e:
            error = f"Failed in {current_step}"
            self._progress_text_info(bar=self.train_bar, text=error, color=ft.Colors.RED)
            print(f"{error}: {str(e)}")
        finally:
            self.list_widget_of_metrics.controls = self._last_metrics_list()
            self.export_model_button.disabled = self._export_button_state()
            self.export_report_button.disabled = self._export_button_state()
            self._set_ui_state(training=False)

    def _save_metrics(self, model, y, y_pred) -> bool:
        try:
            if self.model.settings is None:
                self.model.settings = {}

            self.model.settings.update({
                "last_model": self.model.name,
                "last_task": self.model.task,
                "training_date": datetime.now().isoformat()
            })

            if hasattr(model, 'best_params_'):
                self.model.settings["best_params"] = str(model.best_params_)  # Сериализация

            if self.model.task == "Classification":
                self.model.metrics = {
                    'train_f1': float(f1_score(y, y_pred, average='weighted')),
                    'train_accuracy': float(accuracy_score(y, y_pred))
                }
            else:
                self.model.metrics = {
                    'train_mse': float(mean_squared_error(y, y_pred)),
                    'train_r2': float(r2_score(y, y_pred))
                }
            return True
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")
            return False

    def _export_model_dialog(self, e):
        ExportModelDialog(
            page=self.page,
            model=self.model,
        ).show()

    def _export_report(self, e):
        if self.y_true is None or self.y_pred is None:
            return
        self.file_picker.get_directory_path("Choose the folder for export")

    def _last_metrics_list(self) -> list[ft.Control]:
        """
        returns list if ft.Text of model metrics
        (first ft.Text is label)
        """
        if self.model.metrics:
            metrics_list: list[ft.Control] = [
                self._info_row(metric_name, f"{metric_value:.3f}")
                for metric_name, metric_value in self.model.metrics.items()
            ]
            metrics_list.insert(0, ft.Text("Last metrics", size=35))
            metrics_list.append(ft.Divider(height=1, thickness=2, color=ft.Colors.GREY_300))
            return metrics_list
        else:
            return []

    def _set_ui_state(
            self,
            training: bool = True
    ) -> None:
        """
        Blocking or unblocking interface
        :param training: True - blocking, False - unblocking
        """
        if training:
            self.buttons_action.disabled = True
            self.buttons_action.opacity = 0
        else:
            self.buttons_action.disabled = False
            self.buttons_action.opacity = 1
        self.page.update()

    def _info_row(
            self,
            label: str,
            value: any,
            label_size: int = 20,
            value_size: int = 20
    ) -> ft.Row:
        return ft.Row(
            controls=[
                ft.Text(f"{label}", size=label_size),
                ft.Text(f"{value}", size=value_size, color=ft.Colors.LIGHT_BLUE)
            ]
        )

    def _progress_text_info(
            self,
            bar: ft.Text,
            text: str,
            color: ft.Colors
    ) -> None:
        bar.value = text
        bar.color = color
        self.page.update()

    def _export_button_state(self) -> bool:
        if self.model.ml_model is None:
            return True
        else:
            return False

    def _on_export_report_button_clicked(self, e: ft.FilePickerResultEvent):
        if self.y_true is None or self.y_pred is None:
            print("l")
            return
        if e.path:
            selected_folder = e.path
            if self.model.task == "Classification":
                roc_curve = mv.roc_curve(self.y_true, self.y_pred)
                confusion_matrix = mv.confusion_matrix(self.y_true, self.y_pred)
                list_of_images = [roc_curve,confusion_matrix]
            else:
                line_compare = mv.regression_line_compare(self.y_true, self.y_pred)
                list_of_images = [line_compare]
            date_now = str(datetime.now()).replace(":","_").replace(" ","_")
            name_pdf = f"{self.model.task}_model_report_{date_now}.pdf"
            pdf.generate(
                images=list_of_images,
                metrics=self.model.metrics,
                output_path=os.path.join(selected_folder,name_pdf),
                model=self.model,
                dataset=self.ds_manager.get_dataset(self.model.dataset_id)
            )

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Проверка данных перед обучением"""
        if X.size == 0 or y.size == 0:
            print("Error: Empty dataset")
            return False

        if X.shape[0] != y.shape[0]:
            print(f"Error: Mismatched samples. Features: {X.shape[0]}, Target: {y.shape[0]}")
            return False

        return True

    def _validate_model(self, model) -> bool:
        """Проверка модели перед обучением"""
        if model is None:
            print("Error: Model not selected")
            return False

        if not hasattr(model, 'fit'):
            print("Error: Selected object is not a valid sklearn model")
            return False

        return True


    def _close(self, e):
        self.page.close(self.dialog)
        self.page.update()



