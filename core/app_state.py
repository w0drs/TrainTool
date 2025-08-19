import flet as ft

from core.datasets.datasets_manager import DatasetManager
from core.models.model_manager import ModelManager
from ui.views.main_view import create_main_content
from ui.views.settings_view import create_settings_content


class AppState:

    def __init__(
            self,
            page: ft.Page,
            grid: ft.GridView,
            on_add_click: callable,
            dataset_manager: DatasetManager,
            model_manager: ModelManager
    ) -> None:
        """
        The main application window status controller
        :param page: ft.Page instance
        :param grid: ft.GridView, that contain blocks of models
        :param on_add_click: this is func, which adding a new block for grid
        """
        self.page = page
        self.grid = grid
        self.on_add_click = on_add_click
        self.dataset_manager = dataset_manager
        self.model_manager = model_manager

        self._initialize_views()



    def _initialize_views(self):
        """
        func for initialize views components
        """
        self.content: ft.Column = create_main_content(
            grid=self.grid,
            on_add_click=self.on_add_click,
        )
        self.settings: ft.ListView = create_settings_content(
            on_back_click=lambda e: self.toggle_views(False),
            state=self.dataset_manager,
            page=self.page,
        )

    def toggle_views(self, show_settings: bool):
        """
        Switch between main content and settings
        """
        self.content.visible = not show_settings
        self.settings.visible = show_settings
        self.page.update()