import flet as ft

from core.app_state import AppState
from core.datasets.datasets_manager import DatasetManager
from core.models.model_manager import ModelManager
from ui.components.nav_drawer import AppDrawer
from ui.views.grid_view import GridView
from ui.dialogs.add_block_dialog import AddDialog

def main(page: ft.Page):
    page.window.width = 1000
    page.window.height = 600
    page.window.resizable = False
    page.window.full_screen = False
    page.window.maximizable = False
    page.update()

    # deleting old head of program
    page.window.title_bar_hidden = True
    page.window.title_bar_buttons_hidden = True

    # functions for window
    def minimize_window(e):
        page.window.minimized = True
        page.update()

    def close_window(e):
        page.window.close()

    #datasets manager
    dataset_manager = DatasetManager()
    #models manager
    model_manager = ModelManager()

    # initialization the grid of blocks
    gridview = GridView(
        page=page,
        model_manager=model_manager,
        dataset_manager=dataset_manager
    )
    # AppState
    state = AppState(
        page=page,
        grid=gridview.get_grid(),
        on_add_click=lambda e: add_dialog.show(),
        dataset_manager=dataset_manager,
        model_manager=model_manager
    )

    # initialization menu bar
    drawer = AppDrawer(
        page=page,
        on_settings_click=lambda:state.toggle_views(True),
        on_models_click=lambda: state.toggle_views(False),
    )

    # custom program head
    custom_title_bar = ft.Row(
        controls=[
            ft.IconButton(ft.Icons.MENU, on_click=drawer.open),
            ft.WindowDragArea(ft.Container(ft.Text("TrainTool"), padding=10), expand=True),
            ft.IconButton(ft.Icons.MINIMIZE, on_click=minimize_window),
            ft.IconButton(ft.Icons.CLOSE, on_click=close_window),
        ],
        spacing=0,
        tight=True
    )

    # dialog for adding a new model
    add_dialog = AddDialog(
        page,
        gridview.add_block
    )


    # adding all widgets
    page.add(
        ft.Column(
            controls=[
                custom_title_bar,
                state.content,
                state.settings,
            ],
            spacing=0,
            expand=True
        )
    )

if __name__ == "__main__":
    ft.app(
        target=main,
        view=ft.AppView.FLET_APP,
    )