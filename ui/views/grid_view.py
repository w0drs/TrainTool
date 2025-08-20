import flet as ft

from core.datasets.datasets_manager import DatasetManager
from ui.dialogs.block_edit_dialog import EditDialog
from core.models.model_manager import ModelManager
from ui.dialogs.train_model_dialog import TrainEditDialog


class GridView:
    """Grid, which contains all blocks"""
    def __init__(
            self,
            page: ft.Page,
            model_manager: ModelManager,
            dataset_manager: DatasetManager
    ) -> None:
        #Its grid for Ml-model blocks
        self.blocks_grid = ft.GridView(
            expand=True,
            runs_count=6,  # Column count
            spacing=10,    # indent between blocks
            run_spacing=10, # indent between rows
            animate_size=True,
            animate_opacity=True,
        )
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
        self.page = page

    def create_block(
            self,
            data:dict
    ) -> ft.Container:
        """
        Возвращает объект контейнера, где будет вся инфа о Ml-модели
        """
        bd = data.copy() # block_data
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Column(
                        controls=[
                            ft.Text(data.get("name", "no name model"), size=16),
                            ft.Text(data.get("description", "something things")),
                        ],
                        expand=True,  # Растягиваем верхнюю часть
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    ft.Row(
                        controls=[
                            ft.IconButton(
                                icon=ft.Icons.ROCKET_LAUNCH,
                                style=ft.ButtonStyle(
                                    shape=ft.RoundedRectangleBorder(radius=0)
                                ),
                                on_click=lambda e: self._show_train_dialog(e, bd)
                            ),
                            ft.IconButton(
                                icon=ft.Icons.SETTINGS,
                                style=ft.ButtonStyle(
                                    shape=ft.RoundedRectangleBorder(radius=0)
                                ),
                                on_click=lambda e: self._open_edit_dialog(e, bd)
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.END
                    )
                ],
                spacing=0,
                expand=True,
            ),
            alignment=ft.alignment.center,
            width=60,
            height=90,
            bgcolor=data["color"],
            border_radius=10,
            padding=10,
            ink=False,
            animate=ft.Animation(300, ft.AnimationCurve.EASE_IN_OUT),
            on_hover=lambda e: self._animate_padding(e)
        )

    def _animate_padding(self, e):
        e.control.padding = 15 if e.data == "true" else 10
        e.control.update()

    def add_block(
            self,
            data: dict,
    ) -> None:
        """
        adding the block into grid
        :param data: data  - 'color', 'name' and 'description' for block
        """
        new_models_id = self.model_manager.add_model()
        current_data = data.copy()
        current_data["id"] = new_models_id
        current_dataset = self.model_manager.get_model(new_models_id)

        # changing color, name and description in new block
        current_dataset.color = current_data.get('color',ft.Colors.BLACK)
        current_dataset.name = current_data.get('name', 'no name model')
        current_dataset.description = current_data.get('description', 'something things')

        new_block = self.create_block(current_data)
        new_block.data = current_dataset.id

        self.blocks_grid.controls.append(new_block)
        self.page.update()

    def get_grid(self) -> ft.GridView:
        return self.blocks_grid

    def _open_edit_dialog(self, e, block_data: dict):
        """
            Open edit dialog for model (private method)
            Args:
            block_data: dict - 'color', 'name', 'description' and 'id' of model's block
        """
        edit_dialog = EditDialog(
            page=self.page,
            block_data=block_data,
            model_manager=self.model_manager,
            dataset_manager=self.dataset_manager,
            update_func=self._refresh_grid
        )
        edit_dialog.show()

    def _show_train_dialog(self, e, block_data: dict):
        TrainEditDialog(
            page=self.page,
            dataset_manager=self.dataset_manager,
            model=self.model_manager.get_model(block_data.get("id","error"))
        ).show()

    def _refresh_grid(self):
        """Refreshing all blocks"""
        self.blocks_grid.controls.clear()
        for model_id in self.model_manager.get_all():
            model = self.model_manager.get_model(model_id)
            data = {
                "id": model_id,
                "name": model.name,
                "description": model.description,
                "color": model.color
            }
            self.blocks_grid.controls.append(self.create_block(data))
        self.page.update()