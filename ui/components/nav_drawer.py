import flet as ft

class AppDrawer:
    def __init__(self,
                 page: ft.Page,
                 on_settings_click: callable,
                 on_models_click: callable,
    ) -> None:
        self.drawer = ft.NavigationDrawer(
            position=ft.NavigationDrawerPosition.START,
            controls=[
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.IconButton(
                                ft.Icons.MENU,
                                on_click=self.close
                            ),
                            ft.Text("Menu", size=20),
                        ],
                        spacing=2,  # Отступ между элементами в ftRow
                    ),
                    #Отступ кнопок в меню
                    padding=ft.padding.only(top=10, left=10),
                ),
                ft.ElevatedButton(
                    content=ft.Row([
                        ft.Icon(ft.Icons.APPS),
                        ft.Text("models", size=16)
                    ], spacing=10
                    ),
                    on_click=lambda e: self._handle_click(e, on_models_click),
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=0),
                        padding=15,
                    ),
                    height=50,
                ),
                ft.ElevatedButton(
                    content=ft.Row([
                        ft.Icon(ft.Icons.SETTINGS),
                        ft.Text("settings", size=16)
                    ], spacing=10),
                    on_click=lambda e: self._handle_click(e, on_settings_click),
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=0),
                        padding=15,
                    ),
                    height=50,
                ),
            ],
        )
        self.page = page

    def _handle_click(self, e, callback):

        self.close(e)
        callback()

    def open(self, e=None):
        return self.page.open(self.drawer)

    def close(self, e=None):
        return self.page.close(self.drawer)