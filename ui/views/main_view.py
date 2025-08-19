import flet as ft

def create_main_content(
    grid: ft.GridView,
    on_add_click: callable
) -> ft.Column:
    """
    Creating main content for program
    :param grid: GridView with blocks for models
    :param on_add_click: add_button func
    :return: column with main content
    """
    return ft.Column(
        controls=[
            grid,
            ft.Row(
                controls=[
                    ft.ElevatedButton(
                        "Add model",
                        icon=ft.Icons.ADD,
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=0)
                        ),
                        on_click=on_add_click
                    )
                ],
                alignment=ft.MainAxisAlignment.END
            )
        ],
        expand=True
    )