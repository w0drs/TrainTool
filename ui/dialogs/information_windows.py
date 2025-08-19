import flet as ft

def show_banner(
        page: ft.Page,
        content: ft.Control
) -> None:
    def close_banner(e):
        page.close(page.banner)
        page.update()

    banner = ft.Banner(
        bgcolor=ft.Colors.AMBER_100,
        leading=ft.Icon(ft.Icons.WARNING, color=ft.Colors.ORANGE),
        content=content,
        actions=[
            ft.TextButton("Cancel", on_click=close_banner),
        ]
    )
    page.banner = banner
    page.open(banner)
    page.update()

def show_snackbar(
        page: ft.Page,
        content: ft.Text,
        color: ft.Colors = ft.Colors.WHITE,
        time: int = 5
) -> None:
    snack_bar = ft.SnackBar(
        content=content,
        action_color=color,
        duration=time * 1000
    )
    page.snack_bar = snack_bar
    page.open(snack_bar)
    page.update()


def show_error_dialog(
        page: ft.Page,
        title: str,
        message: str,
        button_text: str = "OK",
        bgcolor: ft.Colors = ft.Colors.RED_300,
        icon: ft.Icons = ft.Icons.ERROR_OUTLINE
) -> None:
    """Показывает кастомный диалог с ошибкой.

    Args:
        page: Страница Flet
        title: Заголовок ошибки
        message: Текст сообщения
        button_text: Текст кнопки (по умолчанию "OK")
        bgcolor: Цвет фона (по умолчанию red_300)
        icon: Иконка (по умолчанию error_outline)
    """

    def close_dialog(e):
        page.close(page.error_dialog)
        page.update()

    error_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Row(
            controls=[
                ft.Icon(icon, color=ft.Colors.RED),
                ft.Text(title, weight=ft.FontWeight.BOLD),
            ],
            spacing=10
        ),
        content=ft.Text(message),
        actions=[
            ft.TextButton(
                button_text,
                on_click=close_dialog,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.RED_700,
                    padding=ft.Padding(15, 5, 15, 5),
                )
            ),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
        bgcolor=bgcolor,
        shape=ft.RoundedRectangleBorder(radius=10),
    )
    page.error_dialog = error_dialog
    page.open(error_dialog)
    page.update()