import flet as ft

class AddDialog:
    color_map: dict[str, ft.Colors] = {
        "black": ft.Colors.BLACK,
        "blue-grey": ft.Colors.BLUE_GREY_900,
        "brown": ft.Colors.BROWN_900,
        "deep-orange": ft.Colors.DEEP_ORANGE_900,
        "cyan": ft.Colors.CYAN_900
    }

    def __init__(self, page: ft.Page, add_block_func = None):
        self.page = page
        self.dialog = None

        # Dropdown for changing the colors for blocks
        self.color_dropdown = ft.DropdownM2(
            label="color",
            value="black",
            options=[
                ft.dropdownm2.Option("black"),
                ft.dropdownm2.Option("blue-grey"),
                ft.dropdownm2.Option("brown"),
                ft.dropdownm2.Option("deep-orange"),
                ft.dropdownm2.Option("cyan"),
            ],
            on_change=self.on_color_change,
            width=200,
        )

        # TextEdit for name for blocks
        self.name_field = ft.TextField(
            label="Name",
            max_length=20
        )

        # TextEdit for description for blocks
        self.description_field = ft.TextField(
            label="Description",
            max_length=35
        )

        self.content = ft.Column([
            self.color_dropdown,
            self.name_field,
            self.description_field,

        ])
        self.color = ft.Colors.BLACK
        self.title = "New model"
        self.add_func = add_block_func

    def show(self):
        def apply(e):
            if self.add_func is not None:
                data = {
                    "color": self.color_map.get(self.color, ft.Colors.BLACK),
                    "name": self.name_field.value,
                    "description": self.description_field.value
                }
                self.add_func(data)

                self.name_field.value = ""
                self.description_field.value = ""
                self.color_dropdown.value = "black"

                self.name_field.update()
                self.description_field.update()
                self.color_dropdown.update()

            self.page.close(self.dialog)
            self.page.update()

        self.dialog = ft.AlertDialog(
            title=ft.Text(self.title),
            content=self.content,
            actions=[
                ft.ElevatedButton(
                    "add",
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=0)
                    ),
                    on_click=apply
                )
            ]
        )

        self.page.dialog = self.dialog
        self.page.open(self.dialog)
        self.page.update()

    def close(self):
        if self.dialog:
            self.page.close(self.dialog)
            self.page.update()

    def on_color_change(self, e):
        self.color = e.control.value