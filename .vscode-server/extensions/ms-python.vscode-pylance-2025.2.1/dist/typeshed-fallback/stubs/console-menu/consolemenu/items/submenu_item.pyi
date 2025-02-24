from collections.abc import Callable

from consolemenu.console_menu import ConsoleMenu
from consolemenu.items import MenuItem as MenuItem

class SubmenuItem(MenuItem):
    submenu: ConsoleMenu
    def __init__(
        self,
        text: str | Callable[[], str],
        submenu: ConsoleMenu,
        menu: ConsoleMenu | None = None,
        should_exit: bool = False,
        menu_char: str | None = None,
    ) -> None: ...
    menu: ConsoleMenu
    def set_menu(self, menu: ConsoleMenu) -> None: ...
    def set_up(self) -> None: ...
    def action(self) -> None: ...
    def clean_up(self) -> None: ...
    def get_return(self) -> object: ...
    def get_submenu(self) -> ConsoleMenu: ...
