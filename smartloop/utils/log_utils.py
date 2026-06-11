import logging
from rich.console import Console
from rich.text import Text
from smartloop.constants import LOGO, SLP_ACCENT

def print_logo(console: Console = None, version: str = None):
    """Print the Smartloop logo and framework info to the console."""
    try:
        if console is None:
            console = Console()
        console.print(Text(LOGO, style=f"bold {SLP_ACCENT}"))

        if version:
            console.print(f"[dim]Version:[/dim] [green]{version}[/green]")

        console.print()
    except Exception:
        # Fallback for terminals that don't support Rich or Unicode (e.g. Windows)
        print(f"Smartloop Framework{f' v{version}' if version else ''}")
        print()


class ConsoleLogger(logging.Logger):
    def __init__(self, console: Console):
        self.console = console

    def info(self, msg, *args):
        if args:
            msg = msg % args
        self.console.print(f"[blue]{msg}[/blue]")

    def warning(self, msg, *args):
        if args:
            msg = msg % args
        self.console.print(f"[yellow]{msg}[/yellow]")

    def debug(self, msg, *args):
        if args:
            msg = msg % args
        self.console.print(f"[magenta]{msg}[/magenta]")

    def error(self, msg, *args):
        if args:
            msg = msg % args
        self.console.print(f"[red]{msg}[/red]")
