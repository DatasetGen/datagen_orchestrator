import pyfiglet
from rich.console import Console
from rich.text import Text

console = Console()

def print_banner():
    ascii_banner = pyfiglet.figlet_format("Datagen")
    console.print(Text(ascii_banner, style="bold magenta"))
