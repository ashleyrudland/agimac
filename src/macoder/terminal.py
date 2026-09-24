"""Small terminal presentation layer; never changes prompts or generated tokens."""

import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console(highlight=False)


def banner():
    console.print("[bold cyan]agimac[/bold cyan]  [dim]local intelligence · Apple silicon[/dim]")


def answer(text):
    if sys.stdout.isatty():
        console.print(
            Panel(
                Markdown(text),
                title="agimac",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
            )
        )
    else:
        print("Assistant>", text)


def tool(call, result):
    import json

    # Escape model-generated content by using Text rather than Rich markup.
    from rich.text import Text

    console.print(Text("tool › " + json.dumps(call) + " → " + json.dumps(result), style="yellow"))
