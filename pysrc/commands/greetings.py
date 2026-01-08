from . import commands
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from pytauri.webview import WebviewWindow


class _BaseModel(BaseModel):
    """Base model that accepts camelCase from JS and snake_case from Python."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class GreetRequest(_BaseModel):
    """Greet request model."""
    name: str


# === Commands ===

@commands.command()
async def greet(body: GreetRequest, webview_window: WebviewWindow) -> str:
    """Greet a user with a personalized message."""
    webview_window.set_title(f"Hello {body.name}!")
    return f"Hello, {body.name}! , 测试调用 Python"
