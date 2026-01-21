"""PyTauri Barebones App."""

import sys
from os import environ
from pathlib import Path
from anyio import create_task_group
from anyio.abc import TaskGroup
from anyio.from_thread import start_blocking_portal
from pytauri_plugins import opener
from pytauri_wheel.lib import builder_factory, context_factory
from pysrc.commands import commands

# Configuration
BACKEND_DIR = Path(__file__).parent.absolute()
print(f"BACKEND_DIR: {BACKEND_DIR}")
print(f"Build directory exists: {BACKEND_DIR.joinpath('build').exists()}")
print(f"Index.html exists: {BACKEND_DIR.joinpath('build', 'index.html').exists()}")
DEV_MODE = environ.get("PYTAURI_REACT_DEV") == "1"
# DEV_MODE = True

task_group: TaskGroup


def main() -> int:
    """Run the Tauri app."""
    global task_group

    with (
        start_blocking_portal("asyncio") as portal,
        portal.wrap_async_context_manager(
            portal.call(create_task_group)
        ) as task_group,
    ):
        if DEV_MODE:
            tauri_config = {
                "build": {"frontendDist": "http://localhost:1420"},
            }
        else:
            # Set the frontendDist to the build directory relative to BACKEND_DIR
            tauri_config = {
                "build": {"frontendDist": str(BACKEND_DIR.joinpath("build"))},
            }
        print(f"Tauri config: {tauri_config}")

        # Create context and print its properties
        context = context_factory(BACKEND_DIR, tauri_config=tauri_config)
        print(f"Context created successfully")

        app = builder_factory().build(
            context=context,
            invoke_handler=commands.generate_handler(portal),
            plugins=(opener.init(),),
        )
        return app.run_return()


if __name__ == "__main__":
    sys.exit(main())
