import subprocess
from collections.abc import Callable

from .constants import DVC_BIN
from .layout import DVC_SUFFIX, Layout

Runner = Callable[[list[str]], None]


def subprocess_runner(command: list[str]) -> None:
    subprocess.run(command, check=True)


def publish_commands(layout: Layout, name: str, push: bool) -> list[list[str]]:
    target = layout.dvc_target(name)
    commands = [[DVC_BIN, "add", target]]

    if push:
        commands.append([DVC_BIN, "push", f"{target}{DVC_SUFFIX}"])

    return commands


def publish(layout: Layout, name: str, runner: Runner, push: bool) -> list[list[str]]:
    commands = publish_commands(layout, name, push)

    for command in commands:
        runner(command)

    return commands
