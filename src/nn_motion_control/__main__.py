"""
Command-line interface for ``python -m nn_motion_control``.
"""

import typer
from typer.core import TyperGroup

from . import __version__
from .training.run import CompleteRun

__all__ = ["main"]


class NaturalOrderGroup(TyperGroup):
    """
    Typer group that lists commands in declaration order rather than alphabetically.
    """

    def list_commands(self, ctx):
        return list(self.commands)


cli = typer.Typer(cls=NaturalOrderGroup, pretty_exceptions_enable=False)


def version_callback(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@cli.command()
def model(
    model_cfg_pth: str = typer.Argument(
        default="examples/deltabot/configs/plant_tcn.json",
        help="Path to the artifact run-config JSON.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
) -> None:
    """
    Train and test a model from its artifact config.
    """

    CompleteRun(model_cfg_pth)


@cli.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Print the program version and exit.",
    ),
):
    """
    Neural-network motion-controller builder.
    """


if __name__ == "__main__":
    cli()
