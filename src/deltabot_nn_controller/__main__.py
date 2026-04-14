"""Interface for ``python -m deltabot_nn_controller``."""

import os

import typer

from . import __version__
from .globals import NaturalOrderGroup
from .model_utils.run_all import CompleteRun

__all__ = ["main"]

cli = typer.Typer(cls=NaturalOrderGroup, pretty_exceptions_enable=False)


def version_callback(value):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@cli.command()
def model(
    model_cfg_pth: str = typer.Argument(
        default="src/deltabot_nn_controller/model_zoo/plant_mlp.json",
        help="A relative path to the model run config json",
        exists=True,
        dir_okay=True,
        file_okay=False,
        autocompletion=lambda: [],  # forces autocompletion
    ),
) -> None:
    """
    Completely trains and tests a model config.
    """
    if not model_cfg_pth:
        typer.echo("Error: A model config must be provided.", err=True)

    os.system("clear")

    # Train and Test model
    CompleteRun(model_cfg_pth)


@cli.command()
def test() -> None:
    print("hello world")


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
    Model builder for Deltabot Neural Network configuration
    """


# test with:
#   uv pip install -e .
#   deltabot-nn-controller --help
if __name__ == "__main__":
    cli()
