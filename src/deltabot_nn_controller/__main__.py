"""Interface for ``python -m deltabot_nn_controller``."""

from argparse import ArgumentParser
from collections.abc import Sequence

from model_zoo.json_manager import load_config

from . import __version__

__all__ = ["main"]


def main(args: Sequence[str] | None = None) -> None:
    """Argument parser for the CLI."""
    parser = ArgumentParser()
    parser.add_argument(
        "json_path", help="The path to the json description of the model"
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
    )
    args = parser.parse_args(args)

    load_config(args[0])


if __name__ == "__main__":
    main()
