import subprocess
import sys

from deltabot_nn_controller import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "deltabot_nn_controller", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
