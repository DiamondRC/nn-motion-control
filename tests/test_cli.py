import subprocess
import sys

from nn_motion_control import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "nn_motion_control", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
