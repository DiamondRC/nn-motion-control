import subprocess
import sys

from nn_motion_control import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "nn_motion_control", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__


def test_cli_resource_scores_example_controller():
    # The resource command needs no plant checkpoint or dataset, only
    # the config.
    cfg = "examples/deltabot/configs/controller_10k.json"
    cmd = [sys.executable, "-m", "nn_motion_control", "resource", cfg]
    out = subprocess.check_output(cmd).decode()
    assert "deltabot_controller_10k" in out
    assert "10000 Hz" in out
    assert "Fits" in out


def test_cli_track_is_registered():
    # A full track run needs a trained checkpoint + dataset (not
    # hermetic), assert the command is wired and documents its args
    # so the CLI surface stays covered.
    cmd = [sys.executable, "-m", "nn_motion_control", "track", "--help"]
    out = subprocess.check_output(cmd).decode()
    assert "tracking metrics" in out
    assert "--no-animate" in out
    assert "--samples" in out


def test_cli_champions_lists_registry():
    cmd = [sys.executable, "-m", "nn_motion_control", "champions"]
    out = subprocess.check_output(cmd).decode()
    assert "champions.json" in out
    assert "plant: deltabot_plant_tcn_rollout" in out
    assert "controller: deltabot_controller_10k_general" in out


def test_cli_missing_config_errors_when_not_a_tty():
    # With a piped stdin (no TTY) an omitted config errors
    # deterministically, no hang.
    cmd = [sys.executable, "-m", "nn_motion_control", "resource"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "not a terminal" in (result.stderr + result.stdout)


def test_config_picker_discovers_example_configs():
    from nn_motion_control.cli.select import find_configs, is_interactive

    names = {p.name for p in find_configs()}
    assert "plant_tcn_rollout.json" in names
    assert "controller_10k_general.json" in names
    assert isinstance(is_interactive(), bool)
