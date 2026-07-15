"""
RunConfiguration tests: per-axis channel expansion, I/O-size validation, paths.
"""

import pytest

from nn_motion_control.core.config import RunConfiguration


def test_expands_channels_to_labels_and_weights(config_factory):
    rc = RunConfiguration(config_factory())
    # 5 input channels x 3 axes = 15; 4 target channels x 3 axes = 12.
    assert len(rc.input_params) == 15
    assert rc.input_params[:5] == ["x_pos", "x_vel", "x_acc", "x_jer", "x_DAC_real"]
    assert len(rc.target_params) == 12
    assert next(iter(rc.target_params[0])) == "x_pos_nxt"  # next-step target label


def test_input_size_mismatch_raises(config_factory):
    # First layer declares 15 inputs; 4 channels x 3 axes = 12 breaks the contract.
    cfg = config_factory(inputs=["position", "velocity", "acceleration", "jerk"])
    with pytest.raises(ValueError, match="input size"):
        RunConfiguration(cfg)


def test_target_size_mismatch_raises(config_factory):
    # Last layer emits 12; 3 target channels x 3 axes = 9 breaks the contract.
    cfg = config_factory(
        targets={
            ch: {"predict": "next", "weight": 1}
            for ch in ("position", "velocity", "acceleration")
        }
    )
    with pytest.raises(ValueError, match="output size"):
        RunConfiguration(cfg)


def test_data_and_out_paths_resolved(config_factory, synth_h5):
    rc = RunConfiguration(config_factory())
    assert rc.datafile_dir == synth_h5["path"]  # absolute path passed through
    assert rc.m_save_dir.endswith("/TestModel.pth")
