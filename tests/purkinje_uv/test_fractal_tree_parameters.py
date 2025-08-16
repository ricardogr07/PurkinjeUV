from dataclasses import FrozenInstanceError, replace
import logging
import math
from pathlib import Path

import pytest

from purkinje_uv.fractal_tree_parameters import FractalTreeParameters


def test_parameters_defaults():
    params = FractalTreeParameters()

    assert params.meshfile is None
    assert params.init_node_id == 0
    assert params.second_node_id == 1
    assert params.init_length == 0.1
    assert params.N_it == 10
    assert params.length == 0.1
    assert params.branch_angle == 0.15
    assert params.w == 0.1
    assert params.l_segment == 0.01

    assert isinstance(params.fascicles_angles, list)
    assert isinstance(params.fascicles_length, list)
    assert len(params.fascicles_angles) == 0
    assert len(params.fascicles_length) == 0


def test_parameters_frozen_and_list_mutability():
    params = FractalTreeParameters()

    # Frozen: attribute rebinding should fail
    with pytest.raises(FrozenInstanceError):
        params.meshfile = "sample.obj"
    with pytest.raises(FrozenInstanceError):
        params.init_node_id = 5
    with pytest.raises(FrozenInstanceError):
        params.fascicles_angles = [0.1, 0.2]  # rebinding not allowed
    with pytest.raises(FrozenInstanceError):
        params.fascicles_length = [0.03, 0.04]  # rebinding not allowed

    # List contents remain mutable by design
    params.fascicles_angles.append(0.1)
    params.fascicles_angles.extend([0.2, -0.4])
    params.fascicles_length.extend([0.03, 0.04, 0.05])

    assert params.fascicles_angles == [0.1, 0.2, -0.4]
    assert params.fascicles_length == [0.03, 0.04, 0.05]


def test_parameters_replace_construction():
    base = FractalTreeParameters()
    lseg = 0.01
    out = replace(
        base,
        meshfile="mesh.obj",
        init_node_id=738,
        second_node_id=210,
        l_segment=lseg,
        init_length=0.3,
        length=0.15,
        fascicles_length=[20 * lseg, 40 * lseg],
        fascicles_angles=[-0.4, 0.5],
    )
    assert out.meshfile == "mesh.obj"
    assert out.init_node_id == 738
    assert out.second_node_id == 210
    assert out.l_segment == pytest.approx(0.01)
    assert out.init_length == pytest.approx(0.3)
    assert out.length == pytest.approx(0.15)
    assert out.fascicles_length == [0.2, 0.4]
    assert out.fascicles_angles == [-0.4, 0.5]


# --- Validation error checks (from __post_init__) ---


def test_parameters_invalid_equal_node_ids():
    with pytest.raises(ValueError):
        FractalTreeParameters(init_node_id=5, second_node_id=5)


def test_parameters_invalid_negative_ids():
    with pytest.raises(TypeError):
        FractalTreeParameters(init_node_id=-1)
    with pytest.raises(TypeError):
        FractalTreeParameters(second_node_id=-2)


def test_parameters_invalid_lengths_and_angle():
    with pytest.raises(ValueError):
        FractalTreeParameters(init_length=0.0)  # must be > 0
    with pytest.raises(ValueError):
        FractalTreeParameters(length=-0.1)  # must be > 0
    with pytest.raises(ValueError):
        FractalTreeParameters(w=-0.01)  # must be >= 0
    with pytest.raises(ValueError):
        FractalTreeParameters(l_segment=0.2, init_length=0.1, length=0.15)
    with pytest.raises(ValueError):
        FractalTreeParameters(branch_angle=0.0)
    with pytest.raises(ValueError):
        FractalTreeParameters(branch_angle=math.pi + 1.0)


def test_parameters_invalid_fascicles_mismatch_and_values():
    # Mismatch in list lengths
    with pytest.raises(ValueError):
        FractalTreeParameters(fascicles_angles=[0.1], fascicles_length=[])

    # Nonpositive fascicle length
    with pytest.raises(ValueError):
        FractalTreeParameters(fascicles_angles=[0.1], fascicles_length=[0.0])

    # Non-finite angle
    with pytest.raises(TypeError):
        FractalTreeParameters(fascicles_angles=[float("nan")], fascicles_length=[0.1])


def test_to_dict_and_json_roundtrip():
    lseg = 0.01
    p = FractalTreeParameters(
        meshfile="mesh.obj",
        init_node_id=2,
        second_node_id=3,
        init_length=0.25,
        N_it=8,
        length=0.12,
        branch_angle=0.2,
        w=0.05,
        l_segment=lseg,
        fascicles_angles=[-0.3, 0.6],
        fascicles_length=[20 * lseg, 40 * lseg],
    )

    # dict contains all public fields and is JSON-serializable
    d = p.to_dict()
    assert isinstance(d, dict)
    assert d["meshfile"] == "mesh.obj"
    assert d["fascicles_angles"] == [-0.3, 0.6]
    assert d["fascicles_length"] == [0.2, 0.4]

    # json -> object round-trip preserves equality
    s = p.to_json()
    p2 = FractalTreeParameters.from_json(s)
    assert p2 == p

    # Also test compact JSON (indent=None) parses back equivalently
    s_compact = p.to_json(indent=None)
    assert "\n" not in s_compact
    p3 = FractalTreeParameters.from_json(s_compact)
    assert p3 == p


def test_to_json_file_and_from_dict_ignore_unknown(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    p = FractalTreeParameters(init_node_id=10, second_node_id=11, meshfile="m.obj")

    # Write to file and read back via from_json
    path = tmp_path / "params.json"
    p.to_json_file(str(path))
    data = path.read_text(encoding="utf-8")
    p_loaded = FractalTreeParameters.from_json(data)
    assert p_loaded == p

    # Inject unknown key and use from_dict; should warn and ignore
    caplog.set_level(logging.WARNING, logger="purkinje_uv.fractal_tree_parameters")
    d = p.to_dict()
    d["unknown_key"] = 123  # should be ignored
    p_ignored = FractalTreeParameters.from_dict(d)
    assert p_ignored == p
    assert "Ignoring unknown parameter keys" in caplog.text


def test_logging_on_init(caplog: pytest.LogCaptureFixture):
    # Ensure INFO/DEBUG messages are captured from the module's logger
    caplog.set_level(logging.DEBUG, logger="purkinje_uv.fractal_tree_parameters")

    _ = FractalTreeParameters(
        meshfile="mesh.obj",
        init_node_id=1,
        second_node_id=2,
        fascicles_angles=[0.1],
        fascicles_length=[0.05],
    )

    # Summary at INFO
    assert "Initialized FractalTreeParameters" in caplog.text

    # Full config at DEBUG
    assert "FractalTreeParameters full config:" in caplog.text

    # A small sanity check that some numeric fields appear in the summary line
    assert "init_node_id=1" in caplog.text
    assert "second_node_id=2" in caplog.text
