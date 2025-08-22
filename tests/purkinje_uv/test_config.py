from __future__ import annotations
import numpy as np
from purkinje_uv.config import (
    configure,
    use,
    to_cpu,
    to_device,
    norm,
    cdist,
    rng,
    seed,
    bool_env,
    int_env,
    backend_name,
    is_gpu,
)


def test_bool_env_and_int_env_roundtrip(monkeypatch):
    monkeypatch.setenv("TBOOL", "true")
    assert bool_env("TBOOL", False) is True
    monkeypatch.setenv("TBOOL", "0")
    assert bool_env("TBOOL", True) is False
    monkeypatch.setenv("TINT", "42")
    assert int_env("TINT", 0) == 42


def test_config_cpu_roundtrip():
    configure("cpu", seed=1234)
    a = to_device([1, 2, 3], dtype=float)
    b = to_cpu(a)
    assert isinstance(b, np.ndarray)
    assert np.allclose(b, [1, 2, 3])
    v = to_device([[3.0, 4.0]], dtype=float)
    n = norm(v, axis=1)
    assert np.allclose(to_cpu(n), [5.0])


def test_cdist_small():
    configure("cpu", seed=1234)
    A = to_device([[0.0, 0.0], [1.0, 0.0]])
    B = to_device([[0.0, 0.0]])
    D = cdist(A, B)
    assert np.allclose(to_cpu(D).ravel(), [0.0, 1.0])


def test_use_context_restores_backend():
    prev = backend_name()
    with use("cpu"):
        pass
    assert backend_name() == prev
    assert is_gpu() in (True, False)  # just touch the path


def test_seed_reproducible():
    configure("cpu")
    seed(1234)
    a = to_cpu(rng.random(4))
    seed(1234)
    b = to_cpu(rng.random(4))
    assert np.allclose(a, b)
