from __future__ import annotations
import pytest

import numpy as np
from purkinje_uv.mesh import Mesh


def _gpu_available() -> bool:
    try:
        import cupy as cp  # noqa: F401
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip tests marked 'gpu' when no CUDA device is present."""
    if _gpu_available():
        return
    skip_marker = pytest.mark.skip(reason="GPU not available for CuPy.")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture()
def puv_cpu():
    import purkinje_uv as puv

    with puv.use("cpu", seed=0, strict=False):
        yield puv


@pytest.fixture()
def puv_gpu():
    if not _gpu_available():
        pytest.skip("No CUDA device available for CuPy.")
    import purkinje_uv as puv

    with puv.use("gpu", seed=0, strict=True):
        yield puv


@pytest.fixture
def simple_triangle_mesh():
    """
    Provides a Mesh instance with a single triangle:
        v0 = [0, 0, 0]
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
    """
    verts = np.array(
        [
            [0.0, 0.0, 0.0],  # v0
            [1.0, 0.0, 0.0],  # v1
            [0.0, 1.0, 0.0],  # v2
        ]
    )
    connectivity = np.array([[0, 1, 2]])  # One triangle
    return Mesh(verts=verts, connectivity=connectivity)
