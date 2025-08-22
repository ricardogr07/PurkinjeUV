import numpy as np
import pytest

from purkinje_uv import Mesh
import purkinje_uv as puv


@pytest.mark.gpu
def test_mesh_cpu_vs_gpu_normals_centroids_identical() -> None:
    """Normals & centroids must match across backends."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)

    with puv.use("cpu", seed=0, strict=True):
        m_cpu = Mesh(verts=verts, connectivity=conn)
        normals_cpu = m_cpu.normals.copy()
        cents_cpu = m_cpu.centroids.copy()

    with puv.use("gpu", seed=0, strict=True):
        m_gpu = Mesh(verts=verts, connectivity=conn)
        normals_gpu = m_gpu.normals.copy()
        cents_gpu = m_gpu.centroids.copy()

    np.testing.assert_allclose(normals_gpu, normals_cpu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cents_gpu, cents_cpu, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_mesh_bmatrix_gradient_identical() -> None:
    """Bmatrix and gradient must match across backends for one triangle."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    u = np.array([0.0, 1.0, 2.0], dtype=float)  # per-vertex scalar field on tri 0

    with puv.use("cpu", seed=0, strict=True):
        m_cpu = Mesh(verts=verts, connectivity=conn)
        B_cpu, J_cpu = m_cpu.Bmatrix(0)
        grad_cpu = m_cpu.gradient(0, u)

    with puv.use("gpu", seed=0, strict=True):
        m_gpu = Mesh(verts=verts, connectivity=conn)
        B_gpu, J_gpu = m_gpu.Bmatrix(0)
        grad_gpu = m_gpu.gradient(0, u)

    np.testing.assert_allclose(B_gpu, B_cpu, rtol=1e-12, atol=1e-12)
    assert pytest.approx(J_cpu, rel=1e-12, abs=1e-12) == J_gpu
    np.testing.assert_allclose(grad_gpu, grad_cpu, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_mesh_construction_and_kdtree_query_in_gpu_context() -> None:
    """Mesh should construct in GPU context and KD-tree queries should work."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    conn = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)

    with puv.use("gpu", seed=0, strict=True):
        m = Mesh(verts=verts, connectivity=conn)

        d, idx = m.tree.query([0.1, 0.1, 0.0])
        assert np.isfinite(d)
        assert isinstance(idx, (int, np.integer))

        proj, tri_idx, r, t = m.project_new_point(
            np.array([0.25, 0.25, 0.5]), verts_to_search=2
        )
        assert tri_idx in (0, 1)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= t <= 1.0
        assert (r + t) <= 1.001


@pytest.mark.gpu
def test_mesh_accepts_cupy_input_and_returns_numpy() -> None:
    """If CuPy is available, Mesh should accept CuPy verts/conn and store NumPy."""
    try:
        import cupy as cp  # type: ignore
    except Exception:
        pytest.skip("CuPy not available")

    verts_cp = cp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=cp.float64,
    )
    conn_cp = cp.asarray([[0, 1, 2], [0, 2, 3]], dtype=cp.int32)

    with puv.use("gpu", seed=0, strict=True):
        m = Mesh(verts=verts_cp, connectivity=conn_cp)

        assert isinstance(m.verts, np.ndarray)
        assert isinstance(m.connectivity, np.ndarray)
        assert isinstance(m.normals, np.ndarray)
        assert isinstance(m.centroids, np.ndarray)

        d, idx = m.tree.query([0.9, 0.9, 0.0])
        assert np.isfinite(d)
        assert isinstance(idx, (int, np.integer))
