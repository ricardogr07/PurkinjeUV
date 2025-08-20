import numpy as np
import pytest

from numpy.testing import assert_allclose

from purkinje_uv import Mesh
import purkinje_uv as puv


def _grid(nx: int, ny: int, z: float = 0.0):
    """Axis-aligned unit square grid [0,1]x[0,1] with (nx*ny) vertices, 2*(nx-1)*(ny-1) tris."""
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    verts = np.column_stack([xv.ravel(), yv.ravel(), np.full(xv.size, z, dtype=float)])

    def vid(i, j):  # row-major
        return i * nx + j

    tris = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            v0 = vid(i, j)
            v1 = vid(i, j + 1)
            v2 = vid(i + 1, j)
            v3 = vid(i + 1, j + 1)
            # split square into two triangles [v0, v1, v3] and [v0, v3, v2]
            tris.append([v0, v1, v3])
            tris.append([v0, v3, v2])

    conn = np.asarray(tris, dtype=int)
    return verts.astype(float), conn


def _nearest_node(verts: np.ndarray, target: np.ndarray) -> int:
    dif = verts - target[None, :]
    idx = int(np.argmin(np.linalg.norm(dif, axis=1)))
    return idx


@pytest.mark.gpu
@pytest.mark.parametrize("nx,ny", [(10, 10)])
def test_laplace_cpu_gpu_equivalence_on_grid(nx: int, ny: int) -> None:
    """Solve Laplace with boundary loop Dirichlet BCs; CPU vs GPU must match."""
    verts, conn = _grid(nx, ny)

    with puv.use("cpu", seed=0, strict=True):
        m_cpu = Mesh(verts=verts, connectivity=conn)
        around_nodes, bc_u, bc_v = m_cpu.uv_bc()
        u_cpu = m_cpu.computeLaplace(around_nodes[:-1], bc_u)
        v_cpu = m_cpu.computeLaplace(around_nodes[:-1], bc_v)

    with puv.use("gpu", seed=0, strict=True):
        m_gpu = Mesh(verts=verts, connectivity=conn)
        around_nodes_g, bc_u_g, bc_v_g = m_gpu.uv_bc()
        # ensure identical boundary loop for deterministic comparison
        assert around_nodes_g == around_nodes
        assert_allclose(bc_u_g, bc_u, rtol=1e-14, atol=1e-14)
        assert_allclose(bc_v_g, bc_v, rtol=1e-14, atol=1e-14)
        u_gpu = m_gpu.computeLaplace(around_nodes[:-1], bc_u)
        v_gpu = m_gpu.computeLaplace(around_nodes[:-1], bc_v)

    assert_allclose(u_gpu, u_cpu, rtol=1e-11, atol=1e-11)
    assert_allclose(v_gpu, v_cpu, rtol=1e-11, atol=1e-11)


@pytest.mark.gpu
@pytest.mark.parametrize("nx,ny", [(10, 10)])
def test_geodesic_cpu_gpu_equivalence_on_grid(nx: int, ny: int) -> None:
    """Heat method geodesics with a single interior source; CPU vs GPU must match."""
    verts, conn = _grid(nx, ny)
    src_idx = _nearest_node(verts, np.array([0.5, 0.5, 0.0]))

    with puv.use("cpu", seed=0, strict=True):
        m_cpu = Mesh(verts=verts, connectivity=conn)
        d_cpu, Xs_cpu = m_cpu.computeGeodesic(nodes=[src_idx], nodeVals=[0.0], dt=5.0)

    with puv.use("gpu", seed=0, strict=True):
        m_gpu = Mesh(verts=verts, connectivity=conn)
        d_gpu, Xs_gpu = m_gpu.computeGeodesic(nodes=[src_idx], nodeVals=[0.0], dt=5.0)

    assert_allclose(d_gpu, d_cpu, rtol=1e-10, atol=1e-10)
    assert_allclose(Xs_gpu, Xs_cpu, rtol=1e-10, atol=1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("nx,ny", [(10, 10)])
def test_uvmap_and_uvscaling_cpu_gpu_equivalence(nx: int, ny: int) -> None:
    """UV mapping (two Laplace solves) and UV scaling must match across backends."""
    verts, conn = _grid(nx, ny)

    with puv.use("cpu", seed=0, strict=True):
        m_cpu = Mesh(verts=verts, connectivity=conn)
        m_cpu.uvmap()
        m_cpu.compute_uvscaling()
        uv_cpu = m_cpu.uv.copy()
        uvscale_cpu = m_cpu.uvscaling.copy()

    with puv.use("gpu", seed=0, strict=True):
        m_gpu = Mesh(verts=verts, connectivity=conn)
        m_gpu.uvmap()
        m_gpu.compute_uvscaling()
        uv_gpu = m_gpu.uv.copy()
        uvscale_gpu = m_gpu.uvscaling.copy()

    assert uv_cpu.shape == uv_gpu.shape
    assert_allclose(uv_gpu, uv_cpu, rtol=1e-12, atol=1e-12)
    assert_allclose(uvscale_gpu, uvscale_cpu, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("nx,ny", [(12, 12)])
def test_end_to_end_mesh_gpu_pipeline_and_vtu_write(tmp_path, nx: int, ny: int) -> None:
    """End-to-end on a small grid in a GPU context:
    - KD-tree queries
    - uv_bc, uvmap, uvscaling
    - computeLaplace, computeGeodesic
    - tri2node_interpolation
    - writeVTU
    """
    verts, conn = _grid(nx, ny)

    with puv.use("gpu", seed=0, strict=True):
        m = Mesh(verts=verts, connectivity=conn)

        # KD-tree sanity
        d, idx = m.tree.query([0.33, 0.66, 0.0])
        assert np.isfinite(d)
        assert isinstance(idx, (int, np.integer))

        # Boundary + UV map + scaling
        around_nodes, bc_u, bc_v = m.uv_bc()
        m.uvmap()
        m.compute_uvscaling()
        assert m.uv is not None and m.uv.shape == (verts.shape[0], 2)
        assert m.uvscaling is not None and m.uvscaling.shape == (conn.shape[0],)

        # Laplace directly via boundary loop
        u = m.computeLaplace(around_nodes[:-1], bc_u)
        v = m.computeLaplace(around_nodes[:-1], bc_v)
        assert u.shape == (verts.shape[0],)
        assert v.shape == (verts.shape[0],)

        # Geodesic from center node
        src_idx = _nearest_node(verts, np.array([0.5, 0.5, 0.0]))
        dists, Xs = m.computeGeodesic(nodes=[src_idx], nodeVals=[0.0], dt=5.0)
        assert dists.shape == (verts.shape[0],)
        assert Xs.shape == (conn.shape[0], 3)

        # Interpolate a triangle field to nodes (use uvscale as a test field)
        node_vals = m.tri2node_interpolation(m.uvscaling)
        assert len(node_vals) == verts.shape[0]
        assert np.isfinite(node_vals).all()

        # Write VTU
        out = tmp_path / "grid_gpu_pipeline.vtu"
        m.writeVTU(
            str(out),
            point_data={"u": u, "v": v, "d": dists},
            cell_data={"uvscale": m.uvscaling},
        )
        assert out.exists() and out.stat().st_size > 0
