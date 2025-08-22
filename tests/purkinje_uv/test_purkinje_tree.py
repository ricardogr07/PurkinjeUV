from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest

from purkinje_uv.purkinje_tree import PurkinjeTree
from purkinje_uv.config import to_cpu, use
from vtkmodules.numpy_interface import dataset_adapter as dsa


# -----------------------------------------------------------------------------
# Fixtures: a tiny 1D line tree (4 nodes in a chain)
# -----------------------------------------------------------------------------
@pytest.fixture()
def simple_tree_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 4 points along x-axis, unit spacing (3 edges)
    xyz = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=float,
    )
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=int)
    pmj = np.array([0, 3], dtype=int)  # leaves
    return xyz, edges, pmj


@pytest.fixture()
def tree(simple_tree_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> PurkinjeTree:
    xyz, edges, pmj = simple_tree_data
    return PurkinjeTree(xyz, edges, pmj)


# -----------------------------------------------------------------------------
# Helpers for fake FIM solver
# -----------------------------------------------------------------------------
class _FakeFim:
    """Simple FIM stand-in that computes shortest-time along a line graph.

    Uses D tensor to infer isotropic conduction velocity (cv),
    then does Dijkstra from multiple sources with initial times.
    """

    def __init__(self, xyz: Any, elm: Any, D: Any, device: str = "cpu") -> None:
        # Convert any device arrays to CPU numpy
        self.xyz = np.asarray(to_cpu(xyz))
        self.elm = np.asarray(to_cpu(elm), dtype=int)
        Dcpu = np.asarray(to_cpu(D), dtype=float)
        # D has shape (E, dim, dim) with D[e] = cv * I
        self.cv = float(Dcpu[0, 0, 0]) if Dcpu.size else 1.0

    def comp_fim(self, x0: Any, x0_vals: Any) -> np.ndarray:
        x0 = np.asarray(to_cpu(x0), dtype=int).ravel()
        x0_vals = np.asarray(to_cpu(x0_vals), dtype=float).ravel()

        n = self.xyz.shape[0]
        act = np.full(n, np.inf, dtype=float)

        # edge weights = segment length / cv
        w = (
            np.linalg.norm(self.xyz[self.elm[:, 1]] - self.xyz[self.elm[:, 0]], axis=1)
            / self.cv
        )

        # Build adjacency
        adj = [[] for _ in range(n)]
        for (u, v), wt in zip(self.elm, w):
            adj[u].append((v, wt))
            adj[v].append((u, wt))

        # Multi-source Dijkstra
        import heapq

        heap: list[tuple[float, int]] = []
        for i, t0 in zip(x0, x0_vals):
            act[i] = min(act[i], float(t0))
            heapq.heappush(heap, (float(t0), int(i)))

        while heap:
            t, u = heapq.heappop(heap)
            if t > act[u]:
                continue
            for v, wt in adj[u]:
                nt = t + wt
                if nt < act[v]:
                    act[v] = nt
                    heapq.heappush(heap, (nt, v))
        return act


def _make_fake_solver(xyz: Any, elm: Any, D: Any, device: str = "cpu") -> _FakeFim:
    return _FakeFim(xyz, elm, D, device=device)


def _make_fake_solver_fail_on_cuda(
    xyz: Any, elm: Any, D: Any, device: str = "cpu"
) -> _FakeFim:
    if device != "cpu":
        raise RuntimeError("Simulated GPU failure in test")
    return _FakeFim(xyz, elm, D, device=device)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_init_sets_activation_inf_and_pmj(tree: PurkinjeTree) -> None:
    d = dsa.WrapDataObject(tree.vtk_tree)
    act = np.asarray(d.PointData["activation"])
    assert act.shape[0] == tree.xyz.shape[0]
    assert np.isinf(act).all()
    assert np.array_equal(np.asarray(tree.pmj), np.array([0, 3], dtype=int))


def test_extract_edges(tree: PurkinjeTree) -> None:
    out = tree.extract_edges()
    assert out.shape == tree.connectivity.shape
    assert np.array_equal(out, tree.connectivity)


def test_extract_pmj_methods(
    simple_tree_data: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    xyz, edges, pmj = simple_tree_data
    t = PurkinjeTree(xyz, edges, pmj)
    a = sorted(t.extract_pmj_counter())
    b = sorted(map(int, t.extract_pmj_np_bincount()))
    c = sorted(map(int, t.extract_pmj_np_unique()))
    assert a == [0, 3]
    assert b == [0, 3]
    assert c == [0, 3]


def test_save_vtu_and_meshio_and_pmj(tree: PurkinjeTree, tmp_path: Path) -> None:
    vtu = tmp_path / "tree.vtu"
    tree.save(str(vtu))
    assert vtu.exists() and vtu.stat().st_size > 0

    # meshio save
    vtu2 = tmp_path / "tree_meshio.vtu"
    point_data = {"id": np.arange(tree.xyz.shape[0], dtype=int)}
    nE = tree.connectivity.shape[0]
    cell_data = {"w": [np.ones(nE)]}
    tree.save_meshio(str(vtu2), point_data=point_data, cell_data=cell_data)
    tree.save_meshio(str(vtu2), point_data=point_data, cell_data=cell_data)
    assert vtu2.exists() and vtu2.stat().st_size > 0

    # pmj vtp
    vtp = tmp_path / "pmj.vtp"
    tree.save_pmjs(str(vtp))
    assert vtp.exists() and vtp.stat().st_size > 0


def test_activate_fim_cpu(monkeypatch: pytest.MonkeyPatch, tree: PurkinjeTree) -> None:
    # Patch the solver to our fake CPU implementation
    monkeypatch.setenv("PURKINJE_UV_GPU", "0")
    from purkinje_uv import purkinje_tree as _pt_mod

    monkeypatch.setattr(_pt_mod, "create_fim_solver", _make_fake_solver)

    x0 = np.array([0], dtype=int)
    x0_vals = np.array([0.0], dtype=float)

    act_pmj = tree.activate_fim(x0, x0_vals, return_only_pmj=True)
    # time to reach node 3 along 3 unit edges at cv=2.5 => 3 / 2.5 = 1.2
    assert act_pmj.shape == (2,)
    assert np.isclose(act_pmj[0], 0.0)  # source at node 0
    assert np.isclose(act_pmj[1], 3.0 / 2.5, rtol=1e-6, atol=1e-12)


@pytest.mark.gpu
def test_activate_fim_gpu_fallback(
    monkeypatch: pytest.MonkeyPatch, tree: PurkinjeTree
) -> None:
    """Exercise the GPU path + CPU fallback branch."""
    from purkinje_uv import purkinje_tree as _pt_mod

    monkeypatch.setattr(_pt_mod, "create_fim_solver", _make_fake_solver_fail_on_cuda)

    x0 = np.array([0], dtype=int)
    x0_vals = np.array([0.0], dtype=float)

    # Force GPU; our fake will raise on CUDA, triggering the internal CPU fallback
    with use("gpu"):
        act_pmj = tree.activate_fim(x0, x0_vals, return_only_pmj=True)

    assert np.isclose(act_pmj[0], 0.0)
    assert np.isclose(act_pmj[1], 3.0 / 2.5, rtol=1e-6, atol=1e-12)


@pytest.mark.gpu
def test_activate_fim_gpu_direct(
    monkeypatch: pytest.MonkeyPatch, tree: PurkinjeTree
) -> None:
    """Run happy-path on GPU (our fake accepts device='cuda' and converts arrays)."""
    from purkinje_uv import purkinje_tree as _pt_mod

    monkeypatch.setattr(_pt_mod, "create_fim_solver", _make_fake_solver)

    x0 = np.array([0], dtype=int)
    x0_vals = np.array([0.0], dtype=float)

    with use("gpu"):
        act_pmj = tree.activate_fim(x0, x0_vals, return_only_pmj=True)

    assert np.isclose(act_pmj[0], 0.0)
    assert np.isclose(act_pmj[1], 3.0 / 2.5, rtol=1e-6, atol=1e-12)
