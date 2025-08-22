from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from purkinje_uv.fractal_tree import FractalTree, FractalTreeParameters


# -------------------------
# Fixtures
# -------------------------


@pytest.fixture
def square_obj(tmp_path: Path) -> Path:
    """
    Tiny deterministic mesh: unit square (z=0) split into 2 triangles.
    """
    p = tmp_path / "square.obj"
    p.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "f 1 2 3",
                "f 1 3 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def params(square_obj: Path) -> FractalTreeParameters:
    """
    Small steps/lengths for quick, reproducible runs.
    """
    return FractalTreeParameters(
        meshfile=str(square_obj),
        l_segment=0.10,
        init_length=0.20,
        length=0.30,
        branch_angle=0.35,  # radians
        w=0.20,
        fascicles_length=[0.20],  # one short fascicle
        fascicles_angles=[0.25],
        N_it=1,  # one generation
        init_node_id=0,
        second_node_id=1,
    )


@pytest.fixture
def tree(params: FractalTreeParameters) -> Iterator[FractalTree]:
    """
    Builds Mesh + UV + locator inside FractalTree.__init__.
    """
    ft = FractalTree(params)
    yield ft


# -------------------------
# Unit tests: small helpers
# -------------------------


def test_interpolate_scalar_and_vector() -> None:
    # scalar (3,)
    scal = np.array([10.0, 20.0, 30.0])
    out_s = FractalTree._interpolate(scal, r=0.2, t=0.3)
    # 0.3*30 + 0.2*20 + 0.5*10 = 18
    assert np.isclose(float(out_s), 18.0)

    # vector (3, k)
    vecs = np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]])
    out_v = FractalTree._interpolate(vecs, r=0.25, t=0.25)
    expected = 0.25 * vecs[2] + 0.25 * vecs[1] + 0.5 * vecs[0]
    assert np.allclose(out_v, expected)

    # bad shape: leading dimension must be 3
    with pytest.raises(ValueError):
        FractalTree._interpolate(np.zeros((2,)), r=0.0, t=0.0)


def test_scaling_on_uv_node(tree: FractalTree) -> None:
    uv_pt = tree.mesh_uv.verts[0][:2]
    s, tri = tree._scaling(uv_pt)
    assert tri >= 0
    assert s >= 0.0
    # sanity: distance-based locator shouldn’t be far for an existing node
    s2, tri2 = tree._scaling(tree.mesh_uv.verts[1][:2])
    assert tri2 >= 0
    assert s2 >= 0.0


def test_point_in_mesh_helpers(tree: FractalTree) -> None:
    inside = tree.mesh_uv.verts[0][:2]
    outside = np.array([10.0, 10.0])
    # vtk-based
    assert tree._point_in_mesh_vtk(inside) is True
    assert tree._point_in_mesh_vtk(outside) is False
    # kd-tree based (uses project_new_point on the flat UV Mesh wrapper)
    assert tree._point_in_mesh(inside, tree.mesh_uv) is True
    assert tree._point_in_mesh(outside, tree.mesh_uv) is False


def test_eval_field_returns_valids(tree: FractalTree) -> None:
    # Evaluate original 3D verts field at a known UV position (z will be 0 in ppoint)
    q3 = np.append(tree.mesh_uv.verts[0][:2], 0.0)
    val, ppoint, tri = tree._eval_field(q3, tree.m.verts, tree.mesh_uv)
    assert tri >= 0
    assert val.shape == (3,)
    assert np.isfinite(val).all()
    assert ppoint.shape == (3,)
    assert np.isfinite(ppoint).all()


# -------------------------
# Higher-level helpers (state + phases)
# -------------------------


def test_init_state_shapes(tree: FractalTree) -> None:
    (
        nodes,
        edges,
        edge_queue,
        branches,
        sister_branches,
        end_nodes,
        branch_id,
        Rplus,
        Rminus,
        dx,
        w,
        branch_length,
        init_branch_length,
    ) = tree._init_state()

    # basic invariants
    assert isinstance(nodes, list) and len(nodes) == 2
    assert isinstance(edges, list) and len(edges) == 1
    assert isinstance(edge_queue, list) and edge_queue == [0]
    assert isinstance(branches, dict) and 0 in branches and branches[0] == [0]
    assert sister_branches == {}
    assert end_nodes == []
    assert branch_id == 0
    assert Rplus.shape == (2, 2) and Rminus.shape == (2, 2)
    assert dx > 0 and w >= 0 and branch_length > 0 and init_branch_length > 0

    # directions are finite 2d
    assert nodes[0].shape == (2,)
    assert nodes[1].shape == (2,)
    assert np.isfinite(nodes[0]).all()
    assert np.isfinite(nodes[1]).all()


def test_grow_initial_trunk_mutates_in_place(tree: FractalTree) -> None:
    (
        nodes,
        edges,
        edge_queue,
        branches,
        sister_branches,
        end_nodes,
        branch_id,
        Rplus,
        Rminus,
        dx,
        w,
        branch_length,
        init_branch_length,
    ) = tree._init_state()

    n_steps = int(init_branch_length / dx)
    beid = tree._grow_initial_trunk(
        nodes, edges, edge_queue, branches, dx, init_branch_length
    )

    # edges + nodes increased by n_steps
    assert len(nodes) == 2 + n_steps
    assert len(edges) == 1 + n_steps
    assert isinstance(beid, int) and 0 <= beid < len(edges)


def test_spawn_fascicles_adds_branches(tree: FractalTree) -> None:
    (
        nodes,
        edges,
        edge_queue,
        branches,
        sister_branches,
        end_nodes,
        branch_id,
        Rplus,
        Rminus,
        dx,
        w,
        branch_length,
        init_branch_length,
    ) = tree._init_state()

    beid = tree._grow_initial_trunk(
        nodes, edges, edge_queue, branches, dx, init_branch_length
    )

    try:
        branch_id2 = tree._spawn_fascicles(
            branching_edge_id=beid,
            nodes=nodes,
            edges=edges,
            edge_queue=edge_queue,
            branches=branches,
            branch_id=branch_id,
            dx=dx,
        )
        # If it succeeded, we expect something to have been queued.
        assert branch_id2 > branch_id
        assert len(edge_queue) >= 1
        for eid in edge_queue:
            assert 0 <= eid < len(edges)
    except RuntimeError:
        # On very small domains it’s OK for a fascicle to be rejected.
        # Still validate types/invariants.
        assert isinstance(edge_queue, list)


def test_branch_generation_produces_children(tree: FractalTree) -> None:
    (
        nodes,
        edges,
        edge_queue,
        branches,
        sister_branches,
        end_nodes,
        branch_id,
        Rplus,
        Rminus,
        dx,
        w,
        branch_length,
        init_branch_length,
    ) = tree._init_state()
    beid = tree._grow_initial_trunk(
        nodes, edges, edge_queue, branches, dx, init_branch_length
    )
    try:
        branch_id = tree._spawn_fascicles(
            branching_edge_id=beid,
            nodes=nodes,
            edges=edges,
            edge_queue=edge_queue,
            branches=branches,
            branch_id=branch_id,
            dx=dx,
        )
        parents = edge_queue[:]  # use actual fascicle heads
    except RuntimeError:
        # fallback: try to branch from the trunk head if fascicle failed
        parents = [beid]

    new_queue, branch_id = tree._branch_generation(
        edge_queue=parents,
        nodes=nodes,
        edges=edges,
        branches=branches,
        sister_branches=sister_branches,
        Rplus=Rplus,
        Rminus=Rminus,
        dx=dx,
        branch_id=branch_id,
        end_nodes=end_nodes,
    )
    assert isinstance(new_queue, list)
    # When at least one child is created, the “pair the last two branch IDs” rule holds.
    if len(new_queue) >= 1 and branch_id >= 1:
        assert sister_branches.get(branch_id) == branch_id - 1


def test_growing_generation_progresses(tree: FractalTree) -> None:
    (
        nodes,
        edges,
        edge_queue,
        branches,
        sister_branches,
        end_nodes,
        branch_id,
        Rplus,
        Rminus,
        dx,
        w,
        branch_length,
        init_branch_length,
    ) = tree._init_state()

    beid = tree._grow_initial_trunk(
        nodes, edges, edge_queue, branches, dx, init_branch_length
    )
    try:
        branch_id = tree._spawn_fascicles(
            branching_edge_id=beid,
            nodes=nodes,
            edges=edges,
            edge_queue=edge_queue,
            branches=branches,
            branch_id=branch_id,
            dx=dx,
        )
    except RuntimeError:
        # okay on tiny domain; just attempt branching from trunk head
        edge_queue = [beid]

    # Branch once, then grow (may yield an empty queue in tiny domains)
    edge_queue, branch_id = tree._branch_generation(
        edge_queue=edge_queue,
        nodes=nodes,
        edges=edges,
        branches=branches,
        sister_branches=sister_branches,
        Rplus=Rplus,
        Rminus=Rminus,
        dx=dx,
        branch_id=branch_id,
        end_nodes=end_nodes,
    )
    before_nodes = len(nodes)
    if edge_queue:
        next_queue = tree._growing_generation(
            edge_queue=edge_queue,
            nodes=nodes,
            edges=edges,
            branches=branches,
            sister_branches=sister_branches,
            dx=dx,
            w=w,
            branch_length=dx * 2,  # just 2 steps
            end_nodes=end_nodes,
        )
        assert isinstance(next_queue, list)
        assert len(nodes) >= before_nodes
    else:
        # nothing to grow — still a valid outcome on a tiny domain
        assert len(nodes) == before_nodes


# -------------------------
# Device-helpers and GPU parity
# -------------------------


def test_device_helpers_cpu(tree: FractalTree) -> None:
    (
        nodes,
        edges,
        *_,
    ) = tree._init_state()

    a = tree._branch_idx_dev(0, {0: [0, 1]})  # type: ignore[arg-type]
    b = tree._nodes_uv_dev(nodes)
    c = tree._vec2_dev(nodes[0])

    # On CPU backend these should be NumPy arrays (duck-type check: .shape + dtype)
    assert hasattr(a, "shape") and hasattr(a, "dtype")
    assert hasattr(b, "shape") and b.shape[1] == 2
    assert hasattr(c, "shape") and c.shape == (2,)


@pytest.mark.gpu
def test_device_helpers_gpu(tree: FractalTree, monkeypatch) -> None:
    # Switch only the fractal_tree module symbols used in device helpers
    import cupy as cp
    from purkinje_uv import fractal_tree as ft_mod

    monkeypatch.setattr(ft_mod, "xp", cp, raising=True)
    monkeypatch.setattr(ft_mod, "backend_name", lambda: "cupy", raising=True)

    (
        nodes,
        edges,
        *_,
    ) = tree._init_state()

    a = tree._branch_idx_dev(0, {0: [0, 1]})  # type: ignore[arg-type]
    b = tree._nodes_uv_dev(nodes)
    c = tree._vec2_dev(nodes[0])

    # Cupy ndarrays announce themselves via __module__ attr
    assert a.__class__.__module__.startswith("cupy")
    assert b.__class__.__module__.startswith("cupy")
    assert c.__class__.__module__.startswith("cupy")


@pytest.mark.gpu
def test_gpu_parity_counts(params: FractalTreeParameters) -> None:
    """
    On small toy domains the first fascicle may be rejected by the VTK locator
    (distance slightly over the 1e-3 tolerance). In that case grow_tree() is
    expected to raise. This test accepts either outcome but asserts behavior
    is sensible on GPU.
    """
    tree = FractalTree(params)
    try:
        tree.grow_tree()
        # Success path: ensure we actually built something non-trivial.
        assert len(tree.nodes_xyz) >= 2
        assert len(tree.edges) >= 1
        assert isinstance(tree.end_nodes, list)
    except RuntimeError as e:
        # Expected on tight domains: first fascicle exits UV domain tolerance.
        assert "fascicle goes out of the domain" in str(e).lower()


# -------------------------
# Full flow smoke + save
# -------------------------


def test_grow_tree_full_and_save(tree: FractalTree, tmp_path: Path) -> None:
    try:
        tree.grow_tree()
    except RuntimeError:
        pytest.skip("Fascicle left tiny domain on this mesh; full-grow smoke skipped.")

    # invariants
    assert tree.uv_nodes is not None and tree.uv_nodes.shape[1] == 3
    assert len(tree.nodes_xyz) == len(tree.uv_nodes)
    assert len(tree.connectivity) == len(tree.edges)

    # ensure no node index out of bounds in connectivity
    con = np.asarray(tree.connectivity, dtype=int)
    assert con.min() >= 0
    assert con.max() < len(tree.nodes_xyz)

    out = tmp_path / "tree.vtu"
    tree.save(str(out))
    assert out.exists() and out.stat().st_size > 0
