"""Unit tests for the Branch class.

This test suite verifies that branches grow correctly over a mesh,
including direction initialization, segment-wise projection, and
integration with the Nodes manager.

"""
import numpy as np
import pytest
from unittest.mock import Mock

from purkinje_uv.branch import Branch


def test_branch_growth_success_three_segments():
    """
    Test successful branch growth over 3 segments.

    This test checks that:
    - The branch grows fully when all projections succeed.
    - No collision halts the growth process.
    - The final direction vector is correctly rotated and adjusted using the gradient.
    - The queue, triangle list, and node list are all correctly populated.
    """
    # Create a mock mesh with uniform normals (flat surface)
    mock_mesh = Mock()
    mock_mesh.normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (10, 1))

    # Always return a valid triangle for projection
    mock_mesh.project_new_point.side_effect = lambda pt: (pt, 0, None)

    # Create a mock Nodes object with stubbed methods and attributes
    mock_nodes = Mock()
    mock_nodes.nodes = [np.array([0.0, 0.0, 0.0])] * 10  # Initial position
    mock_nodes.gradient.return_value = np.array([0.0, 1.0, 0.0])  # Gradient along +Y
    mock_nodes.collision.return_value = ("ok", 1.0)  # No collision
    mock_nodes.add_nodes.side_effect = lambda points: list(range(1, len(points) + 1))
    mock_nodes.update_collision_tree.return_value = None
    mock_nodes.end_nodes = []

    # Construct a Branch with Nsegments = 3, angle = 90°, w = 0.5
    branch = Branch(
        mesh=mock_mesh,
        init_node=0,
        init_dir=np.array([1.0, 0.0, 0.0]),  # Initial direction: +X
        init_tri=0,
        length=1.0,
        angle=np.pi / 2,  # Rotate 90° in-plane from init_dir
        w=0.5,  # Blend with gradient
        nodes=mock_nodes,
        brother_nodes=[],
        Nsegments=3,
    )

    # Structural checks
    assert branch.growing is True
    assert len(branch.nodes) == 3  # Initial + 2 new segments
    assert len(branch.queue) == 3
    assert branch.tri == 0  # All projected to same triangle (mocked)

    # Compute expected direction:
    # Step 1: rotate [1,0,0] by 90° toward inplane direction [-Y]
    # Step 2: add gradient [0,1,0], scale by w, and normalize
    init_dir = np.array([1.0, 0.0, 0.0])
    init_normal = np.array([0.0, 0.0, 1.0])
    inplane = -np.cross(init_dir, init_normal)  # [0, -1, 0]
    rotated_dir = (
        np.cos(np.pi / 2) * init_dir + np.sin(np.pi / 2) * inplane
    )  # [0, -1, 0]
    rotated_dir /= np.linalg.norm(rotated_dir)

    grad = np.array([0.0, 1.0, 0.0])
    expected_dir = rotated_dir + 0.5 * grad
    expected_dir /= np.linalg.norm(expected_dir)

    # Numerical check: the direction should match expected
    np.testing.assert_allclose(branch.dir, expected_dir, rtol=1e-5, atol=1e-8)


def test_branch_projection_failure_stops_growth():
    """
    Test that a branch stops growing if projection fails on a segment.

    This test simulates:
    - A successful projection on the first segment (i = 1)
    - A failed projection on the second segment (i = 2), which returns triangle = -1

    It verifies that:
    - The branch stops growing (`growing = False`)
    - Only valid projections are kept in `queue` and `triangles`
    - The correct node is marked as the end node
    - `add_nodes` only receives successfully added points
    """
    # Mock mesh with flat normals
    mock_mesh = Mock()
    mock_mesh.normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (10, 1))

    # Custom projection: succeed once, then fail
    def project_point(pt):
        if pt[0] < 0.5:
            return pt, 0, None  # Success
        else:
            return pt, -1, None  # Failure (invalid triangle)

    mock_mesh.project_new_point.side_effect = project_point

    # Mock Nodes manager
    mock_nodes = Mock()
    mock_nodes.nodes = [np.array([0.0, 0.0, 0.0])] * 10  # Initial position
    mock_nodes.gradient.return_value = np.array([0.0, 1.0, 0.0])
    mock_nodes.collision.return_value = ("ok", 1.0)  # Always safe
    mock_nodes.add_nodes.side_effect = lambda points: list(range(1, len(points) + 1))
    mock_nodes.update_collision_tree.return_value = None
    mock_nodes.end_nodes = []

    # Create a Branch with Nsegments = 3
    branch = Branch(
        mesh=mock_mesh,
        init_node=0,
        init_dir=np.array([1.0, 0.0, 0.0]),
        init_tri=0,
        length=1.0,
        angle=0.0,
        w=0.5,
        nodes=mock_nodes,
        brother_nodes=[],
        Nsegments=3,
    )

    # Structural expectations:
    # - Growth stops at i=2 (due to failed projection)
    # - One new node added (from step i=1)
    # - queue includes [init, projected]
    # - nodes = [init_node, new_node]
    assert branch.growing is False
    assert len(branch.queue) == 2
    assert branch.nodes == [0, 1]
    assert branch.triangles == [0, 0]
    assert mock_nodes.end_nodes == [1]  # Last valid node becomes endpoint

    # Optional: verify only one call to add_nodes, with 1 point
    assert mock_nodes.add_nodes.call_count == 1
    args, _ = mock_nodes.add_nodes.call_args
    assert len(args[0]) == 1


def test_branch_stops_on_collision():
    """
    Test that a branch stops when a collision is detected.

    This test simulates:
    - A successful first projection (i = 1) with a safe collision distance
    - A second projection (i = 2) that also succeeds, but is too close to existing nodes

    It verifies that:
    - The second projected point is removed (popped) from the queue and triangles
    - The branch stops growing (`growing = False`)
    - Only the valid node is added to the node list
    - The last valid node is marked as an end node
    """
    # Mock mesh with flat normals
    mock_mesh = Mock()
    mock_mesh.normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (10, 1))
    mock_mesh.project_new_point.side_effect = lambda pt: (pt, 0, None)

    # Mock Nodes with collision on second step
    mock_nodes = Mock()
    mock_nodes.nodes = [np.array([0.0, 0.0, 0.0])] * 10
    mock_nodes.gradient.return_value = np.array([0.0, 1.0, 0.0])

    # First projection: OK (distance = 1.0), second: too close (distance = 0.01)
    mock_nodes.collision.side_effect = [("ok", 1.0), ("collision", 0.01)]

    mock_nodes.add_nodes.side_effect = lambda points: list(range(1, len(points) + 1))
    mock_nodes.update_collision_tree.return_value = None
    mock_nodes.end_nodes = []

    # Create branch with 3 segments (will stop at 2nd)
    branch = Branch(
        mesh=mock_mesh,
        init_node=0,
        init_dir=np.array([1.0, 0.0, 0.0]),
        init_tri=0,
        length=1.0,
        angle=0.0,
        w=0.5,
        nodes=mock_nodes,
        brother_nodes=[],
        Nsegments=3,
    )

    # Expect early stop due to collision after 1 projected point
    assert branch.growing is False
    assert len(branch.queue) == 2  # second point was popped
    assert len(branch.triangles) == 2
    assert branch.nodes == [0, 1]
    assert mock_nodes.end_nodes == [1]

    # Also check collision call sequence
    assert mock_nodes.collision.call_count == 2


def test_branch_direction_math():
    """
    Test the mathematical correctness of the computed branch direction.

    This test sets up:
    - An initial direction vector along +X ([1, 0, 0])
    - A surface normal along +Z ([0, 0, 1]), so the in-plane direction is [-Y]
    - A 90° rotation (π/2), which turns the initial direction to -Y
    - A gradient along +Y, which is blended into the rotated direction with w = 1.0

    It verifies that:
    - The in-plane rotation is computed correctly using a cross product
    - The final direction vector is normalized and points between -Y and +Y
    - The result matches the expected direction within floating-point tolerance
    """
    # Mock mesh with flat normals and successful projection
    mock_mesh = Mock()
    mock_mesh.normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (10, 1))
    mock_mesh.project_new_point.side_effect = lambda pt: (pt, 0, None)

    # Mock Nodes with constant gradient
    mock_nodes = Mock()
    mock_nodes.nodes = [np.array([0.0, 0.0, 0.0])] * 10
    mock_nodes.gradient.return_value = np.array([0.0, 1.0, 0.0])  # +Y
    mock_nodes.collision.return_value = ("ok", 1.0)
    mock_nodes.add_nodes.side_effect = lambda pts: list(range(1, len(pts) + 1))
    mock_nodes.update_collision_tree.return_value = None
    mock_nodes.end_nodes = []

    # Create a branch with rotation and gradient blending
    branch = Branch(
        mesh=mock_mesh,
        init_node=0,
        init_dir=np.array([1.0, 0.0, 0.0]),  # +X
        init_tri=0,
        length=1.0,
        angle=np.pi / 2,  # rotate 90° counter-clockwise
        w=1.0,  # equal weight with gradient
        nodes=mock_nodes,
        brother_nodes=[],
        Nsegments=2,
    )

    # Manually compute expected direction
    init_dir = np.array([1.0, 0.0, 0.0])
    init_normal = np.array([0.0, 0.0, 1.0])
    inplane = -np.cross(init_dir, init_normal)  # [0, -1, 0]

    rotated_dir = (
        np.cos(np.pi / 2) * init_dir + np.sin(np.pi / 2) * inplane
    )  # [0, -1, 0]
    rotated_dir /= np.linalg.norm(rotated_dir)

    grad = np.array([0.0, 1.0, 0.0])
    expected_dir = rotated_dir + grad  # should be [0, 0, 0]
    expected_dir /= np.linalg.norm(expected_dir)

    # Final direction must match computed one
    np.testing.assert_allclose(branch.dir, expected_dir, rtol=1e-5, atol=1e-8)


def test_add_node_to_queue_success():
    """
    Test that `add_node_to_queue()` successfully adds a new point to the queue.

    This test sets up:
    - A valid projection from the mesh (`triangle >= 0`)
    - An initialized Branch with one point in the queue
    - A known new direction for projection

    It verifies that:
    - The returned value is `True`
    - The projected point is added to `queue`
    - The associated triangle is added to `triangles`
    """

    # Mock mesh with normals and valid projection
    mock_mesh = Mock()
    mock_mesh.normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (10, 1))
    projected_point = np.array([1.0, 1.0, 1.0])
    mock_mesh.project_new_point.return_value = (projected_point, 3, None)

    # Minimal mock Nodes to construct a Branch
    mock_nodes = Mock()
    mock_nodes.nodes = [np.array([0.0, 0.0, 0.0])] * 10
    mock_nodes.gradient.return_value = np.array([0.0, 0.0, 0.0])
    mock_nodes.collision.return_value = ("ok", 1.0)
    mock_nodes.add_nodes.return_value = [1]
    mock_nodes.update_collision_tree.return_value = None
    mock_nodes.end_nodes = []

    # Create a branch with just 1 segment to isolate setup
    branch = Branch(
        mesh=mock_mesh,
        init_node=0,
        init_dir=np.array([1.0, 0.0, 0.0]),
        init_tri=0,
        length=1.0,
        angle=0.0,
        w=0.0,
        nodes=mock_nodes,
        brother_nodes=[],
        Nsegments=1,
    )

    # Add a new projected node
    success = branch.add_node_to_queue(
        mesh=mock_mesh, init_node=branch.queue[0], dir_vec=np.array([1.0, 1.0, 1.0])
    )

    # Expect success and one new entry in both queue and triangles
    assert success is True
    assert branch.queue[-1] is projected_point
    assert branch.triangles[-1] == 3


def test_add_node_to_queue_failure():
    """
    Test that `add_node_to_queue()` handles failed projections correctly.

    This test simulates:
    - A projection that fails by returning `triangle = -1`
    - An initialized Branch with one point in the queue

    It verifies that:
    - The method returns `False`
    - No new point is added to the `queue`
    - No new triangle is added to `triangles`
    """

    # Mock mesh with invalid projection
    mock_mesh = Mock()
    mock_mesh.normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (10, 1))
    mock_mesh.project_new_point.return_value = (np.array([0.0, 0.0, 0.0]), -1, None)

    # Minimal mock Nodes to construct the branch
    mock_nodes = Mock()
    mock_nodes.nodes = [np.array([0.0, 0.0, 0.0])] * 10
    mock_nodes.gradient.return_value = np.array([0.0, 0.0, 0.0])
    mock_nodes.collision.return_value = ("ok", 1.0)
    mock_nodes.add_nodes.return_value = [1]
    mock_nodes.update_collision_tree.return_value = None
    mock_nodes.end_nodes = []

    # Create a minimal branch with 1 segment
    branch = Branch(
        mesh=mock_mesh,
        init_node=0,
        init_dir=np.array([1.0, 0.0, 0.0]),
        init_tri=0,
        length=1.0,
        angle=0.0,
        w=0.0,
        nodes=mock_nodes,
        brother_nodes=[],
        Nsegments=1,
    )

    # Attempt to add a node, expecting failure
    success = branch.add_node_to_queue(
        mesh=mock_mesh, init_node=branch.queue[0], dir_vec=np.array([1.0, 1.0, 1.0])
    )

    # Expect nothing added and return value = False
    assert success is False
    assert len(branch.queue) == 1  # only the original point
    assert len(branch.triangles) == 1  # only the initial triangle


@pytest.mark.gpu
def test_branch_cpu_vs_gpu_identical_results(puv_cpu, puv_gpu) -> None:
    """CPU and GPU Branch growth should produce identical results within tolerance."""
    # Shared mocks (deterministic)
    mock_mesh = Mock()
    mock_mesh.normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (10, 1))
    mock_mesh.project_new_point.side_effect = lambda pt: (pt, 0, None)

    mock_nodes = Mock()
    mock_nodes.nodes = [np.array([0.0, 0.0, 0.0])] * 10
    mock_nodes.gradient.return_value = np.array([0.0, 1.0, 0.0])
    mock_nodes.collision.return_value = ("ok", 1.0)
    mock_nodes.add_nodes.side_effect = lambda pts: list(range(1, len(pts) + 1))
    mock_nodes.update_collision_tree.return_value = None
    mock_nodes.end_nodes = []

    kwargs = dict(
        mesh=mock_mesh,
        init_node=0,
        init_dir=np.array([1.0, 0.0, 0.0]),
        init_tri=0,
        length=1.0,
        angle=np.pi / 3,  # 60 degrees
        w=0.25,
        nodes=mock_nodes,
        brother_nodes=[],
        Nsegments=4,
    )

    # Run on CPU
    with puv_cpu.use("cpu", seed=0):
        b_cpu = Branch(**kwargs)

    # Run on GPU
    with puv_gpu.use("gpu", seed=0, strict=True):
        b_gpu = Branch(**kwargs)

    # Compare final direction and triangles
    np.testing.assert_allclose(b_cpu.dir, b_gpu.dir, rtol=1e-6, atol=1e-8)
    assert b_cpu.triangles == b_gpu.triangles
    assert b_cpu.growing == b_gpu.growing

    # Compare queue point-by-point (they should be NumPy arrays)
    assert len(b_cpu.queue) == len(b_gpu.queue)
    for qc, qg in zip(b_cpu.queue, b_gpu.queue):
        np.testing.assert_allclose(qc, qg, rtol=1e-6, atol=1e-8)


@pytest.mark.gpu
def test_branch_accepts_cupy_init_dir_and_returns_numpy_state(puv_gpu) -> None:
    """Branch should accept CuPy init_dir on GPU but store state as NumPy arrays."""
    import cupy as cp

    mock_mesh = Mock()
    mock_mesh.normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (10, 1))
    mock_mesh.project_new_point.side_effect = lambda pt: (pt, 0, None)

    mock_nodes = Mock()
    mock_nodes.nodes = [np.array([0.0, 0.0, 0.0])] * 10
    mock_nodes.gradient.return_value = np.array([0.0, 1.0, 0.0])
    mock_nodes.collision.return_value = ("ok", 1.0)
    mock_nodes.add_nodes.side_effect = lambda pts: list(range(1, len(pts) + 1))
    mock_nodes.update_collision_tree.return_value = None
    mock_nodes.end_nodes = []

    with puv_gpu.use("gpu", seed=0, strict=True):
        b = Branch(
            mesh=mock_mesh,
            init_node=0,
            init_dir=cp.asarray([1.0, 0.0, 0.0]),  # CuPy input
            init_tri=0,
            length=1.0,
            angle=0.0,
            w=0.0,
            nodes=mock_nodes,
            brother_nodes=[],
            Nsegments=2,
        )

    # The internal state should be NumPy on purpose.
    assert isinstance(b.dir, np.ndarray)
    assert all(isinstance(q, np.ndarray) for q in b.queue)
