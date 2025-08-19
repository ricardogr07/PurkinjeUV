"""Unit tests for the Nodes class.

This test suite verifies the functionality of the Nodes class, including:
- Initialization with a single node
- Adding new nodes and updating KD-trees
- Computing distances from arbitrary points and between nodes
- Collision detection via excluded subsets
- Gradient estimation near existing nodes
"""

from purkinje_uv import Nodes
import numpy as np
from numpy.testing import assert_allclose
import pytest


def test_nodes_initialization():
    """
    Test that the Nodes class initializes correctly with a single node.

    This test ensures that:
    - The initial node is stored in the internal list.
    - The last_node index is set to 0.
    - The end_nodes list is empty.
    - The KD-tree is built and queryable with one node.
    """

    # Create a single node at a known position
    init_node = np.array([1.0, 2.0, 3.0], dtype=float)

    # Initialize the Nodes class
    nodes_obj = Nodes(init_node)

    # Check that the initial node was added
    assert len(nodes_obj.nodes) == 1
    assert np.allclose(nodes_obj.nodes[0], init_node)

    # Check that last_node is correctly set
    assert nodes_obj.last_node == 0

    # end_nodes should be empty upon initialization
    assert nodes_obj.end_nodes == []

    # The KD-tree should return distance 0 to the initial node
    query_distance = nodes_obj.distance_from_point(init_node)
    assert np.isclose(query_distance, 0.0)


def test_add_nodes():
    """
    Test that new nodes can be added to the Nodes object correctly.

    This test verifies that:
    - The initial node is preserved.
    - The new nodes are appended in order.
    - The returned indices are correct.
    - The KD-tree is updated and includes all nodes.
    """

    # Initial node at origin
    init_node = np.array([0.0, 0.0, 0.0])
    nodes_manager = Nodes(init_node)

    # Add a sequence of new nodes
    new_points = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    new_indices = nodes_manager.add_nodes(new_points)

    # Check that the returned indices are consecutive and correct
    assert new_indices == [1, 2, 3]

    # Check total number of nodes
    assert len(nodes_manager.nodes) == 4

    # Check that all new points were added in correct order
    np.testing.assert_array_equal(nodes_manager.nodes[1], new_points[0])
    np.testing.assert_array_equal(nodes_manager.nodes[2], new_points[1])
    np.testing.assert_array_equal(nodes_manager.nodes[3], new_points[2])

    # Check that the KD-tree contains all 4 points by querying a known point
    distance, index = nodes_manager.tree.query([0.0, 1.0, 0.0])
    assert np.isclose(distance, 0.0)
    assert index == 2


def test_distance_from_point():
    """
    Test that the distance from an arbitrary point to the nearest node is computed correctly.

    This test initializes a set of known node positions, adds them to the Nodes object,
    and then queries a point whose distance to a known node can be calculated manually.
    """

    # Initialize with a base node
    init_node = np.array([0.0, 0.0, 0.0])
    nodes_manager = Nodes(init_node)

    # Add a few additional nodes
    nodes_manager.add_nodes(
        [
            np.array([1.0, 0.0, 0.0]),  # index 1
            np.array([0.0, 2.0, 0.0]),  # index 2
            np.array([0.0, 0.0, 3.0]),  # index 3
        ]
    )

    # Query a point close to node index 2: (0.0, 1.5, 0.0)
    query_point = np.array([0.0, 1.5, 0.0])
    distance = nodes_manager.distance_from_point(query_point)

    # Closest node is at (0.0, 2.0, 0.0), so distance should be 0.5
    assert np.isclose(distance, 0.5)


def test_distance_from_node():
    """
    Test the distance from a node to its nearest neighbor in the tree.

    This test verifies that the method correctly returns the distance from a given node
    to its nearest *other* node. The returned value should match the Euclidean distance
    between the closest pair of nodes.
    """

    # Initialize with a single node
    init_node = np.array([0.0, 0.0, 0.0])
    nodes_manager = Nodes(init_node)

    # Add more nodes
    nodes_manager.add_nodes(
        [
            np.array([3.0, 0.0, 0.0]),  # index 1
            np.array([0.0, 4.0, 0.0]),  # index 2
            np.array([0.0, 0.0, 5.0]),  # index 3
        ]
    )

    # Now compute the distance from node 2 to its nearest neighbor
    # Node 2 is at (0, 4, 0), and closest is node 0 at (0, 0, 0) → distance = 4.0
    distance = nodes_manager.distance_from_node(2)

    assert np.isclose(distance, 4.0)


def test_update_collision_tree_basic():
    """
    Test updating the collision tree with a partial exclusion list.

    This verifies that excluded nodes are omitted and remaining nodes are indexed correctly.
    """
    init_node = np.array([0.0, 0.0, 0.0])
    nodes_manager = Nodes(init_node)

    # Add three more nodes (indices 1, 2, 3)
    nodes_manager.add_nodes(
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
    )

    # Exclude node 1 and 2 → keep 0 and 3
    nodes_manager.update_collision_tree(nodes_to_exclude=[1, 2])

    # Should keep nodes 0 and 3
    expected_keys = [0, 3]
    assert sorted(nodes_manager.nodes_to_consider_keys) == sorted(expected_keys)

    # Should match the expected points
    expected_points = [nodes_manager.nodes[i] for i in expected_keys]
    tree_points = nodes_manager.collision_tree.data
    for pt in expected_points:
        assert any(np.allclose(pt, tpt) for tpt in tree_points)


def test_update_collision_tree_all_excluded():
    """
    Test behavior when all nodes are excluded from the collision tree.

    The method should insert a dummy node and set the key to 100000000.
    """
    init_node = np.array([0.0, 0.0, 0.0])
    nodes_manager = Nodes(init_node)
    nodes_manager.add_nodes(
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
    )

    # Exclude all nodes (0, 1, 2)
    nodes_manager.update_collision_tree(nodes_to_exclude=[0, 1, 2])

    # Should only contain the dummy key
    assert nodes_manager.nodes_to_consider_keys == [100000000]

    # Should contain only one point and match the dummy location
    dummy_point = np.array([-1e11, -1e11, -1e11], dtype=float)
    assert nodes_manager.collision_tree.data.shape == (1, 3)
    assert np.allclose(nodes_manager.collision_tree.data[0], dummy_point)


def test_collision_nearest_node():
    """
    Test collision detection with some nodes excluded.

    Verifies that the function returns the index and distance
    to the nearest non-excluded node.
    """
    init_node = np.array([0.0, 0.0, 0.0])
    nodes_manager = Nodes(init_node)
    nodes_manager.add_nodes(
        [
            np.array([2.0, 0.0, 0.0]),  # index 1
            np.array([0.0, 2.0, 0.0]),  # index 2
        ]
    )

    # Exclude node 2 → only nodes 0 and 1 remain for collision
    nodes_manager.update_collision_tree(nodes_to_exclude=[2])

    # Query near node 1 at (2,0,0)
    query_point = np.array([2.1, 0.0, 0.0])
    idx, dist = nodes_manager.collision(query_point)

    assert idx == 1
    assert np.isclose(dist, 0.1)


def test_collision_with_only_dummy_node():
    """
    Test collision behavior when all nodes are excluded.

    Verifies that it returns the dummy index and correct distance to dummy.
    """
    init_node = np.array([0.0, 0.0, 0.0])
    nodes_manager = Nodes(init_node)
    nodes_manager.add_nodes(
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
    )

    # Exclude all nodes
    nodes_manager.update_collision_tree(nodes_to_exclude=[0, 1, 2])

    # Any query will be to dummy node at (-1e11, -1e11, -1e11)
    query_point = np.array([0.0, 0.0, 0.0])
    idx, dist = nodes_manager.collision(query_point)

    assert idx == 100000000
    expected_dist = np.linalg.norm(query_point - np.array([-1e11, -1e11, -1e11]))
    assert np.isclose(dist, expected_dist)


def test_gradient_zero_distance():
    """
    Test that the gradient at the location of an existing node is zero.

    When the query point coincides exactly with a node, the gradient should be (0,0,0).
    """
    init_node = np.array([0.0, 0.0, 0.0])
    nodes_manager = Nodes(init_node)

    grad = nodes_manager.gradient([0.0, 0.0, 0.0])
    expected = np.array([0.0, 0.0, 0.0])
    assert_allclose(grad, expected)


def test_gradient_direction():
    """
    Test that the gradient points away from the closest node.

    The gradient should be a unit vector in the direction from the closest node
    to the query point.
    """
    init_node = np.array([0.0, 0.0, 0.0])
    nodes_manager = Nodes(init_node)
    nodes_manager.add_nodes(
        [
            np.array([1.0, 0.0, 0.0]),  # index 1
            np.array([0.0, 1.0, 0.0]),  # index 2
        ]
    )

    query_point = np.array([0.5, 0.5, 0.0])  # Should be closest to (0,0,0)
    grad = nodes_manager.gradient(query_point)

    expected = query_point / np.linalg.norm(query_point)
    assert_allclose(grad, expected)


@pytest.mark.gpu
def test_nodes_accepts_cupy_inputs_and_returns_numpy_state() -> None:
    """Nodes should accept CuPy arrays for init/add and operate normally."""
    import cupy as cp
    import purkinje_uv as puv

    with puv.use("gpu", seed=0, strict=True):
        init_cp = cp.asarray([0.0, 0.0, 0.0])
        nm = Nodes(init_cp)

        added = nm.add_nodes([cp.asarray([1.0, 0.0, 0.0]), cp.asarray([0.0, 2.0, 0.0])])
        assert added == [1, 2]

        # Internal storage is NumPy by design
        assert all(isinstance(p, np.ndarray) for p in nm.nodes)

        # distance_from_point should accept CuPy input too (to_cpu conversions)
        d = nm.distance_from_point(cp.asarray([0.0, 1.5, 0.0]))
        assert np.isclose(d, 0.5)


@pytest.mark.gpu
def test_nodes_cpu_vs_gpu_distance_and_gradient_identical() -> None:
    """Distances and gradients match between CPU and GPU cases."""
    import cupy as cp
    import purkinje_uv as puv

    # Build a tiny deterministic cloud
    pts_np = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    q_np = np.array([0.25, 0.25, 0.0])

    with puv.use("cpu", seed=0):
        nm_cpu = Nodes(pts_np[0])
        nm_cpu.add_nodes(pts_np[1:])
        d_cpu = nm_cpu.distance_from_point(q_np)
        g_cpu = nm_cpu.gradient(q_np)

    with puv.use("gpu", seed=0, strict=True):
        nm_gpu = Nodes(cp.asarray(pts_np[0]))
        nm_gpu.add_nodes([cp.asarray(p) for p in pts_np[1:]])
        d_gpu = nm_gpu.distance_from_point(cp.asarray(q_np))
        g_gpu = nm_gpu.gradient(cp.asarray(q_np))

    np.testing.assert_allclose(d_gpu, d_cpu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g_gpu, g_cpu, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_update_collision_tree_all_excluded_gpu() -> None:
    """Excluding all nodes on GPU inserts dummy key and behaves as expected."""
    import cupy as cp
    import purkinje_uv as puv

    with puv.use("gpu", seed=0, strict=True):
        nm = Nodes(cp.asarray([0.0, 0.0, 0.0]))
        nm.add_nodes([cp.asarray([1.0, 0.0, 0.0]), cp.asarray([0.0, 1.0, 0.0])])

        ok = nm.update_collision_tree(nodes_to_exclude=[0, 1, 2])
        assert ok is True
        assert nm.nodes_to_consider_keys == [100000000]  # sentinel

        # Collision returns dummy index and the correct distance
        q = np.array([0.0, 0.0, 0.0])
        idx, dist = nm.collision(q)
        assert idx == 100000000
        expected = np.linalg.norm(q - np.array([-1e11, -1e11, -1e11]))
        assert np.isclose(dist, expected)


@pytest.mark.gpu
def test_collision_respects_exclusions_on_gpu() -> None:
    """Nearest index respects exclusions with GPU inputs."""
    import cupy as cp
    import purkinje_uv as puv

    with puv.use("gpu", seed=0, strict=True):
        nm = Nodes(cp.asarray([0.0, 0.0, 0.0]))  # index 0
        nm.add_nodes([cp.asarray([2.0, 0.0, 0.0]), cp.asarray([0.0, 2.0, 0.0])])  # 1,2
        nm.update_collision_tree(nodes_to_exclude=[2])  # keep 0 and 1

        idx, dist = nm.collision(cp.asarray([2.1, 0.0, 0.0]))
        assert idx == 1
        assert np.isclose(dist, 0.1)
