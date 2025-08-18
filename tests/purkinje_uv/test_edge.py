""" Unit tests for the Edge class.

This test suite verifies that the Edge class correctly initializes
its attributes and computes a normalized direction vector between
two nodes in 3D space.
"""
from __future__ import annotations

import numpy as np
import pytest

from purkinje_uv.edge import Edge
import purkinje_uv as puv


def test_edge_direction_unit_vector():
    """
    Test that the direction vector computed by Edge is a unit vector.

    This test sets up:
    - Two nodes at known 3D positions
    - An Edge created between them
    - No parent or branch ID

    It verifies that:
    - The `dir` attribute has norm ≈ 1.0
    - The direction is computed as the difference between node coordinates
    - Other attributes (`n1`, `n2`, `parent`, `branch`) are stored correctly
    """

    # Define a simple 3D coordinate system: node 0 at origin, node 1 at (1, 1, 0)
    nodes = [
        np.array([0.0, 0.0, 0.0]),  # Node 0
        np.array([1.0, 1.0, 0.0]),  # Node 1
    ]

    # Create an edge from node 0 to node 1
    edge = Edge(n1=0, n2=1, nodes=nodes, parent=None, branch=None)

    # Compute norm of the direction vector
    norm = np.linalg.norm(edge.dir)

    # Assert that the direction vector has unit length (normalized)
    np.testing.assert_allclose(
        norm, 1.0, rtol=1e-7, err_msg="Direction is not a unit vector."
    )

    # Also verify that n1 and n2 are correctly assigned
    assert edge.n1 == 0
    assert edge.n2 == 1

    # Parent and branch should remain None
    assert edge.parent is None
    assert edge.branch is None


def test_edge_direction_expected_value():
    """
    Test that the direction vector is computed as the normalized difference of node positions.

    This test sets up:
    - Two 3D nodes: node 0 at (0, 0, 0), node 1 at (3, 4, 0)
    - These define a direction of (3, 4, 0) → norm = 5

    It verifies that:
    - The direction vector equals (3/5, 4/5, 0)
    - The computed vector matches expected values within floating-point tolerance
    """

    # Define two nodes with known offset and magnitude
    nodes = [
        np.array([0.0, 0.0, 0.0]),  # Node 0
        np.array([3.0, 4.0, 0.0]),  # Node 1 (distance = 5.0 units)
    ]

    # Expected normalized direction: (3/5, 4/5, 0)
    expected_dir = np.array([0.6, 0.8, 0.0])

    # Create the edge and extract direction
    edge = Edge(n1=0, n2=1, nodes=nodes, parent=None, branch=None)

    # Assert direction matches expected
    np.testing.assert_allclose(
        edge.dir,
        expected_dir,
        rtol=1e-7,
        err_msg="Direction vector does not match expected normalized result.",
    )


def test_edge_with_parent_and_branch():
    """
    Test that parent and branch IDs are stored and accessible.

    This test:
    - Creates two nodes with simple coordinates.
    - Instantiates an Edge with parent ID 5 and branch ID 2.
    - Verifies that the attributes are stored as given.
    """

    # Define two nodes with arbitrary positions
    nodes = [
        np.array([1.0, 1.0, 1.0]),  # Node 0
        np.array([2.0, 2.0, 2.0]),  # Node 1
    ]

    # Define parent and branch identifiers
    parent_id = 5
    branch_id = 2

    # Create the edge
    edge = Edge(n1=0, n2=1, nodes=nodes, parent=parent_id, branch=branch_id)

    # Verify correct storage
    assert edge.parent == parent_id, "Parent ID not stored correctly."
    assert edge.branch == branch_id, "Branch ID not stored correctly."


def test_edge_direction_zero_vector():
    """
    Test that creating an Edge with identical node coordinates raises an error.

    When both endpoints of an edge are at the same position, the computed direction
    vector has zero magnitude. This should raise a ZeroDivisionError or a ValueError,
    depending on how numpy handles normalization of a zero vector.
    """

    # Both nodes are at the same location
    nodes = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
    ]

    with pytest.raises(ValueError) as exc_info:
        Edge(n1=0, n2=1, nodes=nodes, parent=None, branch=None)

    # Check the error message matches expected
    assert "zero magnitude" in str(exc_info.value).lower()


def _gpu_available() -> bool:
    try:
        import cupy as cp  # noqa: F401
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def gpu_test(fn):
    return pytest.mark.gpu(
        pytest.mark.skipif(not _gpu_available(), reason="GPU not available for CuPy.")(
            fn
        )
    )


@gpu_test
def test_edge_cpu_vs_gpu_identical_direction() -> None:
    nodes = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    ]

    with puv.use("cpu", seed=0):
        edge_cpu = Edge(0, 1, nodes, parent=None, branch=None)
        dir_cpu = edge_cpu.dir.copy()

    with puv.use("gpu", seed=0, strict=True):
        edge_gpu = Edge(0, 1, nodes, parent=None, branch=None)
        dir_gpu = edge_gpu.dir.copy()

    np.testing.assert_allclose(dir_gpu, dir_cpu, rtol=1e-12, atol=1e-12)


@gpu_test
def test_edge_accepts_cupy_input_and_returns_numpy() -> None:
    import cupy as cp

    nodes_cp = [
        cp.asarray([0.0, 0.0, 0.0]),
        cp.asarray([2.0, 0.0, 0.0]),
    ]

    with puv.use("gpu", seed=0, strict=True):
        edge = Edge(0, 1, nodes_cp, parent=None, branch=None)

    assert isinstance(edge.dir, np.ndarray)
    np.testing.assert_allclose(edge.dir, np.array([1.0, 0.0, 0.0]), rtol=0, atol=0)
