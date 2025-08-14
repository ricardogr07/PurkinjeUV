"""Unit tests for Mesh initialization and basic geometry."""

import numpy as np


def test_mesh_initialization(simple_triangle_mesh):
    mesh = simple_triangle_mesh

    # Check vertex and triangle count
    assert mesh.verts.shape == (3, 3), "Expected 3 vertices"
    assert mesh.connectivity.shape == (1, 3), "Expected 1 triangle"

    # Check centroid
    expected_centroid = np.array([[1 / 3, 1 / 3, 0.0]])
    np.testing.assert_allclose(mesh.centroids, expected_centroid, atol=1e-8)

    # Check normal
    expected_normal = np.array([[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(mesh.normals, expected_normal, atol=1e-8)

    # Check node-to-triangle mapping
    assert len(mesh.node_to_tri) == 3
    for tris in mesh.node_to_tri.values():
        assert tris == [0]  # Each node should belong to triangle 0
