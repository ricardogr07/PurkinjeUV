"""Unit tests for projection methods in Mesh class."""

import numpy as np


def test_project_point_inside_triangle(simple_triangle_mesh):
    mesh = simple_triangle_mesh
    point = np.array([0.3, 0.3, 1.0])  # Point above the triangle

    projected_point, tri_idx, r, t = mesh.project_new_point(point)

    # Should be projected inside the triangle
    assert tri_idx == 0
    np.testing.assert_allclose(projected_point[2], 0.0, atol=1e-8)
    assert 0.0 <= r <= 1.0
    assert 0.0 <= t <= 1.0
    assert r + t <= 1.001


def test_project_point_outside_triangle(simple_triangle_mesh):
    mesh = simple_triangle_mesh
    point = np.array([2.0, 2.0, 1.0])  # Far outside the triangle

    projected_point, tri_idx, r, t = mesh.project_new_point(point)

    # Should not be projected into any triangle
    assert tri_idx == -1
