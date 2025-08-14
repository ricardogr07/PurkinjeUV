# conftest.py
import pytest
import numpy as np
from purkinje_uv.mesh import Mesh


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
