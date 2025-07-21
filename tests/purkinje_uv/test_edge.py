import numpy as np
from numpy.testing import assert_allclose

from purkinje_uv.edge import Edge

def test_edge_basic_direction():
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    edge = Edge(0, 1, nodes, parent=None, branch=None)
    
    assert edge.n1 == 0
    assert edge.n2 == 1
    assert edge.parent is None
    assert edge.branch is None
    assert_allclose(edge.dir, np.array([1.0, 0.0, 0.0]))

def test_edge_with_parent_branch():
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    edge = Edge(0, 1, nodes, parent=3, branch=2)
    
    assert edge.parent == 3
    assert edge.branch == 2
    assert_allclose(edge.dir, np.array([0.0, 1.0, 0.0]))

def test_edge_diagonal_direction():
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0]
    ])
    edge = Edge(0, 1, nodes, parent=0, branch=0)
    expected_dir = np.array([1.0, 1.0, 1.0]) / np.linalg.norm([1.0, 1.0, 1.0])
    assert_allclose(edge.dir, expected_dir)
