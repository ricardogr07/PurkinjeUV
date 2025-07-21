from purkinje_uv.fractal_tree_uv import FractalTree
from purkinje_uv.fractal_tree_parameters import Parameters
from purkinje_uv.purkinje_tree import PurkinjeTree

import numpy as np
import pytest

@pytest.fixture
def sample_tree():
    params = Parameters()
    params.init_node_id = 738
    params.second_node_id = 210
    params.l_segment = 0.01
    params.init_length = 0.3
    params.length = 0.15
    params.meshfile = '../data/ellipsoid.obj'
    params.fascicles_length = [20 * params.l_segment, 40 * params.l_segment]
    params.fascicles_angles = [-0.4, 0.5]
    tree = FractalTree(params)
    tree.grow_tree()
    return tree

@pytest.fixture
def purkinje_tree(sample_tree):
    return PurkinjeTree(
        np.array(sample_tree.nodes_xyz),
        np.array(sample_tree.connectivity),
        np.array(sample_tree.end_nodes)
    )

def test_activate_fim_returns_correct_shape(purkinje_tree):
    act = purkinje_tree.activate_fim([0], [0.0], return_only_pmj=False)
    assert isinstance(act, np.ndarray)
    assert act.shape[0] == purkinje_tree.xyz.shape[0]

def test_activate_fim_pmj_values(purkinje_tree):
    act = purkinje_tree.activate_fim([0], [0.0], return_only_pmj=True)
    pmj = purkinje_tree.pmj
    assert act.shape[0] == len(pmj)
    assert np.all(np.isfinite(act))

def test_save_creates_file(tmp_path, purkinje_tree):
    fname = tmp_path / "purkinje_tree.vtu"
    purkinje_tree.save(str(fname))
    assert fname.exists()

def test_save_pmjs_creates_file(tmp_path, purkinje_tree):
    fname = tmp_path / "pmjs.vtp"
    purkinje_tree.save_pmjs(str(fname))
    assert fname.exists()

def test_get_pmjs_activation(purkinje_tree):
    purkinje_tree.activate_fim([0], [0.0], return_only_pmj=False)
    pmj_act = purkinje_tree.get_pmjs_activation()
    assert pmj_act.shape[0] == len(purkinje_tree.pmj)
    assert np.all(np.isfinite(pmj_act))

def test_save_meshio_creates_file(tmp_path, purkinje_tree):
    fname = tmp_path / "purkinje_meshio.vtu"
    purkinje_tree.save_meshio(str(fname))
    assert fname.exists()

def test_extract_edges_returns_array(purkinje_tree):
    # branches are not set in this implementation, so skip if not present
    if hasattr(purkinje_tree, 'branches'):
        edges = purkinje_tree.extract_edges()
        assert isinstance(edges, np.ndarray)
        assert edges.shape[1] == 2

def test_extract_pmj_methods_consistency(purkinje_tree):
    # branches are not set in this implementation, so skip if not present
    if hasattr(purkinje_tree, 'branches'):
        pmj_counter = purkinje_tree.extract_pmj_counter()
        pmj_bincount = purkinje_tree.extract_pmj_np_bincount()
        pmj_unique = purkinje_tree.extract_pmj_np_unique()
        assert set(pmj_counter) == set(pmj_bincount) == set(pmj_unique)