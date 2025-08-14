import numpy as np
import pytest
from pathlib import Path

from purkinje_uv import FractalTree, Parameters, PurkinjeTree

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def test_end_to_end_activation_saves_vtu(tmp_path=None):
    """
    End-to-end sanity check:

    1) Build the fractal tree on the ellipsoid surface
    2) Run Purkinje activation (FIM)
    3) Save a VTU with activation times
    """
    # Paths
    here = Path(__file__).resolve()
    e2e_dir = here.parent
    out_dir = e2e_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mesh path: repo_root/data/ellipsoid.obj
    repo_root = e2e_dir.parent.parent
    meshfile = repo_root / "data" / "ellipsoid.obj"
    assert meshfile.exists(), f"Missing meshfile: {meshfile}"

    # Fractal tree parameters (legacy-equivalent)
    params = Parameters()
    params.init_node_id = 738
    params.second_node_id = 210
    params.l_segment = 0.01
    params.init_length = 0.3
    params.length = 0.15
    params.meshfile = str(meshfile)
    params.fascicles_length = [20 * params.l_segment, 40 * params.l_segment]
    params.fascicles_angles = [-0.4, 0.5]  # radians

    # Grow the tree (UV domain)
    tree = FractalTree(params)
    tree.grow_tree()

    # Build Purkinje tree and activate
    Ptree = PurkinjeTree(
        np.asarray(tree.nodes_xyz),
        np.asarray(tree.connectivity),
        np.asarray(tree.end_nodes),
    )
    act = Ptree.activate_fim([0], [0.0], return_only_pmj=False)
    pmj = Ptree.pmj

    # Basic sanity checks
    assert isinstance(act, np.ndarray)
    assert act.ndim == 1
    assert pmj is not None
    assert np.all((pmj >= 0) & (pmj < act.shape[0]))

    # Save VTU with activation times
    out_file = out_dir / "ellipsoid_purkinje_AT.vtu"
    Ptree.save(str(out_file))
    assert out_file.exists() and out_file.stat().st_size > 0

    # Optional: verify readable via meshio if available
    try:
        import meshio  # type: ignore
    except Exception:
        meshio = None

    if meshio is not None:
        m = meshio.read(str(out_file))
        # Ensure something was written
        assert m.points.shape[0] > 0
        assert any(len(cb.data) > 0 for cb in m.cells)
