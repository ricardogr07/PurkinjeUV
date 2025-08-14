from pathlib import Path
import pytest

from purkinje_uv import FractalTree, Parameters

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def test_generate_ellipsoid_vtu_saves_to_e2e_output():
    # Paths
    here = Path(__file__).resolve()
    e2e_dir = here.parent
    out_dir = e2e_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Where the mesh lives (repo_root/data/ellipsoid.obj)
    repo_root = e2e_dir.parent.parent
    meshfile = repo_root / "data" / "ellipsoid.obj"
    assert meshfile.exists(), f"Missing meshfile: {meshfile}"

    out_file = out_dir / "ellipsoid_purkinje_NEW.vtu"

    # Parameters (same as your previous snippet)
    params = Parameters()
    params.init_node_id = 738
    params.second_node_id = 210
    params.l_segment = 0.01
    params.init_length = 0.3
    params.length = 0.15
    params.meshfile = str(meshfile)
    params.fascicles_length = [20 * params.l_segment, 40 * params.l_segment]
    params.fascicles_angles = [-0.4, 0.5]  # radians

    # Generate & save
    tree = FractalTree(params)
    tree.grow_tree()
    tree.save(str(out_file))

    # Sanity: file exists and is non-empty
    assert out_file.exists(), f"Did not write: {out_file}"
    assert out_file.stat().st_size > 0

    # Optional: if meshio is installed, also verify it’s readable
    try:
        import meshio  # type: ignore
    except Exception:
        meshio = None

    if meshio is not None:
        m = meshio.read(str(out_file))
        assert m.points.shape[0] > 0
        # Ensure at least one cell block exists
        assert any(len(cb.data) > 0 for cb in m.cells)
