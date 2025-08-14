"""
Manual side-by-side comparison of OLD vs NEW VTU using PyVista.
This can be run after e2e tests test_generate_ellipsoid_vtu.py was ran
Usage:
    python -m tests.e2e.visual_compare_vtu
Requires:
    pip install pyvista pyvistaqt or pip install -e ".[e2e]"
"""
from pathlib import Path
import numpy as np
import pyvista as pv

# Default locations inside tests/e2e/output/
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OLD_PATH = OUT / "ellipsoid_purkinje.vtu"
NEW_PATH = OUT / "ellipsoid_purkinje_NEW.vtu"


def pick_scalar(ds):
    preferred = ["Activation", "AT", "activation_time", "time_activation"]
    for name in preferred:
        if name in ds.array_names:
            return name
    if ds.point_data:
        return list(ds.point_data.keys())[0]
    if ds.cell_data:
        return list(ds.cell_data.keys())[0]
    return None


def arr_range(ds, name):
    if not name:
        return None
    a = np.asarray(ds[name])
    a = a[np.isfinite(a)]
    return (float(a.min()), float(a.max())) if a.size else None


def to_tubes(ds, radius=0.003):
    # Tube for line data (no-op for surfaces)
    try:
        return ds.tube(radius=radius)
    except Exception:
        return ds


def main(old_path=OLD_PATH, new_path=NEW_PATH):
    if not Path(old_path).exists():
        raise FileNotFoundError(f"Missing OLD VTU: {old_path}")
    if not Path(new_path).exists():
        raise FileNotFoundError(f"Missing NEW VTU: {new_path}")

    old = pv.read(str(old_path))
    new = pv.read(str(new_path))

    arr_old = pick_scalar(old)
    arr_new = arr_old if (arr_old and arr_old in new.array_names) else pick_scalar(new)

    r1 = arr_range(old, arr_old)
    r2 = arr_range(new, arr_new)
    clim = None
    if r1 and r2:
        clim = (min(r1[0], r2[0]), max(r1[1], r2[1]))

    old_vis = to_tubes(old, radius=0.003)
    new_vis = to_tubes(new, radius=0.003)

    p = pv.Plotter(shape=(1, 2), window_size=(1400, 650))

    # LEFT: OLD
    p.subplot(0, 0)
    p.add_text(f"OLD ({Path(old_path).name})", font_size=14)
    p.add_mesh(
        old_vis,
        scalars=arr_old if arr_old else None,
        clim=clim,
        show_scalar_bar=bool(arr_old),
        scalar_bar_args={"title": arr_old or ""},
    )
    p.show_bounds(all_edges=True, color="white", location="outer", grid=False)

    # RIGHT: NEW
    p.subplot(0, 1)
    p.add_text(f"NEW ({Path(new_path).name})", font_size=14)
    p.add_mesh(
        new_vis,
        scalars=arr_new if arr_new else None,
        clim=clim,
        show_scalar_bar=bool(arr_new),
        scalar_bar_args={"title": arr_new or ""},
    )
    p.show_bounds(all_edges=True, color="white", location="outer", grid=False)

    p.link_views()
    p.subplot(0, 0)
    p.camera_position = "xy"  # try "xy", "xz", or "yz"
    p.show(title="Purkinje VTU: OLD vs NEW (linked views)")


if __name__ == "__main__":
    main()
