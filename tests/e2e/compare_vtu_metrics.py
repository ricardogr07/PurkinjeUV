"""
Compare OLD vs NEW Purkinje VTUs (no GUI).
Usage (from repo root):
    python -m tests.e2e.compare_vtu_metrics \
      --old tests/e2e/output/ellipsoid_purkinje.vtu \
      --new tests/e2e/output/ellipsoid_purkinje_NEW.vtu
Requires:
    meshio, numpy, scipy
Produces:
    tests/e2e/output/vtu_diff_report.txt
    tests/e2e/output/vtu_diff_report.json
"""
from __future__ import annotations
from pathlib import Path
import json
import argparse
import numpy as np
import meshio
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# ---------- helpers ----------
def read_points_edges(vtu_path: Path):
    m = meshio.read(str(vtu_path))
    pts = np.asarray(m.points, dtype=float)
    # Find line-like cells
    edges = None
    for cb in m.cells:
        if cb.type in ("line", "line3"):  # most common for Purkinje
            edges = np.asarray(cb.data, dtype=int)
            break
        if cb.type in ("polyline",):
            # explode polylines into 2-node segments
            polys = np.asarray(cb.data, dtype=object)
            segs = []
            for poly in polys:
                # 'poly' may be a 1D array of node IDs
                arr = np.asarray(poly, dtype=int).ravel()
                if arr.size >= 2:
                    segs.append(np.column_stack([arr[:-1], arr[1:]]))
            if segs:
                edges = np.vstack(segs)
                break
    if edges is None:
        raise ValueError(f"No line/polyline cells found in {vtu_path}")
    return pts, edges


def degree_and_components(n_verts: int, edges: np.ndarray):
    i = edges[:, 0]
    j = edges[:, 1]
    data = np.ones(len(edges), dtype=int)
    # Undirected adj
    A = coo_matrix(
        (np.r_[data, data], (np.r_[i, j], np.r_[j, i])), shape=(n_verts, n_verts)
    ).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    n_comp, labels = connected_components(A, directed=False)
    # cyclomatic number (per component sum is global: E - N + C)
    n_edges = len(edges)
    cyclomatic = int(n_edges - n_verts + n_comp)
    return deg, n_comp, cyclomatic


def length_stats(points: np.ndarray, edges: np.ndarray):
    p0 = points[edges[:, 0]]
    p1 = points[edges[:, 1]]
    seglen = np.linalg.norm(p1 - p0, axis=1)
    return {
        "count": int(seglen.size),
        "total": float(seglen.sum()),
        "min": float(seglen.min()),
        "max": float(seglen.max()),
        "mean": float(seglen.mean()),
        "median": float(np.median(seglen)),
        "p95": float(np.percentile(seglen, 95)),
    }


def bidirectional_nn_dist(A_pts: np.ndarray, B_pts: np.ndarray):
    """Symmetric nearest-neighbor distances between point clouds."""
    if A_pts.size == 0 or B_pts.size == 0:
        return {"A_to_B": None, "B_to_A": None, "sym_max": None}

    treeA = cKDTree(A_pts)
    treeB = cKDTree(B_pts)
    dA, _ = treeB.query(A_pts, k=1)
    dB, _ = treeA.query(B_pts, k=1)

    def stats(d):
        return {
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)),
            "max": float(d.max()),
        }

    outA = stats(dA)
    outB = stats(dB)
    sym_max = max(outA["max"], outB["max"])
    return {"A_to_B": outA, "B_to_A": outB, "sym_max": float(sym_max)}


def first_scalar(ds):
    # prefer typical activation-like names
    preferred = ["Activation", "AT", "activation_time", "time_activation"]
    names = list(ds.point_data.keys()) + list(ds.cell_data.keys())
    for nm in preferred:
        if nm in names:
            return nm
    return names[0] if names else None


def scalar_summary(mesh, name: str | None):
    if not name:
        return None
    # meshio stores arrays in point_data/cell_data dicts of name->array
    arr = None
    if name in mesh.point_data:
        arr = np.asarray(mesh.point_data[name])
    elif name in mesh.cell_data:
        # pick first block
        blocks = mesh.cell_data[name]
        if isinstance(blocks, list) and len(blocks):
            arr = np.asarray(blocks[0])
    if arr is None or arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


# ---------- main ----------
def main(old_path: Path, new_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    old_m = meshio.read(str(old_path))
    new_m = meshio.read(str(new_path))
    old_pts, old_edges = read_points_edges(old_path)
    new_pts, new_edges = read_points_edges(new_path)

    old_deg, old_comp, old_cyc = degree_and_components(old_pts.shape[0], old_edges)
    new_deg, new_comp, new_cyc = degree_and_components(new_pts.shape[0], new_edges)

    report = {
        "files": {"old": str(old_path), "new": str(new_path)},
        "counts": {
            "points": {"old": int(old_pts.shape[0]), "new": int(new_pts.shape[0])},
            "segments": {
                "old": int(old_edges.shape[0]),
                "new": int(new_edges.shape[0]),
            },
            "components": {"old": int(old_comp), "new": int(new_comp)},
            "leaves(deg=1)": {
                "old": int((old_deg == 1).sum()),
                "new": int((new_deg == 1).sum()),
            },
            "junctions(deg>=3)": {
                "old": int((old_deg >= 3).sum()),
                "new": int((new_deg >= 3).sum()),
            },
            "cyclomatic_number": {"old": int(old_cyc), "new": int(new_cyc)},
        },
        "segment_length_stats": {
            "old": length_stats(old_pts, old_edges),
            "new": length_stats(new_pts, new_edges),
        },
        "point_cloud_nn_distance": bidirectional_nn_dist(old_pts, new_pts),
        "scalars": {
            "old": {},
            "new": {},
        },
    }

    # one representative scalar if available
    s_old = first_scalar(old_m)
    s_new = (
        s_old
        if (
            s_old
            and (s_old in old_m.point_data or s_old in old_m.cell_data)
            and (s_old in new_m.point_data or s_old in new_m.cell_data)
        )
        else first_scalar(new_m)
    )

    report["scalars"]["old"][s_old or ""] = scalar_summary(old_m, s_old)
    report["scalars"]["new"][s_new or ""] = scalar_summary(new_m, s_new)

    # Write JSON
    json_path = out_dir / "vtu_diff_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Pretty text
    def line(label, a, b):
        return f"{label:<24} old={a:<12} new={b:<12} Δ={b-a:+}"

    txt = []
    txt.append(f"OLD: {old_path}")
    txt.append(f"NEW: {new_path}")
    txt.append("")
    C = report["counts"]
    txt.append(line("points", C["points"]["old"], C["points"]["new"]))
    txt.append(line("segments", C["segments"]["old"], C["segments"]["new"]))
    txt.append(line("components", C["components"]["old"], C["components"]["new"]))
    txt.append(
        line("leaves(deg=1)", C["leaves(deg=1)"]["old"], C["leaves(deg=1)"]["new"])
    )
    txt.append(
        line(
            "junctions(deg>=3)",
            C["junctions(deg>=3)"]["old"],
            C["junctions(deg>=3)"]["new"],
        )
    )
    txt.append(
        line(
            "cyclomatic_number",
            C["cyclomatic_number"]["old"],
            C["cyclomatic_number"]["new"],
        )
    )
    txt.append("")

    for label in ("old", "new"):
        L = report["segment_length_stats"][label]
        txt.append(
            f"[{label}] length stats: total={L['total']:.6f}  "
            f"min={L['min']:.6f}  max={L['max']:.6f}  "
            f"mean={L['mean']:.6f}  median={L['median']:.6f}  p95={L['p95']:.6f}"
        )
    txt.append("")

    D = report["point_cloud_nn_distance"]
    if D["A_to_B"] and D["B_to_A"]:
        txt.append(
            f"NN distance old->new: mean={D['A_to_B']['mean']:.6g}, "
            f"median={D['A_to_B']['median']:.6g}, p95={D['A_to_B']['p95']:.6g}, max={D['A_to_B']['max']:.6g}"
        )
        txt.append(
            f"NN distance new->old: mean={D['B_to_A']['mean']:.6g}, "
            f"median={D['B_to_A']['median']:.6g}, p95={D['B_to_A']['p95']:.6g}, max={D['B_to_A']['max']:.6g}"
        )
    txt.append("")

    for label, mesh in (("old", old_m), ("new", new_m)):
        sname = next(iter(report["scalars"][label].keys()), None)
        summ = report["scalars"][label][sname] if sname else None
        if sname and summ:
            txt.append(
                f"[{label}] scalar '{sname}': "
                f"min={summ['min']:.6g} max={summ['max']:.6g} "
                f"mean={summ['mean']:.6g} median={summ['median']:.6g} p95={summ['p95']:.6g}"
            )
    txt_path = out_dir / "vtu_diff_report.txt"
    txt_path.write_text("\n".join(txt), encoding="utf-8")

    print(f"Wrote:\n  {txt_path}\n  {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tests/e2e/output"))
    args = ap.parse_args()
    main(args.old, args.new, args.out)
