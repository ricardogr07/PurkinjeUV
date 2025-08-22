"""Module defining the Mesh class for 3D triangular surface meshes.

This module provides:
  - Loading from OBJ or direct arrays.
  - Computation of normals, centroids, connectivity.
  - KD-tree spatial queries.
  - FEM routines (B-matrix, stiffness, mass, force).

Designed for fractal tree growth and surface-based FEM analysis.
"""
from __future__ import annotations

import collections
import logging
from typing import Any, Optional, List, Dict, Tuple, Sequence, DefaultDict
from numpy.typing import NDArray

import numpy as np
import meshio
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve
from .config import xp, to_device, to_cpu, norm, backend_name

_LOGGER = logging.getLogger(__name__)


class Mesh:
    """Handle 3D triangular surface meshes.

    Supports geometry, topology, and finite-element operations.
    It computes normals, centroids, boundary edges; builds KD-trees;
    and provides FEM element routines, geodesic/Laplace solvers,
    UV mapping, and interpolation utilities.

    Args:
        filename (Optional[str]): Path to OBJ file to load mesh from.
        verts (Optional[NDArray[Any]]): Vertex coordinates (n_nodes×3).
        connectivity (Optional[NDArray[Any]]): Triangle indices (n_triangles×3).

    Attributes:
        verts (NDArray[Any]): Vertex array, shape (n_nodes, 3).
        connectivity (NDArray[Any]): Triangle indices, shape (n_triangles, 3).
        normals (NDArray[Any]): Triangle normals, shape (n_triangles, 3).
        node_to_tri (DefaultDict[int, List[int]]): Node→[triangle indices].
        tree (cKDTree): KD-tree over `verts` for nearest-node queries.
        centroids (NDArray[Any]): Triangle centroids, shape (n_triangles, 3).
        boundary_edges (Optional[List[Tuple[int,int]]]): Edges on mesh boundary.
        uv (Optional[NDArray[Any]]): UV coordinates per node, shape (n_nodes, 2).
        triareas (Optional[NDArray[Any]]): Triangle areas, shape (n_triangles,).
        uvscaling (Optional[NDArray[Any]]): UV scaling metric per triangle.
    """

    verts: NDArray[Any]
    connectivity: NDArray[Any]
    normals: NDArray[Any]
    node_to_tri: DefaultDict[int, List[int]]
    tree: cKDTree
    centroids: NDArray[Any]
    boundary_edges: Optional[List[Tuple[int, int]]]
    uv: Optional[NDArray[Any]]
    triareas: Optional[NDArray[Any]]
    uvscaling: Optional[NDArray[Any]]

    def __init__(
        self,
        filename: Optional[str] = None,
        verts: Optional[NDArray[Any]] = None,
        connectivity: Optional[NDArray[Any]] = None,
    ) -> None:
        """Initialize mesh from OBJ or provided arrays.

        If `filename` is given, loads verts and connectivity from OBJ.
        Otherwise, uses provided `verts` and `connectivity` arrays.

        Raises:
            ValueError: If neither `filename` nor both arrays are provided.
        """
        # Load mesh from file if filename is provided
        if filename is not None:
            verts, connectivity = self.loadOBJ(filename)

        # Store verts & connectivity as NumPy arrays
        self.verts = np.asarray(to_cpu(verts), dtype=float)
        self.connectivity = np.asarray(to_cpu(connectivity), dtype=int)

        _LOGGER.debug(
            "Mesh __init__: building geometry on backend=%s (storage on CPU)",
            backend_name(),
        )

        # ---- Geometry on active backend (vectorized) ---------------------------
        # Move to device for heavy math, then bring results back to CPU.
        vxp = xp.asarray(self.verts)  # (n_nodes, 3)
        txp = xp.asarray(self.connectivity)  # (n_tris, 3) indices

        a = vxp[txp[:, 0]]  # (n_tris, 3)
        b = vxp[txp[:, 1]]
        c = vxp[txp[:, 2]]

        u = b - a
        v = c - a

        n = xp.cross(u, v)  # raw normals
        nn = xp.linalg.norm(n, axis=1)  # (n_tris,)
        eps = xp.asarray(1e-12)
        safe = xp.where(nn > eps, nn, 1.0)
        n_unit = n / safe[:, None]

        centroids_xp = (a + b + c) / 3.0

        # Back to CPU for storage / SciPy
        normals_cpu: NDArray[Any] = to_cpu(n_unit)
        nn_cpu: NDArray[Any] = to_cpu(nn)
        deg_mask = nn_cpu <= 1e-12
        if np.any(deg_mask):
            # Zero-out degenerate triangle normals to avoid NaNs
            normals_cpu[deg_mask] = 0.0
            _LOGGER.warning(
                "Mesh __init__: %d degenerate triangle(s) with ~zero area; normals set to 0.",
                int(np.count_nonzero(deg_mask)),
            )

        self.normals = normals_cpu
        self.centroids = to_cpu(centroids_xp)

        # ---- Topology / search structures (CPU) --------------------------------
        self.node_to_tri = collections.defaultdict(list)
        for tri_idx in range(self.connectivity.shape[0]):
            tri = self.connectivity[tri_idx]
            # Map node -> list of triangles
            self.node_to_tri[tri[0]].append(tri_idx)
            self.node_to_tri[tri[1]].append(tri_idx)
            self.node_to_tri[tri[2]].append(tri_idx)

        # KD-tree stays CPU (SciPy)
        self.tree = cKDTree(self.verts)

        # ---- Optional attributes ------------------------------------------------
        self.boundary_edges = None
        self.uv = None
        self.triareas = None
        self.uvscaling = None

        _LOGGER.info(
            "Mesh initialized with %d vertices and %d triangles",
            self.verts.shape[0],
            self.connectivity.shape[0],
        )
        _LOGGER.debug(
            "Computed centroids and normals via %s; stored on CPU (NumPy).",
            backend_name(),
        )

    def _as_np(
        self, a: Any, *, dtype: Optional[type | np.dtype] = None
    ) -> NDArray[Any]:
        """Return `a` as a NumPy array on CPU, optionally cast to `dtype`.

        Args:
            a: Array-like input (NumPy/CuPy/sequence/scalar).
            dtype: Optional dtype to cast to (no copy if not needed).

        Returns:
            NumPy array view/copy on CPU with desired dtype.
        """
        arr = np.asarray(to_cpu(a))
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    def _dot(self, a: Any, b: Any) -> float:
        """Compute a backend dot-product and return a Python float.

        Uses the currently active array backend (`xp`) for the computation,
        then converts the 0-d result to a Python float.

        Args:
            a: First vector/array (NumPy/CuPy/compatible with `xp.dot`).
            b: Second vector/array (NumPy/CuPy/compatible with `xp.dot`).

        Returns:
            The scalar dot-product as a Python float.
        """
        return float(np.asarray(to_cpu(xp.dot(a, b))))

    def loadOBJ(self, filename: str) -> Tuple[NDArray[Any], NDArray[Any]]:
        """Read a Wavefront .obj mesh file and return (verts, connectivity).

        Args:
            filename (str): Path to the .obj file.

        Returns:
            Tuple[NDArray[Any], NDArray[Any]]:
                - verts: Array of shape (n_vertices, 3)
                - connectivity: Array of shape (n_triangles, 3)
        """
        numVerts: int = 0
        verts: list[list[float]] = []
        # norms is parsed but unused; kept here for completeness
        norms: list[list[float]] = []
        connectivity: list[list[int]] = []

        for line in open(filename, "r"):
            vals = line.split()
            if len(vals) > 0:
                if vals[0] == "v":
                    v = list(map(float, vals[1:4]))
                    verts.append(v)
                if vals[0] == "vn":
                    n = list(map(float, vals[1:4]))
                    norms.append(n)
                if vals[0] == "f":
                    con = []
                    for f in vals[1:]:
                        w = f.split("/")
                        #                      print w
                        # OBJ Files are 1-indexed so we must subtract 1 below
                        con.append(int(w[0]) - 1)
                        numVerts += 1
                    connectivity.append(con)
        _LOGGER.info(
            f"Loaded OBJ from {filename} with {len(verts)} vertices and {len(connectivity)} triangles"
        )

        verts_arr: NDArray[Any] = np.array(verts, dtype=float)
        connectivity_arr: NDArray[Any] = np.array(connectivity, dtype=int)
        return verts_arr, connectivity_arr

    def project_new_point(
        self,
        point: NDArray[Any],
        verts_to_search: int = 1,
    ) -> Tuple[NDArray[Any], int, float, float]:
        """Project a point onto the mesh and find its containing triangle.

        Args:
            point (NDArray[Any]): Coordinates to project.
            verts_to_search (int): Number of nearby vertices to search.

        Returns:
            Tuple[NDArray[Any], int, float, float]:
                Projected point, triangle index (−1 if outside), and barycentric coords (r, t).
        """
        if verts_to_search < 1:
            raise ValueError("verts_to_search must be >= 1")

        # Normalize to CPU/NumPy once; KD-tree is CPU-only.
        p = np.asarray(to_cpu(point), dtype=float)

        _LOGGER.debug(
            "project_new_point: input=%s (backend=%s), k=%d",
            p.tolist(),
            backend_name(),
            verts_to_search,
        )

        try:
            _dists, idxs = self.tree.query(p, k=verts_to_search)
        except Exception:
            _LOGGER.exception("KD-tree query failed for point %s", p.tolist())
            raise

        # cKDTree returns scalars for k=1; arrays otherwise—normalize both.
        idxs_arr = np.atleast_1d(idxs)

        # Evaluate candidates in the order returned by KD-tree (nearest first).
        last_result: Tuple[NDArray[Any], int, float, float] = (p, -1, -1.0, -1.0)

        for node_idx in idxs_arr:
            node_int: int = int(node_idx)
            proj_point, intri, r, t = self.project_point_check(p, node_int)
            _LOGGER.debug(
                "project_new_point: candidate node=%d -> tri=%d (r=%.6g, t=%.6g)",
                node_int,
                intri,
                r,
                t,
            )
            last_result = (proj_point, intri, r, t)
            if intri != -1:
                return last_result

        _LOGGER.debug(
            "project_new_point: no containing triangle among %d candidates; "
            "returning last projection tri=%d.",
            len(idxs_arr),
            last_result[1],
        )
        return last_result

    def project_point_check(
        self,
        point: NDArray[Any],
        node: int,
    ) -> Tuple[NDArray[Any], int, float, float]:
        """This function projects any point to the surface defined by the mesh.

        Args:
            point (array): coordinates of the point to project.
            node (int): index of the most close node to the point

        Returns:
             projected_point (array): the coordinates of the projected point that lies in the surface.
             intriangle (int): the index of the triangle where the projected point lies. If the point is outside surface, intriangle=-1.
        """
        EPS = 1e-12

        # Normalize to CPU/NumPy once (KD-tree & arrays live on CPU).
        p = np.asarray(to_cpu(point), dtype=float)

        # Quick validation of node index.
        if node < 0 or node >= self.verts.shape[0]:
            _LOGGER.error("project_point_check: node index out of range: %d", node)
            return p, -1, -1.0, -1.0

        # Candidate triangles attached to the seed vertex.
        triangles_list: List[int] = self.node_to_tri.get(node, [])
        if not triangles_list:
            _LOGGER.warning(
                "project_point_check: vertex %d has no adjacent triangles.", node
            )
            return p, -1, -1.0, -1.0

        d, node_idx = self.tree.query(p)
        _LOGGER.debug(
            "project_point_check: seed node=%d (KD-nearest node=%d, dist=%.6g)",
            node,
            int(node_idx),
            float(d),
        )
        _LOGGER.debug(
            "project_point_check: triangles at node %d -> %s",
            node,
            triangles_list,
        )

        # Vertex normal = average of connected triangle normals.
        vertex_normal = np.sum(self.normals[triangles_list, :], axis=0)
        nrm = float(np.linalg.norm(vertex_normal))
        if nrm < EPS or not np.isfinite(nrm):
            # Fall back to the first triangle normal to avoid degeneracy.
            fallback_tri = int(triangles_list[0])
            vertex_normal = self.normals[fallback_tri, :].copy()
            nrm = float(np.linalg.norm(vertex_normal))
            _LOGGER.debug(
                "project_point_check: degenerate avg normal at node %d; falling back to tri %d normal.",
                node,
                fallback_tri,
            )
        vertex_normal /= nrm

        # Project point onto the plane through the vertex with this normal.
        vec_to_vertex = p - self.verts[node]
        distance_along_normal = float(np.dot(vec_to_vertex, vertex_normal))
        pre_projected_point = p - vertex_normal * distance_along_normal

        # Distance from pre_projected_point to each candidate triangle plane.
        cpp_vals: List[float] = []
        for tri in triangles_list:
            val = float(
                np.dot(
                    pre_projected_point - self.verts[self.connectivity[tri, 0], :],
                    self.normals[tri, :],
                )
            )
            cpp_vals.append(val)
        cpp_arr = np.asarray(cpp_vals, dtype=float)
        tri_arr = np.asarray(triangles_list, dtype=int)

        order = np.abs(cpp_arr).argsort()
        _LOGGER.debug(
            "project_point_check: |CPP| sorted (first 5)=%s",
            np.abs(cpp_arr[order])[:5].tolist(),
        )

        intriangle = -1
        projected_point = pre_projected_point
        r = -1.0
        t = -1.0

        # Test candidate triangles from closest plane to furthest.
        for o in order:
            idx = int(o)
            tri_idx = int(tri_arr[idx])

            # Project onto this triangle's plane.
            projected_pt = pre_projected_point - cpp_arr[idx] * self.normals[tri_idx, :]

            # Triangle edge vectors and local w.
            a = int(self.connectivity[tri_idx, 0])
            b = int(self.connectivity[tri_idx, 1])
            c = int(self.connectivity[tri_idx, 2])

            u = self.verts[b, :] - self.verts[a, :]
            v = self.verts[c, :] - self.verts[a, :]
            w_vec = projected_pt - self.verts[a, :]

            # Robust barycentric test using cross products (signs).
            vxw = np.cross(v, w_vec)
            vxu = np.cross(v, u)
            uxw = np.cross(u, w_vec)

            sign_r = float(np.dot(vxw, vxu))
            sign_t = float(np.dot(uxw, -vxu))

            _LOGGER.debug(
                "project_point_check: tri=%d sign_r=%.6g sign_t=%.6g",
                tri_idx,
                sign_r,
                sign_t,
            )

            if sign_r >= 0.0 and sign_t >= 0.0:
                denom = float(np.linalg.norm(vxu))
                if denom < EPS or not np.isfinite(denom):
                    _LOGGER.debug(
                        "project_point_check: tri=%d degenerate vxu; skipping.", tri_idx
                    )
                    continue

                r_try = float(np.linalg.norm(vxw) / denom)
                t_try = float(np.linalg.norm(uxw) / denom)

                _LOGGER.debug(
                    "project_point_check: tri=%d r=%.6g t=%.6g", tri_idx, r_try, t_try
                )

                # Allow tiny tolerance to accept boundary hits.
                if r_try <= 1.0 and t_try <= 1.0 and (r_try + t_try) <= 1.001:
                    intriangle = tri_idx
                    projected_point = projected_pt
                    r = r_try
                    t = t_try
                    _LOGGER.debug("project_point_check: inside tri=%d", tri_idx)
                    break

        return projected_point, intriangle, r, t

    def writeVTU(
        self,
        filename: str,
        point_data: Optional[Dict[str, NDArray[Any]]] = None,
        cell_data: Optional[Dict[str, NDArray[Any]]] = None,
    ) -> None:
        """Export this mesh (and optional point/cell data) in VTU format.

        Accepts NumPy or CuPy arrays; all data are converted to NumPy on CPU
        before writing. `cell_data` is normalized to meshio's expected shape
        (dict of name -> list-of-arrays, per cell block).

        Args:
            filename: Output path (e.g., ``"mesh.vtu"``).
            point_data: Optional dict of per-node arrays (shape (n_nodes,) or (n_nodes, k)).
            cell_data: Optional dict of per-triangle arrays (shape (n_tris,) or (n_tris, k)).

        Raises:
            ValueError: If provided data have incompatible lengths.
            Exception: If the underlying mesh writer fails.
        """
        try:
            # Ensure core arrays are CPU/NumPy with correct dtypes.
            pts = self._as_np(self.verts, dtype=float)
            con = self._as_np(self.connectivity, dtype=int)

            # Build meshio Mesh (single triangle cell block).
            cells: List[Tuple[str, NDArray[Any]]] = [("triangle", con)]
            m = meshio.Mesh(points=pts, cells=cells)

            # Normalize point_data
            if point_data:
                for name, arr in point_data.items():
                    arr_np = self._as_np(arr)
                    if arr_np.shape[0] != pts.shape[0]:
                        msg = (
                            f"point_data['{name}'] length {arr_np.shape[0]} "
                            f"!= n_nodes {pts.shape[0]}"
                        )
                        _LOGGER.error("writeVTU: %s", msg)
                        raise ValueError(msg)
                    m.point_data[name] = arr_np

            # Normalize cell_data to meshio's structure: dict[str] -> List[np.ndarray]
            if cell_data:
                n_tris = con.shape[0]
                normalized: Dict[str, List[NDArray[Any]]] = {}
                for name, arr in cell_data.items():
                    arr_np = self._as_np(arr)
                    if arr_np.shape[0] != n_tris:
                        msg = (
                            f"cell_data['{name}'] length {arr_np.shape[0]} "
                            f"!= n_tris {n_tris}"
                        )
                        _LOGGER.error("writeVTU: %s", msg)
                        raise ValueError(msg)
                    normalized[name] = [arr_np]
                m.cell_data = normalized

            m.write(filename)
            _LOGGER.info(
                "VTU written to '%s' (nodes=%d, tris=%d, point_data=%d, cell_data=%d)",
                filename,
                pts.shape[0],
                con.shape[0],
                0 if not point_data else len(point_data),
                0 if not cell_data else len(cell_data),
            )

        except Exception:
            _LOGGER.exception("writeVTU failed for '%s'.", filename)
            raise

    def Bmatrix(self, element: int) -> Tuple[NDArray[Any], float]:
        """Compute the B-matrix and Jacobian determinant for a triangle.

        Args:
            element (int): Triangle index.

        Returns:
            Tuple[NDArray[Any], float]:
                - B (2×3 array): Strain-displacement matrix.
                - J (float): Twice the triangle area (Jacobian determinant).
        """
        # Gather triangle vertices on CPU, then send to active device.
        tri_idx = self.connectivity[element]
        node_coords_cpu: NDArray[Any] = self.verts[tri_idx]  # (3, 3) NumPy
        node_coords = to_device(node_coords_cpu, dtype=float)  # (3, 3) on backend

        # Local orthonormal frame (e1 along edge 2->1, e2 within the triangle plane).
        edge21 = node_coords[1] - node_coords[0]
        n_edge21 = float(norm(edge21))
        if n_edge21 <= 0.0 or not np.isfinite(n_edge21):
            _LOGGER.error("Bmatrix(%d): degenerate edge length.", element)
            raise ValueError("Degenerate triangle: zero edge length.")

        e1 = edge21 / n_edge21

        temp = node_coords[2] - node_coords[0]
        proj = self._dot(temp, e1)
        perp = temp - proj * e1
        n_perp = float(norm(perp))
        if n_perp <= 0.0 or not np.isfinite(n_perp):
            _LOGGER.error(
                "Bmatrix(%d): colinear vertices (zero perpendicular).", element
            )
            raise ValueError("Degenerate triangle: vertices nearly colinear.")

        e2 = perp / n_perp

        # Scalar edge projections in local frame.
        x21 = self._dot(edge21, e1)
        x13 = self._dot(node_coords[0] - node_coords[2], e1)
        x32 = self._dot(node_coords[2] - node_coords[1], e1)

        y23 = self._dot(node_coords[1] - node_coords[2], e2)
        y31 = self._dot(node_coords[2] - node_coords[0], e2)
        y12 = self._dot(node_coords[0] - node_coords[1], e2)

        # Jacobian (twice area).
        J = x13 * y23 - y31 * x32
        if not np.isfinite(J) or abs(J) < 1e-15:
            _LOGGER.error("Bmatrix(%d): near-zero area (J=%g).", element, J)
            raise ValueError("Degenerate triangle: near-zero area.")

        # Assemble B on device, then bring back to CPU.
        B_dev = xp.array([[y23, y31, y12], [x32, x13, x21]], dtype=float)
        B: NDArray[Any] = to_cpu(B_dev)

        if _LOGGER.isEnabledFor(logging.DEBUG):
            e1_cpu = to_cpu(e1).tolist()
            e2_cpu = to_cpu(e2).tolist()
            _LOGGER.debug(
                "Bmatrix(elem=%d, backend=%s): e1=%s e2=%s J=%.6e",
                element,
                backend_name(),
                e1_cpu,
                e2_cpu,
                J,
            )

        return B, float(J)

    def gradient(self, element: int, u: NDArray[Any]) -> NDArray[Any]:
        """Compute the gradient of a scalar field over a triangle.

        Args:
            element (int): Triangle index.
            u (NDArray[Any]): Field values at the triangle's three vertices.

        Returns:
            NDArray[Any]: 3-vector of gradients in 3D space.
        """
        tri_idx = self.connectivity[element]

        # Bring vertex coords to device; ensure float dtype.
        node_coords_cpu: NDArray[Any] = self.verts[tri_idx]  # (3, 3) NumPy
        node_coords = to_device(node_coords_cpu, dtype=float)  # backend array

        # Bring u to device (flatten to length-3).
        u_dev = to_device(u, dtype=float).reshape(3)

        # Build local orthonormal frame (e1, e2) within the triangle plane.
        edge21 = node_coords[1] - node_coords[0]
        n_edge21 = float(norm(edge21))
        if n_edge21 <= 0.0 or not np.isfinite(n_edge21):
            _LOGGER.error("gradient(%d): degenerate edge length.", element)
            raise ValueError("Degenerate triangle: zero edge length.")

        e1 = edge21 / n_edge21

        temp = node_coords[2] - node_coords[0]
        proj = self._dot(temp, e1)
        perp = temp - proj * e1
        n_perp = float(norm(perp))
        if n_perp <= 0.0 or not np.isfinite(n_perp):
            _LOGGER.error(
                "gradient(%d): colinear vertices (zero perpendicular).", element
            )
            raise ValueError("Degenerate triangle: vertices nearly colinear.")

        e2 = perp / n_perp
        e3 = xp.cross(e1, e2)

        # Scalar edge projections in local frame.
        x21 = self._dot(edge21, e1)
        x13 = self._dot(node_coords[0] - node_coords[2], e1)
        x32 = self._dot(node_coords[2] - node_coords[1], e1)

        y23 = self._dot(node_coords[1] - node_coords[2], e2)
        y31 = self._dot(node_coords[2] - node_coords[0], e2)
        y12 = self._dot(node_coords[0] - node_coords[1], e2)

        # Jacobian (twice area).
        J = x13 * y23 - y31 * x32
        if not np.isfinite(J) or abs(J) < 1e-15:
            _LOGGER.error("gradient(%d): near-zero area (J=%g).", element, J)
            raise ValueError("Degenerate triangle: near-zero area.")

        # B-matrix on device, then compute in-plane gradient components.
        B_dev = xp.array([[y23, y31, y12], [x32, x13, x21]], dtype=float)  # (2, 3)
        grad_vals_dev = (B_dev @ u_dev) / J  # (2,)

        # Expand to 3-vector in the local frame: [grad_x, grad_y, 0].
        grad_dev = xp.zeros(3, dtype=float)
        grad_dev[:2] = grad_vals_dev

        # Rotate back to global frame with columns [e1 e2 e3].
        R_dev = xp.vstack((e1, e2, e3)).T  # (3, 3)
        result_dev = R_dev @ grad_dev  # (3,)

        result = to_cpu(result_dev)

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "gradient(elem=%d, backend=%s): J=%.6e, grad=%s",
                element,
                backend_name(),
                J,
                result.tolist(),
            )

        return result

    def StiffnessMatrix(self, B: NDArray[Any], J: float) -> NDArray[Any]:
        """Compute the local stiffness matrix for a triangle.

        Args:
            B (NDArray[Any]): B-matrix from `Bmatrix`.
            J (float): Jacobian determinant.

        Returns:
            NDArray[Any]: 3×3 stiffness matrix.
        """
        # Normalize inputs (CPU/NumPy)
        B_np = self._as_np(B, dtype=float)

        if B_np.shape != (2, 3):
            _LOGGER.error("StiffnessMatrix: expected B shape (2,3), got %s", B_np.shape)
            raise ValueError(f"Invalid B shape {B_np.shape}; expected (2, 3).")

        if not (np.isfinite(J) and abs(J) >= 1e-15):
            _LOGGER.error("StiffnessMatrix: invalid J=%r (non-finite or near zero).", J)
            raise ValueError("Degenerate triangle: invalid J for stiffness.")

        K = (B_np.T @ B_np) / (2.0 * J)

        _LOGGER.debug(
            "StiffnessMatrix: J=%.6e, K=%s",
            float(J),
            np.array2string(K, precision=6, suppress_small=True),
        )

        return K

    def MassMatrix(self, J: float) -> NDArray[Any]:
        """Compute the 3×3 (consistent) local mass matrix for a triangle.

        Args:
            J (float): Jacobian determinant.

        Returns:
            NDArray[Any]: 3×3 mass matrix.
        """
        if not (np.isfinite(J) and abs(J) >= 1e-15):
            _LOGGER.error("MassMatrix: invalid J=%r (non-finite or near zero).", J)
            raise ValueError("Degenerate triangle: invalid J for mass.")

        M = (J / 12.0) * np.array(
            [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
            dtype=float,
        )

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "MassMatrix: J=%.6e, M=%s",
                float(J),
                np.array2string(M, precision=6, suppress_small=True),
            )

        return M

    def ForceVector(
        self,
        B: NDArray[Any],
        J: float,
        X: NDArray[Any],
    ) -> NDArray[Any]:
        """Compute the local force vector for a triangle.

        Args:
            B (NDArray[Any]): B-matrix from `Bmatrix`.
            J (float): Jacobian determinant.
            X (NDArray[Any]): Gradient vector from `gradient`.

        Returns:
            NDArray[Any]: Length-3 force vector.
        """
        # Normalize inputs to CPU/NumPy float
        B_np = self._as_np(B, dtype=float)
        X_np = self._as_np(X, dtype=float).reshape(-1)

        # Basic shape validation
        if B_np.ndim != 2 or B_np.shape[1] != 3:
            raise ValueError(f"ForceVector: B must be (2,3) or (k,3); got {B_np.shape}")
        if X_np.ndim != 1 or X_np.shape[0] != B_np.shape[0]:
            raise ValueError(
                f"ForceVector: X must be (B.shape[0],); got {X_np.shape} vs B rows {B_np.shape[0]}"
            )

        # Compute force
        F = (B_np.T @ X_np) / 2.0  # shape (3,)

        _LOGGER.debug(
            "ForceVector: J=%.6e , ||B||=%.6e, ||X||=%.6e, F=%s",
            float(J),
            float(np.linalg.norm(B_np)),
            float(np.linalg.norm(X_np)),
            np.array2string(F, precision=6, suppress_small=True),
        )

        return F

    def computeGeodesic(
        self,
        nodes: Sequence[int],
        nodeVals: Sequence[float],
        filename: Optional[str] = None,
        K: Optional[sp.spmatrix] = None,
        M: Optional[sp.spmatrix] = None,
        dt: float = 10.0,
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        """Compute geodesic distances using the heat method (FEM).

        Args:
            nodes (Sequence[int]): Indices of fixed-temperature nodes.
            nodeVals (Sequence[float]): Temperature values at `nodes`.
            filename (Optional[str]): VTU output path.
            K (Optional[sp.spmatrix]): Preassembled stiffness matrix.
            M (Optional[sp.spmatrix]): Preassembled mass matrix.
            dt (float): Time-step for the heat diffusion.

        Returns:
            Tuple[NDArray[Any], NDArray[Any]]:
                - ATglobal: Per-node geodesic distance.
                - Xs: Per-triangle gradient directions.
        """
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError(
                f"computeGeodesic: dt must be positive and finite; got {dt}"
            )

        n_nodes: int = int(self.verts.shape[0])
        n_elem: int = int(self.connectivity.shape[0])

        if n_nodes == 0 or n_elem == 0:
            raise ValueError("computeGeodesic: empty mesh (no nodes or no elements).")

        # Normalize Dirichlet data
        bc_nodes: List[int] = [int(i) for i in nodes]
        bc_vals: NDArray[Any] = self._as_np(nodeVals, dtype=float).reshape(-1)
        if len(bc_nodes) != bc_vals.shape[0]:
            raise ValueError(
                f"computeGeodesic: nodes (len={len(bc_nodes)}) and nodeVals (len={bc_vals.shape[0]}) mismatch."
            )

        _LOGGER.debug(
            "computeGeodesic: n_nodes=%d n_elem=%d | #Dirichlet=%d dt=%.6g",
            n_nodes,
            n_elem,
            len(bc_nodes),
            dt,
        )

        # -------------------------------------------------------------------------
        # Heat solve (M + dt K) u = u0  (u0: large point sources on bc nodes)
        # -------------------------------------------------------------------------
        if K is None or M is None:
            # Assemble sparse K, M (LIL for efficient increment, then CSR)
            K_lil: sp.lil_matrix = sp.lil_matrix((n_nodes, n_nodes), dtype=float)
            M_lil: sp.lil_matrix = sp.lil_matrix((n_nodes, n_nodes), dtype=float)

            for el, tri in enumerate(self.connectivity):
                tri_idx = np.asarray(tri, dtype=int)
                B, J = self.Bmatrix(el)  # (2,3), scalar
                k_loc = self.StiffnessMatrix(B, J)  # (3,3)
                m_loc = self.MassMatrix(J)  # (3,3)

                for a in range(3):
                    ia = int(tri_idx[a])
                    for b in range(3):
                        ib = int(tri_idx[b])
                        K_lil[ia, ib] += k_loc[a, b]
                        M_lil[ia, ib] += m_loc[a, b]

            K = K_lil.tocsr()
            M = M_lil.tocsr()
            _LOGGER.debug(
                "computeGeodesic: assembled K,M sparsity -> nnz(K)=%d nnz(M)=%d",
                int(K.nnz),
                int(M.nnz),
            )
        else:
            # Ensure CSR for efficient slicing/solving
            K = K.tocsr() if not isinstance(K, sp.csr_matrix) else K
            M = M.tocsr() if not isinstance(M, sp.csr_matrix) else M

        # Large heat sources at bc_nodes
        u0: NDArray[Any] = np.zeros(n_nodes, dtype=float)
        u0[bc_nodes] = 1e6

        A1: sp.spmatrix = (M + dt * K).tocsr()
        try:
            u_vec: NDArray[Any] = spsolve(A1, u0)  # shape (n_nodes,)
        except Exception:
            _LOGGER.exception("computeGeodesic: heat solve failed.")
            raise

        # Column view for element indexing consistency
        u_col: NDArray[Any] = u_vec[:, None]

        # -------------------------------------------------------------------------
        # Per-element normalized gradients of u
        # -------------------------------------------------------------------------
        Xs: NDArray[Any] = np.zeros((n_elem, 3), dtype=float)
        for k, tri in enumerate(self.connectivity):
            tri_idx = np.asarray(tri, dtype=int)
            B, J = self.Bmatrix(k)
            # gradient expects the 3 nodal values for this triangle
            X = self.gradient(k, u_col[tri_idx, 0])  # (3,)
            nX = float(np.linalg.norm(X))
            if nX > 0.0 and np.isfinite(nX):
                Xs[k, :] = X / nX
            else:
                Xs[k, :] = 0.0

        # -------------------------------------------------------------------------
        # Assemble RHS F and solve K * AT = F with Dirichlet BCs
        # -------------------------------------------------------------------------
        F: NDArray[Any] = np.zeros(n_nodes, dtype=float)
        for k, tri in enumerate(self.connectivity):
            tri_idx = np.asarray(tri, dtype=int)
            B, J = self.Bmatrix(k)
            # In-plane quantity for the force formula (same as legacy)
            xnr = B @ u_col[tri_idx, 0]  # (2,)
            n_xnr = float(np.linalg.norm(xnr))
            if n_xnr > 0.0 and np.isfinite(n_xnr):
                xnr /= n_xnr
            f_loc = self.ForceVector(B, J, xnr)  # (3,)
            # Scatter-add local force into global RHS
            for a in range(3):
                F[int(tri_idx[a])] -= f_loc[a]

        # Dirichlet split
        active_nodes: List[int] = [i for i in range(n_nodes) if i not in bc_nodes]
        if not active_nodes:
            # Trivial case: all nodes prescribed
            result_at = np.zeros(n_nodes, dtype=float)
            result_at[bc_nodes] = bc_vals
            if filename is not None:
                self.writeVTU(filename, point_data={"d": result_at})
            return result_at, Xs

        # Kaa * AT_active = F_active - Kab * bc_vals
        try:
            Kaa = K[active_nodes, :][:, active_nodes]
            Kab = K[active_nodes, :][:, bc_nodes]
            rhs = F[active_nodes] - Kab.dot(bc_vals)
            AT_active: NDArray[Any] = spsolve(Kaa, rhs)
        except Exception:
            _LOGGER.exception("computeGeodesic: Dirichlet solve failed.")
            raise

        ATglobal: NDArray[Any] = np.zeros(n_nodes, dtype=float)
        ATglobal[active_nodes] = AT_active
        ATglobal[bc_nodes] = bc_vals

        if filename is not None:
            try:
                self.writeVTU(filename, point_data={"d": ATglobal})
            except Exception:
                _LOGGER.exception(
                    "computeGeodesic: writeVTU failed for '%s'.", filename
                )
                raise

        _LOGGER.debug(
            "computeGeodesic: completed (active=%d, fixed=%d).",
            len(active_nodes),
            len(bc_nodes),
        )
        return ATglobal, Xs

    def computeLaplace(
        self,
        nodes: Sequence[int],
        nodeVals: Sequence[float] | NDArray[Any],
        filename: Optional[str] = None,
    ) -> NDArray[Any]:
        """Solve Laplace's equation with Dirichlet boundary conditions.

        Args:
            nodes (Sequence[int]): Dirichlet node indices.
            nodeVals (Sequence[float] | NDArray[Any]): Boundary values.
            filename (Optional[str]): VTU output path.

        Returns:
            NDArray[Any]: Solution vector of length n_nodes.
        """
        n_nodes: int = int(self.verts.shape[0])
        if n_nodes == 0:
            raise ValueError("computeLaplace: empty mesh (no nodes).")

        # Normalize BC inputs
        bc_nodes: list[int] = [int(i) for i in nodes]
        bc_vals: NDArray[Any] = self._as_np(nodeVals, dtype=float).reshape(-1)
        if len(bc_nodes) != bc_vals.shape[0]:
            raise ValueError(
                f"computeLaplace: nodes (len={len(bc_nodes)}) and nodeVals (len={bc_vals.shape[0]}) mismatch."
            )

        _LOGGER.debug(
            "computeLaplace: n_nodes=%d | #Dirichlet=%d",
            n_nodes,
            len(bc_nodes),
        )

        # Assemble global matrices (CSR); CPU solve via SciPy
        K, _M = self.computeLaplacian()
        K = K.tocsr()

        # Dirichlet split
        active_nodes: list[int] = [i for i in range(n_nodes) if i not in bc_nodes]
        if not active_nodes:
            # Trivial case: all nodes prescribed
            result_t = np.zeros(n_nodes, dtype=float)
            result_t[bc_nodes] = bc_vals
            if filename is not None:
                self.writeVTU(filename, point_data={"u": result_t})
            return result_t

        # Proper blocks: Kaa is square (active×active), Kab is (active×bc)
        Kaa = K[active_nodes, :][:, active_nodes]
        Kab = K[active_nodes, :][:, bc_nodes]

        # RHS with zero forcing: Kaa * T_active = -Kab * bc_vals
        rhs = -Kab.dot(bc_vals)

        try:
            T_active: NDArray[Any] = spsolve(Kaa, rhs)
        except Exception:
            _LOGGER.exception("computeLaplace: Dirichlet solve failed.")
            raise

        # Assemble global solution
        Tglobal: NDArray[Any] = np.zeros(n_nodes, dtype=float)
        Tglobal[active_nodes] = T_active
        Tglobal[bc_nodes] = bc_vals

        if filename is not None:
            try:
                self.writeVTU(filename, point_data={"u": Tglobal})
            except Exception:
                _LOGGER.exception("computeLaplace: writeVTU failed for '%s'.", filename)
                raise

        _LOGGER.debug(
            "computeLaplace: solved (active=%d, fixed=%d).",
            len(active_nodes),
            len(bc_nodes),
        )
        return Tglobal

    def computeLaplacian(self) -> Tuple[sp.spmatrix, sp.spmatrix]:
        """Assemble global stiffness (K) and mass (M) matrices as CSR.

        Returns:
            Tuple[sp.spmatrix, sp.spmatrix]: (K, M) in CSR format.
        """
        n_nodes: int = int(self.verts.shape[0])
        n_tris: int = int(self.connectivity.shape[0])
        if n_nodes == 0 or n_tris == 0:
            raise ValueError("computeLaplacian: empty mesh.")

        # COO builder lists (faster than repeated updates)
        rows_K: list[int] = []
        cols_K: list[int] = []
        data_K: list[float] = []
        rows_M: list[int] = []
        cols_M: list[int] = []
        data_M: list[float] = []

        # Assemble element-by-element
        for elem_idx in range(n_tris):
            tri = self.connectivity[elem_idx]  # shape (3,)

            # Local FEM pieces (B on CPU, J as float)
            B, J = self.Bmatrix(elem_idx)
            K_loc = self.StiffnessMatrix(B, J)  # (3,3) np.ndarray
            M_loc = self.MassMatrix(J)  # (3,3) np.ndarray

            # Map 3x3 block into global COO (row-major order)
            # rows: [t0,t0,t0, t1,t1,t1, t2,t2,t2]
            # cols: [t0,t1,t2, t0,t1,t2, t0,t1,t2]
            tri_rows = np.repeat(tri, 3)
            tri_cols = np.tile(tri, 3)

            rows_K.extend(tri_rows.tolist())
            cols_K.extend(tri_cols.tolist())
            data_K.extend(K_loc.ravel(order="C").tolist())

            rows_M.extend(tri_rows.tolist())
            cols_M.extend(tri_cols.tolist())
            data_M.extend(M_loc.ravel(order="C").tolist())

        # Build CSR once
        K = sp.coo_matrix(
            (data_K, (rows_K, cols_K)), shape=(n_nodes, n_nodes), dtype=float
        ).tocsr()
        M = sp.coo_matrix(
            (data_M, (rows_M, cols_M)), shape=(n_nodes, n_nodes), dtype=float
        ).tocsr()

        _LOGGER.debug(
            "computeLaplacian: assembled K/M (nodes=%d, tris=%d, nnzK=%d, nnzM=%d, backend=%s)",
            n_nodes,
            n_tris,
            K.nnz,
            M.nnz,
            backend_name(),
        )

        return K, M

    def uvmap(self, filename: Optional[str] = None) -> None:
        """Compute UV coordinates by solving Laplace's equation.

        Args:
            filename (Optional[str]): VTU export path for u and v fields.
        """
        # Generate boundary and BCs (NumPy on CPU)
        around_nodes, bc_u, bc_v = self.uv_bc()

        n_nodes = int(self.verts.shape[0])
        _LOGGER.debug(
            "uvmap: boundary nodes=%d (loop includes repeat of start), "
            "mesh nodes=%d",
            len(around_nodes),
            n_nodes,
        )

        # Solve two Laplace problems with Dirichlet BC on the boundary loop
        u = self.computeLaplace(around_nodes[:-1], bc_u)  # length n_nodes
        v = self.computeLaplace(around_nodes[:-1], bc_v)  # length n_nodes

        if u.shape[0] != n_nodes or v.shape[0] != n_nodes:
            raise ValueError(
                f"uvmap: Laplace solutions have wrong length (u={u.shape}, v={v.shape}, "
                f"expected {n_nodes})"
            )

        # Stack into (n_nodes, 2) UV array
        uv_arr = np.vstack([u, v]).T.astype(float, copy=False)
        self.uv = uv_arr

        # Optional VTU export of the scalar fields
        if filename is not None:
            try:
                self.writeVTU(filename, point_data={"u": u, "v": v})
            except Exception:
                _LOGGER.exception("uvmap: writeVTU failed for '%s'", filename)
                raise

        preview = uv_arr[: min(5, uv_arr.shape[0])].tolist()
        _LOGGER.debug(
            "uvmap: completed (backend=%s). UV shape=%s, preview(first %d)=%s",
            backend_name(),
            uv_arr.shape,
            len(preview),
            preview,
        )

    def compute_uvscaling(self) -> None:
        """Compute and validate UV-scaling metric per triangle."""
        if self.uv is None:
            self.uvmap()
        assert self.uv is not None

        n_tris = int(self.connectivity.shape[0])
        uvscale = np.empty(n_tris, dtype=float)

        for e in range(n_tris):
            # B_e (2×3), J_e (float); Bmatrix already guards against degeneracy
            B, J = self.Bmatrix(e)

            # uv values at this triangle's vertices: (3, 2)
            uv_e = self.uv[self.connectivity[e]]  # NumPy indexing on CPU

            # Deformation gradient in UV space: F = (2×3 @ 3×2) / scalar -> (2×2)
            F = (B @ uv_e) / J

            # Metric = average eigenvalue of F^T F = 0.5 * trace(F^T F)
            G = F.T @ F  # (2×2), symmetric PSD
            uvscale[e] = 0.5 * float(np.trace(G))

        self.uvscaling = uvscale

        # Basic sanity check (should be non-negative; allow tiny numerical noise)
        min_val = float(np.min(uvscale))
        if min_val < -1e-12:
            _LOGGER.error(
                "compute_uvscaling: negative metric detected (min=%g) — possible flipped triangles.",
                min_val,
            )
            raise ValueError("Flipped triangles detected — check mesh quality")

        _LOGGER.debug(
            "compute_uvscaling: done (backend=%s). min=%g mean=%g max=%g",
            backend_name(),
            float(np.min(uvscale)),
            float(np.mean(uvscale)),
            float(np.max(uvscale)),
        )

    def detect_boundary(self) -> None:
        """Identify boundary edges (edges in exactly one triangle)."""
        conn = np.asarray(self.connectivity, dtype=int)
        if conn.ndim != 2 or conn.shape[1] != 3:
            raise ValueError(f"connectivity must be (n_tri, 3); got {conn.shape}")

        n_tris = int(conn.shape[0])
        n_verts = int(self.verts.shape[0])

        # Basic index validation
        if (conn < 0).any() or (conn >= n_verts).any():
            _LOGGER.error("detect_boundary: connectivity has out-of-range indices.")
            raise ValueError("Connectivity contains out-of-range vertex indices.")

        edge_count: dict[tuple[int, int], int] = {}
        for tri_idx in range(n_tris):
            a, b, c = map(int, conn[tri_idx])
            # undirected edges
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                edge_count[key] = edge_count.get(key, 0) + 1

        boundary_edges: list[tuple[int, int]] = [
            e for e, k in edge_count.items() if k == 1
        ]
        nonmanifold_edges: list[tuple[int, int]] = [
            e for e, k in edge_count.items() if k > 2
        ]

        self.boundary_edges = boundary_edges

        if nonmanifold_edges:
            _LOGGER.warning(
                "detect_boundary: %d non-manifold edge(s) detected (used by >2 tris).",
                len(nonmanifold_edges),
            )

        _LOGGER.debug(
            "detect_boundary: tris=%d -> boundary_edges=%d (unique undirected edges=%d).",
            n_tris,
            len(boundary_edges),
            len(edge_count),
        )

    def uv_bc(self) -> Tuple[List[int], NDArray[Any], NDArray[Any]]:
        """Generate UV boundary loop and boundary conditions.

        Returns:
            Tuple[List[int], NDArray[Any], NDArray[Any]]:
              around_nodes, bc_u, bc_v.
        """
        if self.boundary_edges is None:
            self.detect_boundary()

        edges = self.boundary_edges or []
        if not edges:
            raise ValueError("uv_bc: no boundary edges available.")

        # Build adjacency for the boundary graph (undirected)
        adj: DefaultDict[int, List[int]] = collections.defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Sanity checks: manifold boundary nodes typically have degree 2
        deg1 = [n for n, nbrs in adj.items() if len(nbrs) == 1]
        deggt2 = [n for n, nbrs in adj.items() if len(nbrs) > 2]
        if deg1:
            _LOGGER.warning(
                "uv_bc: found %d boundary node(s) with degree 1 (open boundary).",
                len(deg1),
            )
        if deggt2:
            _LOGGER.warning(
                "uv_bc: found %d non-manifold boundary node(s) (degree > 2).",
                len(deggt2),
            )

        # Start from the first boundary edge; try to walk a closed loop
        start_u, start_v = edges[0]
        around_nodes: List[int] = [int(start_u), int(start_v)]

        # Walk until we return to start_u (closed loop), or we fail
        # Put a safety cap to avoid infinite loops
        max_steps = len(adj) * 2 + 4
        steps = 0
        while around_nodes[-1] != around_nodes[0]:
            steps += 1
            if steps > max_steps:
                _LOGGER.error(
                    "uv_bc: loop walk exceeded safety cap; boundary may be broken."
                )
                raise ValueError(
                    "UV boundary traversal exceeded safety cap; boundary may be broken."
                )

            prev_node = around_nodes[-2]
            cur_node = around_nodes[-1]
            nbrs = adj[cur_node]
            if not nbrs:
                _LOGGER.error(
                    "uv_bc: dead-end at boundary node %d; boundary may be broken.",
                    cur_node,
                )
                raise ValueError(
                    "UV boundary traversal hit a dead-end; boundary may be broken."
                )

            # Choose the neighbor that's not the previous node (prefer a simple cycle)
            next_candidates = [n for n in nbrs if n != prev_node]
            if not next_candidates:
                # Only possible neighbor is the previous node; this suggests a 2-node loop or dead-end
                # If the previous is the start and we can close, do it; otherwise it's broken.
                if prev_node == around_nodes[0]:
                    around_nodes.append(prev_node)
                    break
                _LOGGER.error(
                    "uv_bc: stuck oscillating between nodes %d and %d.",
                    prev_node,
                    cur_node,
                )
                raise ValueError("UV boundary traversal stuck; boundary may be broken.")

            next_node = int(next_candidates[0])
            around_nodes.append(next_node)

            # Hard cap to avoid traversing more nodes than exist
            if len(around_nodes) > self.verts.shape[0] + 2:
                _LOGGER.error(
                    "uv_bc: traversal exceeded number of vertices (%d); boundary may be broken.",
                    self.verts.shape[0],
                )
                raise ValueError(
                    "UV boundary traversal exceeded mesh size — boundary may be broken"
                )

        # We now have a closed loop with around_nodes[0] == around_nodes[-1]
        if len(around_nodes) < 3:
            _LOGGER.error("uv_bc: degenerate boundary loop with < 3 unique nodes.")
            raise ValueError("Degenerate boundary: need at least 3 unique nodes.")

        # Parameterize by cumulative arc-length along the loop (exclude the last duplicate)
        pts = self.verts  # (n, 3) NumPy
        seg_vecs = pts[around_nodes[1:]] - pts[around_nodes[:-1]]
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)  # length = len(around_nodes) - 1
        cum_lengths = np.cumsum(seg_lengths)
        if not np.isfinite(cum_lengths).all():
            _LOGGER.error("uv_bc: non-finite segment lengths encountered.")
            raise ValueError("Non-finite segment lengths on boundary.")

        total_length = float(cum_lengths[-1])
        if total_length <= 0.0 or not np.isfinite(total_length):
            _LOGGER.error("uv_bc: zero or non-finite total boundary length.")
            raise ValueError("Degenerate boundary length.")

        # Sin/cos pattern along the loop, defined at unique nodes (exclude last duplicate)
        # cum_lengths currently has length N-1 and corresponds to nodes 1..N-1
        # Make lengths aligned to nodes 0..N-2 by prepending 0 (start node distance = 0)
        lengths_for_nodes = np.concatenate([[0.0], cum_lengths[:-1]])  # shape N-1
        theta = 2.0 * np.pi * (lengths_for_nodes / total_length)
        bc_u = np.sin(theta)
        bc_v = np.cos(theta)

        _LOGGER.debug(
            "uv_bc: closed loop with %d unique nodes, total_length=%.6g.",
            len(around_nodes) - 1,
            total_length,
        )

        return around_nodes, bc_u.astype(float), bc_v.astype(float)

    def compute_triareas(self) -> None:
        """Compute triangle areas (J/2) and store in `self.triareas`."""
        tri = self.connectivity  # (n_tri, 3) int
        a = self.verts[tri[:, 0], :]  # (n_tri, 3)
        b = self.verts[tri[:, 1], :]
        c = self.verts[tri[:, 2], :]

        u = b - a
        v = c - a
        cross_uv = np.cross(u, v)  # (n_tri, 3)
        twice_area = np.linalg.norm(cross_uv, axis=1)  # ||u×v||

        triareas = twice_area * 0.5  # area = 0.5 * ||u×v||

        # Handle non-finite/near-zero areas robustly
        bad = ~np.isfinite(triareas)
        if np.any(bad):
            _LOGGER.warning(
                "compute_triareas: %d non-finite triangle area(s) set to 0.",
                int(bad.sum()),
            )
            triareas[bad] = 0.0

        eps = 1e-15
        deg = triareas <= eps
        if np.any(deg):
            _LOGGER.warning(
                "compute_triareas: %d degenerate triangle(s) with near-zero area.",
                int(deg.sum()),
            )

        self.triareas = triareas.astype(float, copy=False)

        _LOGGER.debug(
            "compute_triareas: n=%d  min=%.6g  max=%.6g  mean=%.6g",
            triareas.size,
            float(np.min(triareas)) if triareas.size else 0.0,
            float(np.max(triareas)) if triareas.size else 0.0,
            float(np.mean(triareas)) if triareas.size else 0.0,
        )

    def tri2node_interpolation(self, cell_field: NDArray[Any]) -> List[float]:
        """Interpolate triangle-based field to nodes by area-weighting.

        Args:
            cell_field (NDArray[Any]): Per-triangle values.

        Returns:
            List[float]: Per-node interpolated values.
        """
        # Ensure triangle areas exist
        if (
            self.triareas is None
            or self.triareas.shape[0] != self.connectivity.shape[0]
        ):
            _LOGGER.debug("tri2node_interpolation: computing triangle areas...")
            self.compute_triareas()
        assert self.triareas is not None

        # Normalize inputs to CPU/NumPy
        tri_vals = np.asarray(to_cpu(cell_field), dtype=float).reshape(-1)
        tri = self.connectivity  # (T, 3) ints
        areas = np.asarray(self.triareas, dtype=float).reshape(-1)

        T = tri.shape[0]
        if tri_vals.shape[0] != T:
            raise ValueError(
                f"tri2node_interpolation: cell_field length {tri_vals.shape[0]} "
                f"!= n_triangles {T}"
            )

        n_nodes = self.verts.shape[0]
        num = np.zeros(n_nodes, dtype=float)  # numerator accumulator
        den = np.zeros(n_nodes, dtype=float)  # denominator accumulator

        # Contribution per triangle (area * value)
        contrib = areas * tri_vals  # (T,)

        # Accumulate to each of the 3 nodes per triangle (vectorized with add.at)
        for j in range(3):
            np.add.at(num, tri[:, j], contrib)
            np.add.at(den, tri[:, j], areas)

        # Compute final node values; handle zero-area neighborhoods robustly
        out = np.zeros(n_nodes, dtype=float)
        valid = den > 0.0
        out[valid] = num[valid] / den[valid]

        zero_deg = (~valid).sum()
        if zero_deg:
            _LOGGER.warning(
                "tri2node_interpolation: %d node(s) have zero total incident area; "
                "setting their values to 0.",
                int(zero_deg),
            )

        _LOGGER.debug(
            "tri2node_interpolation: T=%d, N=%d, min=%.6g, max=%.6g, mean=%.6g",
            T,
            n_nodes,
            float(out.min(initial=0.0)),
            float(out.max(initial=0.0)),
            float(out.mean()) if n_nodes else 0.0,
        )

        return [float(x) for x in out]
