"""Module defining the FractalTree class to generate fractal trees within a mesh domain.

This module implements UV-based fractal tree growth using geometric rules,
collision detection, and iterative branching to create a tree structure
embedded in a 3D mesh.
"""

import logging
from typing import DefaultDict, Dict, Any, List, Tuple
import numpy as np
import pyvista as pv
import vtk
from numpy.typing import NDArray
import meshio
from collections import defaultdict

from .mesh import Mesh
from .edge import Edge
from .fractal_tree_parameters import FractalTreeParameters
from .config import xp, backend_name, to_cpu

_LOGGER = logging.getLogger(__name__)


class FractalTree:
    """Fractal-tree generator on a UV-mapped surface.

    This class reproduces the legacy algorithm exactly (VTK-based point-in-mesh
    checks, UV-scaling step sizes, and queue behavior), keeping helpers local
    so the module is self-contained.
    """

    def __init__(self, params: FractalTreeParameters) -> None:
        """Initialize the fractal tree generator.

        This:
        - loads the 3D mesh and computes UV + UV-scaling (via :class:`Mesh`),
        - builds a “flattened UV” mesh (z=0) with the same connectivity,
        - constructs a VTK cell locator over the flattened mesh for closest-point queries,
        - precomputes a node-wise scaling proxy (area-weighted from triangle metrics).

        Args:
            params: Tree growth parameters (mesh file, growth lengths/angles, etc.).

        Raises:
            RuntimeError: If UV coordinates or UV-scaling are unavailable.
            Exception: If building the VTK locator fails.
        """
        self.params = params
        _LOGGER.info("FractalTree: initializing (backend=%s)", backend_name())

        # Load 3D surface mesh and compute UV+scaling (Mesh handles GPU where possible)
        self.m = Mesh(params.meshfile)
        _LOGGER.debug("FractalTree: computing UV-scaling via Mesh...")
        self.m.compute_uvscaling()

        if self.m.uv is None or self.m.uvscaling is None:
            # Mesh.compute_uvscaling guarantees both; guard anyway.
            _LOGGER.error("FractalTree: UV or UV-scaling not available.")
            raise RuntimeError("UV coordinates / UV-scaling must be computed.")

        # Build a flattened UV mesh on z=0 with identical connectivity
        uv = np.asarray(self.m.uv, dtype=float)  # (n_nodes, 2)
        zeros = np.zeros((uv.shape[0], 1), dtype=float)  # (n_nodes, 1)
        uv3 = np.concatenate((uv, zeros), axis=1)  # (n_nodes, 3)
        self.mesh_uv = Mesh(verts=uv3, connectivity=self.m.connectivity)
        _LOGGER.debug(
            "FractalTree: UV mesh created (nodes=%d, tris=%d).",
            self.mesh_uv.verts.shape[0],
            self.mesh_uv.connectivity.shape[0],
        )

        # VTK locator over the flattened UV surface (CPU-side legacy behavior)
        try:
            mpv = pv.read(params.meshfile)
            mpv.points = self.mesh_uv.verts  # overwrite with flattened UV points
            self.loc = vtk.vtkCellLocator()
            self.loc.SetDataSet(mpv)
            self.loc.BuildLocator()
            _LOGGER.debug("FractalTree: VTK cell locator built.")
        except Exception:
            _LOGGER.exception("FractalTree: failed to build VTK locator.")
            raise

        # Area-weighted node scaling proxy (kept for parity with legacy)
        self.scaling_nodes = np.array(
            self.mesh_uv.tri2node_interpolation(self.m.uvscaling), dtype=float
        )
        _LOGGER.debug(
            "FractalTree: scaling_nodes computed (len=%d).", self.scaling_nodes.size
        )

        # Outputs populated by grow_tree()
        self.uv_nodes: NDArray[np.float64] | None = None
        self.nodes_xyz: List[NDArray[np.float64]] = []
        self.edges: List[Edge] = []
        self.end_nodes: List[int] = []
        self.connectivity: List[List[int]] = []

    @staticmethod
    def _interpolate(
        vectors: NDArray[Any],
        r: float,
        t: float,
    ) -> NDArray[np.float64]:
        """Barycentric interpolation over a triangle.

        Given the values at the triangle's vertices ``vectors = [v0, v1, v2]``,
        return ``t*v2 + r*v1 + (1 - r - t)*v0``. Works for scalar fields
        (shape ``(3,)``) and vector/tensor fields (shape ``(3, k)``).

        Args:
            vectors: Values at the three triangle vertices (first dimension must be 3).
            r: Barycentric coordinate associated with vertex 1.
            t: Barycentric coordinate associated with vertex 2.

        Returns:
            Interpolated value as ``float64`` (shape ``()`` for scalars or ``(k,)`` for vectors).

        Raises:
            ValueError: If the leading dimension of ``vectors`` is not 3.
        """
        arr = np.asarray(vectors, dtype=float)
        if arr.shape[0] != 3:
            raise ValueError(
                f"_interpolate expects 'vectors' with leading dimension 3; got shape {arr.shape}"
            )

        w0 = 1.0 - r - t  # weight for v0

        v0, v1, v2 = arr[0], arr[1], arr[2]
        out = t * v2 + r * v1 + w0 * v0
        return np.asarray(out, dtype=float)

    def _eval_field(
        self,
        point: NDArray[Any],
        field: NDArray[Any],
        mesh: Mesh,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Project `point` onto `mesh`, then barycentrically interpolate `field`.

        Args:
            point: 2D/3D point in the same space as the mesh vertices
                (CuPy or NumPy ok; will be converted to NumPy on CPU).
            field: Per-vertex values to interpolate (shape (n_nodes, k) or (n_nodes,)).
            mesh:  Mesh instance used for projection and connectivity.

        Returns:
            (value, projected_point, triangle_index)
            - value: interpolated field at the projected point (float64; shape (k,) or scalar).
            - projected_point: closest point on the surface in the same coords as `point`.
            - triangle_index: index of the containing triangle (>= 0).

        Raises:
            ValueError: If the point projects outside the surface (no containing triangle).
        """
        # KD-tree is CPU-only; ensure NumPy float
        p_cpu = np.asarray(to_cpu(point), dtype=float)

        # project_new_point returns: (projected_point, tri, r, t)
        ppoint, tri, r, t = mesh.project_new_point(p_cpu, verts_to_search=5)
        tri_int = int(tri)

        if tri_int < 0:
            raise ValueError(
                "_eval_field: point projects outside the mesh domain (tri=-1)."
            )

        # Convert field to NumPy and gather the triangle's 3 vertex values
        field_cpu = np.asarray(to_cpu(field), dtype=float)
        tri_conn = mesh.connectivity[tri_int]  # shape (3,)
        vectors = field_cpu[tri_conn]  # shape (3,) or (3, k)

        # Barycentric interpolation (r for v1, t for v2, (1-r-t) for v0)
        val = self._interpolate(vectors, float(r), float(t)).astype(float, copy=False)

        return val, np.asarray(ppoint, dtype=float), tri_int

    def _point_in_mesh(self, point: NDArray[Any], mesh: Mesh) -> bool:
        """Return True if `point` projects onto (i.e., is inside) `mesh`.

        Accepts 2D (x,y) or 3D (x,y,z) points. If 2D is provided, a z=0
        coordinate is appended for projection against a z=0 UV mesh.
        """
        # Normalize to CPU/NumPy and flatten
        p = np.asarray(to_cpu(point), dtype=float).reshape(-1)

        # Allow (x, y) by appending z=0 for UV meshes
        if p.size == 2:
            p3 = np.array([p[0], p[1], 0.0], dtype=float)
        elif p.size == 3:
            p3 = p
        else:
            raise ValueError(
                f"_point_in_mesh: expected 2D or 3D point, got shape {p.shape}"
            )

        try:
            # project_new_point returns (projected_point, tri, r, t)
            _, tri, _, _ = mesh.project_new_point(p3, verts_to_search=5)
            inside = int(tri) >= 0
            _LOGGER.debug(
                "_point_in_mesh: point=%s -> tri=%d inside=%s",
                p3.tolist(),
                int(tri),
                inside,
            )
            return inside
        except Exception:
            _LOGGER.exception(
                "_point_in_mesh: projection failed for point %s", p3.tolist()
            )
            return False

    def _point_in_mesh_vtk(self, point: NDArray[Any], tol: float = 1e-9) -> bool:
        """Return True if `point` lies on the UV-surface per the VTK locator.

        Accepts 2D (x, y) or 3D (x, y, z) coordinates. For 2D inputs, z=0 is appended.
        The VTK query is always performed on CPU. `tol` is a distance threshold in
        world units (UV space) for considering the point "inside".
        """
        # Normalize to CPU/NumPy and flatten to 1D
        p = np.asarray(to_cpu(point), dtype=float).reshape(-1)

        if p.size == 2:
            q = np.array([p[0], p[1], 0.0], dtype=float)
        elif p.size == 3:
            q = p.astype(float, copy=False)
        else:
            raise ValueError(
                f"_point_in_mesh_vtk: expected 2D or 3D point, got shape {p.shape}"
            )

        if not hasattr(self, "loc") or self.loc is None:
            _LOGGER.error("_point_in_mesh_vtk: VTK locator not initialized.")
            return False

        try:
            cell_id = vtk.reference(0)
            sub_id = vtk.reference(0)
            dist = vtk.reference(0.0)
            ppoint = np.zeros(3, dtype=float)

            self.loc.FindClosestPoint(q, ppoint, cell_id, sub_id, dist)
            d = float(dist.get())
            inside = bool(d < tol)

            _LOGGER.debug(
                "_point_in_mesh_vtk: q=%s -> cell=%d sub=%d dist=%.3e tol=%.3e inside=%s",
                q.tolist(),
                int(cell_id.get()),
                int(sub_id.get()),
                d,
                tol,
                inside,
            )
            return inside
        except Exception:
            _LOGGER.exception(
                "_point_in_mesh_vtk: VTK query failed for point %s", q.tolist()
            )
            return False

    def _scaling(self, x: NDArray[Any], tol: float = 1e-3) -> Tuple[float, int]:
        """Return (sqrt(uv-scaling), triangle id) at a UV point.

        The triangle id is -1 when the closest point is farther than the distance
        threshold used by the VTK locator.
        """
        # Normalize input (accept CuPy/NumPy; 2D or 3D)
        p = np.asarray(to_cpu(x), dtype=float).reshape(-1)
        if p.size == 2:
            q = np.array([p[0], p[1], 0.0], dtype=float)
        elif p.size == 3:
            q = p
        else:
            raise ValueError(f"scaling: expected 2D or 3D point, got shape {p.shape}")

        if not hasattr(self, "loc") or self.loc is None:
            raise RuntimeError("Scaling: VTK locator not initialized.")

        # VTK closest-point query (CPU)
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist = vtk.reference(0.0)
        ppoint = np.zeros(3, dtype=float)
        self.loc.FindClosestPoint(q, ppoint, cell_id, sub_id, dist)

        d = float(dist.get())
        tri = int(cell_id.get()) if d <= tol else -1

        # UV scaling must be available (computed by Mesh.compute_uvscaling())
        if self.m.uvscaling is None:
            raise RuntimeError("scaling: UV scaling not computed.")

        # Legacy behavior: if tri == -1, index with -1 (last triangle)
        tri_idx = tri if tri >= 0 else -1
        raw = float(self.m.uvscaling[tri_idx])
        s = float(np.sqrt(max(raw, 0.0)))  # guard against tiny negative roundoff

        _LOGGER.debug(
            "Scaling: q=%s -> dist=%.3e tol=%.3e tri=%d used_idx=%d uvscale=%.6g s=%.6g",
            q.tolist(),
            d,
            tol,
            tri,
            tri_idx,
            raw,
            s,
        )
        return s, tri

    def _init_state(
        self,
    ) -> Tuple[
        List[NDArray[np.float64]],  # nodes
        List[Edge],  # edges
        List[int],  # edge_queue
        DefaultDict[int, List[int]],  # branches
        Dict[int, int],  # sister_branches
        List[int],  # end_nodes
        int,  # branch_id
        NDArray[np.float64],  # Rplus (2x2)
        NDArray[np.float64],  # Rminus (2x2)
        float,  # dx
        float,  # w
        float,  # branch_length
        float,  # init_branch_length
    ]:
        """Build the initial in-UV state for tree growth."""
        # Parameters
        dx = float(self.params.l_segment)
        branch_length = float(self.params.length)
        init_branch_length = float(self.params.init_length)
        theta = float(self.params.branch_angle)
        w = float(self.params.w)

        # Initial 2D UV nodes and direction
        init_node = self.mesh_uv.verts[self.params.init_node_id][:2]
        second_node = self.mesh_uv.verts[self.params.second_node_id][:2]

        s0, tri0 = self._scaling(init_node)
        if tri0 < 0:
            raise RuntimeError("The initial node is outside the domain")

        init_dir = second_node - init_node
        init_dir /= np.linalg.norm(init_dir)

        nodes: List[NDArray[np.float64]] = [
            np.asarray(init_node, dtype=float),
            np.asarray(init_node + dx * init_dir * s0, dtype=float),
        ]

        branch_id = 0
        edges: List[Edge] = [Edge(0, 1, nodes, None, branch_id)]
        edge_queue: List[int] = [0]
        branches: DefaultDict[int, List[int]] = defaultdict(list)
        branches[branch_id].append(0)  # match legacy: seed with node 0
        sister_branches: Dict[int, int] = {}
        end_nodes: List[int] = []

        # Rotation matrices
        Rplus = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=float,
        )
        Rminus = np.array(
            [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]],
            dtype=float,
        )

        return (
            nodes,
            edges,
            edge_queue,
            branches,
            sister_branches,
            end_nodes,
            branch_id,
            Rplus,
            Rminus,
            dx,
            w,
            branch_length,
            init_branch_length,
        )

    def _grow_initial_trunk(
        self,
        nodes: List[NDArray[np.float64]],
        edges: List[Edge],
        edge_queue: List[int],
        branches: DefaultDict[int, List[int]],
        dx: float,
        init_branch_length: float,
    ) -> int:
        """Grow the straight initial trunk for a fixed arc-length.

        Keeps legacy semantics exactly:
        - Pops from the front of `edge_queue`
        - Uses the edge's direction (normalized)
        - Scales the step by local sqrt(uv-scaling)
        - Appends new node/edge and pushes new edge index to the queue
        - Raises if the trunk exits the domain

        Returns:
            The edge id to branch from next (i.e., the first item remaining in the queue).
        """
        n_steps = int(init_branch_length / dx)
        _LOGGER.info("Growing initial trunk: steps=%d, dx=%.6g", n_steps, dx)

        for step in range(n_steps):
            if not edge_queue:
                raise RuntimeError(
                    "Initial trunk: empty edge queue before completing steps."
                )

            edge_id = edge_queue.pop(0)
            edge = edges[edge_id]

            # Move forward along the current edge direction
            new_dir = edge.dir / np.linalg.norm(edge.dir)
            s, tri = self._scaling(nodes[edge.n2])
            if tri < 0:
                raise RuntimeError("The initial branch goes out of the domain")

            new_node = nodes[edge.n2] + new_dir * dx * s
            new_node_id = len(nodes)
            nodes.append(new_node)

            # Keep per-branch node list identical to legacy behavior
            b = edge.branch
            if b is None:
                raise RuntimeError(
                    "Edge has no branch id (internal invariant violated)."
                )
            branches[b].append(new_node_id)

            # Add new edge from the previous tip to the new node
            next_edge_id = len(edges)
            edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, edge.branch))
            edge_queue.append(next_edge_id)

            _LOGGER.debug(
                "Trunk step %d/%d: edge=%d -> new_node_id=%d tri=%d s=%.6g",
                step + 1,
                n_steps,
                edge_id,
                new_node_id,
                tri,
                s,
            )

        if not edge_queue:
            raise RuntimeError("Initial trunk grown but queue is empty; cannot branch.")
        branching_edge_id = edge_queue.pop(0)
        _LOGGER.info("Initial trunk complete; branching_edge_id=%d", branching_edge_id)
        return branching_edge_id

    def _spawn_fascicles(
        self,
        branching_edge_id: int,
        nodes: list[NDArray[np.float64]],
        edges: list[Edge],
        edge_queue: list[int],
        branches: DefaultDict[int, list[int]],
        branch_id: int,
        dx: float,
    ) -> int:
        """Spawn and grow initial fascicles off the branching edge (legacy-parity).

        Mutates `nodes`, `edges`, `edge_queue`, `branches` in-place and returns the
        updated `branch_id`.
        """
        for fasc_len, fasc_ang in zip(
            self.params.fascicles_length, self.params.fascicles_angles
        ):
            # Rotation for the initial fascicle direction
            c = float(np.cos(fasc_ang))
            s = float(np.sin(fasc_ang))
            rotation = np.array([[c, -s], [s, c]], dtype=float)

            base_edge = edges[branching_edge_id]
            new_dir = rotation @ base_edge.dir
            new_dir /= np.linalg.norm(new_dir)

            scale, tri = self._scaling(nodes[base_edge.n2])
            if tri < 0:
                raise RuntimeError("the fascicle goes out of the domain")

            # First node of this fascicle
            new_node = nodes[base_edge.n2] + new_dir * dx * scale
            new_node_id = len(nodes)
            nodes.append(new_node)

            branch_id += 1
            branches[branch_id].append(new_node_id)

            new_edge_id = len(edges)
            edges.append(
                Edge(
                    base_edge.n2,
                    new_node_id,
                    nodes,
                    parent=branching_edge_id,
                    branch=branch_id,
                )
            )
            edge_queue.append(new_edge_id)

            n_steps = int(fasc_len / dx)
            _LOGGER.info(
                "Fascicle: angle=%.6g, length=%.6g, steps=%d (parent_edge=%d, branch=%d)",
                fasc_ang,
                fasc_len,
                n_steps,
                branching_edge_id,
                branch_id,
            )

            # Grow along edge queue (legacy FIFO behavior)
            for step in range(n_steps):
                edge_id = edge_queue.pop(0)
                edge = edges[edge_id]

                step_dir = edge.dir / np.linalg.norm(edge.dir)
                scale_step, tri_step = self._scaling(nodes[edge.n2])
                if tri_step < 0:
                    raise RuntimeError("the fascicle goes out of the domain")

                nxt_node = nodes[edge.n2] + step_dir * dx * scale_step
                nxt_node_id = len(nodes)
                nodes.append(nxt_node)

                b = edge.branch
                if b is None:
                    raise RuntimeError(
                        "Edge has no branch id (internal invariant violated)."
                    )
                branches[b].append(nxt_node_id)

                nxt_edge_id = len(edges)
                edges.append(Edge(edge.n2, nxt_node_id, nodes, edge_id, b))
                edge_queue.append(nxt_edge_id)

                _LOGGER.debug(
                    "Fascicle step %d/%d: edge=%d -> new_node_id=%d (tri=%d, s=%.6g)",
                    step + 1,
                    n_steps,
                    edge_id,
                    nxt_node_id,
                    tri_step,
                    scale_step,
                )

        return branch_id

    def _branch_generation(
        self,
        edge_queue: list[int],
        nodes: list[NDArray[np.float64]],
        edges: list[Edge],
        branches: DefaultDict[int, list[int]],
        sister_branches: dict[int, int],
        Rplus: NDArray[np.float64],
        Rminus: NDArray[np.float64],
        dx: float,
        branch_id: int,
        end_nodes: list[int],
    ) -> tuple[list[int], int]:
        """Create left/right children for every edge in the queue (legacy-parity).

        Pops all items from `edge_queue` and returns the new queue with the
        just-created edges (branching_queue). Updates `branch_id`, `branches`,
        `sister_branches`, and `end_nodes` in-place to match the legacy behavior.
        """
        branching_queue: list[int] = []
        rotations = (Rplus, Rminus)

        _LOGGER.info("Branching generation: #parents=%d", len(edge_queue))

        while edge_queue:
            edge_id = edge_queue.pop(0)
            base_edge = edges[edge_id]

            # Produce two children (Rplus, Rminus)
            for R in rotations:
                new_dir = R @ base_edge.dir
                new_dir /= np.linalg.norm(new_dir)

                scale, _ = self._scaling(nodes[base_edge.n2])
                new_node = nodes[base_edge.n2] + new_dir * dx * scale

                # Legacy domain check: if outside, terminate this parent tip
                if not self._point_in_mesh_vtk(new_node):
                    end_nodes.append(base_edge.n2)
                    _LOGGER.debug(
                        "Branching: parent_edge=%d -> outside domain; end node=%d",
                        edge_id,
                        base_edge.n2,
                    )
                    continue

                new_node_id = len(nodes)
                nodes.append(new_node)

                branch_id += 1
                branches[branch_id].append(new_node_id)

                new_edge_id = len(edges)
                edges.append(
                    Edge(
                        base_edge.n2,
                        new_node_id,
                        nodes,
                        parent=edge_id,
                        branch=branch_id,
                    )
                )
                branching_queue.append(new_edge_id)

                _LOGGER.debug(
                    "Branching: parent_edge=%d -> child_edge=%d branch=%d node=%d",
                    edge_id,
                    new_edge_id,
                    branch_id,
                    new_node_id,
                )

            # LEGACY MAPPING: pair the *last* two branch IDs as sisters,
            # regardless of whether two children were actually created.
            # (This mirrors the original code exactly.)
            sister_branches[branch_id - 1] = branch_id
            sister_branches[branch_id] = branch_id - 1

        _LOGGER.info("Branching complete: new edges queued=%d", len(branching_queue))
        return branching_queue, branch_id

    def _growing_generation(
        self,
        *,
        edge_queue: list[int],
        nodes: list[NDArray[np.float64]],
        edges: list[Edge],
        branches: DefaultDict[int, list[int]],
        sister_branches: dict[int, int],
        dx: float,
        w: float,
        branch_length: float,
        end_nodes: list[int],
    ) -> list[int]:
        """Advance all active tips by arc-length `branch_length` (legacy-parity), accelerating the all-pairs distance calc on the GPU when available."""
        n_steps = int(branch_length / dx)
        _LOGGER.info(
            "Growing phase: steps=%d, dx=%.6g, backend=%s",
            n_steps,
            dx,
            backend_name(),
        )

        for step in range(n_steps):
            growing_queue: list[int] = []

            while edge_queue:
                edge_id = edge_queue.pop(0)
                edge = edges[edge_id]

                b = edge.branch
                if b is None:
                    raise RuntimeError(
                        "Edge has no branch id (internal invariant violated)."
                    )

                pred = nodes[edge.n2]  # 2D (u, v) numpy array

                # --- GPU-accelerated distances to all existing nodes ---
                # NOTE: we rebuild the device array *inside* the loop to preserve legacy
                # semantics (newly added nodes in this step must be visible to later tips).
                nodes_mat_dev = xp.asarray(np.asarray(nodes, dtype=float))  # (N, 2)
                pred_dev = xp.asarray(pred, dtype=float)  # (2,)
                d_dev = xp.linalg.norm(nodes_mat_dev - pred_dev, axis=1)  # (N,)

                # Mask own-branch & sister-branch nodes to a huge value
                own_idx = xp.asarray(np.asarray(branches[b], dtype=int))
                d_dev[own_idx] = 1e9

                sister = sister_branches.get(b)
                if sister is None:
                    raise RuntimeError(f"Sister branch not found for branch {b}.")
                sis_idx = xp.asarray(np.asarray(branches[sister], dtype=int))

                sister_vals = d_dev[sis_idx].copy()
                d_dev[sis_idx] = 1e9

                # Local step-size scaling from the VTK locator (CPU)
                s, _ = self._scaling(nodes[edge.n2])

                # Collision check (min distance vs 0.9 * dx * s)
                min_dist = float(to_cpu(d_dev.min()))
                if min_dist < 0.9 * dx * s:
                    end_nodes.append(edge.n2)
                    _LOGGER.debug(
                        "Grow step %d/%d: edge=%d terminated by collision (min=%.6g, thr=%.6g)",
                        step + 1,
                        n_steps,
                        edge_id,
                        min_dist,
                        0.9 * dx * s,
                    )
                    continue

                # Restore sister distances (legacy parity)
                d_dev[sis_idx] = sister_vals

                # Nearest index (GPU -> CPU scalar)
                nearest_idx = int(to_cpu(xp.argmin(d_dev)))
                denom = float(to_cpu(d_dev[nearest_idx])) or 1e-12  # avoid zero divide

                # Gradient away from nearest disallowed point (CPU math)
                grad_dist = (pred - nodes[nearest_idx]) / denom

                new_dir = edge.dir + w * grad_dist
                new_dir /= np.linalg.norm(new_dir)

                new_node = nodes[edge.n2] + new_dir * dx * s

                # Domain check via VTK locator (CPU)
                if not self._point_in_mesh_vtk(new_node):
                    end_nodes.append(edge.n2)
                    _LOGGER.debug(
                        "Grow step %d/%d: edge=%d terminated by domain exit.",
                        step + 1,
                        n_steps,
                        edge_id,
                    )
                    continue

                # Commit new node/edge (legacy FIFO behavior)
                new_node_id = len(nodes)
                nodes.append(new_node)
                branches[b].append(new_node_id)

                next_edge_id = len(edges)
                edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, b))
                growing_queue.append(next_edge_id)

                _LOGGER.debug(
                    "Grow step %d/%d: edge=%d -> node=%d (branch=%d)",
                    step + 1,
                    n_steps,
                    edge_id,
                    new_node_id,
                    b,
                )

            # Next step iterates over newly created edges
            edge_queue = growing_queue

        return edge_queue

    def _finalize_outputs(
        self,
        *,
        nodes: list[NDArray[np.float64]],
        edges: list[Edge],
        edge_queue: list[int],
        end_nodes: list[int],
    ) -> None:
        """Finalize outputs (UV nodes, connectivity, end nodes, and XYZ mapping)."""
        # Legacy: remaining tips in the queue become end nodes
        if edge_queue:
            end_nodes.extend([edges[e].n2 for e in edge_queue])

        # Save UV-space results
        self.uv_nodes = np.array(nodes, dtype=float)
        self.edges = edges
        self.end_nodes = end_nodes
        self.connectivity = [[e.n1, e.n2] for e in edges]

        _LOGGER.info(
            "Finalize: UV nodes=%d, edges=%d, end_nodes=%d",
            len(nodes),
            len(edges),
            len(end_nodes),
        )

        # Map UV -> XYZ by barycentric interpolation of original 3D verts (legacy)
        self.nodes_xyz = []
        for node_uv in nodes:
            q3 = np.append(node_uv, 0.0)
            f, _, _ = self._eval_field(q3, self.m.verts, self.mesh_uv)
            self.nodes_xyz.append(f.astype(float))

        _LOGGER.debug(
            "Finalize: mapped %d UV nodes to XYZ (first=%s)",
            len(self.nodes_xyz),
            self.nodes_xyz[0].tolist() if self.nodes_xyz else "[]",
        )

    def grow_tree(self) -> None:
        """Generate the Purkinje fractal tree."""
        (
            nodes,
            edges,
            edge_queue,
            branches,
            sister_branches,
            end_nodes,
            branch_id,
            Rplus,
            Rminus,
            dx,
            w,
            branch_length,
            init_branch_length,
        ) = self._init_state()

        branching_edge_id = self._grow_initial_trunk(
            nodes, edges, edge_queue, branches, dx, init_branch_length
        )

        branch_id = self._spawn_fascicles(
            branching_edge_id=branching_edge_id,
            nodes=nodes,
            edges=edges,
            edge_queue=edge_queue,
            branches=branches,
            branch_id=branch_id,
            dx=dx,
        )

        for gen in range(int(self.params.N_it)):
            _LOGGER.info("Generation %d: branching phase", gen)

            edge_queue, branch_id = self._branch_generation(
                edge_queue=edge_queue,
                nodes=nodes,
                edges=edges,
                branches=branches,
                sister_branches=sister_branches,
                Rplus=Rplus,
                Rminus=Rminus,
                dx=dx,
                branch_id=branch_id,
                end_nodes=end_nodes,
            )

            _LOGGER.info("Generation %d: growing phase", gen)
            edge_queue = self._growing_generation(
                edge_queue=edge_queue,
                nodes=nodes,
                edges=edges,
                branches=branches,
                sister_branches=sister_branches,
                dx=dx,
                w=w,
                branch_length=branch_length,
                end_nodes=end_nodes,
            )

        self._finalize_outputs(
            nodes=nodes,
            edges=edges,
            edge_queue=edge_queue,
            end_nodes=end_nodes,
        )

    def save(self, filename: str) -> None:
        """Write the generated line mesh to a VTU file."""
        # Basic emptiness checks
        if not self.nodes_xyz or not self.connectivity:
            _LOGGER.error(
                "save: empty tree (nodes=%d, segments=%d).",
                len(self.nodes_xyz),
                len(self.connectivity),
            )
            raise ValueError("Cannot save: no nodes or connectivity in the tree.")

        # Normalize to CPU/NumPy
        pts = np.asarray(to_cpu(self.nodes_xyz), dtype=float)
        con = np.asarray(to_cpu(self.connectivity), dtype=int)

        # Validate shapes
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"save: nodes_xyz must be (N, 3); got shape {pts.shape}.")
        if con.ndim != 2 or con.shape[1] != 2:
            raise ValueError(
                f"save: connectivity must be (M, 2) line segments; got shape {con.shape}."
            )

        # Build and write VTU
        try:
            m = meshio.Mesh(points=pts, cells=[("line", con)])
            m.write(filename)
            _LOGGER.info(
                "save: wrote VTU '%s' (nodes=%d, segments=%d).",
                filename,
                pts.shape[0],
                con.shape[0],
            )
        except Exception:
            _LOGGER.exception("save: failed to write '%s'.", filename)
            raise
