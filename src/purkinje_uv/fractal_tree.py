"""Module defining the FractalTree class to generate fractal trees within a mesh domain.

This module implements UV-based fractal tree growth using geometric rules,
collision detection, and iterative branching to create a tree structure
embedded in a 3D mesh.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
import pyvista as pv
import vtk
from numpy.typing import NDArray

import meshio

from .mesh import Mesh
from .edge import Edge
from .fractal_tree_parameters import FractalTreeParameters


class FractalTree:
    """Fractal-tree generator on a UV-mapped surface.

    This class reproduces the legacy algorithm exactly (VTK-based point-in-mesh
    checks, UV-scaling step sizes, and queue behavior), keeping helpers local
    so the module is self-contained.
    """

    def __init__(self, params: FractalTreeParameters) -> None:
        self.params = params

        # Load 3D mesh & compute UV + UV-scaling (like legacy)
        self.m = Mesh(params.meshfile)
        print("computing uv map")
        self.m.compute_uvscaling()

        # Build a "UV mesh" flattened on z=0 with same connectivity

        assert self.m.uv is not None, "UV coordinates must be computed"
        uv = np.asarray(self.m.uv, dtype=float)
        zeros = np.zeros((uv.shape[0], 1), dtype=float)

        uv3 = np.concatenate((uv, zeros), axis=1)
        self.mesh_uv = Mesh(verts=uv3, connectivity=self.m.connectivity)

        # Prepare a VTK locator over the flat UV surface (legacy approach)
        mpv = pv.read(params.meshfile)
        mpv.points = self.mesh_uv.verts  # overwrite points with flattened UV
        self.loc = vtk.vtkCellLocator()
        self.loc.SetDataSet(mpv)
        self.loc.BuildLocator()

        # Kept for parity with legacy (even if not used by current scaling)
        self.scaling_nodes = np.array(
            self.mesh_uv.tri2node_interpolation(self.m.uvscaling)
        )

        # Outputs (filled in grow_tree)
        self.uv_nodes: NDArray[np.float64] | None = None
        self.nodes_xyz: List[NDArray[np.float64]] = []
        self.edges: List[Edge] = []
        self.end_nodes: List[int] = []
        self.connectivity: List[List[int]] = []

    # ---------- self-contained legacy helpers ----------

    @staticmethod
    def _interpolate(vectors: NDArray[Any], r: float, t: float) -> NDArray[np.float64]:
        # barycentric: v = t*v2 + r*v1 + (1 - r - t)*v0
        return (t * vectors[2] + r * vectors[1] + (1.0 - r - t) * vectors[0]).astype(
            float
        )

    def _eval_field(
        self,
        point: NDArray[Any],
        field: NDArray[Any],
        mesh: Mesh,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
        # project_new_point returns: ppoint, tri, r, t
        ppoint, tri, r, t = mesh.project_new_point(point, 5)
        val = self._interpolate(field[mesh.connectivity[tri]], float(r), float(t))
        return val, ppoint.astype(float), int(tri)

    def _point_in_mesh(self, point: NDArray[Any], mesh: Mesh) -> bool:
        q = np.append(point, 0.0)
        _, tri, _, _ = mesh.project_new_point(q, 5)
        return int(tri) >= 0

    def _point_in_mesh_vtk(self, point: NDArray[Any]) -> bool:
        q = np.append(point, 0.0)
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist = vtk.reference(0.0)
        ppoint = np.zeros(3, dtype=float)
        self.loc.FindClosestPoint(q, ppoint, cell_id, sub_id, dist)
        return bool(dist.get() < 1e-9)

    # ---------- legacy scaling that uses the locator ----------
    def scaling(self, x: NDArray[Any]) -> Tuple[float, int]:
        """Return (sqrt(uv-scaling), triangle id) at a UV point.

        The triangle id is -1 when the closest point is farther than the distance
        threshold used by the VTK locator.
        """
        q = np.append(x, 0.0)
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist = vtk.reference(0.0)
        ppoint = np.zeros(3, dtype=float)
        self.loc.FindClosestPoint(q, ppoint, cell_id, sub_id, dist)
        if dist.get() > 1e-3:
            tri = -1
        else:
            tri = int(cell_id.get())
        assert self.m.uvscaling is not None, "UV scaling must be computed"
        s = float(np.sqrt(self.m.uvscaling[int(tri)]))
        return s, int(tri)

    def grow_tree(self) -> None:
        """Generate the Purkinje fractal tree using the legacy UV algorithm."""
        branches: dict[int, list[int]] = defaultdict(list)
        branch_id = 0
        end_nodes: list[int] = []
        sister_branches: dict[int, int] = {}

        dx = float(self.params.l_segment)

        # Initial 2D UV nodes and direction
        init_node = self.mesh_uv.verts[self.params.init_node_id][:2]
        second_node = self.mesh_uv.verts[self.params.second_node_id][:2]
        s0, tri0 = self.scaling(init_node)
        if tri0 < 0:
            raise RuntimeError("the initial node is outside the domain")

        init_dir = second_node - init_node
        init_dir /= np.linalg.norm(init_dir)

        nodes: List[NDArray[np.float64]] = [
            init_node.astype(float),
            (init_node + dx * init_dir * s0).astype(float),
        ]
        edges: List[Edge] = [Edge(0, 1, nodes, None, branch_id)]

        edge_queue: List[int] = [0]
        branches[branch_id].append(0)

        branch_length = float(self.params.length)
        init_branch_length = float(self.params.init_length)

        theta = float(self.params.branch_angle)
        w = float(self.params.w)

        Rplus = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=float,
        )
        Rminus = np.array(
            [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]],
            dtype=float,
        )

        # --- grow initial trunk (legacy) ---
        for _ in range(int(init_branch_length / dx)):
            edge_id = edge_queue.pop(0)
            edge = edges[edge_id]
            new_dir = edge.dir / np.linalg.norm(edge.dir)
            s, tri = self.scaling(nodes[edge.n2])
            if tri < 0:
                raise RuntimeError("the initial branch goes out of the domain")
            new_node = nodes[edge.n2] + new_dir * dx * s
            new_node_id = len(nodes)
            nodes.append(new_node)
            branches[edge.branch].append(new_node_id)  # type: ignore[index]
            edge_queue.append(len(edges))
            edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, edge.branch))

        branching_edge_id = edge_queue.pop(0)

        # --- fascicles (legacy) ---
        for fasc_len, fasc_ang in zip(
            self.params.fascicles_length, self.params.fascicles_angles
        ):
            Rotation = np.array(
                [
                    [np.cos(fasc_ang), -np.sin(fasc_ang)],
                    [np.sin(fasc_ang), np.cos(fasc_ang)],
                ],
                dtype=float,
            )
            edge = edges[branching_edge_id]
            new_dir = Rotation @ edge.dir
            new_dir /= np.linalg.norm(new_dir)
            s, tri = self.scaling(nodes[edge.n2])
            if tri < 0:
                raise RuntimeError("the fascicle goes out of the domain")
            new_node = nodes[edge.n2] + new_dir * dx * s
            new_node_id = len(nodes)
            nodes.append(new_node)
            branch_id += 1
            branches[branch_id].append(new_node_id)
            edge_queue.append(len(edges))
            edges.append(
                Edge(edge.n2, new_node_id, nodes, parent=edge_id, branch=branch_id)
            )
            for _ in range(int(fasc_len / dx)):
                edge_id = edge_queue.pop(0)
                edge = edges[edge_id]
                new_dir = edge.dir / np.linalg.norm(edge.dir)
                s, tri = self.scaling(nodes[edge.n2])
                if tri < 0:
                    raise RuntimeError("the fascicle goes out of the domain")
                new_node = nodes[edge.n2] + new_dir * dx * s
                new_node_id = len(nodes)
                nodes.append(new_node)
                branches[edge.branch].append(new_node_id)  # type: ignore[index]
                edge_queue.append(len(edges))
                edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, edge.branch))

        # --- generations (legacy) ---
        for gen in range(int(self.params.N_it)):
            print("generation", gen)
            # Branching step
            branching_queue: List[int] = []
            while edge_queue:
                edge_id = edge_queue.pop(0)
                edge = edges[edge_id]
                for R in (Rplus, Rminus):
                    new_dir = R @ edge.dir
                    new_dir /= np.linalg.norm(new_dir)
                    s, _ = self.scaling(nodes[edge.n2])
                    new_node = nodes[edge.n2] + new_dir * dx * s
                    if not self._point_in_mesh_vtk(new_node):
                        end_nodes.append(edge.n2)
                        continue
                    new_node_id = len(nodes)
                    nodes.append(new_node)
                    branching_queue.append(len(edges))
                    branch_id += 1
                    branches[branch_id].append(new_node_id)
                    edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, branch_id))
                sister_branches[branch_id - 1] = branch_id
                sister_branches[branch_id] = branch_id - 1

            edge_queue = branching_queue

            # Growing step
            for _ in range(int(branch_length / dx)):
                growing_queue: List[int] = []
                new_nodes_batch: List[NDArray[np.float64]] = []

                while edge_queue:
                    edge_id = edge_queue.pop(0)
                    edge = edges[edge_id]

                    pred = nodes[edge.n2]
                    # vectorized "collision" distances, excluding own branch & sister
                    all_dist = np.linalg.norm(nodes - pred, axis=1)
                    all_dist[branches[edge.branch]] = 1e9  # type: ignore[index]
                    sister = sister_branches[edge.branch]  # type: ignore[index]
                    sister_vals = all_dist[branches[sister]]
                    all_dist[branches[sister]] = 1e9

                    s, _ = self.scaling(nodes[edge.n2])
                    if all_dist.min() < 0.9 * dx * s:
                        end_nodes.append(edge.n2)
                        continue

                    # gradient = direction away from nearest disallowed point
                    all_dist[branches[sister]] = sister_vals  # restore sister
                    nearest_idx = int(np.argmin(all_dist))
                    grad_dist = (pred - nodes[nearest_idx]) / float(
                        all_dist[nearest_idx]
                    )

                    new_dir = edge.dir + w * grad_dist
                    new_dir /= np.linalg.norm(new_dir)
                    new_node = nodes[edge.n2] + new_dir * dx * s
                    if not self._point_in_mesh_vtk(new_node):
                        end_nodes.append(edge.n2)
                        continue

                    new_node_id = len(nodes)
                    nodes.append(new_node)
                    new_nodes_batch.append(new_node)
                    growing_queue.append(len(edges))
                    branches[edge.branch].append(new_node_id)  # type: ignore[index]
                    edges.append(
                        Edge(edge.n2, new_node_id, nodes, edge_id, edge.branch)
                    )

                edge_queue = growing_queue

        end_nodes += [edges[e].n2 for e in edge_queue]

        self.uv_nodes = np.array(nodes, dtype=float)
        self.edges = edges
        self.end_nodes = end_nodes
        self.connectivity = [[e.n1, e.n2] for e in edges]

        # Map UV -> XYZ by barycentric interpolation of original 3D verts (legacy)
        self.nodes_xyz = []
        for node_uv in nodes:
            q3 = np.append(node_uv, 0.0)
            f, _, _ = self._eval_field(q3, self.m.verts, self.mesh_uv)
            self.nodes_xyz.append(f.astype(float))

    def save(self, filename: str) -> None:
        """Write the generated line mesh to a VTU file."""
        line = meshio.Mesh(
            np.array(self.nodes_xyz, dtype=float),
            [("line", np.array(self.connectivity, dtype=int))],
        )
        line.write(filename)
