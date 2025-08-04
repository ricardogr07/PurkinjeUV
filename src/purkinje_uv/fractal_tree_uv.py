"""Module defining the FractalTree class to generate fractal trees within a mesh domain.

This module implements UV-based fractal tree growth using geometric rules,
collision detection, and iterative branching to create a tree structure
embedded in a 3D mesh.
"""

import logging
from collections import defaultdict
from typing import Any, Tuple, List, Dict, DefaultDict, Optional, Sequence
from numpy.typing import NDArray

import meshio
import numpy as np
import pyvista as pv
import vtk

from .edge import Edge
from .mesh import Mesh
from .fractal_tree_parameters import Parameters

logger = logging.getLogger(__name__)


class FractalTree:
    """Generate a fractal tree structure within a mesh using UV mapping.

    This class handles:
    - Loading and embedding a mesh in UV space.
    - Iterative growth of a main branch and fascicles.
    - Collision avoidance via KD-trees.
    - Reconstruction of 3D node coordinates from UV.
    - Exporting the final tree as a line mesh.

    Attributes:
        mesh (Mesh): Original 3D mesh loaded from file.
        mesh_uv (Mesh): Mesh mapped into UV coordinates, embedded as 3D.
        loc (vtk.vtkCellLocator): VTK locator for mesh spatial queries.
        scaling_nodes (NDArray[Any]): Node-wise UV scaling factors.
        params (Parameters): Growth parameters and mesh settings.
        uv_nodes (NDArray[Any]): Final node coordinates in UV space.
        edges (List[Edge]): List of edges defining the tree connectivity.
        end_nodes (List[int]): Indices of terminal (leaf) nodes.
        connectivity (List[List[int]]): Edge pairs for mesh export.
        nodes_xyz (List[NDArray[Any]]): Final node coordinates in 3D space.
    """

    mesh: Mesh
    mesh_uv: Mesh
    loc: vtk.vtkCellLocator
    scaling_nodes: NDArray[Any]
    params: Parameters
    uv_nodes: NDArray[Any]
    edges: List[Edge]
    end_nodes: List[int]
    connectivity: List[List[int]]
    nodes_xyz: List[NDArray[Any]]

    def __init__(self, params: Parameters) -> None:
        """Initialize the fractal tree generator with given parameters.

        Args:
            params (Parameters): Configuration object with meshfile, branch lengths,
                angles, and other settings.

        Raises:
            TypeError: If `params` is not a Parameters instance.
            ValueError: If `params.meshfile` is not provided.
            RuntimeError: On missing UV map or scaling after computation.
        """
        if not isinstance(params, Parameters):
            raise TypeError(
                "The parameters must be an instance of the Parameters class"
            )
        if not getattr(params, "meshfile", None):
            raise ValueError("Parameter 'meshfile' must be provided")
        logger.info("Initializing FractalTree with parameters: %r", params)
        self.params = params

        # Load mesh
        meshfile = params.meshfile
        assert meshfile is not None, "Parameter 'meshfile' must be provided"
        self.mesh: Mesh = Mesh(meshfile)

        # Compute UV scaling
        logger.info("Computing UV map")
        self.mesh.compute_uvscaling()

        # Embed UV into 3D
        if self.mesh.uv is None:
            raise RuntimeError("UV map missing after compute_uvscaling")
        uv_2d: NDArray[Any] = self.mesh.uv
        uv_3d: NDArray[Any] = np.concatenate(
            (uv_2d, np.zeros((uv_2d.shape[0], 1), dtype=float)), axis=1
        )
        self.mesh_uv: Mesh = Mesh(
            verts=uv_3d,
            connectivity=self.mesh.connectivity,
        )
        logger.info("UV‐embedded mesh created: %d verts", self.mesh_uv.verts.shape[0])

        # Build VTK locator
        mpv: Any = pv.read(meshfile)
        mpv.points = self.mesh_uv.verts
        self.loc = vtk.vtkCellLocator()
        self.loc.SetDataSet(mpv)
        self.loc.BuildLocator()

        # Compute scaling nodes
        if self.mesh.uvscaling is None:
            raise RuntimeError("UV scaling missing")
        scaling_list: List[float] = self.mesh_uv.tri2node_interpolation(
            self.mesh.uvscaling
        )
        self.scaling_nodes: NDArray[Any] = np.array(scaling_list, dtype=float)

    def _interpolate(
        self,
        vectors: Sequence[NDArray[Any]],
        r: float,
        t: float,
    ) -> NDArray[Any]:
        """Interpolate three vectors using barycentric weights.

        Args:
            vectors (Sequence[NDArray[Any]]): List of three vectors.
            r (float): Weight for the second vector.
            t (float): Weight for the third vector.

        Returns:
            NDArray[Any]: Weighted sum: t*vectors[2] + r*vectors[1] + (1−r−t)*vectors[0].
        """
        result: NDArray[Any] = (
            t * vectors[2] + r * vectors[1] + (1 - r - t) * vectors[0]
        )
        return result

    def _eval_field(
        self,
        point: NDArray[Any],
        field: NDArray[Any],
        mesh: Mesh,
    ) -> Tuple[NDArray[Any], NDArray[Any], int]:
        """Project a point onto `mesh` and interpolate `field` there.

        Args:
            point (NDArray[Any]): (3,)-array of XYZ coordinates.
            field (NDArray[Any]): (n_nodes, …) array of values at mesh nodes.
            mesh (Mesh): Mesh supporting `project_new_point`.

        Returns:
            Tuple[NDArray[Any], NDArray[Any], int]:
                - interpolated field value at projection,
                - projected 3D point,
                - triangle index containing the projection.
        """
        ppoint, tri, r, t = mesh.project_new_point(point, 5)
        interpolated: NDArray[Any] = self._interpolate(
            field[mesh.connectivity[tri]], r, t
        )
        return interpolated, ppoint, tri

    def _point_in_mesh(self, point: NDArray[Any], mesh: Mesh) -> bool:
        """Check if a 2D point lies inside `mesh` by barycentric projection.

        Args:
            point (NDArray[Any]): (2,)-array of UV coordinates.
            mesh (Mesh): Mesh for testing.

        Returns:
            bool: True if the point is inside, False otherwise.
        """
        point_3d: NDArray[Any] = np.append(point, np.zeros(1, dtype=float))
        _, tri, _, _ = mesh.project_new_point(point_3d, 5)
        return tri >= 0

    def _point_in_mesh_vtk(
        self,
        point: NDArray[Any],
        loc: vtk.vtkCellLocator,
    ) -> bool:
        """Check mesh containment using VTK’s cell locator.

        Args:
            point (NDArray[Any]): (2 or 3)-array of coordinates.
            loc (vtk.vtkCellLocator): Locator built on the UV-mesh.

        Returns:
            bool: True if within tolerance of a cell, False otherwise.
        """
        # Ensure the point is 3D: if 2D, append zero z-coordinate
        point_3d: NDArray[Any] = np.append(point, np.zeros(1, dtype=float))

        cellId = vtk.reference(0)
        subId = vtk.reference(0)
        d = vtk.reference(0.0)
        ppoint: NDArray[Any] = np.zeros(3, dtype=float)

        loc.FindClosestPoint(point_3d, ppoint, cellId, subId, d)
        inside: bool = bool(d.get() < 1e-9)
        return inside

    def _scaling(self, x: NDArray[Any]) -> Tuple[float, int]:
        """Compute scaling factor and triangle index for UV point `x`.

        Args:
            x (NDArray[Any]): (2,)-array of UV coordinates.

        Returns:
            Tuple[float, int]: (sqrt(scaling), triangle index) or (-1.0, -1) if outside.
        """
        x3d: NDArray[Any] = np.append(x, np.zeros(1, dtype=float))
        cellId = vtk.reference(0)
        subId = vtk.reference(0)
        d = vtk.reference(0.0)
        ppoint: NDArray[Any] = np.zeros(3, dtype=float)

        # Ensure UV scaling was computed so indexing is safe
        assert self.mesh.uvscaling is not None, "UV scaling must be initialized"
        # Single declaration of tri avoids multiple redeclarations
        tri: int

        self.loc.FindClosestPoint(x3d, ppoint, cellId, subId, d)
        if d.get() > 1e-3:
            tri = -1
        else:
            tri = cellId.get()

        scaling_value: float = float(np.sqrt(self.mesh.uvscaling[tri]))
        return scaling_value, tri

    def _add_node(
        self,
        nodes: List[NDArray[Any]],
        new_node: NDArray[Any],
    ) -> int:
        """Append `new_node` to `nodes` list and return its index."""
        nodes.append(new_node)
        return len(nodes) - 1

    def _compute_new_direction(
        self,
        dir: NDArray[Any],
        rotation: NDArray[Any],
        grad_dist: Optional[NDArray[Any]] = None,
        w: float = 0.0,
    ) -> NDArray[Any]:
        """Rotate `dir` by `rotation` and optionally add `w*grad_dist`, then normalize."""
        result: NDArray[Any] = np.matmul(rotation, dir)
        if grad_dist is not None and w != 0.0:
            result = result + w * grad_dist
        normed: NDArray[Any] = result / np.linalg.norm(result)
        return normed

    def _can_grow_to(
        self,
        new_node: NDArray[Any],
        loc: vtk.vtkCellLocator,
    ) -> bool:
        """Return True if `new_node` lies inside the UV-mesh via VTK locator."""
        return self._point_in_mesh_vtk(new_node, loc)

    def _collision_check(
        self,
        nodes: Sequence[NDArray[Any]],
        branches: DefaultDict[int, List[int]],
        sister_branches: Dict[int, int],
        edge: Edge,
        dx: float,
        s: float,
    ) -> Tuple[bool, Optional[NDArray[Any]]]:
        """Test whether extending `edge` by `dx*s` collides with existing nodes.

        Returns:
            Tuple[bool, Optional[NDArray[Any]]]:
                - True if collision detected (no gradient).
                - False and a repulsion gradient vector otherwise.
        """
        pred_node: NDArray[Any] = nodes[edge.n2]  # predicted new point location
        # Stack nodes for vectorized distance computation
        nodes_arr: NDArray[Any] = np.vstack(nodes)
        all_dist: NDArray[Any] = np.linalg.norm(nodes_arr - pred_node, axis=1)

        # Ignore self-branch and sister-branch nodes in collision detection
        branch_id: Optional[int] = edge.branch
        if branch_id is not None:
            all_dist[branches[branch_id]] = 1e9
            sister: Optional[int] = sister_branches.get(branch_id)
            if sister is not None:
                all_dist[branches[sister]] = 1e9

        min_dist: float = float(all_dist.min())
        min_index: int = int(all_dist.argmin())

        # If too close: mark as collision
        if min_dist < 0.9 * dx * s:
            return True, None

        # Compute gradient direction (unit vector)
        grad_dist: NDArray[Any] = (pred_node - nodes_arr[min_index]) / min_dist
        return False, grad_dist

    def _grow_initial_branch(
        self,
        edge_queue: List[int],
        edges: List[Edge],
        nodes: List[NDArray[Any]],
        branches: DefaultDict[int, List[int]],
        dx: float,
        init_branch_length: float,
    ) -> int:
        """Grow and queue nodes along the main branch for length `init_branch_length`.

        Args:
            edge_queue (List[int]): Queue of edge indices to be processed.
            edges (List[Edge]): List of existing edges in the tree.
            nodes (List[np.ndarray]): List of node coordinates.
            branches (defaultdict): Dictionary mapping branch indices to lists of node indices.
            dx (float): Step size for growing the branch.
            init_branch_length (float): Total length to grow the initial branch.

        Returns:
            int: The ID of the initial edge that was grown.

        Raises:
            ValueError: If the initial branch goes out of the domain as determined by the `_scaling` method.
        """
        edge_id: int = edge_queue.pop(0)
        edge: Edge = edges[edge_id]
        for _ in range(int(init_branch_length / dx)):
            s, tri = self._scaling(nodes[edge.n2])
            if tri < 0:
                raise ValueError("the initial branch goes out of the domain")
            new_node: NDArray[Any] = nodes[edge.n2] + edge.dir * dx * s
            new_node_id: int = self._add_node(nodes, new_node)

            # Retrieve and assert a concrete branch_id before use
            branch_id = edge.branch
            assert branch_id is not None, "Edge.branch must be set"

            branches[branch_id].append(new_node_id)
            edge_queue.append(len(edges))
            # Create new Edge with concrete int branch_id
            edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, branch_id))
        return edge_id

    def _grow_fascicles(
        self,
        branching_edge_id: int,
        edges: List[Edge],
        nodes: List[NDArray[Any]],
        branches: DefaultDict[int, List[int]],
        edge_queue: List[int],
        branch_id: int,
        dx: float,
    ) -> None:
        """Branch off short fascicles at `branching_edge_id`, then queue their growth.

        Args:
            branching_edge_id (int): The index of the edge from which fascicles will branch.
            edges (list): List of Edge objects representing the current edges in the tree.
            nodes (list): List of node coordinates (e.g., numpy arrays) representing the tree nodes.
            branches (dict): Dictionary mapping branch IDs to lists of node indices.
            edge_queue (list): Queue of edge indices to process for fascicle growth.
            branch_id (int): The current branch identifier to assign new fascicles.
            dx (float): The step size for fascicle growth.

        Raises:
            ValueError: If a fascicle grows outside the domain (tri < 0).

        Returns:
            None

        Notes:
            - Fascicle lengths and angles are taken from self.params.fascicles_length and self.params.fascicles_angles.
            - New nodes and edges are added to the tree as fascicles are grown.
            - The function modifies the edges, nodes, branches, and edge_queue in-place.
        """
        for fascicle_length, fascicles_angle in zip(
            self.params.fascicles_length, self.params.fascicles_angles
        ):
            # Build 2D rotation matrix for this fascicle
            Rotation: NDArray[Any] = np.array(
                [
                    [np.cos(fascicles_angle), -np.sin(fascicles_angle)],
                    [np.sin(fascicles_angle), np.cos(fascicles_angle)],
                ],
                dtype=float,
            )
            edge: Edge = edges[branching_edge_id]
            new_dir: NDArray[Any] = self._compute_new_direction(edge.dir, Rotation)
            s: float
            tri: int
            s, tri = self._scaling(nodes[edge.n2])
            if tri < 0:
                raise ValueError("The fascicle goes out of the domain.")
            new_node: NDArray[Any] = nodes[edge.n2] + new_dir * dx * s
            new_node_id: int = self._add_node(nodes, new_node)
            branch_id += 1
            branches[branch_id].append(new_node_id)
            edge_queue.append(len(edges))
            edges.append(
                Edge(edge.n2, new_node_id, nodes, branching_edge_id, edge.branch)
            )

            # Grow the fascicle along its length
            for _ in range(int(fascicle_length / dx)):
                edge_id: int = edge_queue.pop(0)
                edge = edges[edge_id]

                # For inner segments, no additional rotation
                inner_rotation: NDArray[Any] = np.eye(2, dtype=float)
                new_dir = self._compute_new_direction(edge.dir, inner_rotation)
                s, tri = self._scaling(nodes[edge.n2])
                if tri < 0:
                    raise ValueError("The fascicle goes out of the domain.")
                new_node = nodes[edge.n2] + new_dir * dx * s
                new_node_id = self._add_node(nodes, new_node)

                # Ensure branch is not None before using it as a dict key
                branch_inner = edge.branch
                assert branch_inner is not None, "Edge.branch must be set"
                branches[branch_inner].append(new_node_id)
                edge_queue.append(len(edges))

                edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, branch_inner))

    def _grow_generations(
        self,
        edges: List[Edge],
        nodes: List[NDArray[Any]],
        branches: DefaultDict[int, List[int]],
        edge_queue: List[int],
        branch_id: int,
        dx: float,
        branch_length: float,
        w: float,
        Rplus: NDArray[Any],
        Rminus: NDArray[Any],
        end_nodes: List[int],
        sister_branches: Dict[int, int],
    ) -> None:
        """Iterate branching and growth steps for `self.params.N_it` generations."""
        for gen in range(self.params.N_it):
            logger.info(f"Generation {gen}")
            branching_queue: List[int] = []
            # Branching step
            while edge_queue:
                edge_id: int = edge_queue.pop(0)
                edge: Edge = edges[edge_id]
                for R in (Rplus, Rminus):
                    new_dir: NDArray[Any] = self._compute_new_direction(edge.dir, R)
                    s: float
                    tri: int
                    s, tri = self._scaling(nodes[edge.n2])
                    new_node: NDArray[Any] = nodes[edge.n2] + new_dir * dx * s
                    if not self._can_grow_to(new_node, self.loc):
                        end_nodes.append(edge.n2)
                        continue
                    new_node_id: int = self._add_node(nodes, new_node)
                    branching_queue.append(len(edges))
                    branch_id += 1
                    branches[branch_id].append(new_node_id)
                    edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, branch_id))
                sister_branches[branch_id - 1] = branch_id
                sister_branches[branch_id] = branch_id - 1

            # Prepare for the growing step
            edge_queue[:] = branching_queue

            # Growing step
            for _ in range(int(branch_length / dx)):
                growing_queue: List[int] = []
                while edge_queue:
                    edge_id = edge_queue.pop(0)
                    edge = edges[edge_id]
                    s, tri = self._scaling(nodes[edge.n2])
                    collision, grad_dist = self._collision_check(
                        nodes, branches, sister_branches, edge, dx, s
                    )
                    if collision:
                        end_nodes.append(edge.n2)
                        continue
                    new_dir = self._compute_new_direction(
                        edge.dir, np.eye(2, dtype=float), grad_dist, w
                    )
                    new_node = nodes[edge.n2] + new_dir * dx * s
                    if not self._can_grow_to(new_node, self.loc):
                        end_nodes.append(edge.n2)
                        continue
                    new_node_id = self._add_node(nodes, new_node)
                    growing_queue.append(len(edges))

                    # Ensure branch is not None before indexing
                    branch_inner = edge.branch
                    assert branch_inner is not None, "Edge.branch must be set"
                    branches[branch_inner].append(new_node_id)

                    # Create new Edge with concrete branch_inner
                    edges.append(
                        Edge(edge.n2, new_node_id, nodes, edge_id, branch_inner)
                    )

                edge_queue[:] = growing_queue

    def grow_tree(self) -> None:
        """Run the full sequence: initial branch, fascicles, generations, and 3D mapping."""
        # Initialization
        branches: DefaultDict[int, List[int]] = defaultdict(list)
        branch_id: int = 0
        end_nodes: List[int] = []
        sister_branches: Dict[int, int] = {}
        dx: float = self.params.l_segment

        # Initial nodes & direction
        init_node: NDArray[Any] = self.mesh_uv.verts[self.params.init_node_id][:2]
        second_node: NDArray[Any] = self.mesh_uv.verts[self.params.second_node_id][:2]
        s: float
        tri: int
        s, tri = self._scaling(init_node)
        if tri < 0:
            raise ValueError("The initial node is outside the domain")
        init_dir: NDArray[Any] = second_node - init_node
        init_dir = init_dir / np.linalg.norm(init_dir)

        nodes: List[NDArray[Any]] = [
            init_node,
            init_node + dx * init_dir * s,
        ]
        edges: List[Edge] = [Edge(0, 1, nodes, None, branch_id)]
        edge_queue: List[int] = [0]
        branches[branch_id].append(0)

        branch_length: float = self.params.length
        init_branch_length: float = self.params.init_length
        theta: float = self.params.branch_angle
        w: float = self.params.w

        Rplus: NDArray[Any] = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=float,
        )
        Rminus: NDArray[Any] = np.array(
            [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]],
            dtype=float,
        )

        # Grow the main branch
        branching_edge_id = self._grow_initial_branch(
            edge_queue, edges, nodes, branches, dx, init_branch_length
        )

        # Add fascicles
        self._grow_fascicles(
            branching_edge_id, edges, nodes, branches, edge_queue, branch_id, dx
        )

        # Iterate generations
        self._grow_generations(
            edges,
            nodes,
            branches,
            edge_queue,
            branch_id,
            dx,
            branch_length,
            w,
            Rplus,
            Rminus,
            end_nodes,
            sister_branches,
        )

        # Finalize end nodes & connectivity
        end_nodes.extend([edges[e].n2 for e in edge_queue])
        self.uv_nodes: NDArray[Any] = np.array(nodes, dtype=float)
        self.edges: List[Edge] = edges
        self.end_nodes: List[int] = end_nodes
        self.connectivity: List[List[int]] = [[ed.n1, ed.n2] for ed in edges]

        # Map UV nodes back to 3D coordinates
        self.nodes_xyz: List[NDArray[Any]] = []
        for node in nodes:
            node_3d: NDArray[Any] = np.append(node, np.zeros(1, dtype=float))
            val, _, _ = self._eval_field(node_3d, self.mesh.verts, self.mesh_uv)
            self.nodes_xyz.append(val)

    def save(self, filename: str) -> None:
        """Write the final tree connectivity to `filename` using meshio."""
        try:
            # Convert stored node coordinates and edge connectivity to typed arrays
            line_nodes: NDArray[Any] = np.array(self.nodes_xyz, dtype=float)
            connectivity_arr: NDArray[Any] = np.array(self.connectivity, dtype=int)

            # Prepare meshio cells list: one cell block of type 'line'
            cells: List[Tuple[str, NDArray[Any]]] = [("line", connectivity_arr)]

            # Create and write the mesh
            mesh_out = meshio.Mesh(points=line_nodes, cells=cells)
            mesh_out.write(filename)
            logger.info("Fractal tree mesh successfully saved to %s", filename)
        except Exception as e:
            logger.error("Failed to save fractal tree mesh to %s: %s", filename, e)
