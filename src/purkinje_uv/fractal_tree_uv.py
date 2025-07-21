import logging
from collections import defaultdict
from typing import Any, Tuple, List

import meshio
import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree

from .edge import Edge
from .mesh import Mesh

class FractalTree:
    """
    FractalTree generates a fractal tree structure within a given mesh domain using UV mapping and geometric rules.
    Attributes
    m : Mesh
        The mesh object loaded from the provided mesh file.
    mesh_uv : Mesh
        The mesh object in UV space.
    loc : vtk.vtkCellLocator
        VTK cell locator for efficient spatial queries.
    scaling_nodes : np.ndarray
        Array of scaling factors for mesh nodes.
    params : Any
        Parameters for tree growth and mesh configuration.
    uv_nodes : np.ndarray
        Array of node coordinates in UV space.
    edges : List[Edge]
        List of edges representing the tree branches.
    end_nodes : List[int]
        List of indices of terminal nodes.
    connectivity : List[List[int]]
        List of edge connectivity pairs.
    nodes_xyz : List[np.ndarray]
        List of node coordinates in XYZ space.
    Methods
    -------
    grow_tree() -> None
        Generates the fractal tree structure by iteratively growing and branching according to geometric and collision rules.
    save(filename: str) -> None
    """

    def __init__(self, params: Any) -> None:
        """
        Initializes the fractal tree UV mapping object.
        Args:
            params (Any): An object containing parameters for mesh file path and other settings.
        Attributes:
            m (Mesh): The original mesh loaded from the file specified in params.
            mesh_uv (Mesh): A mesh object with UV coordinates extended to 3D.
            loc (vtk.vtkCellLocator): VTK cell locator for spatial queries on the mesh.
            scaling_nodes (np.ndarray): Array of node scaling values interpolated from the UV scaling.
            params (Any): Stores the input parameters for later use.
        Side Effects:
            - Prints 'computing uv map' to the console.
            - Builds a VTK cell locator for the mesh.
        Raises:
            Any exceptions raised by Mesh, pv.read, or VTK methods will propagate.
        """
        self.mesh = Mesh(params.meshfile)
        print('computing uv map')
        self.mesh.compute_uvscaling()

        self.mesh_uv = Mesh(verts = np.concatenate((self.mesh.uv,np.zeros((self.mesh.uv.shape[0],1))), axis =1), connectivity= self.mesh.connectivity)
        mpv = pv.read(params.meshfile)
        mpv.points = self.mesh_uv.verts
        self.loc = vtk.vtkCellLocator()
        self.loc.SetDataSet(mpv)
        self.loc.BuildLocator()
        self.scaling_nodes = np.array(self.mesh_uv.tri2node_interpolation(self.m.uvscaling))
        self.params = params
    
    def _interpolate(self, vectors: np.ndarray, r: float, t: float) -> np.ndarray:
        """
        Interpolates between three vectors using barycentric coordinates.

        Args:
            vectors (list or array-like): A sequence of three vectors to interpolate between.
            r (float): The barycentric coordinate corresponding to the second vector.
            t (float): The barycentric coordinate corresponding to the third vector.

        Returns:
            The interpolated vector as a linear combination of the input vectors.

        Note:
            The first vector's weight is computed as (1 - r - t).
        """
        return t*vectors[2] + r*vectors[1] + (1-r-t)*vectors[0]

    def _eval_field(
        self,
        point: np.ndarray,
        field: np.ndarray,
        mesh: Mesh
    ) -> Tuple[Any, Any, int]:
        """
        Evaluates the field at a given point by projecting the point onto the mesh and interpolating the field value.

        Args:
            point (np.ndarray): The coordinates of the point where the field is to be evaluated.
            field (np.ndarray): The field values defined on the mesh nodes.
            mesh (Mesh): The mesh object containing connectivity and projection methods.

        Returns:
            Tuple[Any, Any, int]: A tuple containing:
                - The interpolated field value at the projected point.
                - The projected point coordinates.
                - The index of the triangle in the mesh where the point was projected.
        """
        ppoint,tri,r,t = mesh.project_new_point(point, 5)
        return self._interpolate(field[mesh.connectivity[tri]], r, t), ppoint, tri

    def _point_in_mesh(self, point: np.ndarray, mesh: Mesh) -> bool:
        """
        Determines whether a given point is inside the specified mesh.

        Args:
            point (np.ndarray): The 2D coordinates of the point to check.
            mesh (Mesh): The mesh object to test against.

        Returns:
            bool: True if the point is inside the mesh, False otherwise.
        """
        point = np.append(point, np.zeros(1))
        _,tri,_,_ = mesh.project_new_point(point, 5)
        return tri >= 0

    def _point_in_mesh_vtk(self, point: np.ndarray, loc: vtk.vtkCellLocator) -> bool:
        """
        Determines whether a given point is inside a mesh using VTK's cell locator.

        Args:
            point (np.ndarray): The 3D coordinates of the point to check.
            loc (vtk.vtkCellLocator): The VTK cell locator associated with the mesh.

        Returns:
            bool: True if the point is inside the mesh (within a tolerance), False otherwise.
        """
        point = np.append(point, np.zeros(1))
        cellId = vtk.reference(0)
        subId  = vtk.reference(0)
        d = vtk.reference(0.0)
        ppoint = np.zeros(3)
        loc.FindClosestPoint(point, ppoint, cellId, subId, d)
        return d.get() < 1e-9

    def _scaling(self, x: np.ndarray) -> tuple[float, int]:
        """
        Calculates the scaling factor and triangle index for a given point.
        Appends a zero to the input array `x`, finds the closest point on the mesh using VTK,
        and determines the corresponding triangle index. If the distance to the closest point
        is greater than 1e-3, returns -1 as the triangle index. Otherwise, returns the scaling
        factor from `self.mesh.uvscaling` for the found triangle.
        Args:
            x (np.ndarray): Input point coordinates.
        Returns:
            tuple[float, int]: A tuple containing the square root of the scaling factor and the triangle index.
        """
        x = np.append(x, np.zeros(1))
        cellId = vtk.reference(0)
        subId  = vtk.reference(0)
        d = vtk.reference(0.0)
        ppoint = np.zeros(3)
        self.loc.FindClosestPoint(x, ppoint, cellId, subId, d)
        if d.get() > 1e-3:
            tri = -1
        else:
            tri = cellId.get()

        return np.sqrt(self.mesh.uvscaling[tri]), tri

    def _add_node(self, nodes: List[np.ndarray], new_node: np.ndarray) -> int:
        """
        Adds a new node to the nodes list and returns its index.
        Args:
            nodes (List[np.ndarray]): List of existing nodes.
            new_node (np.ndarray): The new node to add.
        Returns:
            int: The index of the newly added node.
        """
        nodes.append(new_node)
        return len(nodes) - 1

    def _compute_new_direction(self, dir: np.ndarray, rotation: np.ndarray, grad_dist: np.ndarray = None, w: float = 0.0) -> np.ndarray:
        """
        Computes a new direction vector by applying a rotation and optionally adding a weighted gradient.
        Args:
            dir (np.ndarray): The current direction vector.
            rotation (np.ndarray): The rotation matrix to apply.
            grad_dist (np.ndarray, optional): Gradient direction for collision avoidance.
            w (float, optional): Weight for the gradient direction.
        Returns:
            np.ndarray: The normalized new direction vector.
        """
        new_dir = np.matmul(rotation, dir)
        if grad_dist is not None and w != 0.0:
            new_dir = new_dir + w * grad_dist
        return new_dir / np.linalg.norm(new_dir)

    def _can_grow_to(self, new_node: np.ndarray, loc: vtk.vtkCellLocator) -> bool:
        """
        Checks if a new node can be grown to (i.e., is inside the mesh domain).
        Args:
            new_node (np.ndarray): The candidate node position.
            loc (vtk.vtkCellLocator): The mesh locator.
        Returns:
            bool: True if the node is inside the mesh, False otherwise.
        """
        return self._point_in_mesh_vtk(new_node, loc)

    def _grow_initial_branch(
        self,
        edge_queue: List[int],
        edges: List[Edge],
        nodes: List[np.ndarray],
        branches: defaultdict,
        dx: float,
        init_branch_length: float
    ) -> int:
        """
        Grows the initial branch of the fractal tree by iteratively adding new nodes and edges.

        This method pops the first edge from the edge queue and extends it by repeatedly adding new nodes
        along the direction of the edge, scaled by a factor determined by the `_scaling` method. Each new node
        is appended to the corresponding branch and a new edge is created and added to the edge queue.

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
        edge_id = edge_queue.pop(0)
        edge = edges[edge_id]
        for _ in range(int(init_branch_length / dx)):
            s, tri = self._scaling(nodes[edge.n2])
            if tri < 0:
                raise ValueError("the initial branch goes out of the domain")
            new_node = nodes[edge.n2] + edge.dir * dx * s
            new_node_id = self._add_node(nodes, new_node)
            branches[edge.branch].append(new_node_id)
            edge_queue.append(len(edges))
            edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, edge.branch))
        return edge_id

    def _grow_fascicles(
        self,
        branching_edge_id: int,
        edges: list,
        nodes: list,
        branches: dict,
        edge_queue: list,
        branch_id: int,
        dx: float
    ) -> None:
        """
        Grows fascicles from a given branching edge in a fractal tree structure.

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
        for fascicle_length, fascicles_angle in zip(self.params.fascicles_length, self.params.fascicles_angles):
            Rotation = np.array([
                [np.cos(fascicles_angle), -np.sin(fascicles_angle)],
                [np.sin(fascicles_angle),  np.cos(fascicles_angle)]
            ])
            edge = edges[branching_edge_id]
            new_dir = self._compute_new_direction(edge.dir, Rotation)
            s, tri = self._scaling(nodes[edge.n2])
            if tri < 0:
                raise ValueError("the fascicle goes out of the domain")
            new_node = nodes[edge.n2] + new_dir * dx * s
            new_node_id = self._add_node(nodes, new_node)
            branch_id += 1
            branches[branch_id].append(new_node_id)
            edge_queue.append(len(edges))
            edges.append(Edge(edge.n2, new_node_id, nodes, branching_edge_id, edge.branch))
            # Grow the fascicle
            for _ in range(int(fascicle_length / dx)):
                edge_id = edge_queue.pop(0)
                edge = edges[edge_id]
                new_dir = self._compute_new_direction(edge.dir, np.eye(2))
                s, tri = self._scaling(nodes[edge.n2])
                if tri < 0:
                    raise ValueError("the fascicle goes out of the domain")
                new_node = nodes[edge.n2] + new_dir * dx * s
                new_node_id = self._add_node(nodes, new_node)
                branches[edge.branch].append(new_node_id)
                edge_queue.append(len(edges))
                edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, edge.branch))

    def _grow_generations(
        self,
        edges: list,
        nodes: list,
        branches: dict,
        edge_queue: list,
        branch_id: int,
        dx: float,
        branch_length: float,
        w: float,
        Rplus: np.ndarray,
        Rminus: np.ndarray,
        end_nodes: list,
        sister_branches: dict
    ) -> None:
        """
        Simulates the iterative growth of a fractal tree structure over multiple generations.
        This method alternates between branching and growing steps for a specified number of generations,
        updating the tree's edges, nodes, and branches. It handles branching at each edge, checks for collisions,
        and manages the addition of new nodes and edges. The method also tracks end nodes and sister branches.
        Args:
            edges (list): List of Edge objects representing the current tree edges.
            nodes (list): List of node coordinates (e.g., numpy arrays).
            branches (dict): Dictionary mapping branch IDs to lists of node indices.
            edge_queue (list): List of edge indices to process in the current generation.
            branch_id (int): Current branch identifier, incremented for new branches.
            dx (float): Step size for growth.
            branch_length (float): Total length for each branch to grow.
            w (float): Weight parameter for direction computation.
            Rplus (np.ndarray): Rotation matrix for one branching direction.
            Rminus (np.ndarray): Rotation matrix for the other branching direction.
            end_nodes (list): List to collect indices of nodes where growth ends.
            sister_branches (dict): Dictionary mapping branch IDs to their sister branch IDs.
        Returns:
            None
        Raises:
            None
        Notes:
            - The method modifies the input lists and dictionaries in place.
            - Uses helper methods for direction computation, scaling, collision checking, and node addition.
            - Assumes existence of Edge class and cKDTree for spatial queries.
        """
        for gen in range(self.params.N_it):
            print('generation', gen)
            branching_queue = []
            # Branching step
            while edge_queue:
                edge_id = edge_queue.pop(0)
                edge = edges[edge_id]
                for R in [Rplus, Rminus]:
                    new_dir = self._compute_new_direction(edge.dir, R)
                    s, tri = self._scaling(nodes[edge.n2])
                    new_node = nodes[edge.n2] + new_dir * dx * s
                    if not self._can_grow_to(new_node, self.loc):
                        end_nodes.append(edge.n2)
                        continue
                    new_node_id = self._add_node(nodes, new_node)
                    branching_queue.append(len(edges))
                    branch_id += 1
                    branches[branch_id].append(new_node_id)
                    edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, branch_id))
                sister_branches[branch_id - 1] = branch_id
                sister_branches[branch_id] = branch_id - 1

            edge_queue = branching_queue

            # Growing step
            for _ in range(int(branch_length / dx)):
                growing_queue = []
                tree = cKDTree(nodes)
                new_nodes = []
                while edge_queue:
                    edge_id = edge_queue.pop(0)
                    edge = edges[edge_id]
                    s, tri = self._scaling(nodes[edge.n2])
                    collision, grad_dist = self._collision_check(nodes, branches, sister_branches, edge, dx, s)
                    if collision:
                        end_nodes.append(edge.n2)
                        continue
                    new_dir = self._compute_new_direction(edge.dir, np.eye(2), grad_dist, w)
                    new_node = nodes[edge.n2] + new_dir * dx * s
                    if not self._can_grow_to(new_node, self.loc):
                        end_nodes.append(edge.n2)
                        continue
                    new_node_id = self._add_node(nodes, new_node)
                    new_nodes.append(new_node)
                    growing_queue.append(len(edges))
                    branches[edge.branch].append(new_node_id)
                    edges.append(Edge(edge.n2, new_node_id, nodes, edge_id, edge.branch))
                edge_queue = growing_queue

    def grow_tree(self):

        # Initialization
        branches = defaultdict(list)
        branch_id = 0
        end_nodes = []
        sister_branches = {}
        dx = self.params.l_segment

        # Initial nodes and direction
        init_node = self.mesh_uv.verts[self.params.init_node_id][:2]
        second_node = self.mesh_uv.verts[self.params.second_node_id][:2]
        s, tri = self._scaling(init_node)
        if tri < 0:
            raise ValueError("The initial node is outside the domain")
        init_dir = second_node - init_node
        init_dir /= np.linalg.norm(init_dir)
        nodes = [init_node, init_node + dx * init_dir * s]
        edges = [Edge(0, 1, nodes, None, branch_id)]
        edge_queue = [0]
        branches[branch_id].append(0)

        branch_length = self.params.length
        init_branch_length = self.params.init_length
        theta = self.params.branch_angle
        w = self.params.w

        Rplus = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        Rminus = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])

        # Grow initial branch
        branching_edge_id = self._grow_initial_branch(edge_queue, edges, nodes, branches, dx, init_branch_length)
        
        # Grow fascicles from the initial branch
        self._grow_fascicles(branching_edge_id, edges, nodes, branches, edge_queue, branch_id, dx)

        self._grow_generations(edges, nodes, branches, edge_queue, branch_id, dx, branch_length, w, Rplus, Rminus, end_nodes, sister_branches)
        
        # Finalize end nodes and connectivity
        end_nodes += [edges[edge].n2 for edge in edge_queue]
        self.uv_nodes = np.array(nodes)
        self.edges = edges
        self.end_nodes = end_nodes
        self.connectivity = [[edge.n1, edge.n2] for edge in edges]

        # Map UV nodes to XYZ space
        self.nodes_xyz = []
        for node in nodes:
            n = np.append(node, np.zeros(1))
            f, _, tri = self._eval_field(n, self.mesh.verts, self.mesh_uv)
            self.nodes_xyz.append(f)

    def save(self, filename: str) -> None:
        """
        Saves the fractal tree structure as a mesh file.

        Parameters
        ----------
        filename : str
            The path to the file where the mesh will be saved.
        """
        try:
            line = meshio.Mesh(np.array(self.nodes_xyz), [('line', np.array(self.connectivity))])
            line.write(filename)
            logging.info(f"Fractal tree mesh saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save fractal tree mesh to {filename}: {e}")
