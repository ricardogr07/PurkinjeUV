"""Module defining the Mesh class for 3D triangular surface meshes.

This module provides:
  - Loading from OBJ or direct arrays.
  - Computation of normals, centroids, connectivity.
  - KD-tree spatial queries.
  - FEM routines (B-matrix, stiffness, mass, force).

Designed for fractal tree growth and surface-based FEM analysis.
"""

import collections
import logging
from typing import Any, Optional, List, Dict, Tuple, Sequence, DefaultDict
from numpy.typing import NDArray

import numpy as np
import meshio
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


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
        self.verts: NDArray[Any] = np.array(verts)
        self.connectivity: NDArray[Any] = np.array(connectivity)

        # Compute triangle normals
        self.normals: NDArray[Any] = np.zeros(self.connectivity.shape)

        # Build node-to-triangle connectivity dictionary
        self.node_to_tri: DefaultDict[int, List[int]] = collections.defaultdict(list)
        for i in range(self.connectivity.shape[0]):
            for j in range(3):
                self.node_to_tri[self.connectivity[i, j]].append(i)

            # Compute triangle normal
            u = (
                self.verts[self.connectivity[i, 1], :]
                - self.verts[self.connectivity[i, 0], :]
            )
            v = (
                self.verts[self.connectivity[i, 2], :]
                - self.verts[self.connectivity[i, 0], :]
            )
            n = np.cross(u, v)
            self.normals[i, :] = n / np.linalg.norm(n)

        # Build KD-tree for fast nearest-node queries
        self.tree: cKDTree = cKDTree(self.verts)

        # Compute centroids for each triangle
        self.centroids: NDArray[Any] = (
            self.verts[self.connectivity[:, 0], :]
            + self.verts[self.connectivity[:, 1], :]
            + self.verts[self.connectivity[:, 2], :]
        ) / 3.0

        # Initialize optional attributes
        self.boundary_edges: Optional[List[Tuple[int, int]]] = None
        self.uv: Optional[NDArray[Any]] = None
        self.triareas: Optional[NDArray[Any]] = None
        self.uvscaling: Optional[NDArray[Any]] = None

        logger.info(
            f"Mesh initialized with {self.verts.shape[0]} vertices and {self.connectivity.shape[0]} triangles"
        )

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
        logger.info(
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
        _, idxs = self.tree.query(point, verts_to_search)
        if verts_to_search > 1:
            for node_idx in idxs:
                node_int: int = int(node_idx)
                projected_point, intriangle, r, t = self.project_point_check(
                    point, node_int
                )
                if intriangle != -1:
                    return projected_point, intriangle, r, t
        else:
            node_int = int(idxs)
            projected_point, intriangle, r, t = self.project_point_check(
                point, node_int
            )
        return projected_point, intriangle, r, t

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
        # Print closest point info in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            d, node_idx = self.tree.query(point)
            logger.debug(f"Closest distance: {d}, Closest node: {node_idx}")

        # Get triangles list connected to that node
        triangles_list: List[int] = self.node_to_tri[node]
        logger.debug(f"Node {node} is connected to triangles: {triangles_list}")

        # Compute the vertex normal as the average of the triangle normals.
        vertex_normal: NDArray[Any] = np.sum(self.normals[triangles_list, :], axis=0)
        vertex_normal = vertex_normal / np.linalg.norm(vertex_normal)

        # Project to the point to the closest vertex plane
        vec_to_vertex: NDArray[Any] = point - self.verts[node]
        distance_along_normal: float = float(np.dot(vec_to_vertex, vertex_normal))
        pre_projected_point: NDArray[Any] = (
            point - vertex_normal * distance_along_normal
        )

        # Calculate the distance from point to plane (Closest point projection)
        CPP: List[float] = []
        for tri in triangles_list:
            val: float = float(
                np.dot(
                    pre_projected_point - self.verts[self.connectivity[tri, 0], :],
                    self.normals[tri, :],
                )
            )
            CPP.append(val)
        CPP_arr: NDArray[Any] = np.array(CPP, dtype=float)

        logger.debug(f"CPP={CPP}")

        triangles_arr: NDArray[Any] = np.array(triangles_list, dtype=int)

        # Sort from closest to furthest
        order: NDArray[Any] = np.abs(CPP_arr).argsort()
        logger.debug(f"CPP sorted: {CPP_arr[order]}")

        # Check if point is in triangle
        intriangle: int = -1
        projected_point: NDArray[Any] = pre_projected_point
        r: float = -1.0
        t: float = -1.0

        for o in order:
            idx: int = int(o)
            tri_idx: int = int(triangles_arr[idx])

            projected_pt: NDArray[Any] = (
                pre_projected_point - CPP_arr[idx] * self.normals[tri_idx, :]
            )

            u: NDArray[Any] = (
                self.verts[self.connectivity[tri_idx, 1], :]
                - self.verts[self.connectivity[tri_idx, 0], :]
            )

            v: NDArray[Any] = (
                self.verts[self.connectivity[tri_idx, 2], :]
                - self.verts[self.connectivity[tri_idx, 0], :]
            )

            w: NDArray[Any] = (
                projected_pt - self.verts[self.connectivity[tri_idx, 0], :]
            )

            logger.debug(
                f"Check orthogonality: np.dot(w, self.normals[{tri_idx}, :]) = "
                f"{np.dot(w, self.normals[tri_idx, :])}"
            )

            vxw: NDArray[Any] = np.cross(v, w)
            vxu: NDArray[Any] = np.cross(v, u)
            uxw: NDArray[Any] = np.cross(u, w)
            sign_r: float = float(np.dot(vxw, vxu))
            sign_t: float = float(np.dot(uxw, -vxu))

            logger.debug(f"sign_r={sign_r}, sign_t={sign_t}")

            if sign_r >= 0 and sign_t >= 0:
                r = float(np.linalg.norm(vxw) / np.linalg.norm(vxu))
                t = float(np.linalg.norm(uxw) / np.linalg.norm(vxu))

                logger.debug(f"Sign ok: r={r}, t={t}")

                if r <= 1 and t <= 1 and (r + t) <= 1.001:
                    logger.debug(f"In triangle {tri_idx}")
                    intriangle = tri_idx
                    projected_point = projected_pt
                    break
        return projected_point, intriangle, r, t

    def writeVTU(
        self,
        filename: str,
        point_data: Optional[Dict[str, NDArray[Any]]] = None,
        cell_data: Optional[Dict[str, NDArray[Any]]] = None,
    ) -> None:
        """Export this mesh (and optional point/cell data) in VTU format."""
        # Define cells for meshio: (cell_type, connectivity array)
        cells: List[Tuple[str, NDArray[Any]]] = [("triangle", self.connectivity)]
        m_out = meshio.Mesh(self.verts, cells)

        if point_data is not None:
            m_out.point_data = point_data
        if cell_data is not None:
            m_out.cell_data = cell_data

        m_out.write(filename)

    def Bmatrix(self, element: int) -> Tuple[NDArray[Any], float]:
        """Compute the B-matrix and Jacobian determinant for a triangle.

        Args:
            element (int): Triangle index.

        Returns:
            Tuple[NDArray[Any], float]:
                - B (2×3 array): Strain-displacement matrix.
                - J (float): Twice the triangle area (Jacobian determinant).
        """
        # Extract vertex coordinates for this triangle
        nodeCoords: NDArray[Any] = self.verts[self.connectivity[element]]

        # Build local orthonormal frame (e1, e2)
        edge21: NDArray[Any] = nodeCoords[1, :] - nodeCoords[0, :]
        e1: NDArray[Any] = edge21 / np.linalg.norm(edge21)

        temp: NDArray[Any] = nodeCoords[2, :] - nodeCoords[0, :]
        proj: float = float(np.dot(temp, e1))
        perp: NDArray[Any] = temp - proj * e1
        e2: NDArray[Any] = perp / np.linalg.norm(perp)

        # Compute scalar edge projections
        x21: float = float(np.dot(edge21, e1))
        x13: float = float(np.dot(nodeCoords[0, :] - nodeCoords[2, :], e1))
        x32: float = float(np.dot(nodeCoords[2, :] - nodeCoords[1, :], e1))

        y23: float = float(np.dot(nodeCoords[1, :] - nodeCoords[2, :], e2))
        y31: float = float(np.dot(nodeCoords[2, :] - nodeCoords[0, :], e2))
        y12: float = float(np.dot(nodeCoords[0, :] - nodeCoords[1, :], e2))

        # Compute Jacobian (twice the triangle area)
        J: float = x13 * y23 - y31 * x32

        # Assemble B‐matrix
        B: NDArray[Any] = np.array([[y23, y31, y12], [x32, x13, x21]], dtype=float)

        return B, J

    def gradient(self, element: int, u: NDArray[Any]) -> NDArray[Any]:
        """Compute the gradient of a scalar field over a triangle.

        Args:
            element (int): Triangle index.
            u (NDArray[Any]): Field values at the triangle's three vertices.

        Returns:
            NDArray[Any]: 3-vector of gradients in 3D space.
        """
        node_coords: NDArray[Any] = self.verts[self.connectivity[element]]
        edge_vec: NDArray[Any] = node_coords[1, :] - node_coords[0, :]
        e1: NDArray[Any] = edge_vec / np.linalg.norm(edge_vec)

        temp: NDArray[Any] = node_coords[2, :] - node_coords[0, :]
        proj: float = float(np.dot(temp, e1))
        perp: NDArray[Any] = temp - proj * e1
        e2: NDArray[Any] = perp / np.linalg.norm(perp)

        e3: NDArray[Any] = np.cross(e1, e2)

        x21: float = float(np.dot(edge_vec, e1))
        x13: float = float(np.dot(node_coords[0, :] - node_coords[2, :], e1))
        x32: float = float(np.dot(node_coords[2, :] - node_coords[1, :], e1))

        y23: float = float(np.dot(node_coords[1, :] - node_coords[2, :], e2))
        y31: float = float(np.dot(node_coords[2, :] - node_coords[0, :], e2))
        y12: float = float(np.dot(node_coords[0, :] - node_coords[1, :], e2))

        J: float = x13 * y23 - y31 * x32
        B: NDArray[Any] = np.array([[y23, y31, y12], [x32, x13, x21]], dtype=float)

        grad: NDArray[Any] = np.zeros(3, dtype=float)
        grad_vals: NDArray[Any] = np.dot(B, u) / J
        grad[:2] = grad_vals

        R: NDArray[Any] = np.vstack((e1, e2, e3)).T
        result: NDArray[Any] = np.dot(R, grad)
        return result

    def StiffnessMatrix(self, B: NDArray[Any], J: float) -> NDArray[Any]:
        """Compute the local stiffness matrix for a triangle.

        Args:
            B (NDArray[Any]): B-matrix from `Bmatrix`.
            J (float): Jacobian determinant.

        Returns:
            NDArray[Any]: 3×3 stiffness matrix.
        """
        result: NDArray[Any] = np.dot(B.T, B) / (2.0 * J)
        return result

    def MassMatrix(self, J: float) -> NDArray[Any]:
        """Compute the local mass matrix for a triangle.

        Args:
            J (float): Jacobian determinant.

        Returns:
            NDArray[Any]: 3×3 mass matrix.
        """
        result: NDArray[Any] = (
            np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=float)
            * J
            / 12
        )
        return result

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
        result: NDArray[Any] = np.dot(B.T, X) / 2.0
        return result

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
        nNodes: int = self.verts.shape[0]
        nElem: int = self.connectivity.shape[0]

        #        K = sp.lil_matrix((nNodes, nNodes))
        #        M = sp.lil_matrix((nNodes, nNodes))

        F: NDArray[Any] = np.zeros((nNodes, 1), dtype=float)
        u0: NDArray[Any] = np.zeros((nNodes, 1), dtype=float)
        u0[nodes] = 1e6

        # dt = 10.0

        if (K is None) or (M is None):
            K = np.zeros((nNodes, nNodes), dtype=float)
            M = np.zeros((nNodes, nNodes), dtype=float)
            for el, tri in enumerate(self.connectivity):
                j, i = np.meshgrid(tri, tri)
                B, J = self.Bmatrix(el)
                k_mat = self.StiffnessMatrix(B, J)
                m_mat = self.MassMatrix(J)
                K[i, j] += k_mat
                M[i, j] += m_mat

        activeNodes: List[int] = list(range(nNodes))
        for known in nodes:
            activeNodes.remove(known)

        jActive, iActive = np.meshgrid(activeNodes, activeNodes)
        jKnown, iKnown = np.meshgrid(nodes, activeNodes)

        A1: sp.spmatrix = sp.csr.csr_matrix(M + dt * K)
        u: NDArray[Any] = spsolve(A1, u0)[:, None]
        #  u = np.linalg.solve(M + dt*K,u0)

        Xs: NDArray[Any] = np.zeros((nElem, 3), dtype=float)
        Js: NDArray[Any] = np.zeros((nElem, 1), dtype=float)

        for k, tri in enumerate(self.connectivity):
            j, i = np.meshgrid(tri, tri)
            B, J = self.Bmatrix(k)
            Js[k] = J
            X: NDArray[Any] = self.gradient(k, u[tri, 0])
            Xs[k, :] = X / np.linalg.norm(X)
            Xnr: NDArray[Any] = np.dot(B, u[tri, 0])
            Xnr /= np.linalg.norm(Xnr)
            f: NDArray[Any] = self.ForceVector(B, J, Xnr)
            F[tri, 0] -= f

        A2: sp.spmatrix = sp.csr.csr_matrix(K[iActive, jActive])
        AT: NDArray[Any] = spsolve(
            A2, F[activeNodes, 0] - np.dot(K[iKnown, jKnown], nodeVals)
        )
        #  AT = np.linalg.solve(K[iActive, jActive],F[activeNodes,0]-np.dot(K[iKnown, jKnown],nodeVals))

        ATglobal: NDArray[Any] = np.zeros(nNodes, dtype=float)
        ATglobal[activeNodes] = AT
        ATglobal[nodes] = nodeVals

        if filename is not None:
            self.writeVTU(filename, point_data={"d": ATglobal})

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
        nNodes = self.verts.shape[0]

        K: sp.spmatrix
        M: sp.spmatrix
        K, M = self.computeLaplacian()

        F: NDArray[Any] = np.zeros((nNodes, 1), dtype=float)

        activeNodes: List[int] = list(range(nNodes))

        for known in nodes:
            activeNodes.remove(known)

        jActive: NDArray[Any]
        iActive: NDArray[Any]
        jActive, iActive = np.meshgrid(activeNodes, activeNodes)

        jKnown: NDArray[Any]
        iKnown: NDArray[Any]
        jKnown, iKnown = np.meshgrid(nodes, activeNodes)

        T: NDArray[Any] = spsolve(
            K[iActive, jActive],
            F[activeNodes, 0] - K[iKnown, jKnown].dot(nodeVals),
        )

        Tglobal: NDArray[Any] = np.zeros(nNodes, dtype=float)
        Tglobal[activeNodes] = T
        Tglobal[nodes] = nodeVals

        if filename is not None:
            self.writeVTU(filename, point_data={"u": Tglobal})

        return Tglobal

    def computeLaplacian(self) -> Tuple[sp.spmatrix, sp.spmatrix]:
        """Assemble global stiffness (K) and mass (M) matrices as CSR.

        Returns:
            Tuple[sp.spmatrix, sp.spmatrix]: (K, M) in CSR format.
        """
        nNodes: int = self.verts.shape[0]

        K: sp.spmatrix = sp.lil_matrix((nNodes, nNodes))
        M: sp.spmatrix = sp.lil_matrix((nNodes, nNodes))

        for elem_idx, tri in enumerate(self.connectivity):
            j: NDArray[Any]
            i: NDArray[Any]
            j, i = np.meshgrid(tri, tri)

            B: NDArray[Any]
            J: float
            B, J = self.Bmatrix(elem_idx)

            K_local: NDArray[Any] = self.StiffnessMatrix(B, J)
            K[i, j] += K_local

            M_local: NDArray[Any] = self.MassMatrix(J)
            M[i, j] += M_local

        return K.tocsr(), M.tocsr()

    def uvmap(self, filename: Optional[str] = None) -> None:
        """Compute UV coordinates by solving Laplace's equation.

        Args:
            filename (Optional[str]): VTU export path for u and v fields.
        """
        around_nodes: list[int]
        bc_u: NDArray[Any]
        bc_v: NDArray[Any]
        around_nodes, bc_u, bc_v = self.uv_bc()

        u: NDArray[Any] = self.computeLaplace(around_nodes[:-1], bc_u)
        v: NDArray[Any] = self.computeLaplace(around_nodes[:-1], bc_v)

        if filename is not None:
            self.writeVTU(filename, point_data={"u": u, "v": v})
        uv_arr: NDArray[Any] = np.vstack([u, v]).T
        self.uv = uv_arr

    def compute_uvscaling(self) -> None:
        """Compute and validate UV-scaling metric per triangle."""
        if self.uv is None:
            self.uvmap()
        assert self.uv is not None

        metrics: List[NDArray[Any]] = []

        for e in range(self.connectivity.shape[0]):
            B, J = self.Bmatrix(e)
            uv_e: NDArray[Any] = self.uv[self.connectivity[e]]
            F: NDArray[Any] = np.matmul(B, uv_e) / J
            metrics.append(np.matmul(F.T, F))

        eigvals, _ = np.linalg.eig(metrics)
        uvscale_arr: NDArray[Any] = (eigvals[:, 0] + eigvals[:, 1]) / 2
        self.uvscaling = uvscale_arr

        if uvscale_arr.min() < 0:
            logger.error("Flipped triangles detected — check mesh quality")
            raise ValueError("Flipped triangles detected — check mesh quality")

    def detect_boundary(self) -> None:
        """Identify boundary edges (edges in exactly one triangle)."""
        edge_dict: DefaultDict[Tuple[int, int], List[int]] = collections.defaultdict(
            list
        )

        for tri_idx, el in enumerate(self.connectivity):
            for offset in ((0, 1), (1, 2), (2, 0)):
                n1, n2 = el[offset[0]], el[offset[1]]
                edge_key: Tuple[int, int] = (min(n1, n2), max(n1, n2))
                edge_dict[edge_key].append(tri_idx)

        boundary_edges: List[Tuple[int, int]] = []

        for edge_key, tris in edge_dict.items():
            if len(tris) == 1:
                boundary_edges.append(edge_key)

        self.boundary_edges = boundary_edges

    def uv_bc(self) -> Tuple[List[int], NDArray[Any], NDArray[Any]]:
        """Generate UV boundary loop and boundary conditions.

        Returns:
            Tuple[List[int], NDArray[Any], NDArray[Any]]:
              around_nodes, bc_u, bc_v.
        """
        if self.boundary_edges is None:
            self.detect_boundary()
        assert self.boundary_edges is not None

        boundary_node2edge: DefaultDict[
            int, List[Tuple[int, int]]
        ] = collections.defaultdict(list)
        for edge in self.boundary_edges:
            boundary_node2edge[edge[0]].append(edge)
            boundary_node2edge[edge[1]].append(edge)

        around_nodes: List[int] = list(self.boundary_edges[0])
        last_edge: Tuple[int, int] = self.boundary_edges[0]

        while around_nodes[0] != around_nodes[-1]:
            edges: List[Tuple[int, int]] = boundary_node2edge[around_nodes[-1]].copy()
            edges.remove(last_edge)
            new_nodes: List[int] = list(edges[0])
            new_nodes.remove(around_nodes[-1])
            around_nodes.append(new_nodes[0])
            last_edge = edges[0]
            if len(around_nodes) >= self.verts.shape[0]:
                logger.error(
                    "UV boundary traversal exceeded mesh size — boundary may be broken"
                )
                raise ValueError(
                    "UV boundary traversal exceeded mesh size — boundary may be broken"
                )

        lengths: NDArray[Any] = np.cumsum(
            np.linalg.norm(
                self.verts[around_nodes[:-1]] - self.verts[around_nodes[1:]],
                axis=1,
            )
        )
        total_length: float = float(lengths[-1])
        bc_u: NDArray[Any] = np.sin(2 * np.pi * lengths / total_length)
        bc_v: NDArray[Any] = np.cos(2 * np.pi * lengths / total_length)

        return around_nodes, bc_u, bc_v

    def compute_triareas(self) -> None:
        """Compute triangle areas (J/2) and store in `self.triareas`."""
        triareas_list: List[float] = []
        for e in range(self.connectivity.shape[0]):
            _, J = self.Bmatrix(e)
            triareas_list.append(J / 2.0)

        triareas_arr: NDArray[Any] = np.array(triareas_list, dtype=float)
        self.triareas = triareas_arr

    def tri2node_interpolation(self, cell_field: NDArray[Any]) -> List[float]:
        """Interpolate triangle-based field to nodes by area-weighting.

        Args:
            cell_field (NDArray[Any]): Per-triangle values.

        Returns:
            List[float]: Per-node interpolated values.
        """
        if self.triareas is None:
            self.compute_triareas()
        assert self.triareas is not None

        node_field: List[float] = []
        for i in range(self.verts.shape[0]):
            tris: List[int] = self.node_to_tri[i]
            areas: NDArray[Any] = self.triareas[tris]
            fields: NDArray[Any] = cell_field[tris]
            nodal_val: float = float(np.sum(areas * fields) / np.sum(areas))
            node_field.append(nodal_val)

        return node_field

    # TODO Is this deprecated?
    # def compute_uvscaling_nodes(self):
    #     uvmesh =
    #     if self.uvscaling is None:
    #         self.compute_uvscaling()
    #     self.uvscaling_nodes = self.tri2node_interpolation(self.uvscaling)
