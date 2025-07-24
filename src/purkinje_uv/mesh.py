import collections
import logging
import numpy as np
import meshio
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


class Mesh:
    """
    Mesh class for handling 3D triangular surface meshes, supporting geometry, topology, and finite element operations.
    This class loads a Wavefront .obj file or accepts vertices/connectivity directly, computes triangle normals,
    builds connectivity structures, and provides methods for geometric queries and finite element calculations.
        filename (str, optional): Path to the .obj file to load mesh data from.
        verts (array-like, optional): Vertex coordinates (n_nodes x 3).
        connectivity (array-like, optional): Triangle connectivity (n_triangles x 3).
        verts (np.ndarray): Array of vertex coordinates (n_nodes x 3).
        connectivity (np.ndarray): Array of triangle vertex indices (n_triangles x 3).
        normals (np.ndarray): Array of triangle normals (n_triangles x 3).
        node_to_tri (dict): Maps node indices to lists of adjacent triangle indices.
        tree (scipy.spatial.cKDTree): KD-tree for fast nearest-node queries.
        centroids (np.ndarray): Triangle centroid coordinates (n_triangles x 3).
        boundary_edges (list): List of boundary edges as (node1, node2) tuples.
        uv (np.ndarray): UV coordinates for each node (n_nodes x 2).
        triareas (np.ndarray): Area of each triangle (n_triangles,).
        uvscaling (np.ndarray): UV scaling metric per triangle (n_triangles,).

    Methods:
        loadOBJ(filename):
            Load mesh data from a Wavefront .obj file.
        project_new_point(point, verts_to_search=1):
            Project a point onto the mesh and find the containing triangle.
        project_point_check(point, node):
            Project a point onto the surface near a given node and check triangle inclusion.
        writeVTU(filename, point_data=None, cell_data=None):
            Export mesh and data to VTU format using meshio.
        Bmatrix(element):
            Compute the B matrix and Jacobian for a triangle (finite element method).
        gradient(element, u):
            Compute the gradient of a scalar field over a triangle.
        StiffnessMatrix(B, J):
            Compute the local stiffness matrix for a triangle.
        MassMatrix(J):
            Compute the local mass matrix for a triangle.
        ForceVector(B, J, X):
            Compute the local force vector for a triangle.
        computeGeodesic(nodes, nodeVals, filename=None, K=None, M=None, dt=10.0):
            Compute geodesic distances from specified nodes using FEM.
        computeLaplace(nodes, nodeVals, filename=None):
            Solve Laplace's equation with Dirichlet boundary conditions.
        computeLaplacian():
            Assemble global stiffness and mass matrices for the mesh.
        uvmap(filename=None):
            Compute UV mapping for the mesh using Laplace's equation.
        compute_uvscaling():
            Compute UV scaling metric for each triangle.
        detect_boundary():
            Identify boundary edges of the mesh.
        uv_bc():
            Generate UV boundary conditions for boundary nodes.
        compute_triareas():
            Compute area for each triangle.
        tri2node_interpolation(cell_field):
            Interpolate triangle-based field to nodes using area-weighted averaging.
    Notes:
        - Mesh must be a valid triangular surface mesh.
        - Normals are computed per triangle; ensure correct orientation in .obj file.
        - Some methods require scipy, numpy, and meshio.
        - Designed for use in fractal tree growth and finite element analysis on surfaces.

    Args:
        filename (str): the path and filename of the .obj file with the mesh.

    Attributes:
        verts (array): a numpy array that contains all the nodes of the mesh. verts[i,j], where i is the node index and j=[0,1,2] is the coordinate (x,y,z).
        connectivity (array): a numpy array that contains all the connectivity of the triangles of the mesh. connectivity[i,j], where i is the triangle index and j=[0,1,2] is node index.
        normals (array): a numpy array that contains all the normals of the triangles of the mesh. normals[i,j], where i is the triangle index and j=[0,1,2] is normal coordinate (x,y,z).
        node_to_tri (dict): a dictionary that relates a node to the triangles that it is connected. It is the inverse relation of connectivity. The triangles are stored as a list for each node.
        tree (scipy.spatial.cKDTree): a k-d tree to compute the distance from any point to the closest node in the mesh.
    """

    def __init__(
        self,
        filename: str = None,
        verts: np.ndarray = None,
        connectivity: np.ndarray = None,
    ):
        """
        Initialize the Mesh object.

        Args:
            filename (str, optional): Path to the .obj file to load mesh data from.
            verts (np.ndarray, optional): Vertex coordinates (n_nodes x 3).
            connectivity (np.ndarray, optional): Triangle connectivity (n_triangles x 3).

        Attributes:
            verts (np.ndarray): Array of vertex coordinates (n_nodes x 3).
            connectivity (np.ndarray): Array of triangle vertex indices (n_triangles x 3).
            normals (np.ndarray): Array of triangle normals (n_triangles x 3).
            node_to_tri (defaultdict): Maps node indices to lists of adjacent triangle indices.
            tree (scipy.spatial.cKDTree): KD-tree for fast nearest-node queries.
            centroids (np.ndarray): Triangle centroid coordinates (n_triangles x 3).
            boundary_edges (list): List of boundary edges as (node1, node2) tuples.
            uv (np.ndarray): UV coordinates for each node (n_nodes x 2).
            triareas (np.ndarray): Area of each triangle (n_triangles,).
            uvscaling (np.ndarray): UV scaling metric per triangle (n_triangles,).
        """
        # Load mesh from file if filename is provided
        if filename is not None:
            verts, connectivity = self.loadOBJ(filename)
        # Store vertices and connectivity as numpy arrays
        self.verts: np.ndarray = np.array(verts)
        self.connectivity: np.ndarray = np.array(connectivity)
        # Initialize normals array for each triangle
        self.normals: np.ndarray = np.zeros(self.connectivity.shape)
        # Build node-to-triangle connectivity dictionary
        self.node_to_tri: collections.defaultdict = collections.defaultdict(list)
        for i in range(len(self.connectivity)):
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
        self.centroids: np.ndarray = (
            self.verts[self.connectivity[:, 0], :]
            + self.verts[self.connectivity[:, 1], :]
            + self.verts[self.connectivity[:, 2], :]
        ) / 3.0
        # Initialize optional attributes
        self.boundary_edges: list = None
        self.uv: np.ndarray = None
        self.triareas: np.ndarray = None
        self.uvscaling: np.ndarray = None

        logger.info(
            f"Mesh initialized with {self.verts.shape[0]} vertices and {self.connectivity.shape[0]} triangles"
        )

    def loadOBJ(self, filename):
        """
        This function reads a .obj mesh file

        Args:
            filename (str): the path and filename of the .obj file.

        Returns:
             verts (array): a numpy array that contains all the nodes of the mesh. verts[i,j], where i is the node index and j=[0,1,2] is the coordinate (x,y,z).
             connectivity (array): a numpy array that contains all the connectivity of the triangles of the mesh. connectivity[i,j], where i is the triangle index and j=[0,1,2] is node index.
        """
        numVerts = 0
        verts = []
        norms = []
        connectivity = []
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
        return verts, connectivity

    def project_new_point(self, point, verts_to_search=1):
        """This function receives a triangle and project it on the mesh in order to get the index of the triangle where
        the projected point lies

        Args:
            point (array): coordinates of the point to project.
            verts_to_search (int): the number of verts wich the point is going to be projected
        Returns:
             intriangle (int): the index of the triangle where the projected point lies. If the point is outside surface, intriangle=-1.
        """
        d, nodes = self.tree.query(point, verts_to_search)
        if verts_to_search > 1:
            for node in nodes:
                projected_point, intriangle, r, t = self.project_point_check(
                    point, node
                )
                if intriangle != -1:
                    return projected_point, intriangle, r, t
        else:
            projected_point, intriangle, r, t = self.project_point_check(point, nodes)
        return projected_point, intriangle, r, t

    def project_point_check(self, point, node):
        """This function projects any point to the surface defined by the mesh.

        Args:
            point (array): coordinates of the point to project.
            node (int): index of the most close node to the point
        Returns:
             projected_point (array): the coordinates of the projected point that lies in the surface.
             intriangle (int): the index of the triangle where the projected point lies. If the point is outside surface, intriangle=-1.
        """
        # Get the closest point
        # d, node=self.tree.query(point)
        # print d, node
        # Get triangles connected to that node
        triangles = self.node_to_tri[node]
        # print triangles
        # Compute the vertex normal as the avergage of the triangle normals.
        vertex_normal = np.sum(self.normals[triangles], axis=0)
        # Normalize
        vertex_normal = vertex_normal / np.linalg.norm(vertex_normal)
        # Project to the point to the closest vertex plane
        pre_projected_point = point - vertex_normal * np.dot(
            point - self.verts[node], vertex_normal
        )
        # Calculate the distance from point to plane (Closest point projection)
        CPP = []
        for tri in triangles:
            CPP.append(
                np.dot(
                    pre_projected_point - self.verts[self.connectivity[tri, 0], :],
                    self.normals[tri, :],
                )
            )
        CPP = np.array(CPP)
        #   print 'CPP=',CPP
        triangles = np.array(triangles)
        # Sort from closest to furthest
        order = np.abs(CPP).argsort()
        # print CPP[order]
        # Check if point is in triangle
        intriangle = -1
        for o in order:
            i = triangles[o]
            #      print i
            projected_point = pre_projected_point - CPP[o] * self.normals[i, :]
            #      print projected_point
            u = (
                self.verts[self.connectivity[i, 1], :]
                - self.verts[self.connectivity[i, 0], :]
            )
            v = (
                self.verts[self.connectivity[i, 2], :]
                - self.verts[self.connectivity[i, 0], :]
            )
            w = projected_point - self.verts[self.connectivity[i, 0], :]
            #     print 'check ortogonality',np.dot(w,self.normals[i,:])
            vxw = np.cross(v, w)
            vxu = np.cross(v, u)
            uxw = np.cross(u, w)
            sign_r = np.dot(vxw, vxu)
            sign_t = np.dot(uxw, -vxu)
            r = t = -1
            #    print sign_r,sign_t
            if sign_r >= 0 and sign_t >= 0:
                r = np.linalg.norm(vxw) / np.linalg.norm(vxu)
                t = np.linalg.norm(uxw) / np.linalg.norm(vxu)
                #   print 'sign ok', r , t
                if r <= 1 and t <= 1 and (r + t) <= 1.001:
                    #      print 'in triangle',i
                    intriangle = i
                    break
        return projected_point, intriangle, r, t

    def writeVTU(self, filename, point_data=None, cell_data=None):
        cells = [("triangle", self.connectivity)]
        m_out = meshio.Mesh(self.verts, cells)
        if point_data is not None:
            m_out.point_data = point_data
        if cell_data is not None:
            m_out.cell_data = cell_data

        m_out.write(filename)

    def Bmatrix(self, element):
        nodeCoords = self.verts[self.connectivity[element]]
        e1 = (nodeCoords[1, :] - nodeCoords[0, :]) / np.linalg.norm(
            nodeCoords[1, :] - nodeCoords[0, :]
        )
        e2 = (nodeCoords[2, :] - nodeCoords[0, :]) - np.dot(
            (nodeCoords[2, :] - nodeCoords[0, :]), e1
        ) * e1
        e2 = e2 / np.linalg.norm(e2)  # normalize

        x21 = np.dot(nodeCoords[1, :] - nodeCoords[0, :], e1)
        x13 = np.dot(nodeCoords[0, :] - nodeCoords[2, :], e1)
        x32 = np.dot(nodeCoords[2, :] - nodeCoords[1, :], e1)

        y23 = np.dot(nodeCoords[1, :] - nodeCoords[2, :], e2)
        y31 = np.dot(nodeCoords[2, :] - nodeCoords[0, :], e2)
        y12 = np.dot(nodeCoords[0, :] - nodeCoords[1, :], e2)

        J = x13 * y23 - y31 * x32

        B = np.array([[y23, y31, y12], [x32, x13, x21]])

        return B, J

    def gradient(self, element, u):
        nodeCoords = self.verts[self.connectivity[element]]
        e1 = (nodeCoords[1, :] - nodeCoords[0, :]) / np.linalg.norm(
            nodeCoords[1, :] - nodeCoords[0, :]
        )
        e2 = (nodeCoords[2, :] - nodeCoords[0, :]) - np.dot(
            (nodeCoords[2, :] - nodeCoords[0, :]), e1
        ) * e1
        e2 = e2 / np.linalg.norm(e2)  # normalize
        e3 = np.cross(e1, e2)

        x21 = np.dot(nodeCoords[1, :] - nodeCoords[0, :], e1)
        x13 = np.dot(nodeCoords[0, :] - nodeCoords[2, :], e1)
        x32 = np.dot(nodeCoords[2, :] - nodeCoords[1, :], e1)

        y23 = np.dot(nodeCoords[1, :] - nodeCoords[2, :], e2)
        y31 = np.dot(nodeCoords[2, :] - nodeCoords[0, :], e2)
        y12 = np.dot(nodeCoords[0, :] - nodeCoords[1, :], e2)

        B = np.array([[y23, y31, y12], [x32, x13, x21]])
        J = x13 * y23 - y31 * x32

        grad = np.zeros(3)
        grad[:2] = np.dot(B, u) / J

        R = np.vstack((e1, e2, e3)).T
        # Rinv = np.linalg.inv(R)

        return np.dot(R, grad)

    def StiffnessMatrix(self, B, J):
        return np.dot(B.T, B) / (2.0 * J)

    def MassMatrix(self, J):
        # return np.eye(3)*J/3.
        return np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) * J / 12

    def ForceVector(self, B, J, X):
        return np.dot(B.T, X) / 2.0

    def computeGeodesic(self, nodes, nodeVals, filename=None, K=None, M=None, dt=10.0):
        nNodes = self.verts.shape[0]
        nElem = self.connectivity.shape[0]

        #        K = sp.lil_matrix((nNodes, nNodes))
        #        M = sp.lil_matrix((nNodes, nNodes))

        F = np.zeros((nNodes, 1))

        u0 = np.zeros((nNodes, 1))

        u0[nodes] = 1e6

        # dt = 10.0

        if (K is None) or (M is None):
            K = np.zeros((nNodes, nNodes))
            M = np.zeros((nNodes, nNodes))
            for el, tri in enumerate(self.connectivity):
                j, i = np.meshgrid(tri, tri)
                B, J = self.Bmatrix(el)
                k = self.StiffnessMatrix(B, J)
                m = self.MassMatrix(J)
                K[i, j] += k
                M[i, j] += m

        activeNodes = list(range(nNodes))
        for known in nodes:
            activeNodes.remove(known)

        jActive, iActive = np.meshgrid(activeNodes, activeNodes)

        jKnown, iKnown = np.meshgrid(nodes, activeNodes)

        A1 = sp.csr.csr_matrix(M + dt * K)
        u = spsolve(A1, u0)[:, None]
        #  u = np.linalg.solve(M + dt*K,u0)

        Xs = np.zeros((nElem, 3))
        Js = np.zeros((nElem, 1))

        for k, tri in enumerate(self.connectivity):
            j, i = np.meshgrid(tri, tri)
            B, J = self.Bmatrix(k)
            Js[k] = J
            X = self.gradient(k, u[tri, 0])
            Xs[k, :] = X / np.linalg.norm(X)
            Xnr = np.dot(B, u[tri, 0])  # not rotated
            Xnr /= np.linalg.norm(Xnr)
            f = self.ForceVector(B, J, Xnr)
            F[tri, 0] -= f
        A2 = sp.csr.csr_matrix(K[iActive, jActive])
        AT = spsolve(A2, F[activeNodes, 0] - np.dot(K[iKnown, jKnown], nodeVals))
        #  AT = np.linalg.solve(K[iActive, jActive],F[activeNodes,0]-np.dot(K[iKnown, jKnown],nodeVals))

        ATglobal = np.zeros(nNodes)

        ATglobal[activeNodes] = AT
        ATglobal[nodes] = nodeVals

        if filename is not None:
            self.writeVTU(filename, point_data={"d": ATglobal})

        return ATglobal, Xs

    def computeLaplace(self, nodes, nodeVals, filename=None):
        nNodes = self.verts.shape[0]

        K, M = self.computeLaplacian()
        F = np.zeros((nNodes, 1))

        activeNodes = list(range(nNodes))
        for known in nodes:
            activeNodes.remove(known)

        jActive, iActive = np.meshgrid(activeNodes, activeNodes)

        jKnown, iKnown = np.meshgrid(nodes, activeNodes)

        T = spsolve(
            K[iActive, jActive], F[activeNodes, 0] - K[iKnown, jKnown].dot(nodeVals)
        )

        Tglobal = np.zeros(nNodes)

        Tglobal[activeNodes] = T
        Tglobal[nodes] = nodeVals

        if filename is not None:
            self.writeVTU(filename, point_data={"u": Tglobal})

        return Tglobal

    def computeLaplacian(self):
        nNodes = self.verts.shape[0]

        K = sp.lil_matrix((nNodes, nNodes))
        M = sp.lil_matrix((nNodes, nNodes))

        for k, tri in enumerate(self.connectivity):
            j, i = np.meshgrid(tri, tri)
            B, J = self.Bmatrix(k)
            k = self.StiffnessMatrix(B, J)
            K[i, j] += k
            m = self.MassMatrix(J)
            M[i, j] += m

        return K.tocsr(), M.tocsr()

    def uvmap(self, filename=None):
        # this is only for meshes with one hole
        around_nodes, bc_u, bc_v = self.uv_bc()
        u = self.computeLaplace(around_nodes[:-1], bc_u)
        v = self.computeLaplace(around_nodes[:-1], bc_v)

        if filename is not None:
            self.writeVTU(point_data={"u": u, "v": v})
        uv = np.vstack([u, v]).T
        self.uv = uv

    def compute_uvscaling(self):
        if self.uv is None:
            self.uvmap()
        metrics = []
        for e in range(self.connectivity.shape[0]):
            B, J = self.Bmatrix(e)
            uv_e = self.uv[self.connectivity[e]]
            F = np.matmul(B, uv_e) / J
            metrics.append(np.matmul(F.T, F))
        eigvals, _ = np.linalg.eig(metrics)
        self.uvscaling = (eigvals[:, 0] + eigvals[:, 1]) / 2
        if self.uvscaling.min() < 0:
            logger.error("Flipped triangles detected — check mesh quality")
            raise ValueError("Flipped triangles detected — check mesh quality")

    def detect_boundary(self):
        edge_dict = collections.defaultdict(list)

        for e, el in enumerate(self.connectivity):
            for edge in [[0, 1], [1, 2], [2, 0]]:
                edge_dict[(min(el[edge]), max(el[edge]))].append(e)
        boundary_edges = []

        for edge, tris in edge_dict.items():
            if len(tris) == 1:
                boundary_edges.append(edge)

        self.boundary_edges = boundary_edges

    def uv_bc(self):

        if self.boundary_edges is None:
            self.detect_boundary()

        boundary_node2edge = collections.defaultdict(list)
        for edge in self.boundary_edges:
            boundary_node2edge[edge[0]].append(edge)
            boundary_node2edge[edge[1]].append(edge)

        around_nodes = list(self.boundary_edges[0])
        last_edge = self.boundary_edges[0]

        while around_nodes[0] != around_nodes[-1]:
            edges = boundary_node2edge[around_nodes[-1]].copy()
            edges.remove(last_edge)
            nodes = list(edges[0])
            nodes.remove(around_nodes[-1])
            around_nodes.append(nodes[0])
            last_edge = edges[0]
            if len(around_nodes) >= self.verts.shape[0]:
                logger.error(
                    "UV boundary traversal exceeded mesh size — boundary may be broken"
                )
                raise ValueError("UV boundary traversal exceeded mesh size — boundary may be broken")

        lengths = np.cumsum(
            np.linalg.norm(
                self.verts[around_nodes[:-1]] - self.verts[around_nodes[1:]], axis=1
            )
        )
        total_length = lengths[-1]
        bc_u = np.sin(2 * np.pi * lengths / total_length)
        bc_v = np.cos(2 * np.pi * lengths / total_length)

        return around_nodes, bc_u, bc_v

    def compute_triareas(self):
        self.triareas = []
        for e in range(self.connectivity.shape[0]):
            B, J = self.Bmatrix(e)
            self.triareas.append(J / 2)
        self.triareas = np.array(self.triareas)

    def tri2node_interpolation(self, cell_field):
        if self.triareas is None:
            self.compute_triareas()
        node_field = []
        for i in range(self.verts.shape[0]):
            tris = self.node_to_tri[i]
            areas = self.triareas[tris]
            fields = cell_field[tris]
            nodal_val = np.sum(areas * fields) / np.sum(areas)
            node_field.append(nodal_val)
        return node_field

    # def compute_uvscaling_nodes(self):
    #     uvmesh =
    #     if self.uvscaling is None:
    #         self.compute_uvscaling()
    #     self.uvscaling_nodes = self.tri2node_interpolation(self.uvscaling)
