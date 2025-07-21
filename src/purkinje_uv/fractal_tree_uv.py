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
    
    def _interpolate(self, vectors, r, t):
        return t*vectors[2] + r*vectors[1] + (1-r-t)*vectors[0]

    def _eval_field(self, point, field, mesh):
        ppoint,tri,r,t = mesh.project_new_point(point, 5)
        return self._interpolate(field[mesh.connectivity[tri]], r, t), ppoint, tri

    def _point_in_mesh(self, point, mesh):
        point = np.append(point, np.zeros(1))
        _,tri,_,_ = mesh.project_new_point(point, 5)
        return tri >= 0

    def _point_in_mesh_vtk(self, point, loc):
        point = np.append(point, np.zeros(1))
        cellId = vtk.reference(0)
        subId  = vtk.reference(0)
        d = vtk.reference(0.0)
        ppoint = np.zeros(3)
        loc.FindClosestPoint(point, ppoint, cellId, subId, d)
        return d.get() < 1e-9

    def _scaling(self,x):
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
    
    def grow_tree(self):
        branches = defaultdict(list)
        branch_id = 0

        end_nodes = []

        sister_branches = {}

        dx = self.params.l_segment
         
        init_node = self.mesh_uv.verts[self.params.init_node_id][:2]
        second_node = self.mesh_uv.verts[self.params.second_node_id][:2]
        s, tri = self._scaling(init_node) 
        if tri < 0:
            raise "the initial node is outside the domain"
        init_dir = second_node - init_node
        init_dir /= np.linalg.norm(init_dir)
        nodes = [init_node, init_node + dx*init_dir*s]
        edges = [Edge(0,1,nodes,None,branch_id)]


        edge_queue = [0]
        branches[branch_id].append(0)

        branch_length = self.params.length
        init_branch_length = self.params.init_length


        theta = self.params.branch_angle
        w = self.params.w

        Rplus = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        Rminus = np.array([[np.cos(-theta), -np.sin(-theta)],[np.sin(-theta), np.cos(-theta)]])




        for i in range(int(init_branch_length/dx)):

            edge_id = edge_queue.pop(0)
            edge = edges[edge_id]
            new_dir = edge.dir 
            new_dir /= np.linalg.norm(new_dir)
            s, tri = self._scaling(nodes[edge.n2]) 
            if tri < 0:
                raise "the initial branch goes out of the domain"
            new_node = nodes[edge.n2] + new_dir*dx*s
            new_node_id = len(nodes)
            nodes.append(new_node)
            branches[edge.branch].append(new_node_id)
            edge_queue.append(len(edges))
            edges.append(Edge(edge.n2, new_node_id,nodes, edge_id, edge.branch))

        branching_edge_id = edge_queue.pop(0)


        for fascicle_length, fascicles_angle in zip(self.params.fascicles_length, self.params.fascicles_angles):
            Rotation = np.array([[np.cos(fascicles_angle), -np.sin(fascicles_angle)],[np.sin(fascicles_angle), np.cos(fascicles_angle)]])    
            edge = edges[branching_edge_id]
            new_dir = np.matmul(Rotation, edge.dir)
            new_dir /= np.linalg.norm(new_dir)
            s, tri = self._scaling(nodes[edge.n2]) 
            if tri < 0:
                raise "the fascicle goes out of the domain"
            new_node = nodes[edge.n2] + new_dir*dx*s
            new_node_id = len(nodes)
            nodes.append(new_node)
            branch_id += 1
            branches[branch_id].append(new_node_id)
            edge_queue.append(len(edges))
            edges.append(Edge(edge.n2, new_node_id,nodes, edge_id, edge.branch))
            for i in range(int(fascicle_length/dx)):
                edge_id = edge_queue.pop(0)
                edge = edges[edge_id]
                new_dir = edge.dir 
                new_dir /= np.linalg.norm(new_dir)
                s, tri = self._scaling(nodes[edge.n2]) 
                if tri < 0:
                    raise "the fascicle goes out of the domain"
                new_node = nodes[edge.n2] + new_dir*dx*s
                new_node_id = len(nodes)
                nodes.append(new_node)
                branches[edge.branch].append(new_node_id)
                edge_queue.append(len(edges))
                edges.append(Edge(edge.n2, new_node_id,nodes, edge_id, edge.branch))



        for gen in range(self.params.N_it):
            print('generation', gen)
            branching_queue = []
            while len(edge_queue) > 0:
                edge_id = edge_queue.pop(0)
                edge = edges[edge_id]
                for R in [Rplus, Rminus]:
                    new_dir = np.matmul(R,edge.dir)
                    new_dir /= np.linalg.norm(new_dir)
                    s, tri = self._scaling(nodes[edge.n2]) 
                    new_node = nodes[edge.n2] + new_dir*dx*s
                    # aa = point_in_mesh(new_node, self.mesh_uv)
                    # bb = point_in_mesh_vtk(new_node, self.loc)
                    # if aa != bb:
                    #     print(new_node,aa,bb)
                    if not self._point_in_mesh_vtk(new_node, self.loc):
                        end_nodes.append(edge.n2)
                        continue
                    new_node_id = len(nodes)
                    nodes.append(new_node)
                    branching_queue.append(len(edges))
                    branch_id += 1
                    branches[branch_id].append(new_node_id)
                    edges.append(Edge(edge.n2, new_node_id,nodes, edge_id, branch_id))
                sister_branches[branch_id - 1] = branch_id
                sister_branches[branch_id] = branch_id - 1


            edge_queue = branching_queue

            for i in range(int(branch_length/dx)):
                growing_queue = []
                tree = cKDTree(nodes)
                new_nodes = []
                last_node = len(nodes)
                while len(edge_queue) > 0:
                    edge_id = edge_queue.pop(0)
                    edge = edges[edge_id]
                    # collision detection
                    # temp_nodes = np.array(nodes)
                    # temp_nodes[branches[edge.branch]] = np.array([1e9,1e9])
                    # temp_nodes[branches[sister_branches[edge.branch]]] = np.array([1e9,1e9])
                    # tree = cKDTree(temp_nodes)


                    # nodes[branches[edge.branch]] = np.array([1e9,1e9])
                    # nodes[branches[sister_branches[edge.branch]]] = np.array([1e9,1e9])
                    
                    pred_node = nodes[edge.n2]# + dx*edge.dir
  
                    all_dist = np.linalg.norm(nodes - pred_node, axis = 1)
                    all_dist[branches[edge.branch]] = 1e9
                    sister_dist = all_dist[branches[sister_branches[edge.branch]]]
                    all_dist[branches[sister_branches[edge.branch]]] = 1e9
                    s, tri = self._scaling(nodes[edge.n2]) 

                    if all_dist.min() < 0.9*dx*s:
                        end_nodes.append(edge.n2)
                        continue

                    # grad calculation
                    all_dist[branches[sister_branches[edge.branch]]] = sister_dist
                    # temp_nodes = np.array(new_nodes)


                    # temp_nodes = np.array(nodes)
                    # temp_nodes[branches[edge.branch]] = np.array([1e9,1e9])
                    # ttree = cKDTree(temp_nodes)
                    # pred_node = nodes[edge.n2]# + dx*edge.dir
                    # dist, closest = ttree.query(pred_node)
                    # for d, c in zip(dist, closest):
                    #     if c not in [edge.n2] + branches[edge.branch]:
                    #         break
                    grad_dist = (pred_node - nodes[np.argmin(all_dist)])/(all_dist.min())
                   # grad_dist = (pred_node - closest_node)/d_final


                    new_dir = edge.dir + w*grad_dist
                    new_dir /= np.linalg.norm(new_dir)
                    new_node = nodes[edge.n2] + new_dir*dx*s
                    new_node_id = len(nodes)
                    # aa = point_in_mesh(new_node, self.mesh_uv)
                    # bb = point_in_mesh_vtk(new_node, self.loc)
                    # if aa != bb:
                    #     print(new_node,aa,bb)
                    if not self._point_in_mesh_vtk(new_node, self.loc):            
                        end_nodes.append(edge.n2)
                        continue
                    nodes.append(new_node)
                    new_nodes.append(new_node)
                    growing_queue.append(len(edges))
                    branches[edge.branch].append(new_node_id)
                    edges.append(Edge(edge.n2, new_node_id,nodes, edge_id, edge.branch))
                edge_queue = growing_queue

        end_nodes += [edges[edge].n2 for edge in edge_queue]

        self.uv_nodes = np.array(nodes)
        self.edges = edges
        self.end_nodes = end_nodes

        self.connectivity = []
        for edge in edges:
            self.connectivity.append([edge.n1, edge.n2])

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
