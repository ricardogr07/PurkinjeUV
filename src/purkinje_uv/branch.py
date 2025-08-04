"""Module defining the Branch class for fractal tree growth on a mesh.

This module contains the Branch class, which represents a single branch in the fractal tree.
"""

import numpy as np
from typing import Any, Sequence
from numpy.typing import NDArray
from .nodes import Nodes
from .mesh import Mesh
import logging

logger = logging.getLogger(__name__)


class Branch:
    """Represents a branch of the fractal tree on a mesh.

    Attributes:
        mesh (Mesh): Mesh on which the branch grows.
        queue (list[NDArray[Any]]): Coordinates of points queued for growth.
        triangles (list[int]): Triangle indices for each queued point.
        nodes (list[int]): Node indices in the global Nodes manager.
        growing (bool): Whether the branch is still growing.
    """

    def __init__(
        self,
        mesh: Mesh,
        init_node: int,
        init_dir: NDArray[Any],
        init_tri: int,
        length: float,
        angle: float,
        w: float,
        nodes: Nodes,
        brother_nodes: Sequence[int],
        Nsegments: int,
    ) -> None:
        """Initialize a Branch.

        Args:
            mesh (Mesh): Mesh on which the branch grows.
            init_node (int): Index of the initial node in the mesh.
            init_dir (NDArray[Any]): Direction vector for the initial growth.
            init_tri (int): Index of the triangle containing the initial node.
            length (float): Total length of the branch.
            angle (float): Growth angle parameter.
            w (float): Weight for gradient adjustment.
            nodes (Nodes): Nodes manager to track new nodes.
            brother_nodes (Sequence[int]): Indices of brother nodes.
            Nsegments (int): Number of segments to divide the branch.
        """
        #        self.nnodes=0
        self.child = [0, 0]
        self.dir = np.array([0.0, 0.0, 0.0])
        self.nodes = []
        self.triangles = []
        #        self.normal=np.array([0.0,0.0,0.0])
        self.queue = []
        self.growing = True
        init_normal = mesh.normals[init_tri]
        nodes.update_collision_tree(brother_nodes)
        #        global_nnodes=len(nodes.nodes)

        #  R=np.array([[np.cos(angle),-np.sin(angle)],[ np.sin(angle), np.cos(angle)]])
        inplane = -np.cross(init_dir, init_normal)
        dir = np.cos(angle) * init_dir + np.sin(angle) * inplane
        dir = dir / np.linalg.norm(dir)
        self.nodes.append(init_node)
        self.queue.append(nodes.nodes[init_node])
        self.triangles.append(init_tri)
        grad = nodes.gradient(self.queue[0])
        dir = (dir + w * grad) / np.linalg.norm(dir + w * grad)
        #    print nodes.nodes[init_node]+dir*l/Nsegments
        for i in range(1, Nsegments):
            intriangle = self.add_node_to_queue(
                mesh, self.queue[i - 1], dir * length / Nsegments
            )
            # print 'intriangle',intriangle
            if not intriangle:
                print("Point not in triangle", i)
                #                print self.queue[i-1]+dir*l/50.
                self.growing = False
                break
            collision = nodes.collision(self.queue[i])
            if collision[1] < length / 5.0:
                # print("Collision",i, collision)
                self.growing = False
                self.queue.pop()
                self.triangles.pop()
                break
            grad = nodes.gradient(self.queue[i])
            normal = mesh.normals[self.triangles[i], :]
            # Project the gradient to the surface
            grad = grad - (np.dot(grad, normal)) * normal
            dir = (dir + w * grad) / np.linalg.norm(dir + w * grad)

        nodes_id = nodes.add_nodes(self.queue[1:])
        for x in nodes_id:
            self.nodes.append(x)

        if not self.growing:
            nodes.end_nodes.append(self.nodes[-1])
        self.dir = dir
        # #print self.triangles
        self.tri = self.triangles[-1]

    # TODO: Can this be deprecated?
    # Uncomment the following lines for a closed network
    #   if shared_node is not -1:
    #      self.nodes.append(shared_node)

    def add_node_to_queue(
        self,
        mesh: Mesh,
        init_node: NDArray[Any],
        dir: NDArray[Any],
    ) -> bool:
        """Project a new node onto the mesh and add it to the growth queue.

        This method projects a direction from a starting point onto the mesh surface.
        If the projected point lies within a mesh triangle, it is appended to the queue and triangles list.

        Args:
            mesh (Mesh): Mesh on which the branch grows.
            init_node (NDArray[Any]): Coordinates of the last node in the branch.
            dir (NDArray[Any]): Direction vector from the initial node to the new node.

        Returns:
            bool: True if the projected node lies within a mesh triangle; False otherwise.
        """
        logger.debug('node trying to project %s', init_node + dir)
        point, triangle, *_ = mesh.project_new_point(init_node + dir)
        logger.debug(
            'Projected point %s, dist %s', point, np.linalg.norm(point - init_node)
        )
        if triangle >= 0:
            self.queue.append(point)
            self.triangles.append(triangle)
            success = True
        else:
            logger.debug('Projection failed: point %s, triangle %s', point, triangle)
            success = False
        logger.debug('Success? %s', success)
        return success
