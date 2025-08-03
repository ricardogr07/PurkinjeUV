import numpy as np
from typing import Any, Sequence
from numpy.typing import NDArray
from .nodes import Nodes
from .mesh import Mesh


class Branch:
    """Class that contains a branch of the fractal tree

    Args:
        mesh: an object of the mesh class, where the fractal tree will grow
        init_node (int): initial node to grow the branch. This is an index that refers to a node in the nodes.nodes array.
        init_dir (array): initial direction to grow the branch. In general, it refers to the direction of the last segment of the mother brach.
        init_tri (int): the index of triangle of the mesh where the init_node sits.
        length (float): total length of the branch
        angle (float): angle (rad) with respect to the init_dir in the plane of the init_tri triangle
        w (float): repulsitivity parameter. Controls how much the branches repel each other.
        nodes: the object of the class nodes that contains all the nodes of the existing branches.
        brother_nodes (list): the nodes of the brother and mother branches, to be excluded from the collision detection between branches.
        Nsegments (int): number of segments to divide the branch.

    Attributes:
        child (list): contains the indexes of the child branches. It is not assigned when created.
        dir (array): vector direction of the last segment of the branch.
        nodes (list): contains the node indices of the branch. The node coordinates can be retrieved using nodes.nodes[i]
        triangles (list): contains the indices of the triangles from the mesh where every node of the branch lies.
        tri (int): triangle index where last node sits.
        growing (bool): False if the branch collide or is out of the surface. True otherwise.

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

    # Uncomment the following lines for a closed network
    #   if shared_node is not -1:
    #      self.nodes.append(shared_node)

    def add_node_to_queue(
        self,
        mesh: Any,
        init_node: NDArray[Any],
        dir: NDArray[Any],
    ) -> bool:
        """Functions that projects a node in the mesh surface and it to the queue is it lies in the surface.

        Args:
            mesh: an object of the mesh class, where the fractal tree will grow
            init_node (array): vector that contains the coordinates of the last node added in the branch.
            dir (array): vector that contains the direction from the init_node to the node to project.

        Return:
            success (bool): true if the new node is in the triangle.

        """
        # print 'node trying to project', init_node+dir
        point, triangle = mesh.project_new_point(init_node + dir)
        # print 'Projected point', point, 'dist', np.linalg.norm(point-init_node)
        if triangle >= 0:
            self.queue.append(point)
            self.triangles.append(triangle)
            success = True
        else:
            #            print point, triangle
            success = False
        # print 'Success? ',success
        return success
