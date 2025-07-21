import numpy as np
from scipy.spatial import cKDTree


class Nodes:
    """A class containing the nodes of the branches plus some fuctions to compute distance related quantities.

    Args:
        init_node (array): an array with the coordinates of the initial node of the first branch.

    Attributes:
        nodes (list): list of arrays containing the coordinates of the nodes
        last_node (int): last added node.
        end_nodes (list): a list containing the indices of all end nodes (nodes that are not connected) of the tree.
        tree (scipy.spatial.cKDTree): a k-d tree to compute the distance from any point to the closest node in the tree. It is updated once a branch is finished.
        collision_tree (scipy.spatial.cKDTree): a k-d tree to compute the distance from any point to the closest node in the tree, except from the brother and mother branches. It is used to check collision between branches.

    """

    def __init__(self, init_node):
        self.nodes = []
        self.nodes.append(init_node)
        self.last_node = 0
        self.end_nodes = []
        self.tree = cKDTree(self.nodes)

    def add_nodes(self, queue):
        """This function stores a list of nodes of a branch and returns the node indices. It also updates the tree to compute distances.

        Args:
            queue (list): a list of arrays containing the coordinates of the nodes of one branch.

        Returns:
            nodes_id (list): the indices of the added nodes.
        """
        nodes_id = []
        for point in queue:
            self.nodes.append(point)
            self.last_node += 1
            nodes_id.append(self.last_node)
        self.tree = cKDTree(self.nodes)
        return nodes_id

    def distance_from_point(self, point):
        """This function returns the distance from any point to the closest node in the tree.

        Args:
            point (array): the coordinates of the point to calculate the distance from.

        Returns:
            d (float): the distance between point and the closest node in the tree.
        """
        d, node = self.tree.query(point)
        return d

    def distance_from_node(self, node):
        """This function returns the distance from any node to the closest node in the tree.

        Args:
            node (int): the index of the node to calculate the distance from.

        Returns:
            d (float): the distance between specified node and the closest node in the tree.
        """
        d, node = self.tree.query(self.nodes[node])
        return d

    def update_collision_tree(self, nodes_to_exclude):
        """This function updates the collision_tree excluding a list of nodes from all the nodes in the tree. If all the existing nodes are excluded, one distant node is added.

        Args:
            nodes_to_exclude (list): contains the nodes to exclude from the tree. Usually it should be the mother and the brother branch nodes.

        Returns:
            none
        """
        nodes = set(range(len(self.nodes)))
        nodes = nodes.difference(nodes_to_exclude)
        nodes_to_consider = [self.nodes[x] for x in nodes]
        self.nodes_to_consider_keys = [x for x in nodes]
        if len(nodes_to_consider) == 0:
            nodes_to_consider = [
                np.array([-100000000000.0, -100000000000.0, -100000000000.0])
            ]
            self.nodes_to_consider_keys = [100000000]
            print("no nodes to consider")
        self.collision_tree = cKDTree(nodes_to_consider)

    def collision(self, point):
        """This function returns the distance between one point and the closest node in the tree and the index of the closest node using the collision_tree.

        Args:
            point (array): the coordinates of the point to calculate the distance from.

        Returns:
            collision (tuple): (distance to the closest node, index of the closest node)
        """
        d, node = self.collision_tree.query(point)
        collision = (self.nodes_to_consider_keys[node], d)
        return collision

    def gradient(self, point):
        """This function returns the gradient of the distance from the existing points of the tree from any point. It uses a central finite difference approximation.

        Args:
            point (array): the coordinates of the point to calculate the gradient of the distance from.

        Returns:
            grad (array): (x,y,z) components of gradient of the distance.
        """

        if True:
            point = np.array(point)
            d, idx = self.tree.query(point)
            if np.isclose(d, 0.0):
                return np.array([0, 0, 0])
            p2 = self.nodes[idx]
            # print(point,p2,self.tree.query(point))
            grad2 = (point - p2) / np.linalg.norm(p2 - point)
            return grad2

        delta = 0.001
        dx = np.array([delta, 0, 0])
        dy = np.array([0.0, delta, 0.0])
        dz = np.array([0.0, 0.0, delta])
        distx_m = self.distance_from_point(point - dx)
        distx_p = self.distance_from_point(point + dx)
        disty_m = self.distance_from_point(point - dy)
        disty_p = self.distance_from_point(point + dy)
        distz_m = self.distance_from_point(point - dz)
        distz_p = self.distance_from_point(point + dz)
        grad = np.array(
            [
                (distx_p - distx_m) / (2 * delta),
                (disty_p - disty_m) / (2 * delta),
                (distz_p - distz_m) / (2 * delta),
            ]
        )

        return grad
