import numpy as np
from scipy.spatial import cKDTree
from typing import Any, List, Sequence, Tuple, Union
from numpy.typing import NDArray


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

    nodes: List[NDArray[Any]]
    end_nodes: List[int]
    tree: cKDTree

    def __init__(self, init_node: NDArray[Any]) -> None:
        self.nodes = [init_node]
        self.last_node = 0
        self.end_nodes = []
        self.tree = cKDTree(self.nodes)

    def add_nodes(self, queue: Sequence[NDArray[Any]]) -> List[int]:
        """This function stores a list of nodes of a branch and returns the node indices. It also updates the tree to compute distances.

        Args:
            queue (list): a list of arrays containing the coordinates of the nodes of one branch.

        Returns:
            nodes_id (list): the indices of the added nodes.
        """
        nodes_id: List[int] = []
        for point in queue:
            self.nodes.append(point)
            self.last_node += 1
            nodes_id.append(self.last_node)
        self.tree = cKDTree(self.nodes)
        return nodes_id

    def distance_from_point(self, point: Union[NDArray[Any], Sequence[float]]) -> float:
        """This function returns the distance from any point to the closest node in the tree.

        Args:
            point (array): the coordinates of the point to calculate the distance from.

        Returns:
            d (float): the distance between point and the closest node in the tree.
        """
        res = self.tree.query(point)
        d = float(res[0])
        return d

    def distance_from_node(self, node: int) -> float:
        """This function returns the distance from any node to the closest node in the tree.

        Args:
            node (int): the index of the node to calculate the distance from.

        Returns:
            d (float): the distance between specified node and the closest node in the tree.
        """
        res = self.tree.query(self.nodes[node])
        d = float(res[0])
        return d

    def update_collision_tree(self, nodes_to_exclude: Sequence[int]) -> None:
        """
        Update the collision_tree by excluding specified nodes.
        If no nodes remain, insert one distant dummy point so the KD‐tree never ends up empty.

        Args:
            nodes_to_exclude (Sequence[int]): Indices of nodes to exclude.
        """

        nodes: set[int] = set(range(len(self.nodes)))
        nodes = nodes.difference(nodes_to_exclude)
        nodes_to_consider: List[NDArray[Any]] = [self.nodes[i] for i in nodes]
        self.nodes_to_consider_keys = list(nodes)

        if len(nodes_to_consider) == 0:
            dummy: NDArray[Any] = np.array([-1e11, -1e11, -1e11], dtype=float)
            nodes_to_consider = [dummy]
            self.nodes_to_consider_keys = [100000000]
            print("no nodes to consider")

        self.collision_tree = cKDTree(nodes_to_consider)

    def collision(
        self, point: Union[NDArray[Any], Sequence[float]]
    ) -> Tuple[int, float]:
        """
        Return (node_index, distance) for the closest node to `point` using the collision_tree.
        """
        res = self.collision_tree.query(point)
        d = float(res[0])
        idx = int(res[1])
        return (self.nodes_to_consider_keys[idx], d)

    def gradient(self, point: Union[NDArray[Any], Sequence[float]]) -> NDArray[Any]:
        """This function returns the gradient of the distance from the existing points of the tree from any point. It uses a central finite difference approximation.

        Args:
            point (array): the coordinates of the point to calculate the gradient of the distance from.

        Returns:
            grad (array): (x,y,z) components of gradient of the distance.
        """

        arr = np.array(point, dtype=float)
        res = self.tree.query(arr)
        d = float(res[0])
        idx = int(res[1])

        if np.isclose(d, 0.0):
            zero_grad: NDArray[Any] = np.array([0.0, 0.0, 0.0], dtype=float)
            return zero_grad

        p2 = self.nodes[idx]
        diff = arr - p2
        grad2: NDArray[Any] = diff / np.linalg.norm(p2 - arr)
        return grad2

        # TODO is this deprecated?
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
