"""Module defining the Nodes class for managing tree nodes and spatial queries.

This module provides the Nodes class, which stores branch nodes and offers methods
to compute distances, collisions, and gradients via KD‐trees.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Any, List, Sequence, Tuple, Union
from numpy.typing import NDArray


class Nodes:
    """Manage nodes and compute spatial queries for a tree structure.

    The Nodes class stores branch nodes and provides methods to compute
    distances, collisions, and gradients using k-d trees.

    Attributes:
        nodes (List[NDArray[Any]]): Coordinates of all nodes.
        last_node (int): Index of the most recently added node.
        end_nodes (List[int]): Indices of terminal nodes (not connected).
        tree (cKDTree): KD-tree of all nodes for nearest-neighbor queries.
        collision_tree (cKDTree): KD-tree excluding certain nodes for collision checks.
    """

    nodes: List[NDArray[Any]]
    end_nodes: List[int]
    tree: cKDTree

    def __init__(self, init_node: NDArray[Any]) -> None:
        """Initialize with a single starting node.

        Args:
            init_node (NDArray[Any]): Coordinates of the initial branch node.
        """
        self.nodes = [init_node]
        self.last_node = 0
        self.end_nodes = []
        self.tree = cKDTree(self.nodes)

    def add_nodes(self, queue: Sequence[NDArray[Any]]) -> List[int]:
        """Append a sequence of new nodes and rebuild the KD-tree.

        Args:
            queue (Sequence[NDArray[Any]]): Coordinates of nodes to add.

        Returns:
            List[int]: Indices of the newly added nodes.
        """
        nodes_id: List[int] = []
        for point in queue:
            self.nodes.append(point)
            self.last_node += 1
            nodes_id.append(self.last_node)
        self.tree = cKDTree(self.nodes)
        return nodes_id

    def distance_from_point(self, point: Union[NDArray[Any], Sequence[float]]) -> float:
        """Compute distance from an arbitrary point to the nearest node.

        Args:
            point (Union[NDArray[Any], Sequence[float]]): Query coordinates.

        Returns:
            float: Distance to the closest node.
        """
        res = self.tree.query(point)
        d = float(res[0])
        return d

    def distance_from_node(self, node: int) -> float:
        """Compute distance from one node to its nearest neighbor in the tree.

        Args:
            node (int): Index of the node to query.

        Returns:
            float: Distance to the closest other node.
        """
        res = self.tree.query(self.nodes[node], k=2)
        d = float(res[0][1])
        return d

    def update_collision_tree(self, nodes_to_exclude: Sequence[int]) -> None:
        """Rebuild the collision tree excluding specified nodes.

        If all nodes are excluded, inserts a distant dummy point
        so the KD-tree remains non-empty.

        Args:
            nodes_to_exclude (Sequence[int]): Indices to omit from collision checks.
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
        """Find the nearest node (excluding excluded ones) to a query point.

        Args:
            point (Union[NDArray[Any], Sequence[float]]): Query coordinates.

        Returns:
            Tuple[int, float]: (node_index, distance) to the closest node.
        """
        res = self.collision_tree.query(point)
        d = float(res[0])
        idx = int(res[1])
        return (self.nodes_to_consider_keys[idx], d)

    def gradient(self, point: Union[NDArray[Any], Sequence[float]]) -> NDArray[Any]:
        """Approximate the gradient of the distance field at a point.

        Uses a central difference if needed, but by default returns a unit vector
        pointing away from the nearest node.

        Args:
            point (Union[NDArray[Any], Sequence[float]]): Query coordinates.

        Returns:
            NDArray[Any]: (dx, dy, dz) gradient components.
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
