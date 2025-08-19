"""Module defining the Nodes class for managing tree nodes and spatial queries.

This module provides the Nodes class, which stores branch nodes and offers methods
to compute distances, collisions, and gradients via KD‐trees.
"""


from __future__ import annotations

import logging
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .config import backend_name, to_cpu

_LOGGER = logging.getLogger(__name__)
_DUMMY_POINT = np.array([-1e11, -1e11, -1e11], dtype=float)
_DUMMY_KEY: int = 100_000_000


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
    collision_tree: cKDTree
    nodes_to_consider_keys: list[int]

    def __init__(self, init_node: NDArray[Any]) -> None:
        """Initialize with a single starting node.

        Args:
            init_node (NDArray[Any]): Coordinates of the initial branch node.
        """
        init_np = np.asarray(to_cpu(init_node), dtype=float)

        # Core state
        self.nodes: list[NDArray[Any]] = [init_np]
        self.last_node: int = 0
        self.end_nodes: list[int] = []

        # KD-trees (SciPy / CPU)
        self.tree = cKDTree(self.nodes)
        self.collision_tree = cKDTree(self.nodes)  # default: all nodes considered
        self.nodes_to_consider_keys: list[int] = [0]

        _LOGGER.debug(
            "Nodes initialized with 1 node on backend=%s; first=%s",
            backend_name(),
            init_np.tolist(),
        )

    def add_nodes(self, queue: Sequence[NDArray[Any]]) -> List[int]:
        """Append a sequence of new nodes and rebuild the KD-tree.

        Args:
            queue (Sequence[NDArray[Any]]): Coordinates of nodes to add.

        Returns:
            List[int]: Indices of the newly added nodes.
        """
        if not queue:
            _LOGGER.debug("add_nodes called with empty queue; no nodes added.")
            return []

        nodes_id: List[int] = []
        for point in queue:
            p = np.asarray(to_cpu(point), dtype=float)
            self.nodes.append(p)
            self.last_node += 1
            nodes_id.append(self.last_node)

        self.tree = cKDTree(self.nodes)

        _LOGGER.debug(
            "Added %d nodes; total=%d. Rebuilt main KD-tree.",
            len(nodes_id),
            len(self.nodes),
        )
        return nodes_id

    def distance_from_point(self, point: Union[NDArray[Any], Sequence[float]]) -> float:
        """Compute distance from an arbitrary point to the nearest node.

        Args:
            point (Union[NDArray[Any], Sequence[float]]): Query coordinates.

        Returns:
            float: Distance to the closest node.
        """
        arr = np.asarray(to_cpu(point), dtype=float)
        d, _ = self.tree.query(arr)
        dist = float(d)
        _LOGGER.debug("distance_from_point(%s) -> %g", arr.tolist(), dist)
        return dist

    def distance_from_node(self, node: int) -> float:
        """Compute distance from one node to its nearest neighbor in the tree.

        Args:
            node (int): Index of the node to query.

        Returns:
            float: Distance to the closest other node.
        """
        d, _ = self.tree.query(self.nodes[node], k=2)
        try:
            dist = float(d[1])
        except Exception:
            # Fallback if SciPy returns a scalar or unexpected shape
            dist = float(d)

        _LOGGER.debug("distance_from_node(%d) -> %g", node, dist)
        return dist

    def update_collision_tree(self, nodes_to_exclude: Sequence[int]) -> bool:
        """Rebuild the collision KD-tree excluding specified nodes.

        If all nodes are excluded, inserts a distant dummy point so the KD-tree
        remains non-empty. Returns ``True`` on success; on failure, preserves the
        previous collision tree and returns ``False``.

        Args:
            nodes_to_exclude: Indices to omit from collision checks.

        Returns:
            bool: ``True`` if the collision KD-tree was rebuilt and installed,
            ``False`` if an error occurred (previous tree remains active).
        """
        try:
            total = len(self.nodes)
            # Sanitize + bound-check indices
            exclude_set = {int(i) for i in nodes_to_exclude if 0 <= int(i) < total}
            invalid = len(nodes_to_exclude) - len(exclude_set)
            if invalid:
                _LOGGER.debug(
                    "update_collision_tree: %d invalid exclusion indices ignored (total=%d).",
                    invalid,
                    total,
                )

            include_indices = [i for i in range(total) if i not in exclude_set]
            nodes_to_consider: List[NDArray[Any]] = [
                self.nodes[i] for i in include_indices
            ]
            pending_keys: List[int] = include_indices

            if not nodes_to_consider:
                # Keep the tree non-empty to avoid query errors.
                nodes_to_consider = [_DUMMY_POINT]
                pending_keys = [_DUMMY_KEY]
                _LOGGER.warning(
                    "update_collision_tree: no nodes to consider after exclusions; inserting dummy point."
                )

            # Build new tree first (no state changes yet).
            new_tree = cKDTree(nodes_to_consider)

            # Commit: install both tree and keys atomically.
            self.collision_tree = new_tree
            self.nodes_to_consider_keys = pending_keys

            _LOGGER.debug(
                "Collision KD-tree rebuilt: kept=%d excluded=%d total=%d",
                len(self.nodes_to_consider_keys),
                len(exclude_set),
                total,
            )
            return True

        except Exception:
            # Preserve previous state; surface details for debugging.
            _LOGGER.exception(
                "update_collision_tree failed; keeping previous collision tree."
            )
            return False

    def collision(
        self, point: Union[NDArray[Any], Sequence[float]]
    ) -> Tuple[int, float]:
        """Find the nearest node (excluding excluded ones) to a query point.

        Args:
            point (Union[NDArray[Any], Sequence[float]]): Query coordinates.

        Returns:
            Tuple[int, float]: (node_index, distance) to the closest node.
        """
        arr = np.asarray(to_cpu(point), dtype=float)

        # Ensure a collision tree exists; fall back to full set if needed.
        if not hasattr(self, "collision_tree") or not hasattr(
            self, "nodes_to_consider_keys"
        ):
            _LOGGER.warning(
                "collision(): collision tree not initialized; falling back to full set."
            )
            self.collision_tree = cKDTree(self.nodes if self.nodes else [_DUMMY_POINT])
            self.nodes_to_consider_keys = (
                list(range(len(self.nodes))) if self.nodes else [_DUMMY_KEY]
            )

        try:
            d, idx = self.collision_tree.query(arr)
            int_idx = int(idx)

            # Bounds check the mapping back to global indices.
            if not (0 <= int_idx < len(self.nodes_to_consider_keys)):
                _LOGGER.error(
                    "collision(): index %d out of bounds for keys len=%d",
                    int_idx,
                    len(self.nodes_to_consider_keys),
                )
                return -1, float("inf")

            key = int(self.nodes_to_consider_keys[int_idx])
            dist = float(d)
            _LOGGER.debug("collision(%s) -> (idx=%d, dist=%g)", arr.tolist(), key, dist)
            return (key, dist)

        except Exception:
            _LOGGER.exception("collision() failed; returning (-1, inf).")
            return (-1, float("inf"))

    def gradient(self, point: Union[NDArray[Any], Sequence[float]]) -> NDArray[Any]:
        """Approximate the gradient of the distance field at a point.

        Uses a central difference if needed, but by default returns a unit vector
        pointing away from the nearest node.

        Args:
            point (Union[NDArray[Any], Sequence[float]]): Query coordinates.

        Returns:
            NDArray[Any]: (dx, dy, dz) gradient components.
        """
        arr = np.asarray(to_cpu(point), dtype=float)
        try:
            d, idx = self.tree.query(arr)
            dist = float(d)
            nn_index = int(idx)

            if not np.isfinite(dist) or np.isclose(dist, 0.0):
                _LOGGER.debug(
                    "gradient(%s): non-finite or zero distance -> zero vector",
                    arr.tolist(),
                )
                return np.array([0.0, 0.0, 0.0], dtype=float)

            p2 = self.nodes[nn_index]
            diff = arr - p2
            nrm = float(np.linalg.norm(diff))
            if nrm == 0.0 or not np.isfinite(nrm):
                _LOGGER.debug(
                    "gradient(%s): degenerate diff -> zero vector", arr.tolist()
                )
                return np.array([0.0, 0.0, 0.0], dtype=float)

            grad = diff / nrm
            _LOGGER.debug(
                "gradient(%s): nn=%d dist=%g grad=%s",
                arr.tolist(),
                nn_index,
                dist,
                grad.tolist(),
            )
            return grad

        except Exception:
            _LOGGER.exception("gradient() failed; returning zero vector.")
            return np.array([0.0, 0.0, 0.0], dtype=float)

        # TODO is this deprecated?
        # delta = 0.001
        # dx = np.array([delta, 0, 0])
        # dy = np.array([0.0, delta, 0.0])
        # dz = np.array([0.0, 0.0, delta])
        # distx_m = self.distance_from_point(point - dx)
        # distx_p = self.distance_from_point(point + dx)
        # disty_m = self.distance_from_point(point - dy)
        # disty_p = self.distance_from_point(point + dy)
        # distz_m = self.distance_from_point(point - dz)
        # distz_p = self.distance_from_point(point + dz)
        # grad = np.array(
        #     [
        #         (distx_p - distx_m) / (2 * delta),
        #         (disty_p - disty_m) / (2 * delta),
        #         (distz_p - distz_m) / (2 * delta),
        #     ]
        # )
        #
        # return grad
