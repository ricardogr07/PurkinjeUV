"""Module defining the Edge class for graph edges in a Purkinje tree.

This module provides the Edge class, representing a connection between two nodes,
including its geometric direction and optional parent/branch relationships.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence
from numpy.typing import NDArray

from .config import backend_name, norm, to_cpu, to_device

_LOGGER = logging.getLogger(__name__)


class Edge:
    """Represents an edge between two nodes in a graph.

    Each Edge connects node `n1` to node `n2`, computes the normalized
    direction vector between them, and optionally tracks a parent edge
    and branch association.

    Attributes:
        n1 (int): The ID of the first node.
        n2 (int): The ID of the second node.
        dir (NDArray[Any]): Normalized direction vector from `n1` to `n2`.
        parent (Optional[int]): The parent edge ID, if any.
        branch (Optional[int]): The branch ID, if any.
    """

    def __init__(
        self,
        n1: int,
        n2: int,
        nodes: Sequence[NDArray[Any]],
        parent: Optional[int],
        branch: Optional[int],
    ) -> None:
        """Initialize an Edge with endpoints and compute its direction.

        Args:
            n1 (int): The ID of the first node.
            n2 (int): The ID of the second node.
            nodes (Sequence[NDArray[Any]]): Sequence of node coordinate arrays.
            parent (Optional[int]): The parent edge ID, if any.
            branch (Optional[int]): The branch ID, if any.
        """
        self.n1 = n1  # ids
        self.n2 = n2  # ids

        _LOGGER.debug(
            "Initializing Edge(n1=%d, n2=%d, parent=%s, branch=%s) on backend=%s",
            n1,
            n2,
            str(parent),
            str(branch),
            backend_name(),
        )

        # Compute on the active backend (GPU if configured), then store on CPU.
        p1 = to_device(nodes[n1], dtype=float)
        p2 = to_device(nodes[n2], dtype=float)
        diff = p2 - p1
        mag = float(norm(diff))

        _LOGGER.debug("Edge raw magnitude between nodes %d and %d: %.6e", n1, n2, mag)

        if mag < 1e-12:
            _LOGGER.error(
                "Zero-magnitude direction between nodes %d and %d; cannot form Edge",
                n1,
                n2,
            )
            raise ValueError(
                f"Edge direction vector has zero magnitude between nodes {n1} and {n2}"
            )

        dir_backend = diff / mag
        self.dir: NDArray[Any] = to_cpu(dir_backend)
        self.parent = parent
        self.branch = branch

        _LOGGER.debug(
            "Edge initialized: n1=%d, n2=%d, dir=%s (stored on CPU)",
            self.n1,
            self.n2,
            self.dir.tolist(),
        )

    def __repr__(self) -> str:
        """Return a string representation of the edge."""
        return f"Edge(n1={self.n1}, n2={self.n2}, dir={self.dir.tolist()})"
