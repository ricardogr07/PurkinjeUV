"""Module defining the Edge class for graph edges in a Purkinje tree.

This module provides the Edge class, representing a connection between two nodes,
including its geometric direction and optional parent/branch relationships.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional, Sequence


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

        diff: NDArray[Any] = nodes[n2] - nodes[n1]
        norm: float = float(np.linalg.norm(diff))
        if norm < 1e-12:
            raise ValueError(
                f"Edge direction vector has zero magnitude between nodes {n1} and {n2}"
            )
        self.dir: NDArray[Any] = diff / norm

        self.parent = parent
        self.branch = branch
