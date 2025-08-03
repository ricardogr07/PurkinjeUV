import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional, Sequence


class Edge:
    """
    Represents an edge between two nodes in a graph structure.
    Args:
        n1 (int): The ID of the first node.
        n2 (int): The ID of the second node.
        nodes (np.ndarray): Array of node coordinates.
        parent (Optional[int]): The parent edge ID, if any.
        branch (Optional[int]): The branch ID, if any.
    Attributes:
        n1 (int): The ID of the first node.
        n2 (int): The ID of the second node.
        dir (np.ndarray): The normalized direction vector from n1 to n2.
        parent (Optional[int]): The parent edge ID.
        branch (Optional[int]): The branch ID.
    """

    def __init__(
        self,
        n1: int,
        n2: int,
        nodes: Sequence[NDArray[Any]],
        parent: Optional[int],
        branch: Optional[int],
    ) -> None:
        self.n1 = n1  # ids
        self.n2 = n2  # ids

        diff: NDArray[Any] = nodes[n2] - nodes[n1]
        norm: float = float(np.linalg.norm(diff))
        self.dir: NDArray[Any] = diff / norm

        self.parent = parent
        self.branch = branch
