import numpy as np

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
        nodes: np.ndarray,
        parent: int | None,
        branch: int | None
    ) -> None:
        
        self.n1 = n1 #ids
        self.n2 = n2 #ids

        self.dir = (nodes[n2] - nodes[n1])/np.linalg.norm(nodes[n2] - nodes[n1])
        self.parent = parent
        self.branch = branch
        