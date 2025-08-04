"""Module defining the Parameters class for configuring fractal tree generation.

This module provides the Parameters class, which holds all settings
for generating a fractal tree structure.
"""

from typing import List


class Parameters:
    """Holds settings for generating a fractal tree structure.

    Attributes:
        meshfile (Optional[str]): Path to the mesh OBJ file.
        init_node_id (int): Index of the initial node in the mesh.
        second_node_id (int): Index of the second node, which sets the initial growth direction.
        init_length (float): Length of the first branch.
        N_it (int): Number of branch generations (iterations).
        length (float): Mean length of tree branches.
        branch_angle (float): Angle (in radians) between consecutive branches.
        w (float): Weight parameter for branch divergence.
        l_segment (float): Approximate length of each branch segment.
        fascicles_angles (List[float]): Angles (in radians) for each fascicle branch.
        fascicles_length (List[float]): Lengths for each fascicle branch; matches `fascicles_angles`.

    Notes:
        - `second_node_id` should not be adjacent to `init_node_id` in the mesh to avoid numeric instability.
        - Fascicles represent straight sub-branches inspired by cardiac fascicles.
        - Set parameters to prevent negative branch lengths when randomization is applied.
    """

    def __init__(self) -> None:
        self.meshfile = None
        self.init_node_id = 0
        self.second_node_id = 1
        self.init_length = 0.1
        # Number of iterations (generations of branches)
        self.N_it = 10
        # Median length of the branches
        self.length = 0.1
        # Standard deviation of the length
        # Min length to avoid negative length
        self.branch_angle = 0.15
        self.w = 0.1
        # Length of the segments (approximately, because the lenght of the branch is random)
        self.l_segment = 0.01

        ###########################################
        # Fascicles data
        ###########################################
        self.fascicles_angles: List[float] = []  # rad
        self.fascicles_length: List[float] = []
