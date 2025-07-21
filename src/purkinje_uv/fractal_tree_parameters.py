class Parameters:
    """
    Class to specify the parameters for generating a fractal tree structure.
        meshfile (str): Path and filename to the mesh OBJ file.
        init_node_id (int): Index of the initial node in the mesh.
        second_node_id (int): Index of the second node, used to determine the initial direction of the tree.
        init_length (float): Length of the first branch.
        N_it (int): Number of generations (iterations) of branches.
        length (float): Average length of the branches in the tree.
        branch_angle (float): Angle (in radians) between the previous branch and the new branch.
        w (float): Repulsivity parameter controlling branch divergence.
        l_segment (float): Approximate length of segments composing each branch (interpreted as element length in a finite element mesh).
        fascicles_angles (list of float): Angles (in radians) for each fascicle branch, relative to the initial branch.
        fascicles_length (list of float): Lengths for each fascicle branch; must match the size of fascicles_angles.
    Notes:
        - The second_node_id should not be directly connected to init_node_id by a single edge in the mesh to avoid numerical issues.
        - Fascicles are straight branches with different lengths and angles, motivated by the fascicles of the left ventricle.
        - To avoid negative branch lengths, ensure randomization parameters are set appropriately.
    """
    def __init__(self):
        self.meshfile = None
        self.init_node_id= 0
        self.second_node_id = 1
        self.init_length=0.1
        #Number of iterations (generations of branches)
        self.N_it=10
        #Median length of the branches
        self.length=.1
        #Standard deviation of the length
        #Min length to avoid negative length
        self.branch_angle=0.15
        self.w=0.1
        #Length of the segments (approximately, because the lenght of the branch is random)
        self.l_segment=.01

        ###########################################
        # Fascicles data
        ###########################################
        self.fascicles_angles=[] #rad
        self.fascicles_length=[]