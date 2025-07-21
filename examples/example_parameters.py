import numpy as np

from purkinje_uv.fractal_tree_parameters import Parameters as BaseParameters

class Parameters(BaseParameters):
    """Class to specify the parameters of the fractal tree.
            
    Attributes:
        meshfile (str): path and filename to obj file name.
        filename (str): name of the output files.
        init_node (numpy array): the first node of the tree.
        second_node (numpy array): this point is only used to calculate the initial direction of the tree and is not included in the tree. Please avoid selecting nodes that are connected to the init_node by a single edge in the mesh, because it causes numerical issues.
        init_length (float): length of the first branch.
        N_it (int): number of generations of branches.
        length (float): average lenght of the branches in the tree.
        std_length (float): standard deviation of the length. Set to zero to avoid random lengths.
        min_length (float): minimum length of the branches. To avoid randomly generated negative lengths.
        branch_angle (float): angle with respect to the direction of the previous branch and the new branch.
        w (float): repulsivity parameter.
        l_segment (float): length of the segments that compose one branch (approximately, because the lenght of the branch is random). It can be interpreted as the element length in a finite element mesh.
        Fascicles (bool): include one or more straigth branches with different lengths and angles from the initial branch. It is motivated by the fascicles of the left ventricle. 
        fascicles_angles (list): angles with respect to the initial branches of the fascicles. Include one per fascicle to include.
        fascicles_length (list): length  of the fascicles. Include one per fascicle to include. The size must match the size of fascicles_angles.
        save (bool): save text files containing the nodes, the connectivity and end nodes of the tree.
        save_paraview (bool): save a .vtu paraview file. The tvtk module must be installed.
        
    """
    def __init__(self):
        super().__init__()

        # Extend with additional attributes used in simulations or examples
        self.meshfile='sphere.obj'
        self.filename = "sphere-line"
        self.init_node = np.array([-1.0, 0.0, 0.0])
        self.second_node = np.array([-0.964, 0.0, 0.266])
        
        # Standard deviation and minimum branch length for randomized growth
        self.std_length = np.sqrt(0.2) * self.length

        #Min length to avoid negative length
        self.min_length = self.length / 10.0
        
        # Use fascicles and visualization options
        self.Fascicles = True
        self.save = True
        self.save_paraview = True

        ###########################################
        # Fascicles data
        ###########################################
        self.fascicles_angles=[-1.5,.2] #rad
        self.fascicles_length=[.5,.5]