"""The purkinje_uv package provides tools for generating Purkinje fiber networks.

This package offers:
  - Fractal tree construction on surface meshes.
  - Activation solvers for Purkinje networks.
  - Finite-element utilities for surface analysis.

Submodules:
  - branch: Branch growth logic.
  - edge: Edge abstraction for tree connectivity.
  - fractal_tree: FractalTree generation.
  - fractal_tree_parameters: Parameter container for tree generation.
  - mesh: Mesh class with geometry and FEM utilities.
  - nodes: Node management and collision/gradient utilities.
  - purkinje_tree: Activation solver on the Purkinje network.

Classes:
  Branch, Edge, FractalTree, FractalTreeParameters, Mesh, Nodes, PurkinjeTree

Utilities:
  IGBReader, VTUWriter, vtkutils
"""

from .config import (
    config,
    configure,
    use,
    is_gpu,
    backend_name,
    seed,
    xp,
    rng,
    cdist,
    to_cpu,
    to_device,
    norm,
    set_log_level,
)

from purkinje_uv.branch import Branch
from purkinje_uv.edge import Edge
from purkinje_uv.fractal_tree import FractalTree
from purkinje_uv.fractal_tree_parameters import FractalTreeParameters
from purkinje_uv.mesh import Mesh
from purkinje_uv.nodes import Nodes
from purkinje_uv.purkinje_tree import PurkinjeTree

from utils.igb_reader import IGBReader
from utils.paraview_writer import VTUWriter
from utils.vtkutils import (
    vtk_unstructuredgrid_from_list,
    vtkIGBReader,
    vtk_extract_boundary_surfaces,
)

__all__ = [
    # Core classes
    "Branch",
    "Edge",
    "FractalTree",
    "FractalTreeParameters",
    "Mesh",
    "Nodes",
    "PurkinjeTree",
    # Utilities
    "IGBReader",
    "VTUWriter",
    "vtk_unstructuredgrid_from_list",
    "vtkIGBReader",
    "vtk_extract_boundary_surfaces",
    # Configuration and backend
    "config",
    "configure",
    "use",
    "is_gpu",
    "backend_name",
    "seed",
    "xp",
    "rng",
    "cdist",
    "to_cpu",
    "to_device",
    "norm",
    "set_log_level",
]
