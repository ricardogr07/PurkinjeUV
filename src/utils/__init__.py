"""The utils package contains file readers and VTK exporters used by purkinje_uv.

Submodules:
  - igb_reader: IGBReader for binary image grid files.
  - paraview_writer: VTUWriter for writing line-based meshes.
  - vtkutils: Helpers for VTK numpy interoperability.

Utilities:
  IGBReader, VTUWriter, vtk_unstructuredgrid_from_list
"""

from utils.igb_reader import IGBReader
from utils.paraview_writer import VTUWriter
from utils.vtkutils import (
    vtk_unstructuredgrid_from_list,
    vtkIGBReader,
    vtk_extract_boundary_surfaces,
)

__all__ = [
    "IGBReader",
    "VTUWriter",
    "vtk_unstructuredgrid_from_list",
    "vtkIGBReader",
    "vtk_extract_boundary_surfaces",
]
