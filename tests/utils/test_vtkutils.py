import numpy as np
import vtk
import pytest
from purkinje_uv import (
    IGBReader,
    vtk_unstructuredgrid_from_list,
    vtkIGBReader,
    vtk_extract_boundary_surfaces,
)


def test_vtk_unstructuredgrid_from_list_creates_line():
    """Test creation of a VTK line from two points.

    Verifies that `vtk_unstructuredgrid_from_list` correctly builds a
    `vtkUnstructuredGrid` when given two 3D points and a single line cell.

    Asserts:
        - Output is an instance of `vtkUnstructuredGrid`.
        - Grid contains exactly two points.
        - Grid contains one cell of type `VTK_LINE`.
    """
    # Define two points in 3D
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    # One line cell connecting point 0 and 1
    cells = np.array([[0, 1]])

    vtk_type = vtk.VTK_LINE

    grid = vtk_unstructuredgrid_from_list(xyz, cells, vtk_type)

    assert isinstance(grid, vtk.vtkUnstructuredGrid)
    assert grid.GetNumberOfPoints() == 2
    assert grid.GetNumberOfCells() == 1
    assert grid.GetCellType(0) == vtk.VTK_LINE


@pytest.mark.parametrize("cell_centered", [True, False])
def test_vtkIGBReader_reads_dummy_file(tmp_path, monkeypatch, cell_centered):
    """Test `vtkIGBReader` on a dummy float32 file with mocked header.

    This test:
        - Creates a dummy 2×2×2 binary file with 8 float32 values.
        - Mocks `IGBReader.read_header` to return a compatible header.
        - Reads the file using `vtkIGBReader`.
        - Verifies output is a `vtkImageData` with expected dimensions.

    Args:
        tmp_path (Path): pytest fixture for temp directory.
        monkeypatch: pytest fixture to override the header reader.
        cell_centered (bool): Tests both cell-centered and point-centered data.
    """
    # Mock header to simulate 2×2×2 grid of float32 voxels
    mock_header = {
        "x": 2,
        "y": 2,
        "z": 2,
        "type": "float",
    }
    monkeypatch.setattr(IGBReader, "read_header", lambda _: mock_header)

    # Create binary file with 8 float32 values
    dummy_data = np.arange(8, dtype=np.float32)
    file_path = tmp_path / "dummy.igb"
    dummy_data.tofile(file_path)

    # Read using vtkIGBReader
    img = vtkIGBReader(
        str(file_path),
        name="cell",
        cell_centered=cell_centered,
        scale=1.5,
        origin=2.0,
    )

    assert isinstance(img, vtk.vtkImageData)

    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    assert spacing == (1.5, 1.5, 1.5)
    assert origin == (2.0, 2.0, 2.0)

    if cell_centered:
        assert img.GetCellData().GetScalars() is not None
        assert img.GetCellData().GetScalars().GetNumberOfTuples() == 8
    else:
        assert img.GetPointData().GetScalars() is not None
        assert img.GetPointData().GetScalars().GetNumberOfTuples() == 8


@pytest.mark.parametrize("extent_offset", [0, 5])
def test_vtk_extract_boundary_surfaces_with_offset_extent(extent_offset):
    """Test `vtk_extract_boundary_surfaces` with non-zero extent origin.

    Verifies that the reshape logic works correctly regardless of offset
    in the VTK extent, e.g., [5,7] instead of [0,2].
    """
    BLOOD_CODE = 104
    LV_CODE = 99
    RV_CODE = 111

    # Create 2×2×2 array of labeled voxels
    data = np.zeros((2, 2, 2), dtype=np.uint8)
    data[0, 0, 0] = BLOOD_CODE
    data[0, 0, 1] = LV_CODE
    data[0, 1, 0] = RV_CODE

    extent = [
        extent_offset,
        extent_offset + 2,  # x: 2 cells
        extent_offset,
        extent_offset + 2,  # y: 2 cells
        extent_offset,
        extent_offset + 2,  # z: 2 cells
    ]

    img = vtk.vtkImageData()
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)
    img.SetExtent(*extent)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    img.GetCellData().SetScalars(None)

    flat = data.ravel(order="F")
    arr = vtk.vtkUnsignedCharArray()
    arr.SetName("cell")
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(flat.size)
    for i, val in enumerate(flat):
        arr.SetTuple1(i, val)

    img.GetCellData().SetScalars(arr)

    surf = vtk_extract_boundary_surfaces(img, triangulate=False)

    # Assertions
    assert isinstance(surf, vtk.vtkPolyData)
    side_array = surf.GetPointData().GetArray("side")
    assert side_array is not None

    side_values = [
        int(side_array.GetTuple1(i)) for i in range(side_array.GetNumberOfTuples())
    ]
    assert 1 in side_values
    assert 2 in side_values
