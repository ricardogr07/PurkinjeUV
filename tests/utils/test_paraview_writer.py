import os
import xml.etree.ElementTree as ET
import pytest
from tempfile import TemporaryDirectory

from utils.paraview_writer import VTUWriter


@pytest.fixture
def sample_data():
    nodes = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0)]
    elements = [(0, 1), (1, 2)]
    return nodes, elements


def test_file_creation_and_parsability(sample_data):
    nodes, elements = sample_data
    with TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test.vtu")
        VTUWriter.write_line_vtu(nodes, elements, filename)

        assert os.path.isfile(filename)

        try:
            tree = ET.parse(filename)
            root = tree.getroot()
        except ET.ParseError:
            pytest.fail("Output VTU file is not valid XML")

        assert root.tag == "VTKFile"
        assert root.attrib["type"] == "UnstructuredGrid"


def test_number_of_points_and_cells(sample_data):
    nodes, elements = sample_data
    with TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test.vtu")
        VTUWriter.write_line_vtu(nodes, elements, filename)

        tree = ET.parse(filename)
        root = tree.getroot()

        piece = root.find(".//Piece")
        assert piece is not None
