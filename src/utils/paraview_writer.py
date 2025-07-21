import xml.etree.cElementTree as ET

class VTUWriter:
    @staticmethod
    def write_line_vtu(nodes, elements, filename):
        file = ET.Element("VTKFile", {
            'type': 'UnstructuredGrid',
            'version': '0.1',
            'byte_order': 'BigEndian'
        })

        unstructured_grid = ET.SubElement(file, 'UnstructuredGrid')
        piece = ET.SubElement(unstructured_grid, 'Piece', {
            'NumberOfPoints': str(len(nodes)),
            'NumberOfCells': str(len(elements))
        })

        # Points
        points = ET.SubElement(piece, 'Points')
        point_data = ET.SubElement(points, 'DataArray', {
            'type': 'Float32',
            'NumberOfComponents': '3',
            'Format': 'ascii'
        })
        point_data.text = '\n'.join(f"{x} {y} {z}" for x, y, z in nodes)

        # Cells
        cells = ET.SubElement(piece, 'Cells')
        connectivity = ET.SubElement(cells, 'DataArray', {
            'type': 'Int32',
            'Name': 'connectivity',
            'Format': 'ascii'
        })
        types = ET.SubElement(cells, 'DataArray', {
            'type': 'Int32',
            'Name': 'types',
            'Format': 'ascii'
        })
        offsets = ET.SubElement(cells, 'DataArray', {
            'type': 'Int32',
            'Name': 'offsets',
            'Format': 'ascii'
        })

        connectivity.text = '\n'.join(f"{a} {b}" for a, b in elements)
        types.text = '\n'.join(['3'] * len(elements))  # 3 = VTK_LINE
        offsets.text = '\n'.join(str(i) for i in range(2, 2 * len(elements) + 1, 2))

        # Write to file
        tree = ET.ElementTree(file)
        tree.write(filename)
