"""Module providing IGBReader for reading IGB (Image Grid Binary) files.

This module defines the IGBReader class with methods to parse IGB headers
and to load the binary data into NumPy arrays with optional scaling.
"""

import numpy as np
from typing import Dict, Any, Tuple, Union
from numpy.typing import NDArray


class IGBReader:
    """Read and parse Image Grid Binary (IGB) files.

    Provides static methods to extract header metadata and to load the
    3D/4D data into NumPy arrays, with optional scaling.

    Attributes:
        _DTYPES (Dict[str, Any]): Mapping from IGB type names to NumPy dtypes.
    """

    _DTYPES = {
        "byte": np.uint8,
        "char": np.int8,
        "short": np.int16,
        "long": np.int32,
        "float": np.float32,
        "double": np.float64,
    }

    @staticmethod
    def read_header(filename: str) -> Dict[str, Any]:
        """Read and parse the header of an IGB file.

        Args:
            filename (str): Path to the IGB file.

        Returns:
            Dict[str, Any]: Parsed header with metadata and comments.

        Raises:
            RuntimeError: If `filename` is empty.
        """
        if not filename:
            raise RuntimeError("No filename specified")

        with open(filename, "rb") as f:
            buf = f.read(1024)

        # Split out the ASCII header block
        lines = buf.decode(errors="ignore").split("\x00", 1)[0].split("\r\n")
        comments = [line.strip()[2:] for line in lines if line.startswith("#")]

        # Collect all non-comment fields, then split on the first ":" in each
        fields = sum(
            (
                line.split()
                for line in (line.strip() for line in lines if not line.startswith("#"))
                if line
            ),
            [],
        )

        header: Dict[str, Any] = {}
        for part in fields:
            key, val = part.split(":", 1)
            header[key] = val
        header["comments"] = comments

        # Convert integer fields
        for key in "xyzt":
            if key in header:
                header[key] = int(header[key])

        # Convert float fields
        for key in ["zero", "facteur"]:
            if key in header:
                header[key] = float(header[key])

        return header

    @staticmethod
    def read(
        filename: str,
        convert_to_float: bool = False,
        return_header: bool = False,
    ) -> Union[NDArray[Any], Tuple[NDArray[Any], Dict[str, Any]]]:
        """Read binary data from an IGB file into a NumPy array.

        Args:
            filename (str): Path to the IGB file.
            convert_to_float (bool): If True, apply scaling using 'zero' and 'facteur'.
            return_header (bool): If True, return a tuple of (data, header).

        Returns:
            Union[NDArray[Any], Tuple[NDArray[Any], Dict[str, Any]]]:
                The data array or a (data, header) tuple if `return_header` is True.
        """
        hdr = IGBReader.read_header(filename)
        nx, ny, nz = hdr["x"], hdr["y"], hdr["z"]
        nt = hdr.get("t", 1)
        shape = (nt, nz, ny, nx) if nt > 1 else (nz, ny, nx)
        dtype = IGBReader._DTYPES[hdr["type"]]

        data = np.fromfile(
            filename, dtype=dtype, count=nx * ny * nz * nt, offset=1024
        ).reshape(shape)

        if convert_to_float:
            facteur = hdr.get("facteur", 1.0)
            zero = hdr.get("zero", 0.0)
            data = facteur * data + zero

        return (data, hdr) if return_header else data
