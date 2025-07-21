import numpy as np


class IGBReader:
    """
    IGBReader provides static methods to read and parse IGB (Image Grid Binary) files.

    Attributes:
        _DTYPES (dict): Mapping of IGB type strings to NumPy dtypes.

    Methods:
        read_header(filename: str) -> dict:
            Reads and parses the header of an IGB file, returning metadata and comments.
        read(filename: str, convert_to_float: bool = False, return_header: bool = False):
            Reads the binary data from an IGB file into a NumPy array, with optional scaling and
            header return.
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
    def read_header(filename: str) -> dict:
        """
        Reads and parses the header of an IGB file.

        Args:
            filename (str): Path to the IGB file.

        Returns:
            dict: Parsed header with metadata and comments.

        Raises:
            RuntimeError: If filename is not specified.
        """
        if not filename:
            raise RuntimeError("No filename specified")

        with open(filename, "rb") as f:
            buf = f.read(1024)

        lines = buf.decode(errors="ignore").split("\x00", 1)[0].split("\r\n")
        comments = [line.strip()[2:] for line in lines if line.startswith("#")]
        fields = sum(
            (
                line.split()
                for line in (line.strip() for line in lines if not line.startswith("#"))
                if line
            ),
            [],
        )
        header = dict(part.split(":") for part in fields)
        header["comments"] = comments

        for key in "xyzt":
            if key in header:
                header[key] = int(header[key])
        for key in ["zero", "facteur"]:
            if key in header:
                header[key] = float(header[key])

        return header

    @staticmethod
    def read(
        filename: str, convert_to_float: bool = False, return_header: bool = False
    ):
        """
        Reads an IGB file into a NumPy array.

        Args:
            filename (str): Path to the IGB file.
            convert_to_float (bool): If True, apply scaling using 'zero' and 'facteur'.
            return_header (bool): If True, return both data and header.

        Returns:
            np.ndarray or (np.ndarray, dict): The data array or a tuple with header.
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
