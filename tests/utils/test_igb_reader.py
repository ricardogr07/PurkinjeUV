import pytest
import numpy as np
from io import BytesIO
from unittest.mock import patch

from purkinje_uv.igb_reader import IGBReader


@pytest.fixture
def minimal_igb_header():
    """
    Returns a minimal valid IGB header as a byte string (1024 bytes).
    """
    lines = [
        "# Comment line",
        "x:4",
        "y:2",
        "z:1",
        "type:float",
        "zero:0.0",
        "facteur:1.0",
    ]
    header_str = "\r\n".join(lines)
    return header_str.encode().ljust(1024, b"\x00")


@pytest.fixture
def igb_data_bytes():
    # 4x2x1 = 8 floats, values 0.0 to 7.0
    return np.arange(8, dtype=np.float32).tobytes()


def test_read_header(minimal_igb_header):
    fake_file = BytesIO(minimal_igb_header)
    with patch("builtins.open", return_value=fake_file):
        header = IGBReader.read_header("fake.igb")

    assert header["x"] == 4
    assert header["y"] == 2
    assert header["z"] == 1
    assert header["type"] == "float"
    assert header["zero"] == 0.0
    assert header["facteur"] == 1.0
    assert isinstance(header["comments"], list)
    assert header["comments"][0] == "Comment line"


def test_read_data(minimal_igb_header, igb_data_bytes):
    fake_file = BytesIO(minimal_igb_header + igb_data_bytes)
    with patch("builtins.open", return_value=fake_file):
        with patch("numpy.fromfile", return_value=np.arange(8, dtype=np.float32)):
            data = IGBReader.read("fake.igb")

    assert isinstance(data, np.ndarray)
    assert data.shape == (1, 2, 4)  # (z, y, x)
    assert np.allclose(data.flatten(), np.arange(8, dtype=np.float32))


def test_read_with_header(minimal_igb_header, igb_data_bytes):
    fake_file = BytesIO(minimal_igb_header + igb_data_bytes)
    with patch("builtins.open", return_value=fake_file):
        with patch("numpy.fromfile", return_value=np.arange(8, dtype=np.float32)):
            data, header = IGBReader.read("fake.igb", return_header=True)

    assert isinstance(data, np.ndarray)
    assert isinstance(header, dict)
    assert header["x"] == 4


def test_read_with_scaling(minimal_igb_header, igb_data_bytes):
    hdr_scaled = minimal_igb_header.replace(b"zero:0.0", b"zero:1.0").replace(
        b"facteur:1.0", b"facteur:2.0"
    )
    fake_file = BytesIO(hdr_scaled + igb_data_bytes)
    with patch("builtins.open", return_value=fake_file):
        with patch("numpy.fromfile", return_value=np.arange(8, dtype=np.float32)):
            data = IGBReader.read("fake.igb", convert_to_float=True)

    expected = 2.0 * np.arange(8) + 1.0
    assert np.allclose(data.flatten(), expected)


def test_missing_filename_raises():
    with pytest.raises(RuntimeError):
        IGBReader.read_header(None)
