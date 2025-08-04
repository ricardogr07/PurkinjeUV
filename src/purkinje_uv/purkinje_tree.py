"""Module for eikonal activation on a Purkinje fiber network.

This module defines the PurkinjeTree class, which wraps a line-based
Purkinje fiber mesh, computes activation times using the Fast Iterative
Method (FIM), and provides utilities for exporting results to VTK and
meshio formats.
"""

import numpy as np
from collections import Counter
from itertools import chain
import logging
from typing import Any, Sequence, Optional, Dict, List
from numpy.typing import NDArray

import meshio
from fimpy import create_fim_solver

import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa
from utils.vtkutils import vtk_unstructuredgrid_from_list

logger = logging.getLogger(__name__)


class PurkinjeTree:
    """Perform eikonal activation on a Purkinje network using FIM solver."""

    def __init__(
        self,
        nodes: Sequence[NDArray[Any]],
        connectivity: Sequence[Sequence[int]],
        end_nodes: Sequence[int],
    ) -> None:
        """Initialize a PurkinjeTree.

        Constructs the VTK representation of a Purkinje fiber network,
        sets up activation array, and stores Purkinje-myocardial junctions.

        Args:
            nodes (Sequence[NDArray[Any]]): List of (x, y, z) coordinates for each node.
            connectivity (Sequence[Sequence[int]]): List of (start, end) index pairs defining line elements.
            end_nodes (Sequence[int]): Indices of Purkinje-myocardial junction nodes.
        """
        self.connectivity = np.array(connectivity, dtype=int)
        self.xyz = np.array(nodes)

        # We keep the tree in VTK for data transfer

        self.vtk_tree = vtk_unstructuredgrid_from_list(
            self.xyz, self.connectivity, vtk.VTK_LINE
        )

        # reset activation
        act = np.empty(len(self.xyz))
        act.fill(np.inf)
        d = dsa.WrapDataObject(self.vtk_tree)
        d.PointData.append(act, "activation")

        # save PMJs
        self.pmj = end_nodes

        # conduction velocity
        self.cv = 2.5  # [m/s]

        logger.info(
            f"PurkinjeTree initialized with {self.xyz.shape[0]} nodes"
            f" and {self.connectivity.shape[0]} edges"
        )

    def activate_fim(
        self,
        x0: NDArray[Any],
        x0_vals: NDArray[Any],
        return_only_pmj: bool = True,
    ) -> NDArray[Any]:
        """Compute activation times using the Fast Iterative Method (FIM).

        Args:
            x0 (NDArray[Any]): Array of source node indices.
            x0_vals (NDArray[Any]): Activation values at each source node.
            return_only_pmj (bool): If True, return activation times only at PMJ nodes.

        Returns:
            NDArray[Any]: Activation times for all nodes or only PMJ nodes.
        """
        logger.info("Activating Purkinje tree with FIM solver")

        xyz = self.xyz
        elm = self.connectivity

        ve = np.ones(elm.shape[0])
        D = self.cv * np.eye(xyz.shape[1])[np.newaxis] * ve[..., np.newaxis, np.newaxis]

        fim = create_fim_solver(xyz, elm, D, device="cpu")
        act: NDArray[Any] = fim.comp_fim(x0, x0_vals)

        # update activation in VTK
        da = dsa.WrapDataObject(self.vtk_tree)
        da.PointData["activation"][:] = act

        if return_only_pmj:
            return act[self.pmj]
        else:
            return act

    def save(self, fname: str) -> None:
        """Write the current activation state to a VTK UnstructuredGrid file.

        Args:
            fname (str): Path to the output .vtu file.
        """
        logger.info(f"Saving PurkinjeTree to VTK at {fname}")

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self.vtk_tree)
        writer.Update()

    def save_pmjs(self, fname: str) -> None:
        """Export PMJ nodes and their activation values as a VTP file.

        Args:
            fname (str): Path to the output .vtp file.
        """
        logger.info(f"Saving PMJs to VTP at {fname}")

        xyz = self.xyz[self.pmj]
        da = dsa.WrapDataObject(self.vtk_tree)
        act = da.PointData["activation"]

        mesh = meshio.Mesh(
            points=xyz, cells={"vertex": np.arange(xyz.shape[0])[:, np.newaxis]}
        )
        mesh.point_data = {"activation": act[self.pmj]}
        # mesh.cell_data  = cell_data or {}

        mesh.write(fname)

    def get_pmjs_activation(self) -> NDArray[Any]:
        """Retrieve activation times at the Purkinje-myocardial junction nodes.

        Returns:
            NDArray[Any]: Activation times indexed by PMJ node order.
        """
        da = dsa.WrapDataObject(self.vtk_tree)
        act: NDArray[Any] = da.PointData["activation"]
        return act[self.pmj]

    def save_meshio(
        self,
        fname: str,
        point_data: Optional[Dict[str, Any]] = None,
        cell_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Export the Purkinje tree as a meshio Mesh with line cells.

        Args:
            fname (str): Path to the output file (format inferred by extension).
            point_data (Optional[Dict[str, Any]]): Optional per-node data arrays.
            cell_data (Optional[Dict[str, Any]]): Optional per-cell data arrays.
        """
        logger.info(f"Saving PurkinjeTree to meshio format at {fname}")

        xyz = self.xyz
        edges = self.extract_edges()
        mesh = meshio.Mesh(points=xyz, cells={"line": edges})
        mesh.point_data = point_data or {}
        mesh.cell_data = cell_data or {}

        mesh.write(fname)

    def extract_edges(self) -> NDArray[Any]:
        """Return the edge connectivity array for the Purkinje tree.

        Returns:
            NDArray[Any]: Array of shape (n_edges, 2) listing node index pairs.
        """
        return self.connectivity

    def extract_pmj_counter(self) -> List[int]:
        """Compute PMJ node indices using a pure Python counter on connectivity.

        Returns:
            List[int]: Nodes with degree equal to one.
        """
        # Flatten our connectivity array into individual node IDs
        flattened = chain.from_iterable(self.connectivity.tolist())
        counts = Counter(flattened)

        # Leaf nodes appear exactly once in the edge list
        return [node for node, deg in counts.items() if deg == 1]

    def extract_pmj_np_bincount(self) -> NDArray[Any]:
        """Compute PMJ node indices by numpy bincount on flattened connectivity.

        Returns:
            NDArray[Any]: Nodes with degree equal to one.
        """
        # Flatten the connectivity array (shape (E,2)) into a 1D sequence of node indices
        flat = self.connectivity.ravel()
        counts = np.bincount(flat)

        # Nodes with count == 1 are leaves
        return np.where(counts == 1)[0]

    def extract_pmj_np_unique(self) -> NDArray[Any]:
        """Compute PMJ node indices via numpy unique and count filtering.

        Returns:
            NDArray[Any]: Nodes with degree equal to one.
        """
        # Flatten the connectivity array into a 1D sequence of node indices
        flat = self.connectivity.ravel()
        nn, cnt = np.unique(flat, return_counts=True)

        # Nodes with exactly one connection are leaves
        leaves: NDArray[Any] = nn[cnt == 1]
        return leaves
