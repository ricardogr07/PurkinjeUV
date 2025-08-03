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
    "Class for eikonal solver on Purkinje tree"

    def __init__(
        self,
        nodes: Sequence[NDArray[Any]],
        connectivity: Sequence[Sequence[int]],
        end_nodes: Sequence[int],
    ) -> None:
        "Init from FractalTree generator"

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
        '''Activate tree with FIM solver.

        Args:
            x0: Starting node indices (array of ints).
            x0_vals: Values at starting nodes (array of floats).
            return_only_pmj: If True, return activation only at PMJ nodes.

        Returns:
            Numpy array of activation values.
        '''

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
        """Save tree to VTK file.

        Args:
            fname: Output VTK file path.
        """
        logger.info(f"Saving PurkinjeTree to VTK at {fname}")

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self.vtk_tree)
        writer.Update()

    def save_pmjs(self, fname: str) -> None:
        "Save the junctions as VTP"

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
        "Return the current activation values at PMJs"

        da = dsa.WrapDataObject(self.vtk_tree)
        act: NDArray[Any] = da.PointData["activation"]
        return act[self.pmj]

    def save_meshio(
        self,
        fname: str,
        point_data: Optional[Dict[str, Any]] = None,
        cell_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        "Save with meshio"

        logger.info(f"Saving PurkinjeTree to meshio format at {fname}")

        xyz = self.xyz
        edges = self.extract_edges()
        mesh = meshio.Mesh(points=xyz, cells={"line": edges})
        mesh.point_data = point_data or {}
        mesh.cell_data = cell_data or {}

        mesh.write(fname)

    def extract_edges(self) -> NDArray[Any]:
        """Return the list of edges from the original connectivity array."""
        return self.connectivity

    def extract_pmj_counter(self) -> List[int]:
        """Compute leaf-node IDs (degree == 1) entirely in Python."""

        # Flatten our connectivity array into individual node IDs
        flattened = chain.from_iterable(self.connectivity.tolist())
        counts = Counter(flattened)

        # Leaf nodes appear exactly once in the edge list
        return [node for node, deg in counts.items() if deg == 1]

    def extract_pmj_np_bincount(self) -> NDArray[Any]:
        """Compute leaf-node IDs (degree == 1) using numpy bin-count."""

        # Flatten the connectivity array (shape (E,2)) into a 1D sequence of node indices
        flat = self.connectivity.ravel()
        counts = np.bincount(flat)

        # Nodes with count == 1 are leaves
        return np.where(counts == 1)[0]

    def extract_pmj_np_unique(self) -> NDArray[Any]:
        """Compute leaf-node IDs (degree == 1) using numpy unique."""

        # Flatten the connectivity array into a 1D sequence of node indices
        flat = self.connectivity.ravel()
        nn, cnt = np.unique(flat, return_counts=True)

        # Nodes with exactly one connection are leaves
        leaves: NDArray[Any] = nn[cnt == 1]
        return leaves
