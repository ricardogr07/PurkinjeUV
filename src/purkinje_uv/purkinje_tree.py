import numpy as np
from collections import Counter
from itertools import chain
import logging

import meshio
from fimpy.solver import FIMPY

import vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa
from utils.vtkutils import vtk_unstructuredgrid_from_list

logger = logging.getLogger(__name__)


class PurkinjeTree:
    "Class for eikonal solver on Purkinje tree"

    def __init__(self, nodes, connectivity, end_nodes):
        "Init from FractalTree generator"

        self.connectivity = connectivity
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
            f"PurkinjeTree initialized with {self.xyz.shape[0]} nodes and {self.connectivity.shape[0]} edges"
        )

    def activate_fim(self, x0, x0_vals, return_only_pmj=True):
        "Activate tree with fim-python"

        logger.info("Activating Purkinje tree with FIM solver")

        xyz = self.xyz
        elm = self.connectivity

        ve = np.ones(elm.shape[0])
        D = self.cv * np.eye(xyz.shape[1])[np.newaxis] * ve[..., np.newaxis, np.newaxis]

        fim = FIMPY.create_fim_solver(xyz, elm, D, device="cpu")
        act = fim.comp_fim(x0, x0_vals)

        # update activation in VTK
        da = dsa.WrapDataObject(self.vtk_tree)
        da.PointData["activation"][:] = act

        if return_only_pmj:
            return act[self.pmj]
        else:
            return act

    def save(self, fname):
        "Save to VTK"

        logger.info(f"Saving PurkinjeTree to VTK at {fname}")

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self.vtk_tree)
        writer.Update()

    def save_pmjs(self, fname):
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

    def get_pmjs_activation(self):
        "Return the current activation values at PMJs"

        da = dsa.WrapDataObject(self.vtk_tree)
        act = da.PointData["activation"]
        return act[self.pmj]

    def save_meshio(self, fname, point_data=None, cell_data=None):
        "Save with meshio"

        logger.info(f"Saving PurkinjeTree to meshio format at {fname}")

        xyz = self.xyz
        edges = self.extract_edges()
        mesh = meshio.Mesh(points=xyz, cells={"line": edges})
        mesh.point_data = point_data or {}
        mesh.cell_data = cell_data or {}

        mesh.write(fname)

    def extract_edges(self):
        "List of edges from branches"

        # edges in each branch
        bedges = chain.from_iterable(
            zip(b.nodes[0:-1], b.nodes[1:])
            for b in self.branches.values()
            if len(b.nodes) > 1
        )
        # collect all edges
        edges = np.array(list(bedges))

        return edges

    def extract_pmj_counter(self):
        "Pure Python version"

        t = chain.from_iterable(
            (b.nodes[0], b.nodes[-1])
            for b in self.branches.values()
            if len(b.nodes) > 1
        )
        c = Counter(t)
        enodes = [k for k, v in c.items() if v == 1 and v != self.branches[0].nodes[0]]

        return enodes

    def extract_pmj_np_bincount(self):
        "End-nodes of the tree or junctions"

        t = chain.from_iterable(
            (b.nodes[0], b.nodes[-1])
            for b in self.branches.values()
            if len(b.nodes) > 1
        )
        c = np.bincount(np.fromiter(t, dtype=int))
        enodes = np.where(c == 1)[0]
        # we remove the entry point
        enodes = np.delete(enodes, self.branches[0].nodes[0])

        return enodes

    def extract_pmj_np_unique(self):

        t = chain.from_iterable(
            (b.nodes[0], b.nodes[-1])
            for b in self.branches.values()
            if len(b.nodes) > 1
        )
        nn, cnt = np.unique(np.fromiter(t, dtype=int), return_counts=True)
        enodes = nn[cnt == 1]
        enodes = np.delete(enodes, self.branches[0].nodes[0])

        return enodes
