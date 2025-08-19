"""Module defining the Branch class for fractal tree growth on a mesh.

This module contains the Branch class, which represents a single branch in the fractal tree.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from purkinje_uv import backend_name, norm, to_cpu, to_device, xp
from .mesh import Mesh
from .nodes import Nodes

_LOGGER = logging.getLogger(__name__)


class Branch:
    """Represents a branch of the fractal tree on a mesh.

    Attributes:
        mesh (Mesh): Mesh on which the branch grows.
        queue (list[NDArray[Any]]): Coordinates of points queued for growth.
        triangles (list[int]): Triangle indices for each queued point.
        nodes (list[int]): Node indices in the global Nodes manager.
        growing (bool): Whether the branch is still growing.
    """

    def __init__(
        self,
        mesh: Mesh,
        init_node: int,
        init_dir: NDArray[Any],
        init_tri: int,
        length: float,
        angle: float,
        w: float,
        nodes: Nodes,
        brother_nodes: Sequence[int],
        Nsegments: int,
    ) -> None:
        """Initialize a Branch.

        Args:
            mesh (Mesh): Mesh on which the branch grows.
            init_node (int): Index of the initial node in the mesh.
            init_dir (NDArray[Any]): Direction vector for the initial growth.
            init_tri (int): Index of the triangle containing the initial node.
            length (float): Total length of the branch.
            angle (float): Growth angle parameter.
            w (float): Weight for gradient adjustment.
            nodes (Nodes): Nodes manager to track new nodes.
            brother_nodes (Sequence[int]): Indices of brother nodes.
            Nsegments (int): Number of segments to divide the branch.
        """
        self.child: list[int] = [0, 0]
        self.dir: NDArray[Any] = np.array([0.0, 0.0, 0.0], dtype=float)
        self.nodes: list[int] = []
        self.triangles: list[int] = []
        self.queue: list[NDArray[Any]] = []
        self.growing: bool = True

        _LOGGER.debug(
            "Initializing Branch(init_node=%d, init_tri=%d, Nsegments=%d) on backend=%s",
            init_node,
            init_tri,
            Nsegments,
            backend_name(),
        )

        init_normal_cpu = mesh.normals[init_tri]

        # Build collision tree before growing.
        nodes.update_collision_tree(brother_nodes)

        # Compute initial in-plane direction on the active backend.
        # Use device copies for vector math; keep stored state on CPU.
        init_dir_dev = to_device(init_dir, dtype=float)
        init_norm_dev = to_device(init_normal_cpu, dtype=float)

        inplane_dev = -xp.cross(init_dir_dev, init_norm_dev)

        # Rotation within tangent plane
        dir_dev = xp.cos(angle) * init_dir_dev + xp.sin(angle) * inplane_dev
        mag0 = float(norm(dir_dev))
        if mag0 < 1e-12:
            raise ValueError("Initial direction magnitude is ~0; cannot grow branch.")
        dir_dev = dir_dev / mag0  # normalized device vector

        # Seed the queue with the provided node position (kept on CPU).
        self.nodes.append(init_node)
        self.queue.append(nodes.nodes[init_node])
        self.triangles.append(init_tri)

        # Gradient step (compute grad on CPU via Nodes, then move to device).
        grad_cpu = nodes.gradient(self.queue[0])
        grad_dev = to_device(grad_cpu, dtype=float)

        upd_dev = dir_dev + w * grad_dev
        mag_upd = float(norm(upd_dev))
        if mag_upd < 1e-12:
            raise ValueError("Initial update direction has ~0 magnitude.")
        dir_dev = upd_dev / mag_upd

        # Grow along segments.
        step_len = float(length) / float(max(Nsegments, 1))
        for i in range(1, Nsegments):
            # Compute step on device, then call mesh.project_new_point on CPU.
            step_cpu = to_cpu(dir_dev) * step_len
            intriangle = self.add_node_to_queue(
                mesh, self.queue[i - 1], step_cpu.astype(float, copy=False)
            )
            if not intriangle:
                _LOGGER.debug(
                    "Point not in triangle at segment %d; stopping growth.", i
                )
                self.growing = False
                break

            # Collision check and early stop if too close.
            collision = nodes.collision(self.queue[i])
            if collision[1] < length / 5.0:
                _LOGGER.debug(
                    "Collision at segment %d (dist=%.6f < %.6f); stopping growth.",
                    i,
                    collision[1],
                    length / 5.0,
                )
                self.growing = False
                self.queue.pop()
                self.triangles.pop()
                break

            # Gradient update (project gradient to tangent plane) on device.
            grad_i_cpu = nodes.gradient(self.queue[i])
            normal_i_cpu = mesh.normals[self.triangles[i], :]

            grad_i_dev = to_device(grad_i_cpu, dtype=float)
            normal_i_dev = to_device(normal_i_cpu, dtype=float)

            # Project gradient onto tangent plane of the surface.
            # grad := grad - (grad·n) n
            dot_gn = xp.dot(grad_i_dev, normal_i_dev)
            grad_tan_dev = grad_i_dev - dot_gn * normal_i_dev

            upd_dev = dir_dev + w * grad_tan_dev
            mag = float(norm(upd_dev))
            if mag < 1e-12:
                _LOGGER.debug(
                    "Vanishing update magnitude at segment %d; stopping growth.", i
                )
                self.growing = False
                break
            dir_dev = upd_dev / mag

        # Register new nodes in global Nodes manager (skip the first, already present).
        nodes_id = nodes.add_nodes(self.queue[1:])
        for nid in nodes_id:
            self.nodes.append(nid)

        if not self.growing and self.nodes:
            nodes.end_nodes.append(self.nodes[-1])

        # Store final direction on CPU for downstream code.
        self.dir = to_cpu(dir_dev).astype(float, copy=False)
        self.tri = self.triangles[-1]

        _LOGGER.debug(
            "Branch initialized: last_tri=%d, n_nodes=%d, growing=%s, final_dir=%s",
            self.tri,
            len(self.nodes),
            self.growing,
            self.dir.tolist(),
        )

    def add_node_to_queue(
        self,
        mesh: Mesh,
        init_node: NDArray[Any],
        dir_vec: NDArray[Any],
    ) -> bool:
        """Project a new node onto the mesh and add it to the growth queue.

        This method projects a direction from a starting point onto the mesh surface.
        If the projected point lies within a mesh triangle, it is appended to the queue and triangles list.

        Args:
            mesh (Mesh): Mesh on which the branch grows.
            init_node (NDArray[Any]): Coordinates of the last node in the branch.
            dir_vec (NDArray[Any]): Direction vector from the initial node to the new node.

        Returns:
            bool: True if the projected node lies within a mesh triangle; False otherwise.
        """
        # Work on CPU for mesh projection; log distances for debugging.
        candidate = init_node + dir_vec
        _LOGGER.debug("Projecting candidate point %s", candidate.tolist())

        point, triangle, *_ = mesh.project_new_point(candidate)

        dist = float(np.linalg.norm(point - init_node))
        _LOGGER.debug("Projected point %s, dist %.6e", point.tolist(), dist)

        if triangle >= 0:
            self.queue.append(point)
            self.triangles.append(int(triangle))
            success = True
        else:
            _LOGGER.debug("Projection failed: point=%s, triangle=%s", point, triangle)
            success = False

        _LOGGER.debug("Projection success? %s", success)
        return success
