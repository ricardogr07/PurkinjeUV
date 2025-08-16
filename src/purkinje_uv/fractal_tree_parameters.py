"""Module defining the FractalTreeParameters class for configuring fractal tree generation.

This module provides the FractalTreeParameters dataclass, which holds all settings
for generating a fractal tree structure.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict, fields
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class FractalTreeParameters:
    """Holds settings for generating a fractal tree structure.

    Attributes
    ----------
    meshfile:
        Path to the mesh file (e.g., OBJ); if ``None``, no mesh is preloaded.
    init_node_id:
        Index of the initial node in the mesh.
    second_node_id:
        Index of the second node, which sets the initial growth direction.
    init_length:
        Length of the first branch.
    N_it:
        Number of branch generations (iterations).
    length:
        Mean length of tree branches.
    branch_angle:
        Angle (in radians) between consecutive branches.
    w:
        Weight parameter for branch divergence.
    l_segment:
        Approximate length of each branch segment.
    fascicles_angles:
        Angles (in radians) for each fascicle branch.
    fascicles_length:
        Lengths for each fascicle branch; matches ``fascicles_angles``.
    """

    meshfile: Optional[str] = None
    init_node_id: int = 0
    second_node_id: int = 1
    init_length: float = 0.1
    # Number of iterations (generations of branches)
    N_it: int = 10
    # Median length of the branches
    length: float = 0.1
    # Angle between branches (radians)
    branch_angle: float = 0.15
    w: float = 0.1
    # Segment length (approximate; branch length may vary)
    l_segment: float = 0.01

    # Fascicles data (lists intentionally left mutable; use default_factory)
    fascicles_angles: List[float] = field(default_factory=list)
    fascicles_length: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        # --- Type checks for indices ---
        if not isinstance(self.init_node_id, int) or self.init_node_id < 0:
            raise TypeError("init_node_id must be a nonnegative integer.")
        if not isinstance(self.second_node_id, int) or self.second_node_id < 0:
            raise TypeError("second_node_id must be a nonnegative integer.")
        if self.second_node_id == self.init_node_id:
            raise ValueError("second_node_id must differ from init_node_id.")

        # --- Scalar numeric validations ---
        for name, val, cond, msg in (
            ("init_length", self.init_length, self.init_length > 0, "must be > 0."),
            ("N_it", self.N_it, self.N_it >= 0, "must be >= 0."),
            ("length", self.length, self.length > 0, "must be > 0."),
            ("w", self.w, self.w >= 0, "must be >= 0."),
            ("l_segment", self.l_segment, self.l_segment > 0, "must be > 0."),
        ):
            if not isinstance(val, Real) or not math.isfinite(float(val)) or not cond:
                raise ValueError(f"{name} {msg}")

        if not isinstance(self.branch_angle, Real) or not math.isfinite(
            float(self.branch_angle)
        ):
            raise TypeError("branch_angle must be a finite real number.")
        if not (0.0 < float(self.branch_angle) <= math.pi):
            raise ValueError("branch_angle must be in the interval (0, π].")

        # Reasonable geometric relation for segments vs. branch lengths
        min_len = min(float(self.init_length), float(self.length))
        if float(self.l_segment) > min_len:
            raise ValueError(
                "l_segment must be <= min(init_length, length). "
                f"Got l_segment={self.l_segment}, min={min_len}."
            )

        # --- Fascicles consistency ---
        if len(self.fascicles_angles) != len(self.fascicles_length):
            raise ValueError(
                "fascicles_angles and fascicles_length must have the same length."
            )

        # Validate each fascicle entry
        for i, ang in enumerate(self.fascicles_angles):
            if not isinstance(ang, Real) or not math.isfinite(float(ang)):
                raise TypeError(
                    f"fascicles_angles[{i}] must be a finite real number (radians)."
                )

        for i, L in enumerate(self.fascicles_length):
            if not isinstance(L, Real) or not math.isfinite(float(L)) or float(L) <= 0:
                raise ValueError(
                    f"fascicles_length[{i}] must be a finite positive number."
                )
        # --- Logging (summary at INFO, full at DEBUG) ---
        logger.info(
            "Initialized FractalTreeParameters: meshfile=%r, init_node_id=%d, "
            "second_node_id=%d, N_it=%d, init_length=%.6g, length=%.6g, "
            "branch_angle=%.6g rad, w=%.6g, l_segment=%.6g, "
            "fascicles=(%d items)",
            self.meshfile,
            self.init_node_id,
            self.second_node_id,
            self.N_it,
            self.init_length,
            self.length,
            self.branch_angle,
            self.w,
            self.l_segment,
            len(self.fascicles_angles),
        )
        logger.debug("FractalTreeParameters full config: %s", self.to_json(indent=None))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of the parameters."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize parameters to a JSON string.

        Parameters
        ----------
        indent:
            Indentation level for pretty printing. If ``None``, the result is compact.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_json_file(self, path: str, indent: Optional[int] = 2) -> None:
        """Write parameters to a JSON file at ``path``."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FractalTreeParameters":
        """Construct parameters from a dictionary.

        Unknown keys are ignored (emits a warning). Validation occurs in ``__post_init__``.
        """
        valid = {f.name for f in fields(cls)}
        unknown = [k for k in data.keys() if k not in valid]
        if unknown:
            logger.warning("Ignoring unknown parameter keys: %s", unknown)
        filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_json(cls, s: str) -> "FractalTreeParameters":
        """Construct parameters from a JSON string."""
        return cls.from_dict(json.loads(s))
