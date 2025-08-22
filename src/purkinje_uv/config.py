"""Global configuration for purkinje-uv CPU/GPU backends.

This module provides a package-wide configuration surface to select and manage
the array backend (NumPy on CPU or CuPy on GPU) without changing public APIs.
It also exposes dynamic proxies (`xp`, `rng`, etc.) that always reflect the
current backend, robust logging, and optional deterministic RNG mirroring for
bitwise-stable CPU↔GPU tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import logging
import os
from typing import Any, Callable, ContextManager, Iterator, Optional


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOGGER = logging.getLogger("purkinje_uv.config")


def _parse_log_level(val: str | int | None, default: int = logging.WARNING) -> int:
    """Parse a logging level string or int into a `logging` level constant.

    Args:
        val: The desired level (e.g., "DEBUG", 10). May be None.
        default: Fallback level if `val` cannot be parsed.

    Returns:
        An integer logging level (e.g., logging.DEBUG).
    """
    if val is None:
        return default
    if isinstance(val, int):
        return val
    try:
        lvl = getattr(logging, str(val).upper())
        if isinstance(lvl, int):
            return lvl
    except Exception:
        pass
    return default


def set_log_level(level: str | int = "WARNING") -> None:
    """Set the module logger level programmatically.

    Args:
        level: A standard logging level name or integer.
    """
    _LOGGER.setLevel(_parse_log_level(level))


# Default level can be overridden by env.
set_log_level(os.getenv("PURKINJE_UV_LOGLEVEL", "WARNING"))


# -----------------------------------------------------------------------------
# Env helpers (JAX-style)
# -----------------------------------------------------------------------------
def bool_env(varname: str, default: bool) -> bool:
    """Read an environment variable and interpret it as a boolean.

    True values: 'y', 'yes', 't', 'true', 'on', '1'.
    False values: 'n', 'no', 'f', 'false', 'off', '0'.

    Args:
        varname: The name of the environment variable.
        default: The default value if the variable is unset.

    Returns:
        A boolean value parsed from the environment.
    """
    val = os.getenv(varname, str(default))
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"invalid truth value {val!r} for environment {varname!r}")


def int_env(varname: str, default: int) -> int:
    """Read an environment variable and interpret it as an integer.

    Args:
        varname: The name of the environment variable.
        default: The default value if the variable is unset.

    Returns:
        The integer value parsed from the environment.
    """
    return int(os.getenv(varname, str(default)))


def _device_env() -> str:
    """Parse PURKINJE_UV_GPU into a device string ('gpu'|'cpu'|'auto').

    Returns:
        One of 'gpu', 'cpu', or 'auto'.
    """
    raw = os.getenv("PURKINJE_UV_GPU", "").strip().lower()
    if raw in {"1", "true", "y", "yes", "on", "gpu"}:
        dev = "gpu"
    elif raw in {"0", "false", "n", "no", "off", "cpu"}:
        dev = "cpu"
    else:
        dev = "auto"
    _LOGGER.debug("Env PURKINJE_UV_GPU=%r -> device=%s", raw, dev)
    return dev


def _deterministic_rng_enabled() -> bool:
    """Return True if deterministic RNG mirroring is requested.

    When enabled (via PURKINJE_UV_DETERMINISTIC_RNG=1), random values are
    generated on CPU (NumPy PCG64) and copied to the active backend. This
    ensures identical CPU/GPU results for tests.
    """
    raw = os.getenv("PURKINJE_UV_DETERMINISTIC_RNG", "").strip().lower()
    return raw in {"1", "true", "y", "yes", "on"}


# -----------------------------------------------------------------------------
# Array backend abstraction
# -----------------------------------------------------------------------------
@dataclass
class ArrayBackend:
    """Descriptor for the active array backend (NumPy or CuPy)."""

    name: str
    is_gpu: bool
    xp: Any
    cdist: Callable[..., Any]
    rng: Any  # np.random.Generator, cp.random.Generator, or a mirror

    def seed(self, s: int = 1234) -> None:
        """Reseed the backend's RNG to a deterministic state.

        Args:
            s: The seed value.
        """
        if self.is_gpu:
            import cupy as cp

            try:
                if hasattr(cp.random, "PCG64") and hasattr(cp.random, "Generator"):
                    self.rng = cp.random.Generator(cp.random.PCG64(s))
                    _LOGGER.debug("Reseeded CuPy Generator(PCG64) with seed=%d", s)
                    return
            except Exception as exc:
                _LOGGER.debug("CuPy PCG64 reseed failed (%r); RNG left unchanged.", exc)
        else:
            import numpy as np

            self.rng = np.random.Generator(np.random.PCG64(s))
            _LOGGER.debug("Reseeded NumPy PCG64 with seed=%d", s)

    def to_cpu(self, a: Any) -> Any:
        """Copy an array to CPU if it is a CuPy array."""
        if self.is_gpu:
            import cupy as cp

            if isinstance(a, cp.ndarray):
                _LOGGER.debug(
                    "Transferring array from GPU->CPU (shape=%s)",
                    getattr(a, "shape", None),
                )
                return cp.asnumpy(a)
        return a

    def to_device(self, a: Any, dtype: Any | None = None) -> Any:
        """Copy an array to the active backend (NumPy or CuPy).

        Args:
            a: Input array-like.
            dtype: Optional dtype to cast to.

        Returns:
            An array on the current backend.
        """
        _LOGGER.debug(
            "Transferring array to %s (dtype=%s, shape=%s)",
            self.name,
            dtype,
            getattr(a, "shape", None),
        )
        return self.xp.asarray(a, dtype=dtype)

    def norm(self, v: Any, axis: int = -1, keepdims: bool = False) -> Any:
        """Compute the L2 norm along `axis` on the active backend."""
        return self.xp.linalg.norm(v, axis=axis, keepdims=keepdims)


def _make_cdist_xp(xp: Any) -> Callable[[Any, Any], Any]:
    """Return a pure-xp pairwise Euclidean distance function.

    Works with both NumPy and CuPy arrays.

    Args:
        xp: The array module (numpy or cupy).

    Returns:
        A function `cdist(A, B)` computing pairwise distances.
    """

    def _cdist(A: Any, B: Any) -> Any:
        A = xp.asarray(A)
        B = xp.asarray(B)
        A2 = xp.sum(A * A, axis=1)[:, None]
        B2 = xp.sum(B * B, axis=1)[None, :]
        D2 = A2 + B2 - 2 * (A @ B.T)
        D2 = xp.maximum(D2, 0)
        return xp.sqrt(D2)

    return _cdist


class _RNGMirrorCPU:
    """Mirror RNG: use NumPy PCG64 on host, return arrays on current backend."""

    def __init__(self, seed: int, xp: Any, logger: logging.Logger) -> None:
        import numpy as _np

        self._xp = xp
        self._np = _np
        self._gen = _np.random.Generator(_np.random.PCG64(seed))
        logger.debug("Initialized _RNGMirrorCPU with PCG64 seed=%d", seed)

    def random(self, size: Any | None = None, dtype: Any | None = None) -> Any:
        """Draw uniform random numbers and return on active backend."""
        a = self._gen.random(size)
        return self._xp.asarray(a, dtype=dtype)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Any | None = None,
        dtype: Any | None = None,
    ) -> Any:
        """Draw normal random numbers and return on active backend."""
        a = self._gen.normal(loc=loc, scale=scale, size=size)
        return self._xp.asarray(a, dtype=dtype)

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: Any | None = None,
        dtype: Any | None = None,
        endpoint: bool = False,
    ) -> Any:
        """Draw integer random numbers and return on active backend."""
        a = self._gen.integers(low, high=high, size=size, endpoint=endpoint)
        return self._xp.asarray(a, dtype=dtype)


def _make_cpu_backend(seed: int = 1234) -> ArrayBackend:
    """Create a CPU (NumPy) backend."""
    import numpy as np

    try:
        from scipy.spatial.distance import cdist as numpy_cdist

        cdist_fn: Callable[..., Any] = numpy_cdist
        _LOGGER.debug("Using SciPy cdist for CPU")
    except Exception as exc:  # pragma: no cover - SciPy is normally present
        _LOGGER.warning("SciPy cdist unavailable (%r); using pure-NumPy cdist.", exc)
        cdist_fn = _make_cdist_xp(np)

    be = ArrayBackend(name="numpy", is_gpu=False, xp=np, cdist=cdist_fn, rng=None)
    be.rng = np.random.Generator(np.random.PCG64(seed))
    _LOGGER.info("Initialized CPU backend (NumPy) with PCG64 seed=%d", seed)
    return be


def _try_make_gpu_backend(seed: int = 1234, *, debug: bool = False) -> ArrayBackend:
    """Create a GPU (CuPy) backend or raise if initialization fails.

    Args:
        seed: RNG seed.
        debug: Unused flag preserved for symmetry; logs already contain detail.

    Returns:
        An initialized `ArrayBackend` for GPU.

    Raises:
        RuntimeError: If no CUDA device is visible to CuPy.
    """
    import cupy as cp

    dev_count = cp.cuda.runtime.getDeviceCount()
    _LOGGER.debug("CuPy detected devices: %d", dev_count)
    if dev_count < 1:
        raise RuntimeError("No CUDA device visible to CuPy")

    try:
        from cupyx.scipy.spatial.distance import cdist as cupy_cdist

        cdist_fn: Callable[..., Any] = cupy_cdist
        _LOGGER.debug("Using cupyx.scipy.spatial.distance.cdist on GPU")
    except Exception as exc:  # pragma: no cover - depends on cupy build
        _LOGGER.warning(
            "cupyx.scipy.spatial.distance.cdist unavailable (%r); "
            "using pure-CuPy cdist.",
            exc,
        )
        cdist_fn = _make_cdist_xp(cp)

    be = ArrayBackend(name="cupy", is_gpu=True, xp=cp, cdist=cdist_fn, rng=None)

    if _deterministic_rng_enabled():
        _LOGGER.info(
            "Deterministic RNG enabled; mirroring NumPy PCG64 → GPU (seed=%d)",
            seed,
        )
        be.rng = _RNGMirrorCPU(seed, xp=cp, logger=_LOGGER)
    else:
        gen: Any | None = None
        try:
            if hasattr(cp.random, "PCG64") and hasattr(cp.random, "Generator"):
                gen = cp.random.Generator(cp.random.PCG64(seed))
                _LOGGER.debug("Using CuPy Generator(PCG64) seed=%d", seed)
        except Exception as exc:
            _LOGGER.debug(
                "CuPy PCG64/Generator path failed (%r); using default_rng.", exc
            )
        if gen is None:
            gen = cp.random.default_rng(seed)
            _LOGGER.debug("Using CuPy default_rng seed=%d", seed)
        be.rng = gen

    _LOGGER.info("Initialized GPU backend (CuPy)")
    return be


def _auto_backend(
    device: str,
    seed: int = 1234,
    *,
    strict: bool = False,
    debug: bool = False,
) -> ArrayBackend:
    """Select and initialize the backend based on `device` and availability.

    Args:
        device: One of 'cpu', 'gpu', or 'auto'.
        seed: RNG seed.
        strict: If True, raise on GPU init failure instead of falling back.
        debug: Unused flag preserved for symmetry; logs already contain detail.

    Returns:
        An initialized `ArrayBackend`.
    """
    _LOGGER.debug("Selecting backend: device=%s strict=%s", device, strict)
    if device == "cpu":
        return _make_cpu_backend(seed)
    if device == "gpu":
        try:
            return _try_make_gpu_backend(seed, debug=debug)
        except Exception as err:
            _LOGGER.error("GPU backend init failed: %r", err)
            if strict:
                raise
            _LOGGER.warning("Falling back to CPU backend.")
            return _make_cpu_backend(seed)
    # auto: prefer GPU, else CPU
    try:
        return _try_make_gpu_backend(seed, debug=debug)
    except Exception as err:
        _LOGGER.warning("Auto GPU init failed (%r); using CPU.", err)
        return _make_cpu_backend(seed)


# -----------------------------------------------------------------------------
# Config singleton + dynamic proxies
# -----------------------------------------------------------------------------
class Config:
    """Global configuration for the purkinje-uv array backend.

    Provides global device selection (cpu/gpu/auto), deterministic seeding, and
    dynamic proxies so code importing `xp`/`rng` always sees the current backend.
    """

    _HAS_DYNAMIC_ATTRIBUTES = True

    def __init__(self) -> None:
        """Initialize config using environment defaults."""
        self._seed_default = int_env("PURKINJE_UV_SEED", 1234)
        device = _device_env()  # 'cpu' | 'gpu' | 'auto'
        self._backend: ArrayBackend = _auto_backend(device, seed=self._seed_default)
        _LOGGER.info(
            "Config initialized: device=%s seed=%d backend=%s",
            device,
            self._seed_default,
            self._backend.name,
        )

    def configure(
        self,
        device: str = "auto",
        *,
        seed: Optional[int] = None,
        strict: bool = False,
        debug: bool = False,
    ) -> Config:
        """Reconfigure the active backend.

        Args:
            device: One of 'cpu', 'gpu', or 'auto'.
            seed: Optional seed (defaults to the environment default).
            strict: If True, raise on GPU init failure instead of fallback.
            debug: If True, include more detailed logs (already enabled globally).

        Returns:
            The `Config` instance (for chaining).
        """
        seed_value = self._seed_default if seed is None else int(seed)
        _LOGGER.info(
            "Reconfiguring: device=%s seed=%d strict=%s debug=%s",
            device,
            seed_value,
            strict,
            debug,
        )
        self._backend = _auto_backend(
            device, seed=seed_value, strict=strict, debug=debug
        )
        return self

    @contextlib.contextmanager
    def use(
        self,
        device: str,
        *,
        seed: Optional[int] = None,
        strict: bool = False,
        debug: bool = False,
    ) -> Iterator[None]:
        """Temporarily switch backend within a context manager.

        Args:
            device: One of 'cpu' or 'gpu'.
            seed: Optional seed for the temporary backend.
            strict: If True, raise on GPU init failure instead of fallback.
            debug: If True, include more detailed logs.

        Yields:
            None. Restores the previous backend on exit.
        """
        prev = self._backend
        try:
            self.configure(device=device, seed=seed, strict=strict, debug=debug)
            yield
        finally:
            self._backend = prev
            _LOGGER.info("Restored previous backend: %s", self._backend.name)

    def seed(self, s: int = 1234) -> None:
        """Reseed the current backend RNG deterministically.

        Args:
            s: The seed value.
        """
        _LOGGER.info("Reseeding RNG to %d on backend=%s", s, self._backend.name)
        self._backend.seed(s)

    @property
    def is_gpu(self) -> bool:
        """Return True if the active backend is a GPU backend."""
        return self._backend.is_gpu

    @property
    def backend_name(self) -> str:
        """Return the name of the active backend ('numpy' or 'cupy')."""
        return self._backend.name

    @property
    def xp(self) -> Any:
        """Return the active array module (NumPy or CuPy)."""
        return self._backend.xp

    @property
    def rng(self) -> Any:
        """Return the active RNG object."""
        return self._backend.rng

    def cdist(self, *args: Any, **kwargs: Any) -> Any:
        """Compute pairwise distances on the active backend."""
        return self._backend.cdist(*args, **kwargs)

    def to_cpu(self, a: Any) -> Any:
        """Copy an array to CPU if needed."""
        return self._backend.to_cpu(a)

    def to_device(self, a: Any, dtype: Any | None = None) -> Any:
        """Copy an array to the active backend."""
        return self._backend.to_device(a, dtype=dtype)

    def norm(self, v: Any, axis: int = -1, keepdims: bool = False) -> Any:
        """Compute the L2 norm on the active backend."""
        return self._backend.norm(v, axis=axis, keepdims=keepdims)


class _XPProxy:
    """Proxy for `xp` that forwards attribute access to the current backend."""

    def __init__(self, _cfg: Config) -> None:
        self._cfg = _cfg

    def __getattr__(self, name: str) -> Any:  # noqa: D401
        return getattr(self._cfg.xp, name)


class _RNGProxy:
    """Proxy for `rng` that forwards attribute access to the current backend."""

    def __init__(self, _cfg: Config) -> None:
        self._cfg = _cfg

    def __getattr__(self, name: str) -> Any:  # noqa: D401
        return getattr(self._cfg.rng, name)


# Singleton & forwards
config = Config()
xp = _XPProxy(config)
rng = _RNGProxy(config)


def cdist(*args: Any, **kwargs: Any) -> Any:
    """Compute pairwise distances on the active backend (module-level)."""
    return config.cdist(*args, **kwargs)


def to_cpu(a: Any) -> Any:
    """Copy an array to CPU if needed (module-level)."""
    return config.to_cpu(a)


def to_device(a: Any, dtype: Any | None = None) -> Any:
    """Copy an array to the active backend (module-level)."""
    return config.to_device(a, dtype=dtype)


def norm(v: Any, axis: int = -1, keepdims: bool = False) -> Any:
    """Compute the L2 norm on the active backend (module-level)."""
    return config.norm(v, axis=axis, keepdims=keepdims)


def is_gpu() -> bool:
    """Return True if the active backend is a GPU backend (module-level)."""
    return config.is_gpu


def backend_name() -> str:
    """Return the name of the active backend (module-level)."""
    return config.backend_name


def configure(
    device: str = "auto",
    *,
    seed: Optional[int] = None,
    strict: bool = False,
    debug: bool = False,
) -> Config:
    """Reconfigure the active backend (module-level)."""
    return config.configure(device, seed=seed, strict=strict, debug=debug)


def use(
    device: str,
    *,
    seed: Optional[int] = None,
    strict: bool = False,
    debug: bool = False,
) -> ContextManager[None]:
    """Temporarily switch backend within a context manager (module-level)."""
    return config.use(device, seed=seed, strict=strict, debug=debug)


def seed(s: int = 1234) -> None:
    """Reseed the current backend RNG deterministically (module-level)."""
    config.seed(s)
