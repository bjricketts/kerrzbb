"""Python bindings for kerrzbb (ctypes over the C ABI in include/kerrzbb.h).

    import numpy as np
    from kerrzbb import KerrzBB

    model = KerrzBB()
    edges = np.geomspace(0.1, 50, 201)
    flux = model(edges, a=0.9, incl=60, mass=10, mdot=1, distance=10)
    flux, jac = model(edges, a=0.9, incl=60, mass=10, mdot=1, distance=10,
                      free=("a", "incl"))

The shared library is located from, in order: the `library` argument, the
KERRZBB_LIBRARY environment variable, and zig-out/lib in the repository.
Build it with `zig build lib -Doptimize=ReleaseSafe`.
"""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

__all__ = ["KerrzBB", "KerrzBBError", "PARAMETERS"]

#: Parameter names in Jacobian-column order (bit k of the free mask).
PARAMETERS = ("eta", "a", "incl", "mass", "mdot", "distance", "fcol", "norm", "r_in")


class _Params(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in PARAMETERS] + [
        ("use_r_in", ctypes.c_int),
        ("limb_darkening", ctypes.c_int),
        ("returning_radiation", ctypes.c_int),
    ]


class _Options(ctypes.Structure):
    _fields_ = [
        ("n_theta", ctypes.c_size_t),
        ("n_rho", ctypes.c_size_t),
        ("n_outer", ctypes.c_size_t),
        ("n_energy", ctypes.c_size_t),
        ("r_break", ctypes.c_double),
        ("r_out", ctypes.c_double),
        ("observer_distance", ctypes.c_double),
        ("n_radii", ctypes.c_size_t),
        ("n_psi", ctypes.c_size_t),
        ("n_chi", ctypes.c_size_t),
        ("r_max", ctypes.c_double),
        ("n_threads", ctypes.c_size_t),
        ("energy_grid", ctypes.c_int),
        ("grid_step", ctypes.c_double),
    ]


class KerrzBBError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(f"kerrzbb error {code}: {message}")
        self.code = code


def _find_library(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("KERRZBB_LIBRARY"):
        candidates.append(Path(os.environ["KERRZBB_LIBRARY"]))
    suffix = ".dylib" if sys.platform == "darwin" else ".dll" if os.name == "nt" else ".so"
    repo = Path(__file__).resolve().parents[2]
    candidates.append(repo / "zig-out" / "lib" / f"libkerrzbb{suffix}")
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError(
        "libkerrzbb not found; build it with `zig build lib` or set KERRZBB_LIBRARY. "
        f"Tried: {', '.join(map(str, candidates))}"
    )


class KerrzBB:
    """Callable kerrzbb model with a cache of the ray tracing.

    Keyword options override the numerical defaults: n_theta, n_rho, n_outer,
    n_energy, r_break, r_out, observer_distance (image), n_radii, n_psi, n_chi,
    r_max (returning radiation), n_threads (0: one per CPU), energy_grid,
    grid_step. Calls that change only mass, mdot, distance, fcol, eta or norm
    reuse the cached ray tracing. An instance is not thread-safe.
    """

    def __init__(self, library=None, **options):
        self._lib = ctypes.CDLL(_find_library(library))
        lib = self._lib
        lib.kzbb_default_options.restype = _Options
        lib.kzbb_free_count.argtypes = [ctypes.c_uint32]
        lib.kzbb_free_count.restype = ctypes.c_int
        lib.kzbb_evaluate.argtypes = [
            ctypes.POINTER(_Params), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(_Options),
        ]
        lib.kzbb_evaluate.restype = ctypes.c_int
        lib.kzbb_model_create.argtypes = [ctypes.POINTER(_Options)]
        lib.kzbb_model_create.restype = ctypes.c_void_p
        lib.kzbb_model_destroy.argtypes = [ctypes.c_void_p]
        lib.kzbb_model_destroy.restype = None
        lib.kzbb_model_evaluate.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_Params), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        lib.kzbb_model_evaluate.restype = ctypes.c_int
        lib.kzbb_status_string.argtypes = [ctypes.c_int]
        lib.kzbb_status_string.restype = ctypes.c_char_p
        lib.kzbb_version.restype = ctypes.c_char_p

        self.options = lib.kzbb_default_options()
        for key, value in options.items():
            if key not in dict(_Options._fields_):
                raise TypeError(f"unknown option {key!r}")
            setattr(self.options, key, int(value) if key == "energy_grid" else value)
        self._model = lib.kzbb_model_create(ctypes.byref(self.options))
        if not self._model:
            raise ValueError("invalid options")

    def __del__(self):
        model = getattr(self, "_model", None)
        if model:
            self._lib.kzbb_model_destroy(model)
            self._model = None

    @property
    def version(self):
        return self._lib.kzbb_version().decode()

    def __call__(self, edges, *, a, incl, mass, mdot, distance, eta=0.0, fcol=1.7,
                 norm=1.0, r_in=None, limb_darkening=False,
                 returning_radiation=False, free=()):
        """Photon flux per bin (photons cm^-2 s^-1) for bin edges in keV.

        Returns `flux` if `free` is empty, otherwise `(flux, jacobian)` with
        `jacobian[b, k] = d flux[b] / d free_k`, the columns ordered as in
        PARAMETERS (not as given in `free`). The inclination is in degrees,
        and so is its derivative.
        """
        edges = np.ascontiguousarray(edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("edges must be a 1-d array with at least two values")
        n_bins = edges.size - 1

        mask = 0
        for name in free:
            if name not in PARAMETERS:
                raise ValueError(f"unknown parameter {name!r}")
            mask |= 1 << PARAMETERS.index(name)
        n_free = self._lib.kzbb_free_count(mask)

        params = _Params(
            eta=eta, a=a, incl=incl, mass=mass, mdot=mdot, distance=distance,
            fcol=fcol, norm=norm, r_in=0.0 if r_in is None else r_in,
            use_r_in=int(r_in is not None), limb_darkening=int(limb_darkening),
            returning_radiation=int(returning_radiation),
        )
        flux = np.empty(n_bins)
        jac = np.empty((n_bins, n_free)) if n_free else None
        dp = ctypes.POINTER(ctypes.c_double)
        status = self._lib.kzbb_model_evaluate(
            self._model, ctypes.byref(params), mask, edges.ctypes.data_as(dp), n_bins,
            flux.ctypes.data_as(dp), jac.ctypes.data_as(dp) if n_free else None,
        )
        if status != 0:
            raise KerrzBBError(status, self._lib.kzbb_status_string(status).decode())
        return (flux, jac) if n_free else flux

    def free_order(self, free):
        """The Jacobian column order for a given set of free parameters."""
        return tuple(p for p in PARAMETERS if p in set(free))
