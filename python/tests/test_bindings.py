"""Tests for the Python bindings. Run with `python -m pytest python/tests`
or directly with `python python/tests/test_bindings.py`."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kerrzbb import PARAMETERS, KerrzBB, KerrzBBError  # noqa: E402

FAST = dict(n_theta=48, n_rho=32, n_outer=16)
EDGES = np.array([0.3, 1.0, 3.0, 8.0])
BASE = dict(eta=0.1, a=0.8, incl=50.0, mass=8.0, mdot=1.2, distance=7.0,
            fcol=1.6, norm=1.0, r_in=3.5)


def test_value_and_jacobian():
    model = KerrzBB(**FAST)
    flux = model(EDGES, **BASE)
    assert flux.shape == (3,) and np.all(flux > 0)

    flux2, jac = model(EDGES, **BASE, free=PARAMETERS)
    assert jac.shape == (3, 9)
    np.testing.assert_array_equal(flux, flux2)

    for k, name in enumerate(PARAMETERS):
        rel = 3e-4 if name in ("a", "incl", "r_in") else 1e-5
        h = rel * max(1.0, abs(BASE[name]))
        up = dict(BASE, **{name: BASE[name] + h})
        dn = dict(BASE, **{name: BASE[name] - h})
        fd = (model(EDGES, **up) - model(EDGES, **dn)) / (2 * h)
        np.testing.assert_allclose(jac[:, k], fd, rtol=3e-5, atol=1e-6 * flux.max(),
                                   err_msg=name)


def test_column_order_follows_parameters():
    model = KerrzBB(**FAST)
    _, full = model(EDGES, **BASE, free=PARAMETERS)
    _, part = model(EDGES, **BASE, free=("fcol", "a"))
    assert model.free_order(("fcol", "a")) == ("a", "fcol")
    np.testing.assert_allclose(part[:, 0], full[:, PARAMETERS.index("a")], rtol=1e-12)
    np.testing.assert_allclose(part[:, 1], full[:, PARAMETERS.index("fcol")], rtol=1e-12)


def test_cache_and_threads():
    serial = KerrzBB(**FAST)
    threaded = KerrzBB(**FAST, n_threads=3)
    for mdot in (1.0, 1.5):
        a = serial(EDGES, **dict(BASE, mdot=mdot), returning_radiation=True)
        b = threaded(EDGES, **dict(BASE, mdot=mdot), returning_radiation=True)
        np.testing.assert_array_equal(a, b)
    try:
        KerrzBB(n_energy=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid options")


def test_errors():
    model = KerrzBB(**FAST)
    try:
        model(EDGES, **dict(BASE, a=0.0, r_in=None))
    except KerrzBBError as err:
        assert err.code == 2
    else:
        raise AssertionError("expected KerrzBBError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
