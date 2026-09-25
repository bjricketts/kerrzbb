#!/usr/bin/env python3
"""Timing of kerrzbb through the Python bindings.

    zig build lib -Doptimize=ReleaseFast
    python validation/benchmark.py

"Cold" calls build the ray tracing from scratch (a new spin or inclination);
"warm" calls change only the accretion rate and reuse the cache.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from kerrzbb import KerrzBB  # noqa: E402

BASE = dict(a=0.9, incl=60.0, mass=10.0, mdot=1.0, distance=10.0)


def timed(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run(n_threads, n_bins, rflag, free):
    edges = np.geomspace(0.1, 50.0, n_bins + 1)
    model = KerrzBB(n_threads=n_threads)
    spins = iter(np.linspace(0.5, 0.95, 50))

    def cold():
        model(edges, **dict(BASE, a=next(spins)), returning_radiation=rflag, free=free)

    mdots = iter(np.linspace(0.5, 2.0, 50))

    def warm():
        model(edges, **dict(BASE, mdot=next(mdots)), returning_radiation=rflag, free=free)

    t_cold = timed(cold)
    warm()  # fill the cache for BASE spin
    t_warm = timed(warm)
    return t_cold, t_warm


def main():
    cpus = os.cpu_count() or 1
    print(f"{'threads':>7} {'bins':>5} {'rflag':>5} {'free':>18} {'cold (s)':>9} {'warm (s)':>9}")
    for n_threads in sorted({1, cpus}):
        for n_bins in (200, 3000):
            for rflag in (False, True):
                for free in ((), ("a", "incl", "mdot")):
                    t_cold, t_warm = run(n_threads, n_bins, rflag, free)
                    label = ",".join(free) or "-"
                    print(f"{n_threads:>7} {n_bins:>5} {int(rflag):>5} {label:>18} {t_cold:>9.3f} {t_warm:>9.3f}")


if __name__ == "__main__":
    main()
