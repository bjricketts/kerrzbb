# kerrzbb

A multi-temperature blackbody model for a thin accretion disc around a Kerr
black hole, following KERRBB (Li, Zimmerman, Narayan & McClintock 2005). The
disc image is ray traced on the fly with [kerrz](https://cosroe.com/kerrz)
instead of being read from a precomputed table, and the spectrum comes with
forward-mode derivatives with respect to every parameter.

Self-irradiation of the disc by returning radiation (kerrbb's `rflag`) is 
included.

## Building

Requires Zig 0.15.2 and a kerrz checkout next to this repository (`../kerrz`).

    zig build test                               # unit tests
    zig build lib -Doptimize=ReleaseSafe         # zig-out/lib/libkerrzbb.{so,dylib} + zig-out/include/kerrzbb.h

`ReleaseSafe` is recommended for the library. kerrz contains assertions that
can fail for extreme rays: in `ReleaseSafe` they stop the process with a
message, while in `ReleaseFast` they are undefined behaviour. `ReleaseSafe`
costs about 10-20% in speed.

## Performance

Release build, one thread, 200 or 3000 bins (`python validation/benchmark.py`):

| | value | value + d/d(a, i, mdot) |
|---|---|---|
| new spin or inclination | 22 ms | 38 ms |
| new spin, with rflag | 80 ms | 125 ms |
| cached (only M, mdot, D, fcol, eta or norm changed) | 17 ms | 25 ms |

The ray tracing is cached per model instance (`kerrbb.Model`,
`kzbb_model_create`, the Python `KerrzBB` object, `KerrzBBCache` in Julia), so
fits that vary only the mass, accretion rate, distance, fcol, eta or norm skip
it. Many narrow bins are handled on an interpolated ln E grid, so the cost does
not grow with the number of bins. `n_threads` (0 for one per CPU) parallelises
the ray tracing, the returning-radiation kernel and the energy sums; four
threads give about 3x. Results are identical for any thread count.

## Parameters

Following XSPEC kerrbb: `eta` (torque), `a` (spin), `incl` (degrees), `mass`
(M_sun), `mdot` (effective accretion rate, 1e18 g/s), `distance` (kpc), `fcol`
(hardening factor), `norm`, and optionally `r_in` (r_g, default the
marginally stable orbit), plus the limb-darkening (`lflag`) and
self-irradiation (`rflag`) switches.

Supported ranges: 1e-3 <= |a| <= 0.9999 and 0 < i <= 89 degrees. Very small
spins are excluded because kerrz's geodesic solutions are unreliable there.

The output is the photon flux per energy bin (photons cm^-2 s^-1). The
Jacobian has one row per bin and one column per free parameter, in the order
above; the inclination derivative is per degree.

## Interfaces

- **Zig**: `kerrzbb.kerrbb.evaluate` (`src/kerrbb.zig`), or `kerrzbb.spectrum`
  for generic dual-number use.
- **C**: `include/kerrzbb.h`; see `examples/example.c`.
- **Python**: `python/kerrzbb` (ctypes + numpy). Set `KERRZBB_LIBRARY` or build
  the library in `zig-out/lib`.

      from kerrzbb import KerrzBB
      flux, jac = KerrzBB()(edges, a=0.9, incl=60, mass=10, mdot=1, distance=10,
                            free=("a", "incl"))

- **Julia**: `julia/KerrzBB` provides `kerrzbb_flux` and `KerrzBBModel`, a
  SpectralFitting.jl additive model. ForwardDiff derivatives are assembled from
  kerrzbb's Jacobian. SpectralFitting and its dependencies are in the
  University of Bristol AstroRegistry, and KerrzBB itself is added by path:

      julia> import Pkg
      julia> Pkg.Registry.add(url = "https://github.com/astro-group-bristol/AstroRegistry")
      julia> Pkg.develop(path = "/path/to/kerrzbb/julia/KerrzBB")
      julia> using KerrzBB   # finds ../../zig-out/lib, or set ENV["KERRZBB_LIBRARY"]

  Run its tests with `Pkg.test("KerrzBB")`.
- **XSPEC**: `xspec/` holds a local model with the same parameters as kerrbb.
  The library is loaded at run time from `KERRZBB_LIBRARY`, and
  `KERRZBB_THREADS` sets the threads per evaluation:

      export KERRZBB_LIBRARY=$PWD/zig-out/lib/libkerrzbb.dylib
      cd xspec && initpackage kerrzbb lmodel_kerrzbb.dat . && hmake
      xspec> lmod kerrzbb /path/to/kerrzbb/xspec

## Validation

    zig build validate                                        # escape fraction of disc emission
    zig build derivative-scan                                 # dual vs finite-difference derivatives
    zig build returning-fractions                             # fates of emitted photons (Li et al. Fig. 2)
    python validation/kerrbb_table_extract.py $HEADAS/../spectral/modelData/kerrbb.fits
    zig build compare -- validation/kerrbb_table.json out.json   # against kerrbb's table nodes
    python validation/plot_comparison.py out.json

For zero torque, kerrzbb agrees with kerrbb's table to about 0.1% without
self-irradiation and to about 1% with it for a <= 0.9.
