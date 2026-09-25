#!/usr/bin/env python3
"""Generate reference spectra from XSPEC's kerrbb for comparison with kerrzbb.

Requires HEASoft with PyXspec (``import xspec``). Run on a machine with
HEASoft initialised, e.g.

    python validation/kerrbb_reference.py --list-grid     # show the table grid
    python validation/kerrbb_reference.py                 # write the reference file

The output (validation/kerrbb_reference.json) is read by
``zig build compare -- validation/kerrbb_reference.json``.

Compare only at parameter values that are nodes of kerrbb's internal table
(spin, inclination, eta), so that XSPEC's interpolation does not enter the
comparison. ``--list-grid`` searches the XSPEC model data directory for the
kerrbb table and prints its axes if it can be read.
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np


def list_grid():
    headas = os.environ.get("HEADAS")
    if not headas:
        raise SystemExit("HEADAS is not set; initialise HEASoft first.")
    roots = [os.path.join(headas, "..", "spectral", "modelData"),
             os.path.join(headas, "spectral", "modelData")]
    files = sorted({f for r in roots for f in glob.glob(os.path.join(r, "*kerr*"))})
    if not files:
        print("No kerrbb data files found under", roots)
        return
    try:
        from astropy.io import fits
    except ImportError:
        fits = None
    for f in files:
        print(f)
        if fits is None or not f.endswith((".fits", ".fits.gz", ".mod")):
            continue
        try:
            with fits.open(f) as hdul:
                hdul.info()
                for hdu in hdul:
                    if hdu.name.upper() == "PARAMETERS":
                        for row in hdu.data:
                            nvals = int(row["NUMBVALS"])
                            print(f"  {row['NAME'].strip()}: {list(row['VALUE'][:nvals])}")
        except Exception as exc:  # noqa: BLE001
            print("  could not read:", exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-grid", action="store_true")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                      "kerrbb_reference.json"))
    parser.add_argument("--spins", type=float, nargs="+",
                        default=[-0.9, 0.5, 0.9, 0.99, 0.998])
    parser.add_argument("--incl", type=float, nargs="+", default=[30.0, 60.0, 85.0],
                        help="inclinations in degrees")
    parser.add_argument("--eta", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--lflag", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--mass", type=float, default=10.0)
    parser.add_argument("--mdot", type=float, default=1.0, help="Mdd in 1e18 g/s")
    parser.add_argument("--dist", type=float, default=10.0, help="kpc")
    parser.add_argument("--hd", type=float, default=1.7)
    parser.add_argument("--emin", type=float, default=0.1)
    parser.add_argument("--emax", type=float, default=50.0)
    parser.add_argument("--nbins", type=int, default=200)
    args = parser.parse_args()

    if args.list_grid:
        list_grid()
        return

    import xspec

    xspec.Xset.chatter = 0
    xspec.AllModels.setEnergies(f"{args.emin} {args.emax} {args.nbins} log")
    model = xspec.Model("kerrbb")

    cases = []
    for a, incl, eta, lflag in itertools.product(args.spins, args.incl, args.eta, args.lflag):
        # kerrbb parameters: eta, a, i, Mbh, Mdd, Dbh, hd, rflag, lflag, norm
        model.setPars({1: eta, 2: a, 3: incl, 4: args.mass, 5: args.mdot,
                       6: args.dist, 7: args.hd, 8: 0, 9: lflag, 10: 1.0})
        edges = list(np.asarray(model.energies(0), dtype=float))
        flux = list(np.asarray(model.values(0), dtype=float))
        cases.append({
            "eta": eta, "a": a, "incl_deg": incl, "mass": args.mass,
            "mdot": args.mdot, "distance": args.dist, "fcol": args.hd,
            "rflag": 0, "lflag": lflag, "norm": 1.0,
            "edges": edges, "flux": flux,
        })
        print(f"a={a:+.4f} i={incl:4.1f} eta={eta:.2f} lflag={lflag}: "
              f"total {sum(flux):.4e} ph/cm^2/s")

    with open(args.out, "w") as fh:
        json.dump({"source": "XSPEC kerrbb (rflag=0)", "cases": cases}, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
