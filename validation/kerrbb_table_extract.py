#!/usr/bin/env python3
"""Extract spectra at grid nodes directly from XSPEC's kerrbb table.

runkbb.f (HEASoft) evaluates kerrbb as

    N(E) = cc * FLUX0(a, i, eta, E / ca),
    ca = Mdd^{1/4} M^{-1/2} fcol,  cc = M^2 / D^2 * fcol^-4 * ca^2,

so FLUX0 is the differential photon spectrum (photons keV^-1 cm^-2 s^-1) for
M = 1 M_sun, Mdd = 1e18 g/s, D = 1 kpc and fcol = 1. At grid nodes this needs
no interpolation in (a, i, eta) or energy, so it is the cleanest reference.

Only numpy is required; the FITS file is read directly.

    python validation/kerrbb_table_extract.py $HEADAS/../spectral/modelData/kerrbb.fits
"""

import argparse
import json
import os

import numpy as np

NG, NS, NTH, NENER = 6, 46, 18, 601
COLUMNS = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}  # (rflag, lflag) -> column


def read_hdus(path):
    """Minimal FITS reader returning (header dict, raw data bytes) per HDU."""
    out = []
    with open(path, "rb") as f:
        while True:
            cards = []
            while True:
                block = f.read(2880)
                if not block:
                    return out
                chunk = [block[i:i + 80].decode("ascii") for i in range(0, 2880, 80)]
                cards += chunk
                if any(c.rstrip() == "END" for c in chunk):
                    break
            header = {}
            for c in cards:
                if c[8:10] == "= ":
                    header[c[:8].strip()] = c[10:].split("/")[0].strip().strip("'").strip()
            naxis = int(header.get("NAXIS", 0))
            size = 0
            if naxis:
                size = abs(int(header["BITPIX"])) // 8
                for k in range(1, naxis + 1):
                    size *= int(header[f"NAXIS{k}"])
            data = f.read(size)
            f.seek((-size) % 2880, 1)
            out.append((header, data))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                      "kerrbb_table.json"))
    parser.add_argument("--spin-index", type=int, nargs="+",
                        default=[1, 5, 11, 15, 17, 21, 27, 35, 45],
                        help="0-based indices into the 46 spin nodes")
    parser.add_argument("--incl-index", type=int, nargs="+", default=[1, 6, 12, 17])
    parser.add_argument("--eta-index", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--lflag", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--rflag", type=int, nargs="+", default=[0])
    parser.add_argument("--list", action="store_true", help="print the grid and exit")
    args = parser.parse_args()

    hdus = read_hdus(args.table)
    grid = np.frombuffer(hdus[1][1], dtype=">f4").reshape(4, 601)
    eta_grid, spin_grid = grid[0, :NG], grid[1, :NS]
    incl_grid, energy = grid[2, :NTH], grid[3, :NENER]
    if args.list:
        print("eta :", eta_grid.tolist())
        print("spin:", spin_grid.tolist())
        print("incl:", incl_grid.tolist())
        print("E   :", energy[0], "...", energy[-1], f"({NENER} log-spaced)")
        return

    rows = np.frombuffer(hdus[2][1], dtype=">f4").reshape(-1, 4)
    cases = []
    for rflag in args.rflag:
        for lflag in args.lflag:
            # Fortran order: FLUX0(NS, NTH, NG, NENER)
            flux0 = rows[:, COLUMNS[(rflag, lflag)]].reshape((NENER, NG, NTH, NS))
            for ie in args.eta_index:
                for ii in args.incl_index:
                    for ia in args.spin_index:
                        cases.append({
                            "eta": float(eta_grid[ie]), "a": float(spin_grid[ia]),
                            "incl_deg": float(incl_grid[ii]),
                            "mass": 1.0, "mdot": 1.0, "distance": 1.0, "fcol": 1.0,
                            "rflag": rflag, "lflag": lflag, "norm": 1.0,
                            "energies": energy.astype(float).tolist(),
                            "density": flux0[:, ie, ii, ia].astype(float).tolist(),
                        })
    with open(args.out, "w") as fh:
        json.dump({"source": f"kerrbb table {args.table}", "cases": cases}, fh)
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
