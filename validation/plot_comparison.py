#!/usr/bin/env python3
"""Plot kerrzbb against XSPEC kerrbb from the output of

    zig build compare -- validation/kerrbb_reference.json validation/kerrbb_comparison.json

Usage: python validation/plot_comparison.py [comparison.json] [key=value,...]
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "validation/kerrbb_comparison.json"
cases = json.load(open(path))["cases"]
if len(sys.argv) > 2:  # optional subset, e.g. "eta=0,lflag=0"
    for cond in sys.argv[2].split(","):
        key, val = cond.split("=")
        cases = [c for c in cases if abs(float(c[key]) - float(val)) < 1e-6]

fig, (ax, rax) = plt.subplots(2, 1, sharex=True, figsize=(7, 7),
                              gridspec_kw={"height_ratios": [2, 1]})
for c in cases:
    if "edges" in c:
        e = np.asarray(c["edges"])
        mid = np.sqrt(e[1:] * e[:-1])
        width = np.diff(e)
    else:  # differential spectra from the kerrbb table
        mid = np.asarray(c["energies"])
        width = np.ones_like(mid)
    ref = np.asarray(c["reference"]) / width
    ours = np.asarray(c["kerrzbb"]) / width
    label = f"a={c['a']:+.3f} i={c['incl_deg']:.0f} eta={c['eta']:.1f} ld={c['lflag']}"
    (line,) = ax.loglog(mid, mid**2 * ref, lw=1, label=label)
    ax.loglog(mid, mid**2 * ours, ls="--", lw=1, color=line.get_color())
    good = mid**2 * ref > 1e-3 * (mid**2 * ref).max()
    rax.semilogx(mid[good], ours[good] / ref[good] - 1, lw=1, color=line.get_color())

ymax = max(line.get_ydata().max() for line in ax.get_lines())
ax.set_ylim(ymax * 1e-6, ymax * 3)
ax.set_ylabel(r"$E^2 N(E)$ (keV$^2$ ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$)")
ax.set_title("solid: XSPEC kerrbb, dashed: kerrzbb")
ax.legend(fontsize=6, ncol=2)
rax.axhline(0, color="k", lw=0.5)
rax.set_ylabel("kerrzbb / kerrbb - 1")
rax.set_xlabel("E (keV)")
fig.tight_layout()
out = path.replace(".json", ".png")
fig.savefig(out, dpi=150)
print("wrote", out)
