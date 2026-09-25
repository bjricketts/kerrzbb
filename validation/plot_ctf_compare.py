"""Figures for the B1 image-plane vs stored Cunningham transfer function comparison.

    zig build ctf-compare -Doptimize=ReleaseFast
    python validation/plot_ctf_compare.py [validation/ctf_compare.json] [outdir]

Writes ctf_cost.png, ctf_tabulated.png, ctf_spectra.png and ctf_gmax.png.
Errors are max |relative error| where E^2 N(E) > 1e-3 of its peak, against B1
at 1024x768. Derivative errors are max |d ln N / d theta - reference| over the
same energies, against B1 256x192 with dual numbers.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "validation/ctf_compare.json")
outdir = Path(sys.argv[2] if len(sys.argv) > 2 else path.parent)
data = json.loads(path.read_text())
E = np.array(data["energies"])
cases = data["cases"]
C_B1, C_CTF = "C0", "C1"


def title(c):
    return f"a = {c['a']:.4g}, i = {c['incl_deg']:g}°"


def mask(ref, frac=1e-3):
    p = E**2 * ref
    return p > frac * p.max()


def err(dens, ref):
    k = mask(ref)
    return np.abs(np.array(dens)[k] / ref[k] - 1).max()


def derr(d, dref, ref):
    k = mask(ref)
    return np.abs((np.array(d)[k] - dref[k]) / ref[k]).max()


def floor(x):
    return max(x, 1e-12)


# 1. Cost: per evaluation (geometry stored) and one-off per (a, i) node.
fig, axes = plt.subplots(2, len(cases), figsize=(4.3 * len(cases), 7.2), sharey=True)
for k, c in enumerate(cases):
    ref = np.array(c["reference"])
    b1 = [m for m in c["exact"] if m["method"] == "B1"]
    ctf = [m for m in c["exact"] if m["method"] == "CTF"]
    eb1 = [floor(err(m["density"], ref)) for m in b1]
    ectf = [floor(err(m["density"], ref)) for m in ctf]
    ax = axes[0, k]
    ax.loglog([m["build_ms"] + m["eval_ms"] for m in b1], eb1, "o-", color=C_B1, label="B1, traced for this (a, i)")
    ax.loglog([m["eval_ms"] for m in b1], eb1, "o--", color=C_B1, mfc="none", label="B1, image stored")
    ax.loglog([m["eval_ms"] for m in ctf], ectf, "s--", color=C_CTF, mfc="none", label="CTF, table stored at this (a, i)")
    for m, e in zip(b1, eb1):
        ax.annotate(m["label"], (m["eval_ms"], e), fontsize=6, xytext=(-4, -9), textcoords="offset points", ha="right")
    for m, e in zip(ctf, ectf):
        ax.annotate(m["label"].split()[0], (m["eval_ms"], e), fontsize=6, xytext=(3, 2), textcoords="offset points")
    # Stored on a kerrbb-spaced (a, i) grid, cubic interpolation, evaluated mid-cell.
    for meth, col in (("B1", C_B1), ("CTF", C_CTF)):
        t = [m for m in c["tabulated"] if m["method"] == meth and m["order"] == 3 and m["factor"] == 1][0]
        ax.plot(t["eval_ms"], err(t["density"], ref), "*", ms=11, color=col, mec="k", mew=0.5,
                label=f"{meth} interpolated from kerrbb-spaced grid" if k == 0 else None)
    ax.set_title(title(c))
    ax.set_xlabel("time per evaluation [ms]")
    ax.axhline(1e-3, color="0.75", lw=0.8, zorder=0)
    ax = axes[1, k]
    ax.loglog([m["build_ms"] for m in b1], eb1, "o-", color=C_B1, label="B1: trace one image")
    ax.loglog([m["build_ms"] for m in ctf], ectf, "s-", color=C_CTF, label="CTF: build one table")
    ax.set_xlabel("one-off cost per (a, i) [ms]")
    ax.axhline(1e-3, color="0.75", lw=0.8, zorder=0)
axes[0, 0].set_ylabel("max |relative error|")
axes[1, 0].set_ylabel("max |relative error|")
axes[0, 0].legend(fontsize=6.5, loc="lower left")
axes[1, 0].legend(fontsize=7, loc="lower left")
fig.suptitle("Spectrum error against cost (one thread; 200-bin evaluation, 0.1–50 keV)")
fig.tight_layout()
fig.savefig(outdir / "ctf_cost.png", dpi=150)

# 2. Stored on an (a, i) grid: interpolation error against grid spacing.
fig, axes = plt.subplots(3, len(cases), figsize=(4.3 * len(cases), 9), sharex=True)
rows = [("value", "max |rel. error| of N(E)"), ("d_a", r"max |error| of $\partial \ln N/\partial a$"),
        ("d_i", r"max |error| of $\partial \ln N/\partial i$ [deg$^{-1}$]")]
for k, c in enumerate(cases):
    ref = np.array(c["reference"])
    exact_b1 = [m for m in c["exact"] if m["label"] == "128x96"][0]
    for j, (what, lab) in enumerate(rows):
        ax = axes[j, k]
        for meth, col in (("B1", C_B1), ("CTF", C_CTF)):
            for order, ls in ((1, ":"), (3, "-")):
                ms = sorted([m for m in c["tabulated"] if m["method"] == meth and m["order"] == order],
                            key=lambda m: -m["factor"])
                f = [m["factor"] for m in ms]
                if what == "value":
                    y = [err(m["density"], ref) for m in ms]
                else:
                    y = [derr(m[what], np.array(c["reference_" + what]), ref) for m in ms]
                ax.loglog(f, np.maximum(y, 1e-12), ls, marker="o" if meth == "B1" else "s", color=col,
                          label=f"{meth}, {'linear' if order == 1 else 'cubic'}")
        if what != "value":
            ax.axhline(floor(derr(exact_b1[what], np.array(c["reference_" + what]), ref)), color=C_B1, lw=0.8, ls="--",
                       label="B1 128x96, dual numbers")
            scale = np.abs(np.array(c["reference_" + what])[mask(ref)] / ref[mask(ref)]).max()
            ax.axhline(scale, color="0.5", lw=0.8, ls="-.", label=r"max |$\partial \ln N$| itself")
        if j == 0:
            ax.set_title(title(c))
        if j == 2:
            ax.set_xlabel("grid spacing / kerrbb spacing")
        if k == 0:
            ax.set_ylabel(lab)
        ax.invert_xaxis()
axes[0, 0].legend(fontsize=7)
axes[1, 0].legend(fontsize=6.5)
fig.suptitle("Geometry stored on an (a, i) grid and interpolated to mid-cell\n"
             "(kerrbb spacing: 0.1 in a below 0.6, factor 0.751 in 1 - a above; 5° in i)")
fig.tight_layout()
fig.savefig(outdir / "ctf_tabulated.png", dpi=150)

# 3. Residuals against energy.
fig, axes = plt.subplots(2, len(cases), figsize=(4.3 * len(cases), 6.2), sharex=True,
                         gridspec_kw=dict(height_ratios=[1.2, 1]))
for k, c in enumerate(cases):
    ref = np.array(c["reference"])
    p = E**2 * ref
    axes[0, k].loglog(E, p, "k")
    axes[0, k].set_ylim(p.max() * 1e-6, p.max() * 3)
    axes[0, k].set_title(title(c))
    ax = axes[1, k]
    pick = [
        (next(m for m in c["exact"] if m["label"] == "64x48"), C_B1, ":", "B1 64x48"),
        (next(m for m in c["exact"] if m["label"] == "128x96"), C_B1, "-", "B1 128x96"),
        (next(m for m in c["exact"] if m["label"].startswith("64r")), C_CTF, "-", "CTF 64 radii, stored"),
        (next(m for m in c["tabulated"] if m["method"] == "CTF" and m["order"] == 3 and m["factor"] == 1), C_CTF, "--",
         "CTF 64 radii, kerrbb-spaced grid, cubic"),
        (next(m for m in c["tabulated"] if m["method"] == "B1" and m["order"] == 3 and m["factor"] == 1), C_B1, "--",
         "B1 128x96, kerrbb-spaced grid, cubic"),
    ]
    for m, col, ls, lab in pick:
        e = np.abs(np.array(m["density"]) / ref - 1)
        e[p < 1e-6 * p.max()] = np.nan
        ax.loglog(E, np.maximum(e, 1e-13), ls, color=col, label=lab)
    ax.set_ylim(1e-11, 1e-1)
    ax.set_xlabel("E [keV]")
axes[0, 0].set_ylabel(r"$E^2 N(E)$")
axes[1, 0].set_ylabel("|relative error|")
axes[1, 0].legend(fontsize=6.5, loc="upper left")
fig.tight_layout()
fig.savefig(outdir / "ctf_spectra.png", dpi=150)

# 4. Maximum blueshift: B1 sample redshifts against the CTF g_max(r).
fig, axes = plt.subplots(1, len(cases), figsize=(4.3 * len(cases), 3.8))
for k, c in enumerate(cases):
    gm = c["gmax"]
    ax = axes[k]
    r, g = np.array(gm["b1_r"]), np.array(gm["b1_g"])
    ax.scatter(r, g, s=1, color=C_B1, alpha=0.35, lw=0, label="B1 128x96 samples")
    ax.plot(gm["radii"], gm["g_max_ctf_corrected"], C_CTF, lw=1.2, label="CTF $g_{max}(r)$")
    bins = np.geomspace(r.min(), 100, 25)
    idx = np.digitize(r, bins)
    top = np.array([(r[idx == j][np.argmax(g[idx == j])], g[idx == j].max()) for j in range(1, len(bins)) if np.any(idx == j)])
    ax.plot(*top.T, "k.", ms=4, label="highest B1 g per bin")
    ax.set_xscale("log")
    ax.set_xlabel("r [GM/c²]")
    ax.set_title(title(c))
    inset = ax.inset_axes([0.5, 0.08, 0.46, 0.34])
    gref = np.interp(np.log(top[:, 0]), np.log(gm["radii"]), gm["g_max_ctf_corrected"])
    inset.semilogx(top[:, 0], top[:, 1] / gref - 1, "k.", ms=3)
    inset.set_title(r"$g^{B1}_{top}/g_{max} - 1$", fontsize=6)
    inset.tick_params(labelsize=5)
axes[0].set_ylabel("g")
axes[0].legend(fontsize=7, loc="upper right", markerscale=4)
fig.tight_layout()
fig.savefig(outdir / "ctf_gmax.png", dpi=150)
print("wrote", *(outdir / n for n in ("ctf_cost.png", "ctf_tabulated.png", "ctf_spectra.png", "ctf_gmax.png")))
