# kerrzbb implementation plan

Goal: reimplement KERRBB (Li, Zimmerman, Narayan & McClintock 2005, ApJS 157, 335; `papers/0411583v2.pdf`) in Zig, ray tracing on the fly with kerrz instead of reading a precomputed table, with forward-mode derivatives of the spectrum with respect to the model parameters.

Equation numbers below refer to the paper.

## 0. Decisions (2026-09-25)
- **Observer-plane integration:** implement both methods. B1 (image-plane contours, one per disc radius) is the production path; B2 (image-plane pixel grid) is an independent cross-check.
- **Returning radiation:** deferred to milestone M5. rflag = 0 must be correct and differentiable first.
- **Interfaces:** an XSPEC local model, a Python wrapper and a Julia (SpectralFitting.jl) binding, all through a single C ABI.
- **Scope beyond kerrbb:**
  - free `r_in ≥ r_ms`;
  - any `η ≥ 0`;
  - optional higher-order images: the n = 1, 2 equatorial crossings. On the B2 path these come from `TracingConfig.winding`. On the B1 path they need their own contour solve per winding.
- **Default behaviour** still reproduces kerrbb: `r_in = r_ms`, primary image only.
- **Spin near zero:** spins with |a| < a_min are rejected with an error, where a_min = 1e-3 (the same value as kerrz's C wrapper clamp). The accepted range is −0.9999 ≤ a ≤ −a_min or a_min ≤ a ≤ 0.9999. The Page–Thorne and ISCO code is exact at a = 0 already, so only the ray-tracing stage needs the check.
- **kerrz dependency:** kerrz is a separate dependency, never vendored.
- **Validation:** compare against XSPEC `kerrbb` on its table grid points only. Call it through ndspec with the local HEASoft install, or through xspectrampoline.

## 1. What KERRBB computes

The observed photon spectrum (eq. 15):

    N(E) = N0 (E/keV)^2 ∫ dΩ̃_obs Υ(cos θ_e) / (exp[µ E / (g f̃_out^{1/4})] − 1)

with

- `N0 ∝ fcol^-4 (M/M☉)^2 (D/kpc)^-2` (eq. 16) and `µ ∝ fcol^-1 (Ṁ_eff/1e18 g s^-1)^-1/4 (M/M☉)^1/2` (eq. 17). These carry all of the dependence on M, D, Ṁ_eff and fcol.
- `dΩ̃_obs = (D/r_g)^2 dΩ_obs`, the solid angle on the observer's sky in units of r_g² (eq. 14, C28).
- `g = E_obs/E_em`, the disc-to-observer redshift of each ray (eq. C21: `g = (E† − ΩL†)/(1 − Ωλ)`, with `E† − ΩL† = χ/Γ`).
- `f̃_out(r) = 8π r_g² F_out / (3 Ṁ_eff)`, the dimensionless emitted flux (eq. 13).
- `Υ = 1` for isotropic emission, `Υ = 1/2 + 3/4 cos θ_e` for limb darkening (eq. D20), where θ_e is the emission angle to the disc normal in the comoving frame (eq. C23).

The emitted flux has three contributions: `F_out = F0 + F_in + F_S` (eq. D17).

- `F0` is Novikov–Thorne/Page–Thorne with a torque `g_in` at `r_in` (eq. D11), where `g_in Ω_in = η ε_in Ṁ` and `Ṁ = Ṁ_eff/(1+η)` (eqs. 2, 4).
- `F_in` and `F_S` come from returning radiation (eqs. D12, D16, D19, D21). Both are *linear* functionals of `F_out`.

KERRBB's defaults are `r_in = r_ms` and `r_out = 1e6 M`. It uses the primary image only (each ray is traced to its first crossing of the equatorial plane). The XSPEC parameters are `eta, a, i, Mbh, Mdd (=Ṁ_eff), Dbh, hd (=fcol), rflag, lflag, norm`.

## 2. Structure

The calculation separates into three stages. Each stage depends on fewer parameters than the one after it, so each can be cached on its own:

| Stage | Depends on | Output |
|---|---|---|
| A. Disc emissivity `f̃_out(r)` | a, η, rflag, lflag | f̃ on a radial grid |
| B. Image-plane samples | a, i | a set of samples `{w_k, r_k, g_k, cos θ_k}` |
| C. Spectrum | all (M, Ṁ_eff, D, fcol only through N0 and µ) | photons per energy bin |

The returning-radiation kernel in stage A depends only on a and lflag. It does not depend on i, η, Ṁ, M, D or fcol.

## 3. Components

Every numerical function is generic over a comptime scalar type `T` (`f64` or kerrz's `DualNumber(f64, N)`), following kerrz's own convention (e.g. `geodesic.NullGeodesic(T)`).

### 3.1 `constants.zig`
Constants in cgs units, plus the derived N0 and µ. Compute these from first principles rather than hard-coding 0.07205 and 0.1202; add a unit test that reproduces the paper's values.

### 3.2 `disc/kinematics.zig`: Keplerian orbits in the equatorial plane
`Ω, E†, L†, Γ, χ, ω, A, Δ` and `dΩ/dr` as functions of `(r, a)`, plus `r_ms`.
- **Reuse from kerrz:** `geometry.KerrMetric` (horizon, ISCO), `orbits.Keplerian`, `orbits.circularFourVelocity` and `orbits.lorentzFactorKeplerian`.
- **Write ourselves:** the closed forms that kerrz does not expose (E†, L†, dΩ/dr).

### 3.3 `disc/novikov_thorne.zig`: `f̃_0(r; a, η, r_in)`
- The Page & Thorne (1974) eq. 15n closed form for `f(r)`, written in terms of `x = √(r/M)` and the roots x1, x2, x3. It is valid for any `x0 = √(r_in/M)`, so `r_in > r_ms` comes for free.
- The torque term of eq. D11, normalised to Ṁ_eff.
- Tests:
  - efficiency `1 − E†(r_ms)` at a = 0 and a = 0.998;
  - `4π∫E† F0 r dr = ε_in Ṁ_eff` (eq. 3), in the no-returning-radiation limit;
  - large-r asymptotes `∝ r^-3` for η = 0 and `∝ r^-7/2` for η → ∞.

### 3.4 `disc/returning.zig`: self-irradiation (rflag = 1)
- **Grid:** for each absorbing radius r_a on a radial grid (about 100 points, log-spaced), build a solid-angle grid over the upper hemisphere of the comoving frame. The grid is non-uniform and concentrated toward the black-hole direction (the paper notes this matters most at large r_a).
- **Tracing** (reverse direction `(π−θ, π+φ)`, eq. D16):
  - **Reuse from kerrz:** `krz_frame` with `circularOrbitVelocity` for the comoving frame; `fromSkyAngles` / `geodesic.skyAnglesToVelocityFrame` to launch rays; `traceToAngle(π/2)` for the first equatorial crossing.
  - Classify each ray as returning (r_e ≥ r_in), captured, or escaped.
  - Compute `ĝ` from eq. C22 and `Υ(θ_e)` at the emitting end.
- **Kernel:** the result is a linear operator `K` (N_r × N_r) with `F_in + F_S = K F_out`. The F_S part is the cumulative integral of eq. D12 over S from eq. D19.
- **Solve:** `F_out = (I − K)^-1 F0`, one dense linear solve instead of the paper's five fixed-point iterations. `K` is cached per (a, lflag).
- **Validation for free:** the capture and escape fractions ι_BH, ι_ret and ι_esc (Figs. 2–3). The paper gives 1.7% returning and 0.66% captured at a = 0, and 27% and 4% at a = 0.9999 (η = 0). Energy conservation (eq. 8) is a second check.

### 3.5 `geometry/image.zig`: observer-plane integration (stage B)
Output: weighted samples such that `∫dΩ̃_obs h(r, g, cosθ) ≈ Σ_k w_k h(r_k, g_k, cosθ_k)`. There are two candidate methods (see Q1).

- **(B1) Image-plane contours per disc radius (production path).**
  - Choose radii r_j on a Gauss–Legendre rule in a mapped variable (e.g. `u = (r_in/r)^p`) from r_in to r_break.
  - For each r_j and image angle ϑ′ on a Gauss–Legendre or trapezoid rule, solve for the image radius ρ with `rootsolve.univariateSolveDual`. Use a dual slot for ρ plus slots for (a, i, r), and use kerrz's `NullGeodesic.fromImpactParameters` + `traceToAngle(π/2)` as the target function (the same construction as kerrz's `TargetRadiusContext`, but keeping the dual part).
  - At the root, compute g (eq. C21 or `redshift.keplerianRedshift`) and cos θ_e (`TraceResult.localAngles`), both as duals.
  - kerrz's `tools.TransferFunction` gives the same contours as plain f64, at about 3–6 ms per radius, so it serves as a value cross-check.
  - **Measure:** `dα dβ = ρ |∂ρ/∂r| dr dϑ′`. The factor ∂ρ/∂r is the r-slot of the root solve. Differentiating that factor with respect to (a, i) needs second derivatives: nest the duals, or take ∂ρ/∂r from a second implicit solve.
  - **Preferred variant: a boundary-fitted image grid.** For each ϑ′, solve only for the ISCO contour radius ρ_in(ϑ′; a, i) and the r_break contour radius ρ_out(ϑ′; a, i) with duals. Then place a Gauss–Legendre rule in s ∈ [0, 1] with `ρ = ρ_in + (ρ_out − ρ_in) s` (clustered toward ρ_in), trace every node directly, and integrate `∫dϑ′ ∫ρ dρ`. This needs only first derivatives. The moving ISCO boundary still enters through ρ_in(a, i). It also replaces the per-node root solves with one direct trace each.
  - **Why this one:** the lower limit r_in = r_ms(a) is an explicit integration limit, so the derivative with respect to a automatically includes the moving-boundary term.
- **(B2) Direct image-plane grid, as in the paper.**
  - Use a `(log r′, ϕ′)` grid on the fictitious disc-plane projection (eqs. C26–C28 and footnote 7). Trace every pixel with `fromImpactParameters` + `traceToAngle(π/2)`, at about 6 µs per ray, so 10⁴–10⁵ rays per spectrum.
  - It is simpler, but the step at the ISCO and horizon edges makes the per-ray derivative miss the boundary term, which biases ∂N/∂a.
- **Outer disc (r > r_break ≈ 10³–10⁴ M).** The kerrz transfer-function tool stops converging somewhere between r = 10⁴ (works, r_err ≈ 1.6e-4) and 10⁵ (hangs). Beyond r_break, use the weak-field limit: `dΩ̃ = cos i r dr dφ`, with g from the Keplerian Doppler shift plus the first-order gravitational redshift. Test for continuity at r_break.
- **Scope:** primary image only, to match KERRBB. Higher-order images could be added as an option later.

### 3.6 `spectrum.zig` (stage C)
- Photon flux per energy bin (XSPEC convention), from an n-point Gauss–Legendre rule in energy within each bin.
- Planck term in an overflow-safe form: `1/expm1(x)`, with a cutoff for x > 700 that is also well defined for dual numbers.
- `norm` multiplies the result.

### 3.7 `kerrbb.zig`: top level
- A `Params(T)` struct and an `eval(T, params, energy_edges, out)` function.
- A cache keyed on (a, i) for stage B and on (a, lflag) for the returning kernel.
- Threading through the kerrz `ThreadPool`, over radii and over absorbing radii.

### 3.8 Autodiff strategy
- Stage B needs duals only in (a, i): `DualNumber(f64, 2)`. The returning kernel needs them only in a.
- M, Ṁ_eff, D and fcol enter only through N0 and µ. Their derivatives are analytic in stage C (`∂/∂µ` of the Planck term), so there is no need to push 7-component duals through the ray tracing. η enters f̃_0 in closed form.
- Output: `N_bin` and the Jacobian `∂N_bin/∂θ` for every free parameter.
- Test: compare against central finite differences on every parameter.

### 3.9 Interfaces (`c_api.zig` and bindings)
Candidates are listed in Q3: a C ABI for an XSPEC local model, a Python ctypes wrapper and a Julia SpectralFitting.jl binding.

### 3.10 Build
- `build.zig` and `build.zig.zon`, with kerrz as a package dependency pinned to a commit.
- Zig 0.15.2, the version kerrz requires. Zig is not currently installed on the Mac.

## 4. Validation targets
1. Disc-flux unit tests (§3.3).
2. a = 0, rflag = lflag = 0: compare against the paper's Fig. 14 (KERRBB vs corrected GRAD) at i = 20°, 60° and 85°.
3. Compare against XSPEC `kerrbb` at its table grid points in (a, i, η), for rflag and lflag each 0 or 1. Run it through ndspec (local HEASoft) or xspectrampoline.
4. Large-radius / low-energy limit: `N ∝ cos i` Newtonian scaling, and ezdiskbb-like behaviour at r ≫ r_in.
5. Returning radiation: ι fractions and spin equilibrium a_eq ≈ 0.9983 (isotropic) and 0.9986 (limb-darkened) (Fig. 4). Fig. 6 check: the rflag = 1 spectrum should match rflag = 0 with Ṁ scaled by about 1.23 (η = 0, a = 0.999).
6. Derivatives vs finite differences; smoothness of ∂N/∂a across the full spin range.

## 5. Milestones
1. **M0:** build skeleton, kerrz dependency, CI with `zig build test`.
2. **M1:** disc kinematics and Novikov–Thorne flux with η.
3. **M2:** image-plane integration (B1, with B2 as a cross-check), plus the outer weak-field region.
4. **M3:** spectrum and energy bins; compare against KERRBB with rflag = lflag = 0.
5. **M4:** derivatives and Jacobian output.
6. **M5:** returning radiation (kernel, linear solve, validation).
7. **M6:** limb darkening, interfaces, caching and threading, performance tuning.

## 6. Notes and issues found so far
- **Reading the kerrz source (commit 4b6d84e) confirmed:**
  - Every numerical routine is generic over `zad.DualNumber(f64, N)`, whose `Algebra` provides add, sub, mult, div, sqrt, cuberoot, pow, powi, exp, log, and the trig and hyperbolic functions.
  - `NullGeodesic(T).fromImpactParameters`, `fromSkyAngles` and `traceToAngle` propagate derivatives, and so do `orbits.circularFourVelocity`, `redshift.keplerianRedshift` and `TraceResult.localAngles`.
  - The `transfer-tables` `Trace` fields are plain `f64`, and `impactOffsetForRadius` discards the dual part. So B1 cannot take derivatives from `CunninghamTransferFunction`, and needs its own contour solver built on `rootsolve.univariateSolveDual`. Because the Newton update carries the extra dual slots, that solver returns the implicit derivatives ∂ρ/∂(a, i, r) when it converges.
  - Direct traces take about 2.6 µs each in a Debug build.
- **Spin near zero:**
  - The `|a| < 1e-3 → 1e-3` clamp lives only in the C wrapper (`wrappers/interface.zig`). The Zig core does not clamp.
  - At a = 0 the geodesic code panics in `Carlson.firstKind` (from `angularCache_normal`). At |a| = 1e-9 it returns wrong radii and NaN derivatives. At a = 1e-6 the values are correct but ∂r/∂a is ill-conditioned (−6.7e4 for one ray, against −7.5e-3 at a = 1e-3).
  - The angular antiderivatives need a small-a expansion. Decision: kerrzbb excludes |a| < 1e-3 (see §0).
- **ISCO derivative:** differentiating kerrz's closed-form `KerrMetric.isco` gives d r_ms/da = 0 at a = 0 (the sign(a)·√ branch). `disc.iscoRadius` takes kerrz's value and recovers the derivative with one Newton step on r² − 6r + 8a√r − 3a² = 0.
- **License:** kerrz stays a separate dependency (a `build.zig.zon` path during development, a pinned URL for releases) and is never vendored. Binaries that link kerrz fall under GPL-3.0; the kerrzbb source stays MIT.

- **M2 findings on kerrz** (to report upstream):
  - **Discontinuity at the case III → case II transition.** Near the critical curve, the equatorial crossing radius jumps where the radial roots change from case III to case II. Example: a = 0.998, i = 65.5°, image angle 3.24, ρ from 1.958 to 1.959 gives r from 1.2340 to 1.2398. Just before the jump, the case III derivative grows.
  - **Spurious escapes near edge-on at small spin.** For |a| ≲ 0.01 near edge-on, rays close to the α axis are wrongly reported as escaping (status `infinity`) at r ≳ 3e3–1e4.
  - **Noise in crossing radii.** Crossing radii carry about 1e-6 relative noise at r ~ 1e4, and about 1e-4 above i ≈ 88°.
  - **Precision loss above r ~ 1e5.** Crossings lose relative accuracy: 3e-4 at 1e6 and 5e-3 at 1e7.
  - **Assertion failure.** For extreme rays, `potentials.determineCase` can fail its assertion `r1.isReal()`. In release builds that assertion is undefined behaviour.

## 7. Development setup
- `build.zig.zon` points to `../kerrz` (a path dependency).
- `zig build test` runs the unit tests. Add `-Dtest-filter=<name>` to run a subset.
- `zig build validate` runs the physical validation (the escape fraction).
- Release builds of kerrz take several minutes to compile.
- The kerrz dependencies (zad, rootsolve, zfits, clippy) resolve from the Zig package cache.

## 8. Status
- M0 is done: build skeleton and a kerrz path dependency.
- M1 is done:
  - `constants.zig`: N0 and µ derived from cgs constants; they reproduce 0.07205 and 0.1202.
  - `quadrature.zig`: Gauss–Legendre rules.
  - `disc.zig`: Keplerian orbit quantities, the smooth ISCO derivative, the Page–Thorne `f` in a form that is regular at a = 0, and `f̃_0` with η.
- The M1 tests cover:
  - the closed form against eq. D13 by quadrature, for a in {−0.9, 0, 0.5, 0.998} and r_in in {1, 1.7}·r_ms;
  - energy conservation `∫E† f̃ r dr = 2ε_in/3` for η in {0, 1, 10};
  - the large-r asymptotes;
  - dual derivatives in (a, η) and in r_ms(a) against finite differences.
- M2 is done: `src/image.zig`.
  - **B1 (`traceImage`):** a boundary-fitted polar image grid. Contour radii come from a safeguarded Newton solve plus explicit implicit-function-theorem derivatives. The closed-form g and cos θ_e (eqs. C21, C23) agree with kerrz's `keplerianRedshift` and `localAngles`. Beyond r_break = 1e4 the grid switches to the weak-field projection.
  - **B2 (`tracePixels`):** a fixed pixel grid, used as the cross-check.
  - **Tests:**
    - contour roots and their derivatives against finite differences;
    - redshift and emission angle against kerrz;
    - projected area of a weakly lensed annulus against π cos i (r_out² − r_in²), agreeing to 1e-7;
    - B1 against B2 on the bolometric integral ∫ g⁴ f̃ dΩ̃: agreement to 7e-7 for (a, i) = (0.9, 30°), (0.9, 75°), (−0.5, 60°) and (0.998, 65.5°);
    - convergence of B1 under grid refinement: 1e-12, and 1e-6 at a = 0.998;
    - bolometric derivatives in (a, i) against finite differences, agreeing to 2e-6. The finite differences are limited by kerrz noise.
  - **Speed:** the default grid of 18,432 samples takes 44 ms in f64 and 99 ms with 2 derivative slots, in a Debug build.
  - **Parameter limits:** 1e-3 ≤ |a| ≤ 0.9999 and 0 < i ≤ 89°.
  - **Validation (`zig build validate`):** fraction of the F0 emission escaping in the primary image, integrated over the sphere. Results: 0.9748 (a = 0.1), 0.9621 (0.5), 0.9157 (0.9), 0.7882 (0.998). The paper gives 0.976 at a = 0, including returning radiation and all image orders.
- M3 is done: `src/spectrum.zig`.
  - `Spectrum(T).init(allocator, Params(T), image.Options)` prepares the emitters.
    - `.density(E)` returns the differential N(E) in photons keV⁻¹ cm⁻² s⁻¹.
    - `.binned(edges, out, n)` returns photons cm⁻² s⁻¹ per bin, computed with Gauss–Legendre nodes in energy.
    - `photonFlux(...)` wraps both.
  - The parameters follow kerrbb: eta, a, i, M, Ṁ_eff, D, fcol, norm, lflag, with an optional r_in.
  - Ray tracing runs with a three-slot dual in (a, i, r_in). The chain rule maps that onto the caller's slots, and the Planck-term derivatives are analytic.
  - Unit tests:
    - total photon flux against the analytic Σ W·2ζ(3)/x0³, agreeing to 1e-6;
    - the low-energy slope E^-2/3;
    - the sign of the limb-darkening effect;
    - all 8 parameter derivatives against finite differences, agreeing to 1e-5.
- **Comparison with XSPEC kerrbb (2026-09-25).** runkbb.f in HEASoft 6.36 evaluates N(E) = cc·FLUX0(a, i, η, E/ca), with ca = Mdd^¼ M^-½ fcol and cc = M²/D² fcol⁻⁴ ca². So the table in `kerrbb.fits` holds the differential spectrum for M = Ṁ = D = fcol = 1.
  - **Table layout:** 46 spins from −1 to 0.9999, 18 inclinations from 0° to 85° in steps of 5°, η in {0, 0.2, …, 1}, and 601 log-spaced energies from 1e-3 to 1e3 keV. The four columns are the (rflag, lflag) combinations.
  - **Method:** `validation/kerrbb_table_extract.py` reads the table nodes (it needs only numpy), and `zig build compare` evaluates `density` at the same nodes. There is no XSPEC interpolation anywhere in this comparison.
  - **Cases:** 216 table nodes. a ∈ {−0.9, −0.5, 0.1, 0.5, 0.6995, 0.9043, 0.9828, 0.9983, 0.9999}, i ∈ {5, 30, 60, 85}°, η ∈ {0, 0.6, 1}, lflag ∈ {0, 1}, rflag = 0.
  - **η = 0:** agreement is better than 0.13% wherever E·N exceeds 1e-3 of its peak and E ≥ 0.002 keV, for every spin, inclination and lflag. There is a constant offset of −0.04%, the same across energies and spins, most likely from different physical constants. At the lowest table energy (1e-3 keV, the outer disc for M = 1) the deviation is −0.6%.
  - **η > 0:** agreement is within 0.1% up to the peak. Above the peak the deviations reach ±2–5% at 10% of the peak and up to ±13% deep in the Wien tail.
    - The sign and size of the deviation vary irregularly with spin. They are zero for a ≥ 0.998, and present even at i = 5°.
    - Our spectra are converged to ≤3e-4 under 4× grid refinement.
    - Working interpretation: kerrbb's table does not resolve the step in emissivity at r_in that a finite torque produces. This is not proven.
  - Plots are in `validation/kerrbb_comparison_eta0.png` and `validation/kerrbb_comparison_eta1.png`.
  - **Binned check against XSPEC (PyXspec, `validation/kerrbb_reference.py`).** 60 cases: M = 10, Ṁ = 1, D = 10, fcol = 1.7, 200 log bins from 0.1 to 50 keV. The error splits into three parts:
    1. **XSPEC's own binning** (runkbb's trapezoid rule on the 0.01-dex table, compared against an accurate integral of the same table): about +1e-4 at the median, rising to +0.5% in the Wien tail (bins at 1e-4 of the peak). It is biased high because the tail is convex.
    2. **kerrzbb against the accurately integrated table at grid nodes:** η = 0 agrees to ≤ 0.08% everywhere, including the −0.035% constant offset. η = 1 shows the tail differences described above. Our binning is therefore consistent with the differential comparison.
    3. **Off-node spins** (0.9, 0.99, 0.998 in that run): XSPEC interpolates the spectra linearly between spin nodes. Doing the same linear interpolation between our own node spectra reproduces the XSPEC deviation: +7.5% against +7.9% for a = 0.9, i = 85°, η = 0; +2.0% against +2.5% at 30°; +1.3% against +1.7% for a = 0.998, i = 85°. The remainder is item 1.
  - The script's defaults now use the spin node values.
- M4 is done: `src/kerrbb.zig`, the top-level interface that the C ABI and bindings will wrap.
  - `evaluate(allocator, Params, FreeSet, edges, flux, ?jacobian, Options)` works in plain f64 parameters in XSPEC units (i in degrees, Ṁ in 1e18 g/s, D in kpc).
    - `FreeSet` is an `EnumSet(Parameter)` over {eta, a, incl, mass, mdot, distance, fcol, norm, r_in}.
    - The Jacobian is row-major, with shape [bin][free parameter in enum order]. The inclination derivative is per degree.
    - The call dispatches at compile time to `DualNumber(f64, N)`, where N is the number of free parameters (0 to 9).
  - Tests:
    - the value-only path is identical to `spectrum.photonFlux`;
    - the full 9-parameter Jacobian (with free r_in) matches finite differences to 2e-5;
    - Jacobian columns are identical (to 1e-12) whether one or all parameters are free;
    - input validation.
  - `zig build derivative-scan`: a spin scan at i = 60° over 105 spins from −0.9998 to 0.99987, comparing dual ∂N/∂a and ∂N/∂i against finite differences (Richardson-extrapolated in a, with a step that scales as (1−a)) in bins at 1, 5 and 15 keV.
    - Below a = 0.99 the agreement is ≤ 1.2e-4 relative.
    - For a ≥ 0.99 the median is 2e-4, with scattered outliers up to 0.85%. Their sign changes from point to point, which is consistent with the kerrz noise and the case II/III jump near the ISCO contour affecting the finite differences.
    - ∂N/∂i agrees to ≤ 1.3e-3, except where ∂N/∂i ≈ 0.
    - (∂N/∂a)/N is smooth across the full range.
  - **Cost (Debug build, 200 bins):** value only 0.50 s; 2 free parameters 0.69 s; 5 free 0.78 s; all 9 free 0.84 s. The energy sum over emitters dominates, and speeding it up is part of M6.
- The interfaces are done (built before M5, at Ben's request).
  - **C ABI:** `src/c_api.zig` and `include/kerrzbb.h`. It provides `kzbb_evaluate` (params, a free-parameter bit mask, edges, flux, and an optional row-major Jacobian), `kzbb_default_options`, `kzbb_status_string` and `kzbb_version`. `zig build lib` builds the shared library and installs the header. `examples/example.c` has been built and run with zig cc.
  - **Python:** `python/kerrzbb` (ctypes and numpy; there is a `pyproject.toml`). `python/tests/test_bindings.py` passes against a Debug build of the library: the 9-parameter Jacobian matches finite differences, column ordering is checked, and errors are mapped to `KerrzBBError`.
  - **Julia:** `julia/KerrzBB` provides a `ccall` wrapper, `kerrzbb_flux`, and `KerrzBBModel`, a SpectralFitting.jl model.
    - The model's `config` field is a closure field, which is non-fitted.
    - `K` is SpectralFitting's normalisation.
    - The ForwardDiff dual output is assembled from kerrzbb's Jacobian for the parameters whose partials are non-zero.
    - This was written against SpectralFitting 0.7.3 (checked from source), but **it has not been run**: Julia is not reachable from the sandbox. Tests are in `julia/KerrzBB/test/runtests.jl`.
  - **XSPEC:** `xspec/lmodel_kerrzbb.dat` and `xspec/kerrzbb_xspec.cxx`.
    - The wrapper loads `KERRZBB_LIBRARY` with dlopen, so `initpackage` and `hmake` need no extra link flags.
    - It has been checked with a small driver against the library. It **has not been built inside XSPEC**.
  - The README now documents building, the parameters, the interfaces and the validation commands.
  - Recommend building the library with `ReleaseSafe`, because kerrz assertions are undefined behaviour in `ReleaseFast`.
- M5 is done: returning radiation (self-irradiation, rflag), in `src/returning.zig`.
  - **Method.** F_in + F_S = K F_out is discretised on 48 radial nodes (r_in to r_max = 1e4, clustered toward r_in). F_out is interpolated as r⁻³ × (piecewise linear in log r). One LU solve gives (I − K_in − D K_S) u = f̃_0, with D the trapezoid form of eq. D12.
    - **Kernel rays.** For each absorbing radius and each azimuth χ around the aberrated black-hole direction, rays are traced backward using the arriving photon's constants (eqs. C14, C15, C17), with theta_sign = −1 and radial_sign = sign(n_r). The radius where they started comes from kerrz `traceToAngle(π/2)` with winding 0.
    - **Energy shift and emission angle:** ĝ = G(r_e)/G(r_a) (eq. C22) and the emission angle from eq. C23, which gives limb darkening when enabled.
    - **Inner edge.** The integrand jumps where backward rays start landing at r_in, because F_out(r_in) = F_in(r_in) > 0. So for each χ the psi grid starts exactly at that edge, psi_in(χ): a coarse scan followed by bisection, with the implicit-function-theorem derivative when the edge is the r_e = r_in contour. The grid is clustered toward psi_in.
    - **Beyond r_max:** F_in ∝ r⁻³, and F_S keeps the form of eq. D12 with the stress integral frozen.
    - **Integration with the spectrum.** The kernel is built with the (a, r_in) geometry slots, lifted to the caller's dual type, and solved with η. It replaces `fluxNoReturn` in the spectrum.
  - **Tests:**
    - eq. D12 carries no net energy for an arbitrary S (1e-4);
    - net power equals ε_in Ṁ_eff with self-irradiation (4e-4);
    - ι_ret is 1.67% at a = 0.001 (paper 1.7%) and 25.6% at a = 0.9999 (paper 27%);
    - the Jacobian of 5 parameters (η, a, i, Ṁ, r_in) with rflag = lflag = 1 matches finite differences to 1e-4;
    - the spin derivative with r_in = r_ms(a) matches to 5e-3. The discretised kernel is not exactly smooth in a: finite differences with h and h/2 already differ by about 1e-3.
  - **Convergence (a = 0.998).**
    - F_in/F0 at r ≥ 3 r_in is stable to 0.3% between the default grid (48 × 48 × 24) and 4× finer grids.
    - It is noisier (about 3%) at 1.5 r_in, from structure near the photon orbit.
    - At a = 0.5 it is converged to 4 digits.
  - **Independent forward check (`zig build returning-fractions`):**
    - Method: rays traced forward from the emission point, with fates weighted by energy at infinity.
    - (ι_ret, ι_BH, ι_esc) results:
      - a = 0.001: (1.67%, 0.68%, 97.66%); the paper has (1.7, 0.66, 97.6).
      - a = 0.9999: (25.8%, 4.0%, 70.2%); the paper has (27, 4, 69).
    - The forward ι_ret agrees with the backward kernel to 1%.
  - **Paper Fig. 6** (a = 0.999, i = 30°, η = 0): the rflag = 1 spectrum equals the rflag = 0 spectrum with Ṁ × 1.23 to within 0.3–2%, as the paper states.
  - **Comparison with the kerrbb table, rflag = 1 (FLUX3/FLUX4), η = 0, lflag 0 and 1, i = 30/60/85°:**
    - a ≤ 0.90: max |dev| ≤ 0.5%.
    - a = 0.98: 0.8%.
    - a = 0.998: 1.5%.
    - a = 0.9999: 2–5%, rising to 8% in the Wien tail at 85°.
    - Above a ≈ 0.9 there is a systematic deficit at low energy (the outer disc), up to −1.6% at a = 0.9999, the same at every inclination. It corresponds to F_in about 5% lower than kerrbb's at large radii. Our forward and backward calculations agree with each other and give ι_ret about 5% (relative) below the paper's 27% at a = 0.9999. The difference therefore most likely lies in Li et al.'s returning-radiation amplitude at high spin, not in kerrzbb, but this is not proven.
    - With η = 1 the deviations are irregular in spin (±4% at low energy), like the rflag = 0, η > 0 case.
  - Timing (Debug build, 200 bins, a = 0.9, no caching yet):
    - value only: 0.95 s with rflag = 1, against 0.52 s without;
    - with (a, i) derivatives: 1.50 s;
    - with 8 free parameters: 1.64 s.
- M6 is done: caching, threading and performance.
  - **Caching.** `spectrum.Cache` holds the image samples, keyed on (a, i, r_in, image options), and the returning-radiation kernel, keyed on (a, r_in, lflag, kernel options). Each is stored in both a plain-f64 and a three-slot-dual variant, and keys ignore the thread count.
    - `kerrbb.Model` wraps the cache.
    - The C API has `kzbb_model_create`, `kzbb_model_evaluate` and `kzbb_model_destroy`.
    - The Python `KerrzBB` object now holds a model.
    - Julia has `KerrzBBCache`, and `KerrzBBModel` carries one.
    - The XSPEC wrapper keeps one model per process; `KERRZBB_THREADS` sets its thread count.
    - Test: cached results are identical to uncached ones, with the expected hit counts.
  - **Threading.** `src/parallel.zig` (plain threads, chunked ranges, first error propagated) is used over image angles, over kernel rows and over energy bins or grid nodes. `n_threads` is set in `spectrum.Options` (0 means one per CPU) and passed through the C API options. Results are bitwise identical to serial runs.
  - **Energy sums.**
    - When bins × nodes is more than twice the number of grid nodes, N(E) is evaluated on a uniform ln E grid (step 0.02). Bins are then integrated with Gauss–Legendre on four-point Lagrange interpolation of ln N and of d ln N/dθ. This matches the direct quadrature to 1e-7 for values and 1e-6 for derivatives with 3000 bins.
    - Emitter merging (`merge_tolerance`) is available but off by default. At a tolerance of 1e-3 it cuts the emitter count about 4× with a 1e-5 change in values, but derivatives change at first order (8e-4).
  - The C options gained the returning-kernel grids, `n_threads`, `energy_grid` and `grid_step`, and invalid options are rejected.
  - **Timing (ReleaseFast, VM with 4 cores, 1 thread; `validation/benchmark.py`):**
    - New (a, i): 22 ms for the value, 38 ms with d/d(a, i, Ṁ).
    - New a with rflag = 1: 79 ms and 125 ms.
    - Cached: 17 ms and 25 ms.
    - The cost is the same for 200 and 3000 bins.
    - Four threads give about 3×.
    - ReleaseSafe is 10–20% slower than ReleaseFast.
    - The library builds in about 12 s.
- **Julia bindings verified (2026-09-25).** The tests pass on Ben's Mac (Julia 1.13, aarch64): the ccall wrapper, the cache, and ForwardDiff through `KerrzBBModel` matching kerrzbb's Jacobian to 1e-12.
  - Installation needs the University of Bristol AstroRegistry, which holds SpectralFitting and MultiLinearInterpolations.
  - `Libdl` is taken from Base, not declared as a dependency, because the Julia 1.13 resolver rejected the stdlib entry.
