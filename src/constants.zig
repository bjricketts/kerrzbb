//! Physical constants (cgs) and the spectral normalisations of Li et al.
//! (2005), eqs. (16) and (17).

const std = @import("std");
const kerrz = @import("kerrz");

pub const G: f64 = 6.67430e-8; // cm^3 g^-1 s^-2
pub const c: f64 = 2.99792458e10; // cm s^-1
pub const h: f64 = 6.62607015e-27; // erg s
pub const k_B: f64 = 1.380649e-16; // erg K^-1
pub const sigma_SB: f64 = 5.670374419e-5; // erg cm^-2 s^-1 K^-4
pub const M_sun: f64 = 1.98847e33; // g
pub const kpc: f64 = 3.0856775814913673e21; // cm
pub const keV: f64 = 1.602176634e-9; // erg

/// Gravitational radius GM/c^2 of one solar mass in cm.
pub const r_g_sun: f64 = G * M_sun / (c * c);

/// N0 of eq. (16) for fcol = 1, M = 1 M_sun, D = 1 kpc, in
/// photons keV^-1 cm^-2 s^-1. Planck photon intensity 2E^2/(h^3 c^2) with E in
/// keV, times the solid angle unit (r_g/D)^2.
pub const N0_unit: f64 = 2.0 * keV * keV * keV / (h * h * h * c * c) *
    (r_g_sun / kpc) * (r_g_sun / kpc);

/// mu of eq. (17) for fcol = 1, Mdot_eff = 1e18 g/s, M = 1 M_sun, in keV^-1.
/// mu = 1 keV / (k T*), with sigma T*^4 = 3 Mdot c^2 / (8 pi r_g^2).
pub const mu_unit: f64 = keV / (k_B * std.math.pow(
    f64,
    3.0 * 1e18 * c * c / (8.0 * std.math.pi * sigma_SB * r_g_sun * r_g_sun),
    0.25,
));

/// Eq. (16). `mass` in M_sun, `distance` in kpc.
pub fn normalisation(comptime T: type, fcol: T, mass: T, distance: T) T {
    const A = T.Algebra;
    return A.mult(
        .promote(N0_unit),
        A.div(A.powi(A.div(mass, distance), 2), A.powi(fcol, 4)),
    );
}

/// Eq. (17). `mdot_eff` in units of 1e18 g/s, `mass` in M_sun.
pub fn temperatureScale(comptime T: type, fcol: T, mdot_eff: T, mass: T) T {
    const A = T.Algebra;
    const mdot_quarter = A.sqrt(A.sqrt(mdot_eff));
    return A.div(
        A.mult(.promote(mu_unit), A.sqrt(mass)),
        A.mult(fcol, mdot_quarter),
    );
}

test "normalisations match Li et al. (2005)" {
    try std.testing.expectApproxEqRel(0.07205, N0_unit, 1e-3);
    try std.testing.expectApproxEqRel(0.1202, mu_unit, 1e-3);

    const D = kerrz.DualNumber(f64, 1);
    var m: D = .promote(10.0);
    m.dx[0] = 1;
    const mu = temperatureScale(D, .promote(1.7), .promote(2.0), m);
    const expected = mu_unit * @sqrt(10.0) / (1.7 * std.math.pow(f64, 2.0, 0.25));
    try std.testing.expectApproxEqRel(expected, mu.x, 1e-12);
    // d mu / d M = mu / (2 M)
    try std.testing.expectApproxEqRel(expected / 20.0, mu.dx[0], 1e-12);
}
