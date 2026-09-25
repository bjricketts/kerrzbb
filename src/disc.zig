//! Thin-disc (Novikov-Thorne / Page-Thorne) emission with a torque at the
//! inner edge, following Li et al. (2005), Appendix D.
//!
//! Units: G = c = M = 1, so radii are in r_g = GM/c^2. The spin `a` is signed;
//! a < 0 describes a retrograde disc. All functions are generic over a kerrz
//! dual-number type `T`.

const std = @import("std");
const kerrz = @import("kerrz");

/// Circular equatorial (Keplerian) orbit quantities at radius `r`
/// (Bardeen, Press & Teukolsky 1972; Page & Thorne 1974).
pub fn Orbit(comptime T: type) type {
    const A = T.Algebra;
    return struct {
        const Self = @This();

        r: T,
        a: T,
        /// Angular velocity, Omega.
        omega: T,
        /// Specific energy at infinity, E-dagger.
        energy: T,
        /// Specific angular momentum, L-dagger.
        ang_mom: T,

        pub fn init(r: T, a: T) Self {
            const sr = A.sqrt(r);
            const r32 = A.mult(r, sr);
            const r34 = A.sqrt(r32);
            const denom = A.mult(
                r34,
                A.sqrt(A.add(
                    A.sub(r32, A.mult(.promote(3), sr)),
                    A.mult(.promote(2), a),
                )),
            );
            const energy = A.div(
                A.add(A.sub(r32, A.mult(.promote(2), sr)), a),
                denom,
            );
            const ang_mom = A.div(
                A.add(
                    A.sub(A.powi(r, 2), A.mult(A.mult(.promote(2), a), sr)),
                    A.powi(a, 2),
                ),
                denom,
            );
            return .{
                .r = r,
                .a = a,
                .omega = A.div(.one, A.add(r32, a)),
                .energy = energy,
                .ang_mom = ang_mom,
            };
        }

        /// dOmega/dr.
        pub fn dOmegaDr(self: Self) T {
            const sr = A.sqrt(self.r);
            const denom = A.add(A.mult(self.r, sr), self.a);
            return A.div(A.mult(.promote(-1.5), sr), A.powi(denom, 2));
        }

        /// E - Omega L (= chi / Gamma), the factor relating the photon energy
        /// at infinity to the energy in the disc frame (eq. C21).
        pub fn energyFactor(self: Self) T {
            return A.sub(self.energy, A.mult(self.omega, self.ang_mom));
        }
    };
}

/// Radius of the marginally stable orbit.
///
/// The value comes from kerrz. Its derivatives are recovered with one Newton
/// step on r^2 - 6r + 8a sqrt(r) - 3a^2 = 0 (Bardeen et al. 1972), which is
/// smooth through a = 0. Differentiating kerrz's closed form directly gives
/// d r_ms / da = 0 at a = 0 because of the sign(a) sqrt(...) branch.
pub fn iscoRadius(comptime T: type, a: T) T {
    const A = T.Algebra;
    const r0 = kerrz.KerrMetric(D0).init(.one, .promote(a.x)).isco.x;
    const r: T = .promote(r0);
    const sr = A.sqrt(r);
    const F = A.sub(
        A.add(A.sub(A.powi(r, 2), A.mult(.promote(6), r)), A.mult(A.mult(.promote(8), a), sr)),
        A.mult(.promote(3), A.powi(a, 2)),
    );
    const dF_dr = 2 * r0 - 6 + 4 * a.x / @sqrt(r0);
    return A.sub(r, A.div(F, .promote(dF_dr)));
}

/// The Page & Thorne (1974) function f(r) (their eq. 15n), defined in Li et
/// al. (2005) eq. (D13), for a disc with inner edge `r_in`. The emitted flux
/// of a zero-torque disc is F = Mdot f / (4 pi r).
///
/// Valid for |a| < 1 and r >= r_in >= r_ms.
pub fn pageThorneF(comptime T: type, r: T, a: T, r_in: T) T {
    const A = T.Algebra;
    const x = A.sqrt(r);
    const x0 = A.sqrt(r_in);

    // Roots of x^3 - 3x + 2a = 0.
    const psi = A.div(A.acos(a), .promote(3));
    const third = std.math.pi / 3.0;
    const roots = [3]T{
        A.mult(.promote(2), A.cos(A.sub(psi, .promote(third)))),
        A.mult(.promote(2), A.cos(A.add(psi, .promote(third)))),
        A.mult(.promote(-2), A.cos(psi)),
    };

    var bracket = A.sub(
        A.sub(x, x0),
        A.mult(A.mult(.promote(1.5), a), A.log(A.div(x, x0))),
    );

    inline for (0..3) |i| {
        const xi = roots[i];
        const xj = roots[(i + 1) % 3];
        const xk = roots[(i + 2) % 3];
        // (x_i - a)^2 / x_i rewritten with x_i^3 - 3 x_i + 2a = 0 as
        // x_i (x_i^2 - 1)^2 / 4, which is regular at a = 0 (x_i = 0).
        const numer = A.mult(
            .promote(0.75),
            A.mult(xi, A.powi(A.sub(A.powi(xi, 2), .one), 2)),
        );
        const denom = A.mult(A.sub(xi, xj), A.sub(xi, xk));
        const log_term = A.log(A.div(A.sub(x, xi), A.sub(x0, xi)));
        bracket = A.sub(bracket, A.mult(A.div(numer, denom), log_term));
    }

    const x2 = A.powi(x, 2);
    const cubic = A.add(
        A.sub(A.powi(x, 3), A.mult(.promote(3), x)),
        A.mult(.promote(2), a),
    );
    return A.div(A.mult(.promote(1.5), bracket), A.mult(x2, cubic));
}

/// Dimensionless emitted flux ftilde_0(r) = 8 pi F0 / (3 Mdot_eff) of eq. (13),
/// without returning radiation (eq. D11). `eta` is the torque parameter of
/// eq. (2) and Mdot_eff = (1 + eta) Mdot. Large-r limits: r^-3 for eta = 0 and
/// r^-7/2 for eta -> infinity.
pub fn fluxNoReturn(comptime T: type, r: T, a: T, r_in: T, eta: T) T {
    const A = T.Algebra;
    const orb = Orbit(T).init(r, a);
    const f = pageThorneF(T, r, a, r_in);
    const accretion = A.div(f, r);

    const orb_in = Orbit(T).init(r_in, a);
    const eps_in = A.sub(.one, orb_in.energy);
    const torque = A.div(
        A.mult(
            A.div(A.mult(eps_in, orb_in.energyFactor()), orb_in.omega),
            orb.dOmegaDr().neg(),
        ),
        A.mult(r, A.powi(orb.energyFactor(), 2)),
    );

    return A.div(
        A.mult(.promote(2.0 / 3.0), A.add(accretion, A.mult(eta, torque))),
        A.add(.one, eta),
    );
}

// ---------------------------------------------------------------------------
// Tests

const quadrature = @import("quadrature.zig");
const D0 = kerrz.DualNumber(f64, 0);
const D1 = kerrz.DualNumber(f64, 1);

fn gl(comptime n: usize) struct { x: [n]f64, w: [n]f64 } {
    var x: [n]f64 = undefined;
    var w: [n]f64 = undefined;
    quadrature.gaussLegendre(&x, &w);
    return .{ .x = x, .w = w };
}

/// f(r) by direct quadrature of eq. (D13), with dL/dr from a dual number.
fn pageThorneNumerical(r: f64, a: f64, r_in: f64) f64 {
    const rule = gl(64);
    // Integrate in x = sqrt(r) to soften the behaviour near r_in.
    const x0 = @sqrt(r_in);
    const x1 = @sqrt(r);
    var sum: f64 = 0;
    for (rule.x, rule.w) |t, w| {
        const x = 0.5 * (x1 - x0) * t + 0.5 * (x1 + x0);
        var rr: D1 = .promote(x * x);
        rr.dx[0] = 1;
        const orb = Orbit(D1).init(rr, .promote(a));
        const integrand = orb.energyFactor().x * orb.ang_mom.dx[0] * 2 * x;
        sum += w * integrand;
    }
    sum *= 0.5 * (x1 - x0);
    const orb = Orbit(D0).init(.promote(r), .promote(a));
    return -orb.dOmegaDr().x / std.math.pow(f64, orb.energyFactor().x, 2) * sum;
}

test "Page-Thorne closed form matches eq. (D13) quadrature" {
    const spins = [_]f64{ -0.9, 0.0, 0.5, 0.998 };
    for (spins) |a| {
        const r_ms = iscoRadius(D0, .promote(a)).x;
        for ([_]f64{ 1.01, 1.5, 3.0, 20.0, 500.0 }) |fac| {
            for ([_]f64{ 1.0, 1.7 }) |rin_fac| {
                const r_in = rin_fac * r_ms;
                const r = fac * r_in;
                const closed = pageThorneF(D0, .promote(r), .promote(a), .promote(r_in)).x;
                const numer = pageThorneNumerical(r, a, r_in);
                try std.testing.expectApproxEqRel(numer, closed, 1e-8);
            }
        }
    }
}

/// Integral of E-dagger ftilde_0 r dr from r_in to (effectively) infinity.
fn emittedPower(a: f64, r_in: f64, eta: f64) f64 {
    const rule = gl(32);
    const t0 = @log(r_in);
    const t1 = @log(1e10);
    const panels = 60;
    const width = (t1 - t0) / panels;
    var sum: f64 = 0;
    for (0..panels) |p| {
        const lo = t0 + width * @as(f64, @floatFromInt(p));
        for (rule.x, rule.w) |t, w| {
            const r = @exp(lo + 0.5 * width * (t + 1));
            const orb = Orbit(D0).init(.promote(r), .promote(a));
            const ft = fluxNoReturn(D0, .promote(r), .promote(a), .promote(r_in), .promote(eta)).x;
            sum += 0.5 * width * w * orb.energy.x * ft * r * r;
        }
    }
    return sum;
}

test "emitted power equals eps_in * Mdot_eff (eq. 3)" {
    for ([_]f64{ -0.7, 0.0, 0.9, 0.998 }) |a| {
        const r_ms = iscoRadius(D0, .promote(a)).x;
        for ([_]f64{ 1.0, 2.0 }) |rin_fac| {
            const r_in = rin_fac * r_ms;
            const eps_in = 1 - Orbit(D0).init(.promote(r_in), .promote(a)).energy.x;
            for ([_]f64{ 0.0, 1.0, 10.0 }) |eta| {
                // 4 pi int E F0 r dr = eps_in Mdot_eff  <=>  int E ft r dr = 2 eps_in / 3
                const p = emittedPower(a, r_in, eta);
                try std.testing.expectApproxEqRel(2.0 * eps_in / 3.0, p, 1e-6);
            }
        }
    }
}

test "large-radius asymptotes" {
    const a: D0 = .promote(0.5);
    const r_in = iscoRadius(D0, a);
    const r: D0 = .promote(1e8);
    const f0 = fluxNoReturn(D0, r, a, r_in, .promote(0.0)).x;
    try std.testing.expectApproxEqRel(1e-24, f0, 1e-3);
    // Pure torque: ftilde ~ r^-7/2
    const f_hi = fluxNoReturn(D0, r, a, r_in, .promote(1e12)).x;
    const f_hi2 = fluxNoReturn(D0, .promote(4e8), a, r_in, .promote(1e12)).x;
    try std.testing.expectApproxEqRel(std.math.pow(f64, 4, -3.5), f_hi2 / f_hi, 1e-4);
}

test "spin derivative matches finite differences" {
    const D2 = kerrz.DualNumber(f64, 2);
    for ([_]f64{ -0.5, 0.0, 0.7, 0.99 }) |a_val| {
        var a: D2 = .promote(a_val);
        a.dx[0] = 1;
        var eta: D2 = .promote(0.3);
        eta.dx[1] = 1;
        const r_in = iscoRadius(D2, a);
        const r: D2 = .promote(1.3 * r_in.x);
        const val = fluxNoReturn(D2, r, a, r_in, eta);

        const h = 1e-6;
        const fd = struct {
            fn eval(av: f64, ev: f64, rv: f64) f64 {
                const aa: D0 = .promote(av);
                return fluxNoReturn(D0, .promote(rv), aa, iscoRadius(D0, aa), .promote(ev)).x;
            }
        };
        const da = (fd.eval(a_val + h, 0.3, r.x) - fd.eval(a_val - h, 0.3, r.x)) / (2 * h);
        const de = (fd.eval(a_val, 0.3 + h, r.x) - fd.eval(a_val, 0.3 - h, r.x)) / (2 * h);
        try std.testing.expectApproxEqRel(da, val.dx[0], 1e-5);
        try std.testing.expectApproxEqRel(de, val.dx[1], 1e-5);
    }
}

test "ISCO derivative is smooth through a = 0" {
    for ([_]f64{ -0.9, -0.3, 0.0, 0.3, 0.9, 0.998 }) |av| {
        var a: D1 = .promote(av);
        a.dx[0] = 1;
        const r = iscoRadius(D1, a);
        const h = 1e-6;
        const fd = (iscoRadius(D0, .promote(av + h)).x - iscoRadius(D0, .promote(av - h)).x) / (2 * h);
        try std.testing.expectApproxEqRel(kerrz.KerrMetric(D0).init(.one, .promote(av)).isco.x, r.x, 1e-12);
        try std.testing.expectApproxEqRel(fd, r.dx[0], 1e-6);
    }
}
