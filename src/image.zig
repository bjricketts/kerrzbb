//! Observer-plane integration (stage B of PLAN.md).
//!
//! Produces weighted samples {w_k, r_k, g_k, cos_k} such that, for any
//! function h of the emission radius, redshift and emission angle,
//!
//!     integral dOmega~_obs h(r, g, cos theta_e)  ~=  sum_k w_k h(r_k, g_k, cos_k),
//!
//! with dOmega~_obs = d alpha d beta in units of r_g^2 (Li et al. 2005, eq. 14).
//! Only the primary image is included, as in KERRBB.
//!
//! Two constructions are provided:
//!
//! - `traceImage` (B1, production): a boundary-fitted polar grid on the image
//!   plane. For each image angle the radii of the r_in and r_break contours
//!   are root-solved, with their parameter derivatives from the implicit
//!   function theorem, and a Gauss-Legendre rule in log(rho) is placed between
//!   them. Beyond r_break a weak-field approximation is used. All samples carry
//!   derivatives with respect to whatever slots `T` holds.
//! - `tracePixels` (B2, cross-check): a fixed polar pixel grid that keeps rays
//!   landing in [r_in, r_break], plus the same outer region. Values only.

const std = @import("std");
const kerrz = @import("kerrz");
const disc = @import("disc.zig");
const quadrature = @import("quadrature.zig");

const KerrMetric = kerrz.KerrMetric;
const FourVector = kerrz.FourVector;
const NullGeodesic = kerrz.NullGeodesic;
const D0 = kerrz.DualNumber(f64, 0);
const D1 = kerrz.DualNumber(f64, 1);

/// kerrz's geodesic solutions are unreliable for |a| below this (PLAN.md §0).
pub const min_abs_spin: f64 = 1e-3;
pub const max_abs_spin: f64 = 0.9999;
/// Nearly tangent rays lose precision in kerrz beyond this inclination.
pub const max_inclination: f64 = std.math.degreesToRadians(89.0);

pub const Error = error{
    SpinOutOfRange,
    InclinationOutOfRange,
    ContourNotBracketed,
    ContourNotConverged,
} || std.mem.Allocator.Error;

pub fn Sample(comptime T: type) type {
    return struct {
        /// Solid-angle weight d alpha d beta, in r_g^2.
        weight: T,
        /// Radius at which the ray crosses the equatorial plane.
        r: T,
        /// Redshift g = E_obs / E_em (eq. E3).
        g: T,
        /// Cosine of the emission angle to the disc normal in the comoving
        /// frame (eq. C23).
        cos_em: T,
    };
}

pub const Options = struct {
    /// Image-plane angles (trapezoid rule, spectrally accurate for periodic
    /// integrands).
    n_theta: usize = 128,
    /// Gauss-Legendre nodes in log(rho) between the r_in and r_break contours.
    n_rho: usize = 96,
    /// Gauss-Legendre nodes in log(r) for the weak-field region.
    n_outer: usize = 48,
    /// Transition to the weak-field approximation. kerrz's equatorial
    /// crossings lose relative accuracy above ~1e5 r_g.
    r_break: f64 = 1e4,
    /// Outer disc radius (KERRBB: 1e6).
    r_out: f64 = 1e6,
    /// Observer radius used to launch rays.
    observer_distance: f64 = 1e12,
};

/// Check the spin and inclination ranges supported by the ray tracing.
pub fn checkParameters(a: f64, incl: f64) Error!void {
    if (@abs(a) < min_abs_spin or @abs(a) > max_abs_spin) return Error.SpinOutOfRange;
    if (!(incl > 0 and incl <= max_inclination)) return Error.InclinationOutOfRange;
}

/// Everything needed to trace rays for one (a, incl).
fn Setup(comptime T: type) type {
    return struct {
        const Self = @This();
        const A = T.Algebra;

        a: T,
        incl: T,
        metric: KerrMetric(T),
        x_obs: FourVector(T),

        fn init(a: T, incl: T, distance: f64) Self {
            return .{
                .a = a,
                .incl = incl,
                .metric = KerrMetric(T).init(.one, a),
                .x_obs = .{ .t = .zero, .r = .promote(distance), .th = incl, .ph = .zero },
            };
        }

        const Hit = struct {
            r: T,
            geod: NullGeodesic(T),
            /// The ray fell into the horizon or escaped without crossing
            /// the equatorial plane.
            captured: bool,
            escaped: bool,
        };

        /// First equatorial crossing of the ray with impact parameters
        /// (alpha, beta).
        fn cross(self: Self, alpha: T, beta: T) Hit {
            const geod = NullGeodesic(T).fromImpactParameters(self.metric, self.x_obs, alpha, beta);
            const res = geod.traceToAngle(self.metric, .promote(std.math.pi / 2.0), .{});
            return .{
                .r = res.r,
                .geod = geod,
                .captured = res.status == .event_horizon or res.status == .infinity,
                .escaped = res.status == .infinity,
            };
        }

        /// Redshift and emission angle for a Keplerian emitter at `r`, from
        /// the constants of motion (eqs. C21, C23): g = (E - Omega L) /
        /// (1 - Omega lambda), cos theta_e = g sqrt(Q) / r.
        fn emission(self: Self, r: T, lambda: T, eta: T) struct { g: T, cos_em: T } {
            const orb = disc.Orbit(T).init(r, self.a);
            const g = A.div(orb.energyFactor(), A.sub(.one, A.mult(orb.omega, lambda)));
            const cos_em = A.div(A.mult(g, A.sqrt(eta)), r);
            return .{ .g = g, .cos_em = cos_em };
        }
    };
}

fn toImpact(comptime T: type, rho: T, theta: f64) [2]T {
    const A = T.Algebra;
    return .{ A.mult(rho, .promote(@cos(theta))), A.mult(rho, .promote(@sin(theta))) };
}

/// Radius (value only) on the image plane along angle `theta` whose primary
/// image lands on `r_target`. Safeguarded Newton iteration, derivative from a
/// one-slot dual.
fn contourValue(a: f64, incl: f64, distance: f64, theta: f64, r_target: f64) Error!f64 {
    const S = Setup(D1).init(.promote(a), .promote(incl), distance);

    const Eval = struct {
        fn f(s: Setup(D1), th: f64, rt: f64, rho: f64) struct { y: f64, dy: f64, valid: bool } {
            const ab = toImpact(D1, D1.promote(rho).diff(0), th);
            const hit = s.cross(ab[0], ab[1]);
            // Rays into the horizon lie inside the shadow: treat them as
            // landing at r = 0. Rays escaping to infinity lie outside the
            // disc image: treat them as landing beyond any target.
            if (hit.escaped) return .{ .y = rt, .dy = 0, .valid = false };
            if (hit.captured or !std.math.isFinite(hit.r.x)) return .{ .y = -rt, .dy = 0, .valid = false };
            return .{ .y = hit.r.x - rt, .dy = hit.r.dx[0], .valid = true };
        }
    };

    // Flat-space estimate of the contour: rho = r / sqrt(cos^2 + sin^2 / cos^2 i).
    const ci = @cos(incl);
    const flat = r_target / @sqrt(@cos(theta) * @cos(theta) + @sin(theta) * @sin(theta) / (ci * ci));

    var hi = 1.5 * flat + 10;
    var f_hi = Eval.f(S, theta, r_target, hi);
    var n: usize = 0;
    while (f_hi.y <= 0) : (n += 1) {
        if (n > 40 or hi > 1e3 * r_target) return Error.ContourNotBracketed;
        hi *= 2;
        f_hi = Eval.f(S, theta, r_target, hi);
    }
    var lo = @min(0.5 * flat, 0.5 * hi);
    var f_lo = Eval.f(S, theta, r_target, lo);
    n = 0;
    while (f_lo.y >= 0) : (n += 1) {
        if (n > 40) return Error.ContourNotBracketed;
        lo *= 0.5;
        f_lo = Eval.f(S, theta, r_target, lo);
    }

    var x = if (f_hi.dy > 0) hi - f_hi.y / f_hi.dy else 0.5 * (lo + hi);
    if (!(x > lo and x < hi)) x = 0.5 * (lo + hi);
    for (0..200) |_| {
        const fx = Eval.f(S, theta, r_target, x);
        if (fx.y == 0) return x;
        if (fx.y > 0) hi = x else lo = x;
        var next = if (fx.dy > 0) x - fx.y / fx.dy else 0.5 * (lo + hi);
        if (!(next > lo and next < hi)) next = 0.5 * (lo + hi);
        if (@abs(next - x) <= 1e-13 * x) {
            // Reject convergence onto a jump (e.g. a spurious escape reported
            // by kerrz for nearly edge-on rays at small spin).
            const check = Eval.f(S, theta, r_target, next);
            // Accept kerrz's noise in the crossing radius (~1e-6 relative at
            // r ~ 1e4 for rays near the alpha axis, ~1e-4 near edge-on) and
            // the small discontinuity kerrz shows where the radial roots
            // change from case III to case II (~2e-3 relative, seen on the
            // ISCO contour at a = 0.998). Reject anything larger.
            if (!check.valid or @abs(check.y) > 1e-2 * r_target) return Error.ContourNotConverged;
            return next;
        }
        x = next;
    }
    return Error.ContourNotConverged;
}

/// Contour radius with derivatives with respect to every slot of `T`, from
/// the implicit function theorem: d rho / dp = -(dF/dp) / (dF/d rho) with
/// F(rho, p) = r_cross(rho, p) - r_target(p).
fn contour(comptime T: type, s: Setup(T), theta: f64, r_target: T, distance: f64) Error!T {
    const rho0 = try contourValue(s.a.x, s.incl.x, distance, theta, r_target.x);
    if (T.N == 0) return .promote(rho0);

    const T1 = T.PushSlot();
    const s1 = Setup(T1).init(s.a.pushSlot(), s.incl.pushSlot(), distance);
    const ab = toImpact(T1, T1.promote(rho0).diff(0), theta);
    const hit = s1.cross(ab[0], ab[1]);
    const F = T1.Algebra.sub(hit.r, r_target.pushSlot());

    var out: T = .promote(rho0);
    inline for (0..T.N) |k| {
        out.dx[k] = -F.dx[k + 1] / F.dx[0];
    }
    return out;
}

fn trapezoidAngle(j: usize, n: usize) f64 {
    return 2.0 * std.math.pi * (@as(f64, @floatFromInt(j)) + 0.5) / @as(f64, @floatFromInt(n));
}

/// Weak-field samples for r_break <= r <= r_out. Rays are straight lines, so
/// the image of the disc point (r, phi) is alpha = r cos phi,
/// beta = r sin phi cos i (eq. C26), with lambda = -alpha sin i and
/// Q = (r^2 - a^2) cos^2 i (eq. C27). Relative corrections are O(r_g / r).
fn outerSamples(
    comptime T: type,
    allocator: std.mem.Allocator,
    s: Setup(T),
    out: []Sample(T),
    opts: Options,
) std.mem.Allocator.Error!void {
    const A = T.Algebra;
    const n_r = opts.n_outer;
    const x = try allocator.alloc(f64, n_r);
    defer allocator.free(x);
    const w = try allocator.alloc(f64, n_r);
    defer allocator.free(w);
    quadrature.gaussLegendre(x, w);

    const log_span = @log(opts.r_out / opts.r_break);
    const cos_i = A.cos(s.incl);
    const sin_i = A.sin(s.incl);
    const dphi = 2.0 * std.math.pi / @as(f64, @floatFromInt(opts.n_theta));

    var k: usize = 0;
    for (0..n_r) |ir| {
        const u = 0.5 * (x[ir] + 1);
        const r_val = opts.r_break * @exp(u * log_span);
        const r: T = .promote(r_val);
        const dr = r_val * log_span * 0.5 * w[ir];
        const eta = A.mult(A.sub(A.powi(r, 2), A.powi(s.a, 2)), A.powi(cos_i, 2));
        for (0..opts.n_theta) |j| {
            const phi = trapezoidAngle(j, opts.n_theta);
            const lambda = A.mult(A.mult(r, .promote(@cos(phi))), sin_i).neg();
            const em = s.emission(r, lambda, eta);
            out[k] = .{
                .weight = A.mult(cos_i, .promote(r_val * dr * dphi)),
                .r = r,
                .g = em.g,
                .cos_em = em.cos_em,
            };
            k += 1;
        }
    }
}

pub fn sampleCount(opts: Options) usize {
    return opts.n_theta * (opts.n_rho + opts.n_outer);
}

/// B1: boundary-fitted image-plane samples. Caller owns the returned slice.
///
/// `a`, `incl` (radians) and `r_in` may carry derivative slots; `r_in` is
/// usually `disc.iscoRadius(T, a)`.
pub fn traceImage(
    comptime T: type,
    allocator: std.mem.Allocator,
    a: T,
    incl: T,
    r_in: T,
    opts: Options,
) Error![]Sample(T) {
    const A = T.Algebra;
    try checkParameters(a.x, incl.x);

    const s = Setup(T).init(a, incl, opts.observer_distance);
    const out = try allocator.alloc(Sample(T), sampleCount(opts));
    errdefer allocator.free(out);

    const x = try allocator.alloc(f64, opts.n_rho);
    defer allocator.free(x);
    const w = try allocator.alloc(f64, opts.n_rho);
    defer allocator.free(w);
    quadrature.gaussLegendre(x, w);
    const dtheta = 2.0 * std.math.pi / @as(f64, @floatFromInt(opts.n_theta));

    var k: usize = 0;
    for (0..opts.n_theta) |j| {
        const theta = trapezoidAngle(j, opts.n_theta);
        const rho_in = try contour(T, s, theta, r_in, opts.observer_distance);
        const rho_out = try contour(T, s, theta, .promote(opts.r_break), opts.observer_distance);
        const log_in = A.log(rho_in);
        const log_span = A.sub(A.log(rho_out), log_in);

        for (0..opts.n_rho) |i| {
            const u = 0.5 * (x[i] + 1);
            const rho = A.exp(A.add(log_in, A.mult(.promote(u), log_span)));
            // d alpha d beta = rho d rho d theta, with d rho = rho log_span du.
            const weight = A.mult(
                A.mult(A.powi(rho, 2), log_span),
                .promote(0.5 * w[i] * dtheta),
            );
            const ab = toImpact(T, rho, theta);
            const hit = s.cross(ab[0], ab[1]);
            if (hit.captured) {
                out[k] = .{ .weight = .zero, .r = r_in, .g = .one, .cos_em = .one };
            } else {
                const em = s.emission(hit.r, hit.geod.lambda, hit.geod.eta);
                out[k] = .{ .weight = weight, .r = hit.r, .g = em.g, .cos_em = em.cos_em };
            }
            k += 1;
        }
    }

    try outerSamples(T, allocator, s, out[k..], opts);
    return out;
}

pub const PixelOptions = struct {
    n_theta: usize = 512,
    n_rho: usize = 1024,
    /// Innermost image radius; must lie inside the shadow.
    rho_min: f64 = 0.5,
    base: Options = .{},
};

/// B2: fixed polar pixel grid (midpoint rule in log(rho) and theta), keeping
/// rays whose first equatorial crossing lies in [r_in, r_break]. Values only.
/// Caller owns the returned slice; unused entries are dropped.
pub fn tracePixels(
    allocator: std.mem.Allocator,
    a: f64,
    incl: f64,
    r_in: f64,
    opts: PixelOptions,
) Error![]Sample(D0) {
    try checkParameters(a, incl);
    const base = opts.base;
    const s = Setup(D0).init(.promote(a), .promote(incl), base.observer_distance);

    // Outer edge: a little beyond the widest flat-space image of r_break.
    const rho_max = 1.2 * base.r_break;
    const log_span = @log(rho_max / opts.rho_min);
    const dlog = log_span / @as(f64, @floatFromInt(opts.n_rho));
    const dtheta = 2.0 * std.math.pi / @as(f64, @floatFromInt(opts.n_theta));

    var list: std.ArrayList(Sample(D0)) = .empty;
    errdefer list.deinit(allocator);

    for (0..opts.n_theta) |j| {
        const theta = trapezoidAngle(j, opts.n_theta);
        for (0..opts.n_rho) |i| {
            const rho = opts.rho_min * @exp(dlog * (@as(f64, @floatFromInt(i)) + 0.5));
            const ab = toImpact(D0, .promote(rho), theta);
            const hit = s.cross(ab[0], ab[1]);
            if (hit.captured or hit.r.x < r_in or hit.r.x >= base.r_break) continue;
            const em = s.emission(hit.r, hit.geod.lambda, hit.geod.eta);
            try list.append(allocator, .{
                .weight = .promote(rho * rho * dlog * dtheta),
                .r = hit.r,
                .g = em.g,
                .cos_em = em.cos_em,
            });
        }
    }

    const n_outer = base.n_theta * base.n_outer;
    const start = list.items.len;
    try list.resize(allocator, start + n_outer);
    try outerSamples(D0, allocator, s, list.items[start..], base);
    return list.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------------
// Tests

const testing = std.testing;

fn deg(x: f64) f64 {
    return std.math.degreesToRadians(x);
}

/// Observed bolometric flux proxy: sum w g^4 ftilde_0(r) (isotropic).
fn bolometric(comptime T: type, samples: []const Sample(T), a: T, r_in: T) T {
    const A = T.Algebra;
    var sum: T = .zero;
    for (samples) |smp| {
        if (smp.weight.x == 0) continue;
        const f = disc.fluxNoReturn(T, smp.r, a, r_in, .zero);
        sum = A.add(sum, A.mult(smp.weight, A.mult(A.powi(smp.g, 4), f)));
    }
    return sum;
}

test "contour lands on target radius, with implicit derivatives" {
    const D2 = kerrz.DualNumber(f64, 2);
    const a_val = 0.7;
    const i_val = deg(40);
    const dist = 1e12;
    for ([_]f64{ 0.3, 1.9, 4.0 }) |theta| {
        const a = D2.promote(a_val).diff(0);
        const incl = D2.promote(i_val).diff(1);
        const r_in = disc.iscoRadius(D2, a);
        const s = Setup(D2).init(a, incl, dist);
        const rho = try contour(D2, s, theta, r_in, dist);

        const s0 = Setup(D0).init(.promote(a_val), .promote(i_val), dist);
        const ab = toImpact(D0, .promote(rho.x), theta);
        try testing.expectApproxEqRel(r_in.x, s0.cross(ab[0], ab[1]).r.x, 1e-9);

        const h = 1e-6;
        const F = struct {
            fn solve(av: f64, iv: f64, th: f64) !f64 {
                const rin = disc.iscoRadius(D0, .promote(av)).x;
                return contourValue(av, iv, 1e12, th, rin);
            }
        };
        const d_a = (try F.solve(a_val + h, i_val, theta) - try F.solve(a_val - h, i_val, theta)) / (2 * h);
        const d_i = (try F.solve(a_val, i_val + h, theta) - try F.solve(a_val, i_val - h, theta)) / (2 * h);
        try testing.expectApproxEqRel(d_a, rho.dx[0], 1e-5);
        try testing.expectApproxEqRel(d_i, rho.dx[1], 1e-5);
    }
}

test "closed-form redshift agrees with kerrz" {
    const s = Setup(D0).init(.promote(0.9), .promote(deg(60)), 1e12);
    for ([_][2]f64{ .{ 4.0, 3.0 }, .{ -7.0, 1.0 }, .{ 2.0, -5.0 }, .{ 20.0, 8.0 } }) |ab| {
        const hit = s.cross(.promote(ab[0]), .promote(ab[1]));
        const res = hit.geod.traceToAngle(s.metric, .promote(std.math.pi / 2.0), .{});
        const em = s.emission(hit.r, hit.geod.lambda, hit.geod.eta);
        const g_kerrz = kerrz.redshift.keplerianRedshiftAsymptotic(D0, s.metric, hit.geod, res);
        const v_disc = kerrz.orbits.circularFourVelocity(D0, s.metric, hit.r);
        const angles = res.localAngles(s.metric, hit.geod, v_disc);
        try testing.expectApproxEqRel(g_kerrz.x, em.g.x, 1e-6);
        // kerrz reports the arriving (time-reversed) direction: theta_e = pi - theta.
        try testing.expectApproxEqRel(-@cos(angles.theta.x), em.cos_em.x, 1e-6);
    }
}

test "weak-lensing projected area" {
    const opts: Options = .{ .n_theta = 64, .n_rho = 32, .n_outer = 32, .r_break = 1e4, .r_out = 1e6 };
    const r_in = 1000.0;
    for ([_]f64{ 20, 75 }) |i_deg| {
        const samples = try traceImage(D0, testing.allocator, .promote(0.5), .promote(deg(i_deg)), .promote(r_in), opts);
        defer testing.allocator.free(samples);
        var area: f64 = 0;
        for (samples) |smp| area += smp.weight.x;
        const flat = std.math.pi * @cos(deg(i_deg)) * (opts.r_out * opts.r_out - r_in * r_in);
        try testing.expectApproxEqRel(flat, area, 1e-3);
    }
}

test "B1 agrees with B2 and converges" {
    for ([_][2]f64{ .{ 0.9, 30 }, .{ 0.9, 75 }, .{ -0.5, 60 }, .{ 0.998, 65.5 } }) |p| {
        const a: D0 = .promote(p[0]);
        const incl: D0 = .promote(deg(p[1]));
        const r_in = disc.iscoRadius(D0, a);

        const b1 = try traceImage(D0, testing.allocator, a, incl, r_in, .{});
        defer testing.allocator.free(b1);
        const b1_fine = try traceImage(D0, testing.allocator, a, incl, r_in, .{ .n_theta = 256, .n_rho = 192, .n_outer = 96 });
        defer testing.allocator.free(b1_fine);
        const b2 = try tracePixels(testing.allocator, a.x, incl.x, r_in.x, .{});
        defer testing.allocator.free(b2);

        const v1 = bolometric(D0, b1, a, r_in).x;
        const v1f = bolometric(D0, b1_fine, a, r_in).x;
        const v2 = bolometric(D0, b2, a, r_in).x;
        // 1e-12 typically; ~1e-6 at a = 0.998 where the ISCO contour crosses
        // kerrz's case III / case II discontinuity.
        try testing.expectApproxEqRel(v1f, v1, 3e-6);
        try testing.expectApproxEqRel(v2, v1, 1e-5);
    }
}

test "bolometric derivatives match finite differences" {
    const D2 = kerrz.DualNumber(f64, 2);
    const opts: Options = .{ .n_theta = 64, .n_rho = 48, .n_outer = 24 };
    for ([_][2]f64{ .{ 0.6, 45 }, .{ -0.3, 70 } }) |p| {
        const a = D2.promote(p[0]).diff(0);
        const incl = D2.promote(deg(p[1])).diff(1);
        const r_in = disc.iscoRadius(D2, a);
        const smp = try traceImage(D2, testing.allocator, a, incl, r_in, opts);
        defer testing.allocator.free(smp);
        const v = bolometric(D2, smp, a, r_in);

        const F = struct {
            fn eval(av: f64, iv: f64, o: Options) !f64 {
                const aa: D0 = .promote(av);
                const rin = disc.iscoRadius(D0, aa);
                const s = try traceImage(D0, testing.allocator, aa, .promote(iv), rin, o);
                defer testing.allocator.free(s);
                return bolometric(D0, s, aa, rin).x;
            }
        };
        // kerrz's traces carry ~1e-10 relative noise, so the finite
        // difference is noise-limited below h ~ 1e-4.
        const h = 3e-4;
        const d_a = (try F.eval(p[0] + h, deg(p[1]), opts) - try F.eval(p[0] - h, deg(p[1]), opts)) / (2 * h);
        const d_i = (try F.eval(p[0], deg(p[1]) + h, opts) - try F.eval(p[0], deg(p[1]) - h, opts)) / (2 * h);
        try testing.expectApproxEqRel(d_a, v.dx[0], 2e-6);
        try testing.expectApproxEqRel(d_i, v.dx[1], 2e-6);
    }
}
