//! Observed photon spectrum (stage C of PLAN.md), Li et al. (2005) eq. (15):
//!
//!     N(E) = norm N0 (E/keV)^2 sum_k w_k Upsilon_k / (exp[mu E / (g_k ftilde_k^{1/4})] - 1)
//!
//! integrated over energy bins with a Gauss-Legendre rule, in
//! photons cm^-2 s^-1 per bin (the XSPEC convention).
//!
//! Derivatives: the ray tracing only depends on (a, i, r_in), so it is run
//! with a three-slot dual and the result is mapped onto the caller's slots by
//! the chain rule. Mass, accretion rate, distance, fcol, eta and norm enter
//! analytically.

const std = @import("std");
const kerrz = @import("kerrz");
const constants = @import("constants.zig");
const disc = @import("disc.zig");
const image = @import("image.zig");
const quadrature = @import("quadrature.zig");

const D0 = kerrz.DualNumber(f64, 0);
const D3 = kerrz.DualNumber(f64, 3);

/// Model parameters, following XSPEC kerrbb where applicable.
pub fn Params(comptime T: type) type {
    return struct {
        /// Torque parameter eta >= 0 (eq. 2).
        eta: T,
        /// Signed dimensionless spin, 1e-3 <= |a| <= 0.9999.
        a: T,
        /// Inclination in radians.
        incl: T,
        /// Black hole mass in M_sun.
        mass: T,
        /// Effective accretion rate Mdot_eff = (1 + eta) Mdot in 1e18 g/s.
        mdot: T,
        /// Distance in kpc.
        distance: T,
        /// Spectral hardening factor f_col (kerrbb `hd`).
        fcol: T,
        /// Normalisation (kerrbb `norm`).
        norm: T,
        /// Inner radius in r_g. `null` means the marginally stable orbit.
        r_in: ?T = null,
        /// Limb-darkened emission (kerrbb `lflag`), eq. (D20).
        limb_darkening: bool = false,
        /// Self-irradiation (kerrbb `rflag`). Not implemented yet (M5).
        returning_radiation: bool = false,
    };
}

pub const Options = struct {
    image: image.Options = .{},
    /// Gauss-Legendre nodes per energy bin.
    n_energy: usize = 4,
};

pub const Error = image.Error || error{
    ReturningRadiationNotImplemented,
    InnerRadiusBelowIsco,
    InvalidParameter,
};

/// Per-sample quantities for the energy sums.
fn Emitter(comptime T: type) type {
    return struct {
        /// norm N0 w Upsilon.
        weight: T,
        /// mu / (g ftilde^{1/4}) in keV^-1 (value only).
        x0: f64,
        /// d x0 / d theta_j.
        dx0: [T.N]f64,

        fn lessThan(_: void, lhs: @This(), rhs: @This()) bool {
            return lhs.x0 < rhs.x0;
        }
    };
}

/// Map a quantity carrying (a, i, r_in) slots onto the caller's slots.
fn lift(comptime T: type, q: anytype, p: Params(T)) T {
    var out: T = .promote(q.x);
    if (T.N == 0) return out;
    inline for (0..T.N) |j| {
        var d = q.dx[0] * p.a.dx[j] + q.dx[1] * p.incl.dx[j];
        if (p.r_in) |r| d += q.dx[2] * r.dx[j];
        out.dx[j] = d;
    }
    return out;
}

/// Prepared emitters for one parameter set. Evaluate the differential
/// spectrum with `density` or bin-integrated fluxes with `binned`.
pub fn Spectrum(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        /// Sorted by x0 so the Wien tail can be cut off early.
        emitters: []Emitter(T),

        pub fn init(
            allocator: std.mem.Allocator,
            params: Params(T),
            opts: image.Options,
        ) Error!Self {
            const A = T.Algebra;
            if (params.returning_radiation) return Error.ReturningRadiationNotImplemented;
            if (params.eta.x < 0 or params.mass.x <= 0 or params.mdot.x <= 0 or
                params.distance.x <= 0 or params.fcol.x <= 0)
                return Error.InvalidParameter;

            // Geometry with slots (a, i, r_in), or plain values.
            const G = if (T.N == 0) D0 else D3;
            const a_g: G = if (T.N == 0) .promote(params.a.x) else G.promote(params.a.x).diff(0);
            const i_g: G = if (T.N == 0) .promote(params.incl.x) else G.promote(params.incl.x).diff(1);
            const r_ms_g = disc.iscoRadius(G, a_g);
            const r_in_g: G = if (params.r_in) |r| blk: {
                if (r.x < r_ms_g.x) return Error.InnerRadiusBelowIsco;
                break :blk if (T.N == 0) .promote(r.x) else G.promote(r.x).diff(2);
            } else r_ms_g;

            const samples = try image.traceImage(G, allocator, a_g, i_g, r_in_g, opts);
            defer allocator.free(samples);

            const r_in = params.r_in orelse lift(T, r_in_g, params);
            const n0 = A.mult(params.norm, constants.normalisation(T, params.fcol, params.mass, params.distance));
            const mu = constants.temperatureScale(T, params.fcol, params.mdot, params.mass);

            const emitters = try allocator.alloc(Emitter(T), samples.len);
            errdefer allocator.free(emitters);
            var n_emit: usize = 0;
            for (samples) |smp| {
                if (smp.weight.x == 0) continue;
                const r = lift(T, smp.r, params);
                const f = disc.fluxNoReturn(T, r, params.a, r_in, params.eta);
                if (!(f.x > 0)) continue;
                const g = lift(T, smp.g, params);
                const tau = A.mult(g, A.sqrt(A.sqrt(f)));
                const upsilon: T = if (params.limb_darkening)
                    A.add(.promote(0.5), A.mult(.promote(0.75), lift(T, smp.cos_em, params)))
                else
                    .one;
                const x0 = A.div(mu, tau);
                var e: Emitter(T) = .{
                    .weight = A.mult(n0, A.mult(lift(T, smp.weight, params), upsilon)),
                    .x0 = x0.x,
                    .dx0 = undefined,
                };
                inline for (0..T.N) |j| e.dx0[j] = x0.dx[j];
                emitters[n_emit] = e;
                n_emit += 1;
            }
            const em = try allocator.realloc(emitters, n_emit);
            std.mem.sort(Emitter(T), em, {}, Emitter(T).lessThan);
            return .{ .allocator = allocator, .emitters = em };
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.emitters);
        }

        /// Differential photon spectrum N(E) in photons keV^-1 cm^-2 s^-1,
        /// with E in keV.
        pub fn density(self: Self, E: f64) T {
            var out: T = .zero;
            for (self.emitters) |e| {
                const x = e.x0 * E;
                if (x > 700) break;
                const em1 = std.math.expm1(x);
                const b = 1 / em1;
                // d b / dx = -e^x / (e^x - 1)^2
                const db = -(em1 + 1) * b * b;
                out.x += e.weight.x * b;
                inline for (0..T.N) |j| {
                    out.dx[j] += e.weight.dx[j] * b + e.weight.x * db * E * e.dx0[j];
                }
            }
            out.x *= E * E;
            inline for (0..T.N) |j| out.dx[j] *= E * E;
            return out;
        }

        /// Photon flux per energy bin (photons cm^-2 s^-1), written to `out`
        /// (length edges.len - 1), with an n-point Gauss-Legendre rule per bin.
        pub fn binned(self: Self, edges: []const f64, out: []T, n_energy: usize) void {
            std.debug.assert(out.len + 1 == edges.len);
            var xg: [32]f64 = undefined;
            var wg: [32]f64 = undefined;
            std.debug.assert(n_energy <= 32);
            quadrature.gaussLegendre(xg[0..n_energy], wg[0..n_energy]);
            for (out, 0..) |*bin, ib| {
                const lo = edges[ib];
                const hi = edges[ib + 1];
                var acc: T = .zero;
                for (0..n_energy) |m| {
                    const E = 0.5 * (hi - lo) * xg[m] + 0.5 * (hi + lo);
                    const d = self.density(E);
                    const wE = 0.5 * (hi - lo) * wg[m];
                    acc.x += wE * d.x;
                    inline for (0..T.N) |j| acc.dx[j] += wE * d.dx[j];
                }
                bin.* = acc;
            }
        }
    };
}

/// Photon flux per energy bin, written to `out` (length edges.len - 1),
/// energies in keV.
pub fn photonFlux(
    comptime T: type,
    allocator: std.mem.Allocator,
    params: Params(T),
    edges: []const f64,
    out: []T,
    opts: Options,
) Error!void {
    const spec = try Spectrum(T).init(allocator, params, opts.image);
    defer spec.deinit();
    spec.binned(edges, out, opts.n_energy);
}

// ---------------------------------------------------------------------------
// Tests

const testing = std.testing;

fn logEdges(allocator: std.mem.Allocator, lo: f64, hi: f64, n: usize) ![]f64 {
    const edges = try allocator.alloc(f64, n + 1);
    for (edges, 0..) |*e, i| {
        e.* = lo * std.math.pow(f64, hi / lo, @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n)));
    }
    return edges;
}

fn defaultParams(comptime T: type) Params(T) {
    return .{
        .eta = .promote(0.0),
        .a = .promote(0.7),
        .incl = .promote(std.math.degreesToRadians(60.0)),
        .mass = .promote(10.0),
        .mdot = .promote(1.0),
        .distance = .promote(10.0),
        .fcol = .promote(1.7),
        .norm = .promote(1.0),
    };
}

const test_image: image.Options = .{ .n_theta = 64, .n_rho = 48, .n_outer = 24 };

test "total photon flux matches the analytic Planck integral" {
    const p = defaultParams(D0);
    const edges = try logEdges(testing.allocator, 1e-7, 100, 400);
    defer testing.allocator.free(edges);
    const out = try testing.allocator.alloc(D0, edges.len - 1);
    defer testing.allocator.free(out);
    try photonFlux(D0, testing.allocator, p, edges, out, .{ .image = test_image, .n_energy = 8 });
    var total: f64 = 0;
    for (out) |o| total += o.x;

    // Analytic: sum_k W_k * 2 zeta(3) / x0_k^3.
    const r_in = disc.iscoRadius(D0, p.a);
    const samples = try image.traceImage(D0, testing.allocator, p.a, p.incl, r_in, test_image);
    defer testing.allocator.free(samples);
    const n0 = constants.normalisation(D0, p.fcol, p.mass, p.distance).x;
    const mu = constants.temperatureScale(D0, p.fcol, p.mdot, p.mass).x;
    var expected: f64 = 0;
    for (samples) |s| {
        if (s.weight.x == 0) continue;
        const f = disc.fluxNoReturn(D0, s.r, p.a, r_in, p.eta).x;
        const x0 = mu / (s.g.x * std.math.pow(f64, f, 0.25));
        expected += n0 * s.weight.x * 2 * 1.2020569031595942 / (x0 * x0 * x0);
    }
    try testing.expectApproxEqRel(expected, total, 1e-6);
}

test "multi-temperature slope N ~ E^-2/3 between T_out and T_max" {
    const p = defaultParams(D0);
    const edges = [_]f64{ 1.0e-3, 1.001e-3, 2.0e-3, 2.002e-3 };
    var out: [3]D0 = undefined;
    try photonFlux(D0, testing.allocator, p, &edges, &out, .{ .image = test_image });
    const n1 = out[0].x / 1e-6;
    const n2 = out[2].x / 2e-6;
    const slope = @log(n2 / n1) / @log(2.0);
    try testing.expectApproxEqAbs(-2.0 / 3.0, slope, 0.02);
}

test "limb darkening brightens low and dims high inclinations" {
    const edges = [_]f64{ 0.5, 5.0 };
    var iso: [1]D0 = undefined;
    var ld: [1]D0 = undefined;
    inline for (.{ 20.0, 80.0 }, .{ true, false }) |i_deg, brighter| {
        var p = defaultParams(D0);
        p.incl = .promote(std.math.degreesToRadians(i_deg));
        try photonFlux(D0, testing.allocator, p, &edges, &iso, .{ .image = test_image });
        p.limb_darkening = true;
        try photonFlux(D0, testing.allocator, p, &edges, &ld, .{ .image = test_image });
        try testing.expect((ld[0].x > iso[0].x) == brighter);
    }
}

test "spectrum derivatives match finite differences for every parameter" {
    const N = 8;
    const D = kerrz.DualNumber(f64, N);
    const edges = [_]f64{ 0.1, 0.5, 2.0, 6.0, 15.0 };
    const n_bins = edges.len - 1;
    const base = defaultParams(D0);
    const values = [N]f64{ 0.3, base.a.x, base.incl.x, base.mass.x, base.mdot.x, base.distance.x, base.fcol.x, 1.3 };
    const steps = [N]f64{ 1e-4, 3e-4, 3e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4 };

    const Build = struct {
        fn params(comptime T: type, v: [N]f64, seed: bool) Params(T) {
            var s: [N]T = undefined;
            inline for (0..N) |j| {
                s[j] = .promote(v[j]);
                if (T.N > 0 and seed) s[j].dx[j % @max(T.N, 1)] = 1;
            }
            return .{
                .eta = s[0], .a = s[1], .incl = s[2], .mass = s[3],
                .mdot = s[4], .distance = s[5], .fcol = s[6], .norm = s[7],
                .limb_darkening = true,
            };
        }
    };

    var dual_out: [n_bins]D = undefined;
    try photonFlux(D, testing.allocator, Build.params(D, values, true), &edges, &dual_out, .{ .image = test_image });

    for (0..N) |j| {
        var vp = values;
        var vm = values;
        vp[j] += steps[j] * @max(1.0, @abs(values[j]));
        vm[j] -= steps[j] * @max(1.0, @abs(values[j]));
        var op: [n_bins]D0 = undefined;
        var om: [n_bins]D0 = undefined;
        try photonFlux(D0, testing.allocator, Build.params(D0, vp, false), &edges, &op, .{ .image = test_image });
        try photonFlux(D0, testing.allocator, Build.params(D0, vm, false), &edges, &om, .{ .image = test_image });
        for (0..n_bins) |b| {
            const fd = (op[b].x - om[b].x) / (vp[j] - vm[j]);
            const scale = @max(@abs(fd), 1e-6 * dual_out[b].x);
            try testing.expectApproxEqAbs(fd, dual_out[b].dx[j], 1e-5 * scale + 1e-12);
        }
    }
}
