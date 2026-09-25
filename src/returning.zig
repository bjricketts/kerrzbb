//! Returning radiation (self-irradiation), Li et al. (2005) §3.1 and
//! Appendix D.
//!
//! The emitted flux satisfies F_out = F0 + F_in[F_out] + F_S[F_out] (eq. D17),
//! where the flux of returning radiation F_in (eq. D16) and the work done by
//! its stress, F_S (eqs. D12, D19), are linear in F_out. Discretising F_out on a
//! radial grid gives a matrix K with F_in + F_S = K F_out, and the solution is
//! one linear solve, (I - K) F_out = F0, instead of the paper's fixed-point
//! iteration. K depends only on the spin, the inner radius and the emission
//! law (isotropic or limb-darkened).
//!
//! Construction of K. For each absorbing radius r_a, the upper hemisphere of
//! the disc frame is sampled with a grid concentrated around the (aberrated)
//! direction of the black hole. Each direction is traced backwards along the
//! arriving photon's path to the equatorial plane. If the ray started on the
//! disc at r_e >= r_in, it contributes
//!
//!     I_in = g_hat^4 F_out(r_e) Upsilon(cos theta_e) / pi,    g_hat = G(r_e) / G(r_a),
//!
//! with G(r) = (E - Omega L)(r) / (1 - Omega(r) lambda) (eqs. C21, C22), to
//! F_in and to the stress S. Everything is in the dimensionless units of
//! eq. (13), ftilde = 8 pi F / (3 Mdot_eff).
//!
//! F_out between grid nodes is interpolated as r^-3 times a function that is
//! piecewise linear in log r. Beyond the last node r_max,
//! F_in ~ F_in(r_max) (r_max / r)^3 and F_S keeps the form of eq. (D12) with
//! the stress integral frozen at its r_max value.

const std = @import("std");
const kerrz = @import("kerrz");
const disc = @import("disc.zig");
const quadrature = @import("quadrature.zig");

pub const Options = struct {
    /// Radial nodes between r_in and r_max, clustered toward r_in.
    n_radii: usize = 48,
    /// Gauss-Legendre nodes in log(psi), the angle from the black hole direction.
    n_psi: usize = 48,
    /// Midpoint nodes in the azimuth around the black hole direction (upper hemisphere).
    n_chi: usize = 24,
    /// Outermost node.
    r_max: f64 = 1e4,
};

pub const Error = std.mem.Allocator.Error || error{SingularSystem};

/// Quantities of the circular orbit at r that the kernel needs.
fn Local(comptime T: type) type {
    const A = T.Algebra;
    return struct {
        const Self = @This();
        orb: disc.Orbit(T),
        /// Gamma (A/r^2)^{1/2}: converts the local azimuthal direction cosine
        /// into specific angular momentum (eqs. C14, D9).
        b: T,
        /// Lorentz factor relative to the locally non-rotating frame.
        gamma: T,
        /// Azimuthal velocity relative to the locally non-rotating frame.
        v: T,

        fn init(r: T, a: T) Self {
            const orb = disc.Orbit(T).init(r, a);
            const r2 = A.powi(r, 2);
            const delta = A.add(A.sub(r2, A.mult(.promote(2), r)), A.powi(a, 2));
            const kerr_a = A.add(A.powi(r2, 2), A.mult(A.powi(a, 2), A.mult(r, A.add(r, .promote(2)))));
            const chi = A.sqrt(A.div(A.mult(r2, delta), kerr_a));
            const gamma = A.div(chi, orb.energyFactor());
            const b = A.mult(gamma, A.div(A.sqrt(kerr_a), r));
            return .{ .orb = orb, .b = b, .gamma = gamma, .v = A.div(orb.ang_mom, b) };
        }

        /// E_inf / E_local for a photon of reduced angular momentum lambda.
        fn redshiftFactor(self: Self, lambda: T) T {
            return A.div(self.orb.energyFactor(), A.sub(.one, A.mult(self.orb.omega, lambda)));
        }

        /// The prefactor of eq. (D12): F_S(r) = prefactor(r) * integral.
        fn stressPrefactor(self: Self) T {
            return A.div(self.orb.dOmegaDr(), A.mult(self.orb.r, A.powi(self.orb.energyFactor(), 2)));
        }
    };
}

/// The discretised operator, K = K_in + D K_S, on `nodes`.
pub fn Kernel(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        a: T,
        r_in: T,
        nodes: []T,
        /// F_in at node i = sum_j k_in[i, j] F_out(node j).
        k_in: []T,
        /// S at node i = sum_j k_s[i, j] F_out(node j).
        k_s: []T,

        pub fn deinit(self: Self) void {
            self.allocator.free(self.nodes);
            self.allocator.free(self.k_in);
            self.allocator.free(self.k_s);
        }

        pub fn n(self: Self) usize {
            return self.nodes.len;
        }

        /// Map every quantity through `ctx.apply(q) U`, e.g. to lift the
        /// derivative slots onto a different dual type.
        pub fn convert(self: Self, comptime U: type, allocator: std.mem.Allocator, ctx: anytype) Error!Kernel(U) {
            const nodes = try allocator.alloc(U, self.nodes.len);
            errdefer allocator.free(nodes);
            const k_in = try allocator.alloc(U, self.k_in.len);
            errdefer allocator.free(k_in);
            const k_s = try allocator.alloc(U, self.k_s.len);
            for (nodes, self.nodes) |*o, q| o.* = ctx.apply(q);
            for (k_in, self.k_in) |*o, q| o.* = ctx.apply(q);
            for (k_s, self.k_s) |*o, q| o.* = ctx.apply(q);
            return .{
                .allocator = allocator,
                .a = ctx.apply(self.a),
                .r_in = ctx.apply(self.r_in),
                .nodes = nodes,
                .k_in = k_in,
                .k_s = k_s,
            };
        }
    };
}

/// Radial grid: r_j = r_in exp(X s_j^2), s_j = j / (n - 1), X = log(r_max / r_in).
fn makeNodes(comptime T: type, nodes: []T, r_in: T, r_max: f64) void {
    const A = T.Algebra;
    const span = A.log(A.div(.promote(r_max), r_in));
    const n = nodes.len;
    for (nodes, 0..) |*r, j| {
        const s = @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(n - 1));
        r.* = A.mult(r_in, A.exp(A.mult(span, .promote(s * s))));
    }
    nodes[0] = r_in;
}

/// Interpolation weights of F_out(r) on the nodes: at most two non-zero,
/// returned as (index, weight) pairs; index = maxInt(usize) marks unused.
fn Weights(comptime T: type) type {
    return struct { idx: [2]usize, w: [2]T };
}

fn weights(comptime T: type, nodes: []const T, r: T) Weights(T) {
    const A = T.Algebra;
    const none = std.math.maxInt(usize);
    const n = nodes.len;
    if (r.x >= nodes[n - 1].x) {
        return .{ .idx = .{ n - 1, none }, .w = .{ A.powi(A.div(nodes[n - 1], r), 3), .zero } };
    }
    // Largest j with nodes[j] <= r.
    var lo: usize = 0;
    var hi: usize = n - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if (nodes[mid].x <= r.x) lo = mid else hi = mid;
    }
    const l0 = A.log(nodes[lo]);
    const t = A.div(A.sub(A.log(r), l0), A.sub(A.log(nodes[hi]), l0));
    return .{
        .idx = .{ lo, hi },
        .w = .{
            A.mult(A.sub(.one, t), A.powi(A.div(nodes[lo], r), 3)),
            A.mult(t, A.powi(A.div(nodes[hi], r), 3)),
        },
    };
}

/// A backward ray from the absorbing radius, launched in the disc frame at
/// angle psi from the black hole direction and azimuth chi around it.
fn Ray(comptime T: type) type {
    const A = T.Algebra;
    return struct {
        const Self = @This();

        geod: kerrz.NullGeodesic(T),
        /// Direction cosines of n (back along the arriving photon) in the
        /// disc frame: radial, azimuthal and vertical.
        nphi: T,
        nz: T,
        r_e: T,
        /// The ray started on the equatorial plane (not at the horizon or
        /// infinity) at a finite radius.
        crossed: bool,

        fn trace(
            metric: kerrz.KerrMetric(T),
            loc: Local(T),
            r_a: T,
            psi: T,
            chi: f64,
        ) Self {
            // Orthonormal basis around the (aberrated) black hole direction,
            // in (e_r, e_phi, e_z) components.
            const e1 = [2]T{ A.div(.one, loc.gamma).neg(), loc.v };
            const e2 = [2]T{ loc.v, A.div(.one, loc.gamma) };
            const c_psi = A.cos(psi);
            const s_psi = A.sin(psi);
            const s_cchi = A.mult(s_psi, .promote(@cos(chi)));
            const nr = A.add(A.mult(c_psi, e1[0]), A.mult(s_cchi, e2[0]));
            const nphi = A.add(A.mult(c_psi, e1[1]), A.mult(s_cchi, e2[1]));
            const nz = A.mult(s_psi, .promote(@sin(chi)));

            // Constants of the arriving photon (direction -n) per unit local
            // energy (eqs. C14, C15, C17).
            const E = A.sub(loc.orb.energy, A.mult(A.mult(loc.b, loc.orb.omega), nphi));
            const L = A.sub(loc.orb.ang_mom, A.mult(loc.b, nphi));
            const Q = A.mult(A.powi(r_a, 2), A.powi(nz, 2));

            // Follow the path back: upward (theta decreasing) and along n_r.
            const x: kerrz.FourVector(T) = .{ .t = .zero, .r = r_a, .th = .promote(std.math.pi / 2.0), .ph = .zero };
            var geod = kerrz.NullGeodesic(T).fromConstantsOfMotion(x, E, L, Q);
            geod.theta_sign = -1;
            geod.radial_sign = if (nr.x >= 0) 1 else -1;
            const res = geod.traceToAngle(metric, .promote(std.math.pi / 2.0), .{});
            const crossed = res.status != .event_horizon and res.status != .infinity and
                std.math.isFinite(res.r.x) and res.r.x > 0;
            return .{ .geod = geod, .nphi = nphi, .nz = nz, .r_e = res.r, .crossed = crossed };
        }
    };
}

/// Smallest psi (at fixed chi) whose backward ray starts on the disc at
/// r_e >= r_in, with derivatives from the implicit function theorem when the
/// boundary is the r_e = r_in contour. Returns null if no ray returns to the
/// disc.
fn innerEdge(
    comptime T: type,
    a: T,
    r_a: T,
    r_in: T,
    chi: f64,
    psi_min: f64,
) ?T {
    const V = kerrz.DualNumber(f64, 0);
    const metric_v = kerrz.KerrMetric(V).init(.one, .promote(a.x));
    const loc_v = Local(V).init(.promote(r_a.x), .promote(a.x));
    const Probe = struct {
        fn onDisc(m: kerrz.KerrMetric(V), l: Local(V), ra: f64, rin: f64, psi: f64, c: f64) bool {
            const ray = Ray(V).trace(m, l, .promote(ra), .promote(psi), c);
            return ray.crossed and ray.r_e.x >= rin;
        }
    };

    // Coarse scan in log(psi), then bisection.
    const n_scan = 48;
    const span = @log(std.math.pi / psi_min);
    var prev = psi_min;
    if (Probe.onDisc(metric_v, loc_v, r_a.x, r_in.x, psi_min, chi)) return .promote(psi_min);
    var found: ?f64 = null;
    for (1..n_scan + 1) |k| {
        const psi = psi_min * @exp(span * @as(f64, @floatFromInt(k)) / n_scan);
        if (Probe.onDisc(metric_v, loc_v, r_a.x, r_in.x, psi, chi)) {
            found = psi;
            break;
        }
        prev = psi;
    }
    var hi = found orelse return null;
    var lo = prev;
    for (0..60) |_| {
        const mid = 0.5 * (lo + hi);
        if (Probe.onDisc(metric_v, loc_v, r_a.x, r_in.x, mid, chi)) hi = mid else lo = mid;
        if (hi - lo <= 1e-14 * hi) break;
    }
    var edge: T = .promote(hi);
    if (T.N == 0) return edge;

    // If the boundary is the r_e = r_in contour, differentiate it:
    // d psi / dp = -(dF/dp) / (dF/d psi) with F = r_e(psi, p) - r_in(p).
    const T1 = T.PushSlot();
    const metric1 = kerrz.KerrMetric(T1).init(.one, a.pushSlot());
    const loc1 = Local(T1).init(r_a.pushSlot(), a.pushSlot());
    const ray = Ray(T1).trace(metric1, loc1, r_a.pushSlot(), T1.promote(hi).diff(0), chi);
    if (!ray.crossed or @abs(ray.r_e.x - r_in.x) > 1e-6 * r_in.x) return edge;
    const F = T1.Algebra.sub(ray.r_e, r_in.pushSlot());
    if (F.dx[0] == 0) return edge;
    inline for (0..T.N) |k| edge.dx[k] = -F.dx[k + 1] / F.dx[0];
    return edge;
}

/// Build the returning-radiation kernel for spin `a` and inner radius `r_in`.
///
/// At each absorbing radius and azimuth chi around the black hole direction,
/// the angle psi runs from the edge psi_in(chi), where backward rays first
/// land on the disc at r_in, to pi. The integrand jumps at psi_in (F_out(r_in)
/// is the finite returning flux there), so the quadrature starts exactly on
/// that edge, clustered toward it; psi_in carries parameter derivatives.
pub fn buildKernel(
    comptime T: type,
    allocator: std.mem.Allocator,
    a: T,
    r_in: T,
    limb_darkening: bool,
    opts: Options,
) Error!Kernel(T) {
    const A = T.Algebra;
    const n = opts.n_radii;
    std.debug.assert(n >= 3);

    const nodes = try allocator.alloc(T, n);
    errdefer allocator.free(nodes);
    makeNodes(T, nodes, r_in, opts.r_max);
    const k_in = try allocator.alloc(T, n * n);
    errdefer allocator.free(k_in);
    const k_s = try allocator.alloc(T, n * n);
    errdefer allocator.free(k_s);
    @memset(k_in, .zero);
    @memset(k_s, .zero);

    const xg = try allocator.alloc(f64, opts.n_psi);
    defer allocator.free(xg);
    const wg = try allocator.alloc(f64, opts.n_psi);
    defer allocator.free(wg);
    quadrature.gaussLegendre(xg, wg);

    const metric = kerrz.KerrMetric(T).init(.one, a);
    const horizon = metric.horizon_radius.x;
    const dchi = std.math.pi / @as(f64, @floatFromInt(opts.n_chi));

    for (0..n) |i| {
        const r_a = nodes[i];
        const loc = Local(T).init(r_a, a);
        const psi_min = 0.02 * @min(1.0, horizon / r_a.x);

        for (0..opts.n_chi) |ic| {
            const chi = (@as(f64, @floatFromInt(ic)) + 0.5) * dchi;
            const psi_in = innerEdge(T, a, r_a, r_in, chi, psi_min) orelse continue;

            // psi = psi_in + (pi - psi_in) (e^{kappa t} - 1) / (e^kappa - 1),
            // clustering nodes on a scale ~0.1 psi_in above the edge.
            const width = A.sub(.promote(std.math.pi), psi_in);
            const kappa = A.log(A.add(.one, A.div(width, A.mult(.promote(0.1), psi_in))));
            const denom = A.sub(A.exp(kappa), .one);

            for (0..opts.n_psi) |ip| {
                const t = 0.5 * (xg[ip] + 1);
                const e_kt = A.exp(A.mult(kappa, .promote(t)));
                const psi = A.add(psi_in, A.div(A.mult(width, A.sub(e_kt, .one)), denom));
                const dpsi_dt = A.div(A.mult(A.mult(width, kappa), e_kt), denom);
                // d Omega = sin(psi) d psi d chi
                const w_ray = A.mult(A.mult(dpsi_dt, A.sin(psi)), .promote(0.5 * wg[ip] * dchi));

                const ray = Ray(T).trace(metric, loc, r_a, psi, chi);
                if (!ray.crossed or ray.r_e.x < r_in.x) continue;
                const r_e = ray.r_e;
                const lambda = ray.geod.lambda;
                const src = Local(T).init(r_e, a);
                const g_hat = A.div(src.redshiftFactor(lambda), loc.redshiftFactor(lambda));
                if (!(g_hat.x > 0)) continue;

                var coeff = A.mult(A.div(w_ray, .promote(std.math.pi)), A.mult(A.powi(g_hat, 4), ray.nz));
                if (limb_darkening) {
                    // cos theta_e = sqrt(eta) (E - Omega L)_e / (r_e |1 - Omega_e lambda|)  (eq. C23)
                    const d = A.abs(A.sub(.one, A.mult(src.orb.omega, lambda)));
                    const cos_e = A.div(A.mult(A.sqrt(ray.geod.eta), src.orb.energyFactor()), A.mult(r_e, d));
                    coeff = A.mult(coeff, A.add(.promote(0.5), A.mult(.promote(0.75), cos_e)));
                }
                const coeff_s = A.mult(coeff, A.mult(loc.b, ray.nphi));

                const wt = weights(T, nodes, r_e);
                inline for (0..2) |k| {
                    const j = wt.idx[k];
                    if (j != std.math.maxInt(usize)) {
                        k_in[i * n + j] = A.add(k_in[i * n + j], A.mult(coeff, wt.w[k]));
                        k_s[i * n + j] = A.add(k_s[i * n + j], A.mult(coeff_s, wt.w[k]));
                    }
                }
            }
        }
    }

    return .{ .allocator = allocator, .a = a, .r_in = r_in, .nodes = nodes, .k_in = k_in, .k_s = k_s };
}

/// The self-consistent emitted flux profile ftilde_out(r).
pub fn Profile(comptime T: type) type {
    const A = T.Algebra;
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        a: T,
        r_in: T,
        eta: T,
        nodes: []T,
        /// ftilde_out at the nodes.
        out: []T,
        /// ftilde_in at the nodes.
        in: []T,
        /// Stress S (same units) at the nodes.
        stress: []T,
        /// Cumulative integral of (E - Omega L) S r dr from r_in to each node.
        stress_integral: []T,

        pub fn deinit(self: Self) void {
            self.allocator.free(self.nodes);
            self.allocator.free(self.out);
            self.allocator.free(self.in);
            self.allocator.free(self.stress);
            self.allocator.free(self.stress_integral);
        }

        fn interpolate(self: Self, values: []const T, r: T) T {
            const w = weights(T, self.nodes, r);
            var v = A.mult(w.w[0], values[w.idx[0]]);
            if (w.idx[1] != std.math.maxInt(usize)) v = A.add(v, A.mult(w.w[1], values[w.idx[1]]));
            return v;
        }

        /// Emitted flux ftilde_out(r) for r >= r_in.
        pub fn flux(self: Self, r: T) T {
            if (r.x < self.r_in.x) return .zero;
            const last = self.nodes.len - 1;
            if (r.x <= self.nodes[last].x) return self.interpolate(self.out, r);
            return A.add(
                disc.fluxNoReturn(T, r, self.a, self.r_in, self.eta),
                A.add(self.returningIn(r), self.stressFlux(r)),
            );
        }

        /// ftilde_in(r).
        pub fn returningIn(self: Self, r: T) T {
            if (r.x < self.r_in.x) return .zero;
            return self.interpolate(self.in, r);
        }

        /// S(r); beyond the last node it is taken to vanish.
        pub fn stressAt(self: Self, r: T) T {
            if (r.x < self.r_in.x or r.x > self.nodes[self.nodes.len - 1].x) return .zero;
            return self.interpolate(self.stress, r);
        }

        /// ftilde_S(r) from eq. (D12).
        pub fn stressFlux(self: Self, r: T) T {
            const last = self.nodes.len - 1;
            const loc = Local(T).init(r, self.a);
            const integral = if (r.x >= self.nodes[last].x)
                self.stress_integral[last]
            else
                self.interpolateIntegral(r);
            return A.mult(loc.stressPrefactor(), integral);
        }

        fn interpolateIntegral(self: Self, r: T) T {
            // Linear in r between nodes.
            var lo: usize = 0;
            while (lo + 1 < self.nodes.len - 1 and self.nodes[lo + 1].x <= r.x) lo += 1;
            const t = A.div(A.sub(r, self.nodes[lo]), A.sub(self.nodes[lo + 1], self.nodes[lo]));
            return A.add(self.stress_integral[lo], A.mult(t, A.sub(self.stress_integral[lo + 1], self.stress_integral[lo])));
        }
    };
}

/// Solve (I - K_in - D K_S) u = ftilde_0 on the kernel's nodes.
pub fn solve(
    comptime T: type,
    allocator: std.mem.Allocator,
    kernel: Kernel(T),
    eta: T,
) Error!Profile(T) {
    const A = T.Algebra;
    const n = kernel.n();
    const nodes = kernel.nodes;
    const a = kernel.a;
    const r_in = kernel.r_in;

    // D: F_S(node i) = P_i * sum_j trap[i, j] (E - Omega L)_j r_j S_j.
    const d = try allocator.alloc(T, n * n);
    defer allocator.free(d);
    @memset(d, .zero);
    const locs = try allocator.alloc(Local(T), n);
    defer allocator.free(locs);
    for (locs, nodes) |*l, r| l.* = Local(T).init(r, a);
    for (1..n) |i| {
        const p = locs[i].stressPrefactor();
        for (0..i) |j| {
            // Trapezoid on [r_j, r_{j+1}] contributes half the width to both ends.
            const half = A.mult(.promote(0.5), A.sub(nodes[j + 1], nodes[j]));
            inline for (.{ j, j + 1 }) |k| {
                const g = A.mult(A.mult(half, locs[k].orb.energyFactor()), nodes[k]);
                d[i * n + k] = A.add(d[i * n + k], A.mult(p, g));
            }
        }
    }

    // M = I - K_in - D K_S, rhs = ftilde_0.
    const m = try allocator.alloc(T, n * n);
    defer allocator.free(m);
    for (0..n) |i| for (0..n) |j| {
        var dk: T = .zero;
        for (0..n) |k| dk = A.add(dk, A.mult(d[i * n + k], kernel.k_s[k * n + j]));
        var v = A.add(kernel.k_in[i * n + j], dk).neg();
        if (i == j) v = A.add(v, .one);
        m[i * n + j] = v;
    };
    const u = try allocator.alloc(T, n);
    errdefer allocator.free(u);
    for (u, nodes) |*x, r| x.* = disc.fluxNoReturn(T, r, a, r_in, eta);
    try luSolve(T, m, u, n);

    const in = try allocator.alloc(T, n);
    errdefer allocator.free(in);
    const stress = try allocator.alloc(T, n);
    errdefer allocator.free(stress);
    for (0..n) |i| {
        var fi: T = .zero;
        var si: T = .zero;
        for (0..n) |j| {
            fi = A.add(fi, A.mult(kernel.k_in[i * n + j], u[j]));
            si = A.add(si, A.mult(kernel.k_s[i * n + j], u[j]));
        }
        in[i] = fi;
        stress[i] = si;
    }
    const integral = try allocator.alloc(T, n);
    errdefer allocator.free(integral);
    integral[0] = .zero;
    for (1..n) |i| {
        const half = A.mult(.promote(0.5), A.sub(nodes[i], nodes[i - 1]));
        const g0 = A.mult(A.mult(locs[i - 1].orb.energyFactor(), nodes[i - 1]), stress[i - 1]);
        const g1 = A.mult(A.mult(locs[i].orb.energyFactor(), nodes[i]), stress[i]);
        integral[i] = A.add(integral[i - 1], A.mult(half, A.add(g0, g1)));
    }

    const nodes_copy = try allocator.dupe(T, nodes);
    return .{
        .allocator = allocator,
        .a = a,
        .r_in = r_in,
        .eta = eta,
        .nodes = nodes_copy,
        .out = u,
        .in = in,
        .stress = stress,
        .stress_integral = integral,
    };
}

/// In-place LU solve with partial pivoting of the n x n row-major system.
fn luSolve(comptime T: type, m: []T, b: []T, n: usize) Error!void {
    const A = T.Algebra;
    for (0..n) |col| {
        var piv = col;
        for (col + 1..n) |r| {
            if (@abs(m[r * n + col].x) > @abs(m[piv * n + col].x)) piv = r;
        }
        if (m[piv * n + col].x == 0) return Error.SingularSystem;
        if (piv != col) {
            for (0..n) |k| std.mem.swap(T, &m[col * n + k], &m[piv * n + k]);
            std.mem.swap(T, &b[col], &b[piv]);
        }
        const inv = A.div(.one, m[col * n + col]);
        for (col + 1..n) |r| {
            const f = A.mult(m[r * n + col], inv);
            if (f.x == 0 and T.N == 0) continue;
            for (col..n) |k| m[r * n + k] = A.sub(m[r * n + k], A.mult(f, m[col * n + k]));
            b[r] = A.sub(b[r], A.mult(f, b[col]));
        }
    }
    var i = n;
    while (i > 0) {
        i -= 1;
        var s = b[i];
        for (i + 1..n) |k| s = A.sub(s, A.mult(m[i * n + k], b[k]));
        b[i] = A.div(s, m[i * n + i]);
    }
}

// ---------------------------------------------------------------------------
// Tests

const testing = std.testing;
const D0 = kerrz.DualNumber(f64, 0);

/// Integrate h(r) r dr from r_in to r_hi in log r (composite Gauss-Legendre).
fn integrate(r_in: f64, r_hi: f64, ctx: anytype) f64 {
    var x: [16]f64 = undefined;
    var w: [16]f64 = undefined;
    quadrature.gaussLegendre(&x, &w);
    const panels = 200;
    const t0 = @log(r_in);
    const width = (@log(r_hi) - t0) / panels;
    var sum: f64 = 0;
    for (0..panels) |p| {
        // Cluster panels toward r_in: t = t0 + width * panels * (p/panels)^2 ...
        const lo = t0 + width * @as(f64, @floatFromInt(p));
        for (x, w) |xi, wi| {
            const r = @exp(lo + 0.5 * width * (xi + 1));
            sum += 0.5 * width * wi * ctx.h(r) * r * r;
        }
    }
    return sum;
}

test "stress flux of eq. (D12) carries no net energy" {
    // For any stress profile S, int (E F_S + Omega S) r dr = 0 over the whole
    // disc, since F_S only redistributes the work done by the stress.
    const a = 0.7;
    const r_in = disc.iscoRadius(D0, .promote(a)).x;
    const K = 200000;
    const h = @log(1e6 / r_in) / @as(f64, K);
    var integral: f64 = 0;
    var net: f64 = 0;
    var scale: f64 = 0;
    var prev_r = r_in;
    var prev_g: f64 = 0; // (E - Omega L) S r at prev_r
    var prev_h: f64 = 0; // (E F_S + Omega S) r at prev_r
    var prev_abs: f64 = 0;
    for (1..K + 1) |k| {
        const r = r_in * @exp(h * @as(f64, @floatFromInt(k)));
        const l = Local(D0).init(.promote(r), .promote(a));
        const stress = (r - r_in) * @exp(-r / 7.0);
        const g = l.orb.energyFactor().x * stress * r;
        integral += 0.5 * (r - prev_r) * (g + prev_g);
        const fs = l.stressPrefactor().x * integral;
        const hh = (l.orb.energy.x * fs + l.orb.omega.x * stress) * r;
        const ha = @abs(l.orb.omega.x * stress) * r;
        net += 0.5 * (r - prev_r) * (hh + prev_h);
        scale += 0.5 * (r - prev_r) * (ha + prev_abs);
        prev_r = r;
        prev_g = g;
        prev_h = hh;
        prev_abs = ha;
    }
    try testing.expectApproxEqAbs(0.0, net / scale, 1e-4);
}

const Budget = struct { emitted: f64, returned: f64, net: f64, eps: f64 };

/// Energy-at-infinity budget of a solved profile (eqs. 6, 8, D22):
/// emitted 4pi int E F_out r dr, returned 4pi int (E F_in - Omega S) r dr,
/// all in units of Mdot_eff (i.e. multiplied by 3/2 from ftilde).
fn budget(profile: Profile(D0), a: f64) Budget {
    const r_in = profile.r_in.x;
    const Em = struct {
        p: Profile(D0),
        a: f64,
        pub fn h(self: @This(), r: f64) f64 {
            const l = Local(D0).init(.promote(r), .promote(self.a));
            return l.orb.energy.x * self.p.flux(.promote(r)).x;
        }
    };
    const Ret = struct {
        p: Profile(D0),
        a: f64,
        pub fn h(self: @This(), r: f64) f64 {
            const l = Local(D0).init(.promote(r), .promote(self.a));
            return l.orb.energy.x * self.p.returningIn(.promote(r)).x - l.orb.omega.x * self.p.stressAt(.promote(r)).x;
        }
    };
    const emitted = 1.5 * integrate(r_in, 1e7, Em{ .p = profile, .a = a });
    const returned = 1.5 * integrate(r_in, 1e7, Ret{ .p = profile, .a = a });
    const eps = 1 - disc.Orbit(D0).init(.promote(r_in), .promote(a)).energy.x;
    return .{ .emitted = emitted, .returned = returned, .net = emitted - returned, .eps = eps };
}

test "returning radiation: energy conservation and returned fractions" {
    // Li et al. (2005) section 3.1: eta = 0, isotropic emission. The fraction
    // of the emitted power that returns to the disc is 1.7% at a = 0 and 27%
    // at a = 0.9999 (their Fig. 2).
    const cases = [_]struct { a: f64, iota_ret: f64 }{
        .{ .a = 0.001, .iota_ret = 0.017 },
        .{ .a = 0.9999, .iota_ret = 0.27 },
    };
    for (cases) |c| {
        const a: D0 = .promote(c.a);
        const r_in = disc.iscoRadius(D0, a);
        const kernel = try buildKernel(D0, testing.allocator, a, r_in, false, .{});
        defer kernel.deinit();
        const profile = try solve(D0, testing.allocator, kernel, .zero);
        defer profile.deinit();
        const b = budget(profile, c.a);
        const iota = b.returned / b.emitted;
        std.debug.print("a={d}: emitted/eps={d:.5} net/eps={d:.5} iota_ret={d:.4} (paper {d})\n", .{ c.a, b.emitted / b.eps, b.net / b.eps, iota, c.iota_ret });
        // Net power must equal eps_in Mdot_eff (eq. 8 with eq. 3).
        try testing.expectApproxEqRel(b.eps, b.net, 5e-3);
        try testing.expectApproxEqRel(c.iota_ret, iota, 0.1);
    }
}
