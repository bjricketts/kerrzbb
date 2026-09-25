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
const returning = @import("returning.zig");
const parallel = @import("parallel.zig");

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
        /// Self-irradiation (kerrbb `rflag`), Appendix D.
        returning_radiation: bool = false,
    };
}

pub const Options = struct {
    image: image.Options = .{},
    /// Grids for the returning-radiation kernel (used when
    /// `returning_radiation` is set).
    returning: returning.Options = .{},
    /// Threads for the ray tracing, the returning-radiation kernel and the
    /// energy sums (0: one per CPU). Overrides the sub-options' `n_threads`.
    n_threads: usize = 1,
    /// Emitters whose x0 = mu / (g ftilde^{1/4}) agree to this relative
    /// tolerance are merged (weighted-mean x0) before the energy sums. The
    /// value of N(E) changes at second order in the tolerance (1e-5 at 1e-3)
    /// but derivatives at first order (8e-4 at 1e-3), so this is off by
    /// default. It suits value-only evaluations: at 1e-3 the emitter count
    /// drops about 4x.
    merge_tolerance: f64 = 0,
    /// Use an interpolated log-energy grid for the energy sums when it needs
    /// fewer evaluations than the per-bin quadrature (many narrow bins). The
    /// grid spacing is `grid_step` in ln E.
    energy_grid: bool = true,
    grid_step: f64 = 0.02,
    /// Gauss-Legendre nodes per energy bin.
    n_energy: usize = 4,
};

pub const Error = image.Error || returning.Error || error{
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

/// Cache of the ray-traced geometry (image samples) and the
/// returning-radiation kernel. Neither depends on the mass, accretion rate,
/// distance, fcol, eta or norm, so fits that vary only those skip the ray
/// tracing. Not thread-safe: use one cache per thread.
pub const Cache = struct {
    allocator: std.mem.Allocator,
    plain: Entries(D0) = .{},
    dual: Entries(D3) = .{},
    /// Number of lookups served from the cache (for diagnostics).
    geometry_hits: usize = 0,
    kernel_hits: usize = 0,

    pub fn init(allocator: std.mem.Allocator) Cache {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Cache) void {
        self.plain.clear(self.allocator);
        self.dual.clear(self.allocator);
    }

    fn entries(self: *Cache, comptime G: type) *Entries(G) {
        return if (G == D0) &self.plain else &self.dual;
    }
};

const GeometryKey = struct {
    a: f64,
    incl: f64,
    r_in: ?f64,
    opts: image.Options,
};

const KernelKey = struct {
    a: f64,
    r_in: ?f64,
    limb_darkening: bool,
    opts: returning.Options,
};

fn Entries(comptime G: type) type {
    return struct {
        const Self = @This();
        geometry: ?struct { key: GeometryKey, samples: []image.Sample(G) } = null,
        kernel: ?struct { key: KernelKey, kernel: returning.Kernel(G) } = null,

        fn clear(self: *Self, allocator: std.mem.Allocator) void {
            if (self.geometry) |g| allocator.free(g.samples);
            if (self.kernel) |k| k.kernel.deinit();
            self.* = .{};
        }
    };
}

/// Merge runs of emitters (sorted by x0) whose x0 agree to `tol`, keeping the
/// summed weight and the weight-averaged x0 and d x0. Returns the new length.
fn mergeEmitters(comptime T: type, em: []Emitter(T), tol: f64) usize {
    if (tol <= 0 or em.len == 0) return em.len;
    const A = T.Algebra;
    var out: usize = 0;
    var i: usize = 0;
    while (i < em.len) {
        var acc = em[i];
        var w_sum = em[i].weight.x;
        var wx = w_sum * em[i].x0;
        var wd: [T.N]f64 = undefined;
        inline for (0..T.N) |k| wd[k] = w_sum * em[i].dx0[k];
        const limit = em[i].x0 * (1 + tol);
        var j = i + 1;
        while (j < em.len and em[j].x0 <= limit) : (j += 1) {
            const w = em[j].weight.x;
            acc.weight = A.add(acc.weight, em[j].weight);
            w_sum += w;
            wx += w * em[j].x0;
            inline for (0..T.N) |k| wd[k] += w * em[j].dx0[k];
        }
        acc.x0 = wx / w_sum;
        inline for (0..T.N) |k| acc.dx0[k] = wd[k] / w_sum;
        em[out] = acc;
        out += 1;
        i = j;
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
        /// Threads for `binned`.
        n_threads: usize = 1,
        energy_grid: bool = true,
        grid_step: f64 = 0.02,

        pub fn init(
            allocator: std.mem.Allocator,
            params: Params(T),
            opts: Options,
        ) Error!Self {
            return initCached(allocator, params, opts, null);
        }

        /// As `init`, reusing (and filling) `cache` for the ray tracing.
        pub fn initCached(
            allocator: std.mem.Allocator,
            params: Params(T),
            opts_in: Options,
            cache: ?*Cache,
        ) Error!Self {
            var opts = opts_in;
            opts.image.n_threads = opts.n_threads;
            opts.returning.n_threads = opts.n_threads;
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

            // Keys ignore the thread count, which does not change results.
            var image_key_opts = opts.image;
            image_key_opts.n_threads = 0;
            var kernel_key_opts = opts.returning;
            kernel_key_opts.n_threads = 0;
            const r_in_key: ?f64 = if (params.r_in) |r| r.x else null;

            const samples: []image.Sample(G) = blk: {
                const key: GeometryKey = .{ .a = params.a.x, .incl = params.incl.x, .r_in = r_in_key, .opts = image_key_opts };
                if (cache) |c| {
                    const e = c.entries(G);
                    if (e.geometry) |g| if (std.meta.eql(g.key, key)) {
                        c.geometry_hits += 1;
                        break :blk g.samples;
                    };
                    const fresh = try image.traceImage(G, c.allocator, a_g, i_g, r_in_g, opts.image);
                    if (e.geometry) |g| c.allocator.free(g.samples);
                    e.geometry = .{ .key = key, .samples = fresh };
                    break :blk fresh;
                }
                break :blk try image.traceImage(G, allocator, a_g, i_g, r_in_g, opts.image);
            };
            defer if (cache == null) allocator.free(samples);

            const r_in = params.r_in orelse lift(T, r_in_g, params);

            // Self-irradiation: the kernel depends on (a, r_in) only, so it is
            // built with the geometry slots and lifted onto T.
            var profile: ?returning.Profile(T) = null;
            defer if (profile) |pr| pr.deinit();
            if (params.returning_radiation) {
                const kernel_g: returning.Kernel(G) = blk: {
                    const key: KernelKey = .{ .a = params.a.x, .r_in = r_in_key, .limb_darkening = params.limb_darkening, .opts = kernel_key_opts };
                    if (cache) |c| {
                        const e = c.entries(G);
                        if (e.kernel) |k| if (std.meta.eql(k.key, key)) {
                            c.kernel_hits += 1;
                            break :blk k.kernel;
                        };
                        const fresh = try returning.buildKernel(G, c.allocator, a_g, r_in_g, params.limb_darkening, opts.returning);
                        if (e.kernel) |k| k.kernel.deinit();
                        e.kernel = .{ .key = key, .kernel = fresh };
                        break :blk fresh;
                    }
                    break :blk try returning.buildKernel(G, allocator, a_g, r_in_g, params.limb_darkening, opts.returning);
                };
                defer if (cache == null) kernel_g.deinit();
                const Lift = struct {
                    p: Params(T),
                    pub fn apply(self: @This(), q: G) T {
                        return lift(T, q, self.p);
                    }
                };
                const kernel = try kernel_g.convert(T, allocator, Lift{ .p = params });
                defer kernel.deinit();
                profile = try returning.solve(T, allocator, kernel, params.eta);
            }
            return fromSamples(G, allocator, params, opts, samples, r_in, profile);
        }

        /// Build the emitters from image-plane samples (any construction, e.g.
        /// the Cunningham transfer-function comparison in validation/) and an
        /// optional self-irradiated flux profile. `samples` carry the
        /// geometry slots of `G` (see `lift`).
        pub fn fromSamples(
            comptime G: type,
            allocator: std.mem.Allocator,
            params: Params(T),
            opts: Options,
            samples: []const image.Sample(G),
            r_in: T,
            profile: ?returning.Profile(T),
        ) Error!Self {
            const A = T.Algebra;
            const n0 = A.mult(params.norm, constants.normalisation(T, params.fcol, params.mass, params.distance));
            const mu = constants.temperatureScale(T, params.fcol, params.mdot, params.mass);

            const emitters = try allocator.alloc(Emitter(T), samples.len);
            errdefer allocator.free(emitters);
            var n_emit: usize = 0;
            for (samples) |smp| {
                if (smp.weight.x == 0) continue;
                const r = lift(T, smp.r, params);
                const f = if (profile) |pr| pr.flux(r) else disc.fluxNoReturn(T, r, params.a, r_in, params.eta);
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
            std.mem.sort(Emitter(T), emitters[0..n_emit], {}, Emitter(T).lessThan);
            const n_merged = mergeEmitters(T, emitters[0..n_emit], opts.merge_tolerance);
            const em = try allocator.realloc(emitters, n_merged);
            return .{
                .allocator = allocator,
                .emitters = em,
                .n_threads = opts.n_threads,
                .energy_grid = opts.energy_grid,
                .grid_step = opts.grid_step,
            };
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
        pub fn binned(self: Self, edges: []const f64, out: []T, n_energy: usize) error{OutOfMemory}!void {
            std.debug.assert(out.len + 1 == edges.len);
            std.debug.assert(n_energy <= 32);
            const h = self.grid_step;
            const u_lo = @log(edges[0]) - 2 * h;
            const u_hi = @log(edges[edges.len - 1]) + 2 * h;
            const n_grid: usize = @as(usize, @intFromFloat(@ceil((u_hi - u_lo) / h))) + 1;
            if (self.energy_grid and 2 * n_grid < out.len * n_energy) {
                return self.binnedGrid(edges, out, n_energy, u_lo, n_grid);
            }

            var xg: [32]f64 = undefined;
            var wg: [32]f64 = undefined;
            quadrature.gaussLegendre(xg[0..n_energy], wg[0..n_energy]);
            const Ctx = struct {
                spec: Self,
                edges: []const f64,
                out: []T,
                xg: []const f64,
                wg: []const f64,
                fn run(c: @This(), start: usize, end: usize) error{}!void {
                    for (start..end) |ib| {
                        const lo = c.edges[ib];
                        const hi = c.edges[ib + 1];
                        var acc: T = .zero;
                        for (c.xg, c.wg) |xm, wm| {
                            const E = 0.5 * (hi - lo) * xm + 0.5 * (hi + lo);
                            const d = c.spec.density(E);
                            const wE = 0.5 * (hi - lo) * wm;
                            acc.x += wE * d.x;
                            inline for (0..T.N) |j| acc.dx[j] += wE * d.dx[j];
                        }
                        c.out[ib] = acc;
                    }
                }
            };
            parallel.forRange(error{}, self.n_threads, out.len, Ctx{
                .spec = self,
                .edges = edges,
                .out = out,
                .xg = xg[0..n_energy],
                .wg = wg[0..n_energy],
            }, Ctx.run) catch unreachable;
        }

        /// Energy sums on a uniform grid in u = ln E (step `grid_step`),
        /// then per-bin Gauss-Legendre on a cubic interpolation of ln N(u)
        /// and of d ln N / d theta_j. ln N is smooth in u even in the Wien
        /// tail, where it is close to -x0 e^u.
        fn binnedGrid(
            self: Self,
            edges: []const f64,
            out: []T,
            n_energy: usize,
            u_lo: f64,
            n_grid: usize,
        ) error{OutOfMemory}!void {
            const h = self.grid_step;
            const Node = struct { log_n: f64, rel: [T.N]f64 };
            const nodes = try self.allocator.alloc(Node, n_grid);
            defer self.allocator.free(nodes);

            const GridCtx = struct {
                spec: Self,
                nodes: []Node,
                u_lo: f64,
                h: f64,
                fn run(c: @This(), start: usize, end: usize) error{}!void {
                    for (start..end) |k| {
                        const E = @exp(c.u_lo + c.h * @as(f64, @floatFromInt(k)));
                        const d = c.spec.density(E);
                        var node: Node = .{ .log_n = @log(@max(d.x, std.math.floatMin(f64))), .rel = undefined };
                        inline for (0..T.N) |j| node.rel[j] = if (d.x > 0) d.dx[j] / d.x else 0;
                        c.nodes[k] = node;
                    }
                }
            };
            parallel.forRange(error{}, self.n_threads, n_grid, GridCtx{
                .spec = self,
                .nodes = nodes,
                .u_lo = u_lo,
                .h = h,
            }, GridCtx.run) catch unreachable;

            var xg: [32]f64 = undefined;
            var wg: [32]f64 = undefined;
            quadrature.gaussLegendre(xg[0..n_energy], wg[0..n_energy]);
            for (out, 0..) |*bin, ib| {
                const lo = edges[ib];
                const hi = edges[ib + 1];
                var acc: T = .zero;
                for (xg[0..n_energy], wg[0..n_energy]) |xm, wm| {
                    const E = 0.5 * (hi - lo) * xm + 0.5 * (hi + lo);
                    const t = (@log(E) - u_lo) / h;
                    // Four-point Lagrange stencil k0 .. k0 + 3 around t.
                    const k0: usize = @min(@as(usize, @intFromFloat(@floor(t))) -| 1, n_grid - 4);
                    const s = t - @as(f64, @floatFromInt(k0));
                    const l = [4]f64{
                        -(s - 1) * (s - 2) * (s - 3) / 6,
                        s * (s - 2) * (s - 3) / 2,
                        -s * (s - 1) * (s - 3) / 2,
                        s * (s - 1) * (s - 2) / 6,
                    };
                    var log_n: f64 = 0;
                    var rel: [T.N]f64 = .{0} ** T.N;
                    inline for (0..4) |q| {
                        const node = nodes[k0 + q];
                        log_n += l[q] * node.log_n;
                        inline for (0..T.N) |j| rel[j] += l[q] * node.rel[j];
                    }
                    const n_e = @exp(log_n);
                    const wE = 0.5 * (hi - lo) * wm;
                    acc.x += wE * n_e;
                    inline for (0..T.N) |j| acc.dx[j] += wE * n_e * rel[j];
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
    const spec = try Spectrum(T).init(allocator, params, opts);
    defer spec.deinit();
    try spec.binned(edges, out, opts.n_energy);
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

test "log-energy grid matches the direct per-bin quadrature" {
    const D = kerrz.DualNumber(f64, 2);
    var p = defaultParams(D);
    p.a = D.promote(0.9).diff(0);
    p.incl = D.promote(1.0).diff(1);
    const edges = try logEdges(testing.allocator, 0.05, 80, 3000);
    defer testing.allocator.free(edges);
    const direct = try testing.allocator.alloc(D, edges.len - 1);
    defer testing.allocator.free(direct);
    const grid = try testing.allocator.alloc(D, edges.len - 1);
    defer testing.allocator.free(grid);
    const spec_direct = try Spectrum(D).init(testing.allocator, p, .{ .image = test_image, .energy_grid = false });
    defer spec_direct.deinit();
    try spec_direct.binned(edges, direct, 4);
    const spec_grid = try Spectrum(D).init(testing.allocator, p, .{ .image = test_image });
    defer spec_grid.deinit();
    try spec_grid.binned(edges, grid, 4);
    var peak: f64 = 0;
    for (direct) |d| peak = @max(peak, d.x);
    for (direct, grid) |d, g| {
        if (d.x < 1e-8 * peak) continue;
        try testing.expectApproxEqRel(d.x, g.x, 1e-7);
        inline for (0..2) |j| try testing.expectApproxEqAbs(d.dx[j], g.dx[j], 1e-6 * @abs(d.dx[j]) + 1e-9 * d.x);
    }
}
