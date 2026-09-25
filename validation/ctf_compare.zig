//! Compare kerrzbb's boundary-fitted image-plane integration (B1) with an
//! integration over stored Cunningham transfer function (CTF) tables built
//! with kerrz, as they would be used in practice: tables computed once per
//! (a, i) node, saved, and interpolated.
//!
//!     zig build ctf-compare -Doptimize=ReleaseFast -- [output.json]
//!     python validation/plot_ctf_compare.py [output.json]
//!
//! Stored table. For each of `n_radii` radii (log-spaced from r_isco to
//! r_break) kerrz traces the r = const contour. We store, on `n_phi` midpoint
//! nodes of phi in [0, pi] with g* = (1 - cos phi) / 2,
//!
//!     S(phi) = sum over both branches of pi f / g = (1/r) dA / (dr dphi),
//!
//! using f = g sqrt(g*(1 - g*)) |d(alpha,beta)/d(r,g*)| / (pi r) from kerrz.
//! Each branch is resampled from the traces with 4-point Lagrange
//! interpolation in phi. The sum over branches is an even, smooth function of
//! phi, so the midpoint rule in phi (Gauss-Chebyshev in g*) converges
//! spectrally. Plus g_min(r) and g_max(r).
//!
//! Evaluation. Gauss-Legendre in log r with `n_r` nodes; 4-point Lagrange in
//! the (uniform) table index for S, g_min and g_max. Tables at other (a, i)
//! are interpolated node-wise (linear or cubic Lagrange) in (x, i), where
//! x = a for a < 0.6 and x = -log(1 - a) above, matching the spacing of
//! kerrbb's table (0.1 in a, then a factor 0.751 in 1 - a; 5 deg in i).
//!
//! kerrz issue worked around here: `isPathological` drops traces with
//! |d(alpha,beta)/d(r,g)| > 1e6 (absolute), while the Jacobian grows as r^2.
//! Beyond r ~ 500 the traces nearest the redshift extrema are lost and
//! g_min/g_max are underestimated (dA/dr 4.6% low at r = 1000, i = 30 deg).
//! `fixExtrema` re-estimates them from a parabola through the extremal trace
//! and its neighbours.
//!
//! Both methods share the weak-field region r > r_break, the same spectrum
//! code and one thread. The reference is B1 at 1024x768; the derivative
//! reference is B1 at 256x192 with dual numbers in (a, i).

const std = @import("std");
const kbb = @import("kerrzbb");
const kerrz = kbb.kerrz;
const D0 = kerrz.DualNumber(f64, 0);
const D2 = kerrz.DualNumber(f64, 2);
const Sample = kbb.image.Sample(D0);
const Table = kerrz.CunninghamTransferFunctionTable(D0);

const r_break = 1e3;
const observer_distance = 1e8;

const Case = struct { a: f64, incl_deg: f64, spin_coord: enum { linear, log } };
const cases = [_]Case{
    .{ .a = 0.55, .incl_deg = 32.5, .spin_coord = .linear },
    .{ .a = 1 - @sqrt((1 - 0.9042770266532898) * (1 - 0.9280869960784912)), .incl_deg = 62.5, .spin_coord = .log },
    .{ .a = 1 - @sqrt((1 - 0.9982540011405945) * (1 - 0.9986879825592041)), .incl_deg = 32.5, .spin_coord = .log },
    .{ .a = 1 - @sqrt((1 - 0.9982540011405945) * (1 - 0.9986879825592041)), .incl_deg = 77.5, .spin_coord = .log },
};
const kerrbb_step_linear = 0.1;
const kerrbb_step_log = -@log(0.751);
const kerrbb_step_incl = 5.0;

fn toCoord(c: Case, a: f64) f64 {
    return switch (c.spin_coord) {
        .linear => a,
        .log => -@log(1 - a),
    };
}
fn fromCoord(c: Case, x: f64) f64 {
    return switch (c.spin_coord) {
        .linear => x,
        .log => 1 - @exp(-x),
    };
}
fn spinStep(c: Case) f64 {
    return switch (c.spin_coord) {
        .linear => kerrbb_step_linear,
        .log => kerrbb_step_log,
    };
}

fn nowMs(timer: *std.time.Timer) f64 {
    return @as(f64, @floatFromInt(timer.read())) / 1e6;
}

fn deg(x: f64) f64 {
    return std.math.degreesToRadians(x);
}

/// Lagrange basis weights for `nodes` at `t`.
fn lagrange(nodes: []const f64, t: f64, out: []f64) void {
    for (out, 0..) |*w, i| {
        var p: f64 = 1;
        for (nodes, 0..) |xj, j| {
            if (j != i) p *= (t - xj) / (nodes[i] - xj);
        }
        w.* = p;
    }
}


fn wrap(x: f64) f64 {
    var y = @mod(x + std.math.pi, 2 * std.math.pi) - std.math.pi;
    if (y <= -std.math.pi) y += 2 * std.math.pi;
    return y;
}
fn vertex(traces: anytype, i: usize) f64 {
    const n = traces.len;
    const t0 = traces[i];
    const tm = traces[(i + n - 1) % n];
    const tp = traces[(i + 1) % n];
    const xm = wrap(tm.image_angle - t0.image_angle);
    const xp = wrap(tp.image_angle - t0.image_angle);
    const sm = (tm.g - t0.g) / xm;
    const sp = (tp.g - t0.g) / xp;
    const a = (sp - sm) / (xp - xm);
    const b = sp - a * xp;
    const x_v = -b / (2 * a);
    if (!(a != 0) or x_v < @min(xm, xp) or x_v > @max(xm, xp)) return t0.g;
    return t0.g - b * b / (4 * a);
}
pub fn fixExtrema(tf: anytype) void {
    const traces = tf.traces;
    const g_max = vertex(traces, tf.g_max_index);
    const g_min = vertex(traces, tf.g_min_index);
    const delta_g = g_max - g_min;
    const prefactor = 1.0 / (std.math.pi * tf.target_radius);
    for (traces) |*t| {
        t.g_star = (t.g - g_min) / delta_g;
        t.f = prefactor * t.g * @sqrt(t.g_star * (1 - t.g_star)) * t.jacobian * delta_g;
    }
    tf.g_min = g_min;
    tf.g_max = g_max;
}

// ---------------------------------------------------------------- CTF table

const CtfTable = struct {
    n_radii: usize,
    n_phi: usize,
    g_min: []f64,
    g_max: []f64,
    /// s[k * n_phi + j]: both branches of pi f / g at radius k, phi node j.
    s: []f64,

    fn alloc(allocator: std.mem.Allocator, n_radii: usize, n_phi: usize) !CtfTable {
        const g_min = try allocator.alloc(f64, n_radii);
        errdefer allocator.free(g_min);
        const g_max = try allocator.alloc(f64, n_radii);
        errdefer allocator.free(g_max);
        const s = try allocator.alloc(f64, n_radii * n_phi);
        @memset(s, 0);
        return .{ .n_radii = n_radii, .n_phi = n_phi, .g_min = g_min, .g_max = g_max, .s = s };
    }

    fn deinit(self: CtfTable, allocator: std.mem.Allocator) void {
        allocator.free(self.g_min);
        allocator.free(self.g_max);
        allocator.free(self.s);
    }
};

const Pt = struct {
    x: f64,
    y: f64,
    fn lessThan(_: void, l: Pt, r: Pt) bool {
        return l.x < r.x;
    }
};

/// Add one branch (traces start..end, cyclic, inclusive) resampled onto the
/// phi midpoint nodes to `out`.
fn addBranch(allocator: std.mem.Allocator, traces: anytype, start: usize, end: usize, out: []f64) !void {
    const n = traces.len;
    const len = (end + n - start) % n + 1;
    const pts_buf = try allocator.alloc(Pt, len);
    defer allocator.free(pts_buf);
    var pts: []Pt = pts_buf[0..0];
    for (0..len) |t| {
        const tr = traces[(start + t) % n];
        // kerrz's autodiff d r / d beta is inaccurate close to the alpha axis
        // (beta = 0): 38% low at 1e-6 rad, 3% at 1e-5, 0.16% at 1e-4, <1e-4
        // beyond 1e-3 (checked against finite differences). kerrz's extremum
        // search places traces there. Skip them.
        if (@abs(@sin(tr.image_angle)) < 1e-3) continue;
        pts = pts_buf[0 .. pts.len + 1];
        pts[pts.len - 1] = .{
            .x = std.math.acos(std.math.clamp(1 - 2 * tr.g_star, -1.0, 1.0)),
            .y = std.math.pi * tr.f / tr.g,
        };
    }
    const len_kept = pts.len;
    std.mem.sort(Pt, pts, {}, Pt.lessThan);
    // Merge near-coincident abscissae (kerrz's extremum search adds traces
    // ~1e-8 apart), which would otherwise make the Lagrange stencil unstable.
    var m: usize = 1;
    var count: f64 = 1;
    for (1..len_kept) |t| {
        if (pts[t].x - pts[m - 1].x > 1e-4) {
            pts[m] = pts[t];
            m += 1;
            count = 1;
        } else {
            pts[m - 1].x = (pts[m - 1].x * count + pts[t].x) / (count + 1);
            pts[m - 1].y = (pts[m - 1].y * count + pts[t].y) / (count + 1);
            count += 1;
        }
    }
    pts = pts[0..m];
    std.debug.assert(m >= 4);
    const n_phi = out.len;
    var xs: [4]f64 = undefined;
    var ws: [4]f64 = undefined;
    var p: usize = 0;
    for (out, 0..) |*o, j| {
        const phi = (@as(f64, @floatFromInt(j)) + 0.5) * std.math.pi / @as(f64, @floatFromInt(n_phi));
        while (p < m and pts[p].x <= phi) p += 1;
        const k0 = std.math.clamp(@as(isize, @intCast(p)) - 2, 0, @as(isize, @intCast(m - 4)));
        const k: usize = @intCast(k0);
        for (0..4) |q| xs[q] = pts[k + q].x;
        lagrange(&xs, phi, &ws);
        var y: f64 = 0;
        var y_lo: f64 = std.math.inf(f64);
        var y_hi: f64 = -std.math.inf(f64);
        for (0..4) |q| {
            y += ws[q] * pts[k + q].y;
            y_lo = @min(y_lo, pts[k + q].y);
            y_hi = @max(y_hi, pts[k + q].y);
        }
        // Safeguard: fall back to linear interpolation if the cubic
        // overshoots far beyond its stencil.
        const span = y_hi - y_lo;
        if (y < y_lo - 0.5 * span or y > y_hi + 0.5 * span) {
            const k_lo = std.math.clamp(p, 1, m - 1) - 1;
            const u = (phi - pts[k_lo].x) / (pts[k_lo + 1].x - pts[k_lo].x);
            y = pts[k_lo].y + u * (pts[k_lo + 1].y - pts[k_lo].y);
        }
        o.* += y;
    }
}

fn tableRadius(r_in: f64, n_radii: usize, k: usize) f64 {
    return r_in * std.math.pow(f64, r_break / r_in, @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(n_radii - 1)));
}

fn buildTable(allocator: std.mem.Allocator, a: f64, incl_deg: f64, n_radii: usize, n_phi: usize) !CtfTable {
    const metric = kerrz.KerrMetric(D0).init(.one, .promote(a));
    const x_obs: kerrz.FourVector(D0) = .{ .t = .zero, .r = .promote(observer_distance), .th = .promote(deg(incl_deg)), .ph = .zero };
    var table = Table.init(metric, x_obs);
    defer table.deinit(allocator);
    const r_in = kbb.disc.iscoRadius(D0, .promote(a)).x;
    const out = try CtfTable.alloc(allocator, n_radii, n_phi);
    errdefer out.deinit(allocator);
    for (0..n_radii) |k| {
        var tf = try table.calculateRadius(allocator, tableRadius(r_in, n_radii, k), .{});
        defer tf.deinit(allocator);
        fixExtrema(&tf);
        out.g_min[k] = tf.g_min;
        out.g_max[k] = tf.g_max;
        const row = out.s[k * n_phi .. (k + 1) * n_phi];
        try addBranch(allocator, tf.traces, tf.g_min_index, tf.g_max_index, row);
        try addBranch(allocator, tf.traces, tf.g_max_index, tf.g_min_index, row);
    }
    return out;
}

/// Weighted sum of tables (node interpolation in (a, i)).
fn combineTables(allocator: std.mem.Allocator, tables: []const CtfTable, coeffs: []const f64) !CtfTable {
    const out = try CtfTable.alloc(allocator, tables[0].n_radii, tables[0].n_phi);
    @memset(out.g_min, 0);
    @memset(out.g_max, 0);
    for (tables, coeffs) |t, c| {
        for (out.g_min, t.g_min) |*o, v| o.* += c * v;
        for (out.g_max, t.g_max) |*o, v| o.* += c * v;
        for (out.s, t.s) |*o, v| o.* += c * v;
    }
    return out;
}

/// Image-plane samples from a stored table: Gauss-Legendre in log r, midpoint
/// nodes in phi, plus the shared weak-field samples.
fn samplesFromTable(allocator: std.mem.Allocator, tab: CtfTable, r_in: f64, n_r: usize, outer: []const Sample) ![]Sample {
    const n_phi = tab.n_phi;
    const N = tab.n_radii;
    const samples = try allocator.alloc(Sample, n_r * n_phi + outer.len);
    const nodes = try allocator.alloc(f64, n_r);
    defer allocator.free(nodes);
    const weights = try allocator.alloc(f64, n_r);
    defer allocator.free(weights);
    kbb.quadrature.gaussLegendre(nodes, weights);
    const L = @log(r_break / r_in);
    const dphi = std.math.pi / @as(f64, @floatFromInt(n_phi));
    const cos_phi = try allocator.alloc(f64, n_phi);
    defer allocator.free(cos_phi);
    for (0..n_phi) |j| cos_phi[j] = @cos((@as(f64, @floatFromInt(j)) + 0.5) * dphi);
    var xs: [4]f64 = undefined;
    var ws: [4]f64 = undefined;
    for (0..n_r) |q| {
        const u = 0.5 * (nodes[q] + 1);
        const r = r_in * @exp(u * L);
        const dr = r * L * 0.5 * weights[q];
        const p = u * @as(f64, @floatFromInt(N - 1));
        const k0: usize = @intCast(std.math.clamp(@as(isize, @intFromFloat(@floor(p))) - 1, 0, @as(isize, @intCast(N - 4))));
        for (0..4) |i| xs[i] = @floatFromInt(k0 + i);
        lagrange(&xs, p, &ws);
        var g_min: f64 = 0;
        var g_max: f64 = 0;
        for (0..4) |i| {
            g_min += ws[i] * tab.g_min[k0 + i];
            g_max += ws[i] * tab.g_max[k0 + i];
        }
        for (0..n_phi) |j| {
            var s: f64 = 0;
            for (0..4) |i| s += ws[i] * tab.s[(k0 + i) * n_phi + j];
            samples[q * n_phi + j] = .{
                .weight = .promote(r * s * dr * dphi),
                .r = .promote(r),
                .g = .promote(g_min + 0.5 * (1 - cos_phi[j]) * (g_max - g_min)),
                .cos_em = .one,
            };
        }
    }
    @memcpy(samples[n_r * n_phi ..], outer);
    return samples;
}

// ------------------------------------------------------------------- B1

const B1Grid = struct { n_theta: usize, n_rho: usize, n_outer: usize };

fn imageOptions(g: B1Grid) kbb.image.Options {
    return .{ .n_theta = g.n_theta, .n_rho = g.n_rho, .n_outer = g.n_outer, .r_break = r_break };
}

fn traceB1(allocator: std.mem.Allocator, a: f64, incl_deg: f64, g: B1Grid) ![]Sample {
    const av: D0 = .promote(a);
    return kbb.image.traceImage(D0, allocator, av, .promote(deg(incl_deg)), kbb.disc.iscoRadius(D0, av), imageOptions(g));
}

fn combineSamples(allocator: std.mem.Allocator, sets: []const []Sample, coeffs: []const f64) ![]Sample {
    const out = try allocator.alloc(Sample, sets[0].len);
    for (out, 0..) |*o, k| {
        var w: f64 = 0;
        var r: f64 = 0;
        var g: f64 = 0;
        var c_em: f64 = 0;
        for (sets, coeffs) |s, c| {
            w += c * s[k].weight.x;
            r += c * s[k].r.x;
            g += c * s[k].g.x;
            c_em += c * s[k].cos_em.x;
        }
        o.* = .{ .weight = .promote(w), .r = .promote(r), .g = .promote(g), .cos_em = .promote(c_em) };
    }
    return out;
}

// -------------------------------------------------------------- spectrum

const Energies = struct { energies: []const f64, edges: []const f64 };

fn params0(a: f64, incl_deg: f64) kbb.spectrum.Params(D0) {
    return .{
        .eta = .zero,
        .a = .promote(a),
        .incl = .promote(deg(incl_deg)),
        .mass = .promote(10),
        .mdot = .promote(1),
        .distance = .promote(10),
        .fcol = .promote(1.7),
        .norm = .one,
    };
}

/// Spectrum from samples: returns the density on `energies` and the time for
/// building the emitters plus one 200-bin evaluation.
fn spectrumFrom(allocator: std.mem.Allocator, a: f64, incl_deg: f64, samples: []const Sample, en: Energies, density: []f64) !f64 {
    const Spec = kbb.spectrum.Spectrum(D0);
    var timer = try std.time.Timer.start();
    const spec = try Spec.fromSamples(D0, allocator, params0(a, incl_deg), .{}, samples, kbb.disc.iscoRadius(D0, .promote(a)), null);
    defer spec.deinit();
    const binned = try allocator.alloc(D0, en.edges.len - 1);
    defer allocator.free(binned);
    try spec.binned(en.edges, binned, 4);
    const ms = nowMs(&timer);
    for (density, en.energies) |*d, E| d.* = spec.density(E).x;
    return ms;
}

/// B1 with dual numbers in (a, i): d N / d a and d N / d i (per degree).
fn b1Derivatives(allocator: std.mem.Allocator, a: f64, incl_deg: f64, g: B1Grid, en: Energies, d_a: []f64, d_i: []f64) !void {
    const Spec = kbb.spectrum.Spectrum(D2);
    const p0 = params0(a, incl_deg);
    const p: kbb.spectrum.Params(D2) = .{
        .eta = .zero,
        .a = D2.promote(a).diff(0),
        .incl = D2.promote(deg(incl_deg)).diff(1),
        .mass = .promote(p0.mass.x),
        .mdot = .promote(p0.mdot.x),
        .distance = .promote(p0.distance.x),
        .fcol = .promote(p0.fcol.x),
        .norm = .one,
    };
    const spec = try Spec.init(allocator, p, .{ .image = imageOptions(g) });
    defer spec.deinit();
    for (en.energies, 0..) |E, k| {
        const v = spec.density(E);
        d_a[k] = v.dx[0];
        d_i[k] = v.dx[1] * std.math.pi / 180.0;
    }
}


// ------------------------------------------------------------------ main

const b1_grids = [_]B1Grid{
    .{ .n_theta = 32, .n_rho = 24, .n_outer = 16 },
    .{ .n_theta = 64, .n_rho = 48, .n_outer = 32 },
    .{ .n_theta = 128, .n_rho = 96, .n_outer = 48 },
    .{ .n_theta = 256, .n_rho = 192, .n_outer = 48 },
    .{ .n_theta = 512, .n_rho = 384, .n_outer = 48 },
};
/// Weak-field grid (r > r_break) shared by the CTF constructions; the same
/// one B1 uses at 64x48, where B1 is already converged to ~1e-7.
const outer_grid: B1Grid = .{ .n_theta = 64, .n_rho = 2, .n_outer = 32 };
const b1_default: B1Grid = .{ .n_theta = 128, .n_rho = 96, .n_outer = 48 };
const b1_reference: B1Grid = .{ .n_theta = 1024, .n_rho = 768, .n_outer = 64 };
const b1_derivative_reference: B1Grid = .{ .n_theta = 256, .n_rho = 192, .n_outer = 48 };

const CtfGrid = struct { n_radii: usize, n_phi: usize, n_r: usize };
const ctf_grids = [_]CtfGrid{
    .{ .n_radii = 16, .n_phi = 16, .n_r = 24 },
    .{ .n_radii = 24, .n_phi = 24, .n_r = 32 },
    .{ .n_radii = 32, .n_phi = 32, .n_r = 48 },
    .{ .n_radii = 48, .n_phi = 48, .n_r = 64 },
    .{ .n_radii = 64, .n_phi = 64, .n_r = 96 },
    .{ .n_radii = 96, .n_phi = 96, .n_r = 128 },
};
/// CTF resolution used for the (a, i) interpolation tests.
const ctf_tabulated: CtfGrid = .{ .n_radii = 64, .n_phi = 64, .n_r = 96 };
const spacing_factors = [_]f64{ 1, 0.5, 0.25 };
const offsets = [_]f64{ -1.5, -0.5, 0.5, 1.5 };

fn writeArray(w: *std.Io.Writer, name: []const u8, v: []const f64) !void {
    try w.print(", \"{s}\": ", .{name});
    try std.json.Stringify.value(v, .{}, w);
}

const Interp = struct {
    case: Case,
    hx: f64,
    hi: f64,
    order: usize,

    fn stencil(self: Interp) []const f64 {
        return if (self.order == 1) offsets[1..3] else offsets[0..];
    }
    fn nodeIndex(self: Interp, j: usize) usize {
        return if (self.order == 1) j + 1 else j;
    }
    /// Coefficients over the 4x4 node grid (zero outside the stencil) at the
    /// offset (dx, di) from the test point.
    fn coeffs(self: Interp, dx: f64, di: f64, out: *[16]f64) void {
        @memset(out, 0);
        const st = self.stencil();
        var wx: [4]f64 = undefined;
        var wi: [4]f64 = undefined;
        lagrange(st, dx / self.hx, wx[0..st.len]);
        lagrange(st, di / self.hi, wi[0..st.len]);
        for (0..st.len) |p| for (0..st.len) |q| {
            out[self.nodeIndex(p) * 4 + self.nodeIndex(q)] = wx[p] * wi[q];
        };
    }
};

const Method = enum { B1, CTF };

/// Density at the offset (dx, di) from the test point, from node data;
/// returns the per-evaluation time.
fn interpolated(
    allocator: std.mem.Allocator,
    method: Method,
    it: Interp,
    tables: []const CtfTable,
    b1_sets: []const []Sample,
    outer_sets: []const []Sample,
    dx: f64,
    di: f64,
    en: Energies,
    density: []f64,
) !f64 {
    const c = it.case;
    const a = fromCoord(c, toCoord(c, c.a) + dx);
    const incl = c.incl_deg + di;
    var cf: [16]f64 = undefined;
    it.coeffs(dx, di, &cf);
    var timer = try std.time.Timer.start();
    const samples = switch (method) {
        .CTF => blk: {
            const tab = try combineTables(allocator, tables, &cf);
            defer tab.deinit(allocator);
            const outer_full = try combineSamples(allocator, outer_sets, &cf);
            defer allocator.free(outer_full);
            break :blk try samplesFromTable(allocator, tab, kbb.disc.iscoRadius(D0, .promote(a)).x, ctf_tabulated.n_r, outer_full[outer_grid.n_theta * 2 ..]);
        },
        .B1 => try combineSamples(allocator, b1_sets, &cf),
    };
    defer allocator.free(samples);
    const t_interp = nowMs(&timer);
    const t_spec = try spectrumFrom(allocator, a, incl, samples, en, density);
    return t_interp + t_spec;
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    const path = if (args.len > 1) args[1] else "validation/ctf_compare.json";

    var energies: [320]f64 = undefined;
    for (&energies, 0..) |*e, k| e.* = 0.01 * std.math.pow(f64, 3e4, @as(f64, @floatFromInt(k)) / 319.0);
    var edges: [201]f64 = undefined;
    for (&edges, 0..) |*e, k| e.* = 0.1 * std.math.pow(f64, 500, @as(f64, @floatFromInt(k)) / 200.0);
    const en: Energies = .{ .energies = &energies, .edges = &edges };
    var density: [320]f64 = undefined;
    var d_a: [320]f64 = undefined;
    var d_i: [320]f64 = undefined;
    var dp: [320]f64 = undefined;
    var dm: [320]f64 = undefined;

    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    var buf: [1 << 16]u8 = undefined;
    var fw = file.writer(&buf);
    const w = &fw.interface;
    try w.writeAll("{\"energies\": ");
    try std.json.Stringify.value(energies, .{}, w);
    try w.writeAll(", \"cases\": [");

    for (cases, 0..) |c, ic| {
        if (ic > 0) try w.writeAll(",");
        try w.print("{{\"a\": {d}, \"incl_deg\": {d}", .{ c.a, c.incl_deg });
        std.debug.print("a={d:.6} i={d}\n", .{ c.a, c.incl_deg });

        // References.
        {
            const s = try traceB1(allocator, c.a, c.incl_deg, b1_reference);
            defer allocator.free(s);
            _ = try spectrumFrom(allocator, c.a, c.incl_deg, s, en, &density);
            try writeArray(w, "reference", &density);
        }
        try b1Derivatives(allocator, c.a, c.incl_deg, b1_derivative_reference, en, &d_a, &d_i);
        try writeArray(w, "reference_d_a", &d_a);
        try writeArray(w, "reference_d_i", &d_i);

        // Shared weak-field samples for the CTF construction.
        const outer_src = try traceB1(allocator, c.a, c.incl_deg, outer_grid);
        defer allocator.free(outer_src);
        const outer = outer_src[outer_grid.n_theta * 2 ..];

        // 1. At the exact (a, i).
        try w.writeAll(", \"exact\": [");
        for (b1_grids, 0..) |g, k| {
            var timer = try std.time.Timer.start();
            const s = try traceB1(allocator, c.a, c.incl_deg, g);
            defer allocator.free(s);
            const t_trace = nowMs(&timer);
            const t_spec = try spectrumFrom(allocator, c.a, c.incl_deg, s, en, &density);
            if (k > 0) try w.writeAll(",");
            try w.print("{{\"method\": \"B1\", \"label\": \"{d}x{d}\", \"samples\": {d}, \"build_ms\": {d}, \"eval_ms\": {d}", .{ g.n_theta, g.n_rho, s.len, t_trace, t_spec });
            try writeArray(w, "density", &density);
            if (g.n_theta == b1_default.n_theta) {
                try b1Derivatives(allocator, c.a, c.incl_deg, g, en, &d_a, &d_i);
                try writeArray(w, "d_a", &d_a);
                try writeArray(w, "d_i", &d_i);
            }
            try w.writeAll("}");
            std.debug.print("  B1 {d}x{d}: trace {d:.1} ms, spectrum {d:.1} ms\n", .{ g.n_theta, g.n_rho, t_trace, t_spec });
        }
        for (ctf_grids) |g| {
            var timer = try std.time.Timer.start();
            const tab = try buildTable(allocator, c.a, c.incl_deg, g.n_radii, g.n_phi);
            defer tab.deinit(allocator);
            const t_build = nowMs(&timer);
            timer.reset();
            const s = try samplesFromTable(allocator, tab, kbb.disc.iscoRadius(D0, .promote(c.a)).x, g.n_r, outer);
            defer allocator.free(s);
            const t_samples = nowMs(&timer);
            const t_spec = try spectrumFrom(allocator, c.a, c.incl_deg, s, en, &density);
            try w.print(",{{\"method\": \"CTF\", \"label\": \"{d}r {d}phi {d}GL\", \"samples\": {d}, \"build_ms\": {d}, \"eval_ms\": {d}", .{ g.n_radii, g.n_phi, g.n_r, s.len, t_build, t_samples + t_spec });
            try writeArray(w, "density", &density);
            try w.writeAll("}");
            std.debug.print("  CTF {d}r {d}phi {d}GL: build {d:.1} ms, eval {d:.1} ms\n", .{ g.n_radii, g.n_phi, g.n_r, t_build, t_samples + t_spec });
        }
        try w.writeAll("]");

        // 2. Stored on an (a, i) grid, evaluated mid-cell.
        try w.writeAll(", \"tabulated\": [");
        var first = true;
        for (spacing_factors) |factor| {
            const hx = spinStep(c) * factor;
            const hi = kerrbb_step_incl * factor;
            var tables: [16]CtfTable = undefined;
            var b1_sets: [16][]Sample = undefined;
            var outer_sets: [16][]Sample = undefined;
            var t_ctf: f64 = 0;
            var t_b1: f64 = 0;
            for (0..4) |p| for (0..4) |q| {
                const a = fromCoord(c, toCoord(c, c.a) + offsets[p] * hx);
                const incl = c.incl_deg + offsets[q] * hi;
                var timer = try std.time.Timer.start();
                tables[p * 4 + q] = try buildTable(allocator, a, incl, ctf_tabulated.n_radii, ctf_tabulated.n_phi);
                t_ctf += nowMs(&timer);
                timer.reset();
                b1_sets[p * 4 + q] = try traceB1(allocator, a, incl, b1_default);
                t_b1 += nowMs(&timer);
                timer.reset();
                outer_sets[p * 4 + q] = try traceB1(allocator, a, incl, outer_grid);
                t_ctf += nowMs(&timer);
            };
            defer for (0..16) |k| {
                tables[k].deinit(allocator);
                allocator.free(b1_sets[k]);
                allocator.free(outer_sets[k]);
            };
            for ([_]Method{ .B1, .CTF }) |method| for ([_]usize{ 1, 3 }) |order| {
                const it: Interp = .{ .case = c, .hx = hx, .hi = hi, .order = order };
                const t_eval = try interpolated(allocator, method, it, &tables, &b1_sets, &outer_sets, 0, 0, en, &density);
                // Derivatives of the interpolant by central differences.
                const ex = 1e-3 * hx;
                _ = try interpolated(allocator, method, it, &tables, &b1_sets, &outer_sets, ex, 0, en, &dp);
                _ = try interpolated(allocator, method, it, &tables, &b1_sets, &outer_sets, -ex, 0, en, &dm);
                const da_dx = (fromCoord(c, toCoord(c, c.a) + ex) - fromCoord(c, toCoord(c, c.a) - ex)) / (2 * ex);
                for (&d_a, dp, dm) |*d, p_, m_| d.* = (p_ - m_) / (2 * ex) / da_dx;
                const ei = 1e-3 * hi;
                _ = try interpolated(allocator, method, it, &tables, &b1_sets, &outer_sets, 0, ei, en, &dp);
                _ = try interpolated(allocator, method, it, &tables, &b1_sets, &outer_sets, 0, -ei, en, &dm);
                for (&d_i, dp, dm) |*d, p_, m_| d.* = (p_ - m_) / (2 * ei);
                if (!first) try w.writeAll(",");
                first = false;
                try w.print("{{\"method\": \"{s}\", \"order\": {d}, \"factor\": {d}, \"spin_step\": {d}, \"incl_step\": {d}, \"build_ms_per_node\": {d}, \"eval_ms\": {d}", .{
                    @tagName(method), order, factor, hx, hi, (if (method == .CTF) t_ctf else t_b1) / 16.0, t_eval,
                });
                try writeArray(w, "density", &density);
                try writeArray(w, "d_a", &d_a);
                try writeArray(w, "d_i", &d_i);
                try w.writeAll("}");
                std.debug.print("  tabulated {s} order {d} factor {d}: eval {d:.1} ms, build/node {d:.1} ms\n", .{ @tagName(method), order, factor, t_eval, (if (method == .CTF) t_ctf else t_b1) / 16.0 });
            };
        }
        try w.writeAll("]");

        // Maximum blueshift: CTF g_max(r) against the default B1 samples.
        try w.writeAll(", \"gmax\": ");
        try writeGmax(allocator, w, c);
        try w.writeAll("}");
        try w.flush();
    }
    try w.writeAll("]}\n");
    try w.flush();
    std.debug.print("wrote {s}\n", .{path});
}

fn writeGmax(allocator: std.mem.Allocator, w: *std.Io.Writer, c: Case) !void {
    const a: D0 = .promote(c.a);
    const metric = kerrz.KerrMetric(D0).init(.one, a);
    const x_obs: kerrz.FourVector(D0) = .{ .t = .zero, .r = .promote(1e8), .th = .promote(std.math.degreesToRadians(c.incl_deg)), .ph = .zero };
    var table = Table.init(metric, x_obs);
    defer table.deinit(allocator);
    const r_in = metric.isco.x;
    const n = 60;
    var radii: [n]f64 = undefined;
    var g_ctf: [n]f64 = undefined;
    var g_fix: [n]f64 = undefined;
    for (0..n) |k| {
        radii[k] = r_in * std.math.pow(f64, 100.0 / r_in, @as(f64, @floatFromInt(k)) / (n - 1));
        const tf = try table.calculateRadiusAndAppend(allocator, radii[k], .{});
        g_ctf[k] = tf.g_max;
        fixExtrema(tf);
        g_fix[k] = tf.g_max;
    }
    // Largest g of the default B1 samples, per image angle, as (r, g) points.
    const b_samples = try traceB1(allocator, c.a, c.incl_deg, b1_default);
    defer allocator.free(b_samples);
    var pts_r: std.ArrayList(f64) = .empty;
    defer pts_r.deinit(allocator);
    var pts_g: std.ArrayList(f64) = .empty;
    defer pts_g.deinit(allocator);
    for (b_samples[0 .. 128 * 96]) |s| {
        if (s.weight.x == 0 or s.r.x > 100) continue;
        try pts_r.append(allocator, s.r.x);
        try pts_g.append(allocator, s.g.x);
    }
    try w.writeAll("{\"radii\": ");
    try std.json.Stringify.value(radii, .{}, w);
    try w.writeAll(", \"g_max_ctf\": ");
    try std.json.Stringify.value(g_ctf, .{}, w);
    try w.writeAll(", \"g_max_ctf_corrected\": ");
    try std.json.Stringify.value(g_fix, .{}, w);
    try w.writeAll(", \"b1_r\": ");
    try std.json.Stringify.value(pts_r.items, .{}, w);
    try w.writeAll(", \"b1_g\": ");
    try std.json.Stringify.value(pts_g.items, .{}, w);
    try w.writeAll("}");
}

