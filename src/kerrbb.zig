//! Top-level model interface: the photon spectrum per energy bin and its
//! Jacobian with respect to any subset of the parameters.
//!
//! This is the entry point the C ABI and language bindings wrap. Parameters
//! follow XSPEC kerrbb (inclination in degrees, Mdot_eff in 1e18 g/s, D in
//! kpc), plus an optional inner radius.

const std = @import("std");
const kerrz = @import("kerrz");
const spectrum = @import("spectrum.zig");
const image = @import("image.zig");

pub const Options = spectrum.Options;
pub const Error = spectrum.Error || error{ JacobianSizeMismatch, FreeInnerRadiusWithoutValue };

/// The model parameters, in the order of the Jacobian columns.
pub const Parameter = enum {
    eta,
    a,
    incl,
    mass,
    mdot,
    distance,
    fcol,
    norm,
    r_in,
};

pub const FreeSet = std.EnumSet(Parameter);

pub const Params = struct {
    /// Torque parameter, >= 0.
    eta: f64 = 0,
    /// Signed spin, 1e-3 <= |a| <= 0.9999.
    a: f64,
    /// Inclination in degrees, 0 < i <= 89.
    incl: f64,
    /// Black hole mass in M_sun.
    mass: f64,
    /// Effective accretion rate in 1e18 g/s.
    mdot: f64,
    /// Distance in kpc.
    distance: f64,
    /// Spectral hardening factor.
    fcol: f64 = 1.7,
    norm: f64 = 1,
    /// Inner radius in r_g; null means the marginally stable orbit.
    r_in: ?f64 = null,
    limb_darkening: bool = false,
    returning_radiation: bool = false,

    fn get(self: Params, p: Parameter) f64 {
        return switch (p) {
            .eta => self.eta,
            .a => self.a,
            .incl => self.incl,
            .mass => self.mass,
            .mdot => self.mdot,
            .distance => self.distance,
            .fcol => self.fcol,
            .norm => self.norm,
            .r_in => self.r_in.?,
        };
    }
};

/// Evaluate the photon flux per bin (photons cm^-2 s^-1) for `edges` in keV.
///
/// If `jacobian` is given it must have length `flux.len * free.count()` and
/// receives d flux[b] / d p_k in row-major order (bin b, then free parameter
/// k in `Parameter` order). The inclination derivative is per degree.
pub fn evaluate(
    allocator: std.mem.Allocator,
    params: Params,
    free: FreeSet,
    edges: []const f64,
    flux: []f64,
    jacobian: ?[]f64,
    opts: Options,
) Error!void {
    return evaluateCached(allocator, params, free, edges, flux, jacobian, opts, null);
}

fn evaluateCached(
    allocator: std.mem.Allocator,
    params: Params,
    free: FreeSet,
    edges: []const f64,
    flux: []f64,
    jacobian: ?[]f64,
    opts: Options,
    cache: ?*spectrum.Cache,
) Error!void {
    std.debug.assert(flux.len + 1 == edges.len);
    if (free.contains(.r_in) and params.r_in == null) return Error.FreeInnerRadiusWithoutValue;
    const n_free = if (jacobian != null) free.count() else 0;
    if (jacobian) |j| if (j.len != flux.len * n_free) return Error.JacobianSizeMismatch;
    inline for (0..@typeInfo(Parameter).@"enum".fields.len + 1) |n| {
        if (n == n_free) return evaluateN(n, allocator, params, free, edges, flux, jacobian, opts, cache);
    }
    unreachable;
}

/// A model instance that caches the ray tracing between calls. The image
/// samples depend only on (a, i, r_in) and the returning-radiation kernel only
/// on (a, r_in, lflag), so fits that vary M, Mdot, D, fcol, eta or norm reuse
/// them. Not thread-safe; use one instance per thread (each call can itself
/// use `options.n_threads` threads).
pub const Model = struct {
    allocator: std.mem.Allocator,
    options: Options,
    cache: spectrum.Cache,

    pub fn init(allocator: std.mem.Allocator, options: Options) Model {
        return .{ .allocator = allocator, .options = options, .cache = .init(allocator) };
    }

    pub fn deinit(self: *Model) void {
        self.cache.deinit();
    }

    /// As `evaluate`, with this model's options and cache.
    pub fn evaluate(
        self: *Model,
        params: Params,
        free: FreeSet,
        edges: []const f64,
        flux: []f64,
        jacobian: ?[]f64,
    ) Error!void {
        return evaluateCached(self.allocator, params, free, edges, flux, jacobian, self.options, &self.cache);
    }
};

fn evaluateN(
    comptime N: usize,
    allocator: std.mem.Allocator,
    params: Params,
    free: FreeSet,
    edges: []const f64,
    flux: []f64,
    jacobian: ?[]f64,
    opts: Options,
    cache: ?*spectrum.Cache,
) Error!void {
    const T = kerrz.DualNumber(f64, N);

    // Seed one derivative slot per free parameter, in enum order.
    var values: std.EnumArray(Parameter, T) = undefined;
    var slot: usize = 0;
    for (std.enums.values(Parameter)) |p| {
        var v: T = .promote(if (p == .r_in and params.r_in == null) 0 else params.get(p));
        if (N > 0 and free.contains(p)) {
            v.dx[slot] = 1;
            slot += 1;
        }
        values.set(p, v);
    }

    const deg: T = .promote(std.math.pi / 180.0);
    const p: spectrum.Params(T) = .{
        .eta = values.get(.eta),
        .a = values.get(.a),
        .incl = T.Algebra.mult(values.get(.incl), deg),
        .mass = values.get(.mass),
        .mdot = values.get(.mdot),
        .distance = values.get(.distance),
        .fcol = values.get(.fcol),
        .norm = values.get(.norm),
        .r_in = if (params.r_in != null) values.get(.r_in) else null,
        .limb_darkening = params.limb_darkening,
        .returning_radiation = params.returning_radiation,
    };

    const out = try allocator.alloc(T, flux.len);
    defer allocator.free(out);
    const spec = try spectrum.Spectrum(T).initCached(allocator, p, opts, cache);
    defer spec.deinit();
    try spec.binned(edges, out, opts.n_energy);

    for (out, 0..) |v, b| {
        flux[b] = v.x;
        if (jacobian) |j| {
            inline for (0..N) |k| j[b * N + k] = v.dx[k];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests

const testing = std.testing;
const test_opts: Options = .{ .image = .{ .n_theta = 64, .n_rho = 48, .n_outer = 24 } };
const test_edges = [_]f64{ 0.2, 0.7, 2.0, 5.0, 12.0 };

fn baseParams() Params {
    return .{ .eta = 0.2, .a = 0.8, .incl = 55, .mass = 8, .mdot = 1.5, .distance = 6, .fcol = 1.6, .norm = 1.1, .r_in = 3.5 };
}

test "value-only evaluation matches the spectrum module" {
    var p = baseParams();
    p.r_in = null;
    var flux: [4]f64 = undefined;
    try evaluate(testing.allocator, p, .initEmpty(), &test_edges, &flux, null, test_opts);

    const D0 = kerrz.DualNumber(f64, 0);
    var ref: [4]D0 = undefined;
    try spectrum.photonFlux(D0, testing.allocator, .{
        .eta = .promote(p.eta), .a = .promote(p.a), .incl = .promote(std.math.degreesToRadians(p.incl)),
        .mass = .promote(p.mass), .mdot = .promote(p.mdot), .distance = .promote(p.distance),
        .fcol = .promote(p.fcol), .norm = .promote(p.norm),
    }, &test_edges, &ref, test_opts);
    for (flux, ref) |f, r| try testing.expectApproxEqRel(r.x, f, 1e-14);
}

test "full Jacobian matches finite differences" {
    const p = baseParams();
    const free = FreeSet.initFull();
    const n_free = free.count();
    var flux: [4]f64 = undefined;
    var jac: [4 * 9]f64 = undefined;
    try evaluate(testing.allocator, p, free, &test_edges, &flux, &jac, test_opts);

    var k: usize = 0;
    for (std.enums.values(Parameter)) |par| {
        // kerrz traces carry ~1e-10 noise; geometric parameters need larger steps.
        const rel: f64 = switch (par) {
            .a, .incl, .r_in => 3e-4,
            else => 1e-5,
        };
        const h = rel * @max(1.0, @abs(p.get(par)));
        var pp = p;
        var pm = p;
        setParam(&pp, par, p.get(par) + h);
        setParam(&pm, par, p.get(par) - h);
        var fp: [4]f64 = undefined;
        var fm: [4]f64 = undefined;
        try evaluate(testing.allocator, pp, .initEmpty(), &test_edges, &fp, null, test_opts);
        try evaluate(testing.allocator, pm, .initEmpty(), &test_edges, &fm, null, test_opts);
        for (0..4) |b| {
            const fd = (fp[b] - fm[b]) / (2 * h);
            const tol = 2e-5 * @max(@abs(fd), 1e-3 * flux[b]);
            testing.expectApproxEqAbs(fd, jac[b * n_free + k], tol) catch |err| {
                std.debug.print("parameter {s}, bin {d}: dual {e} fd {e}\n", .{ @tagName(par), b, jac[b * n_free + k], fd });
                return err;
            };
        }
        k += 1;
    }
}

fn setParam(p: *Params, par: Parameter, v: f64) void {
    switch (par) {
        .eta => p.eta = v,
        .a => p.a = v,
        .incl => p.incl = v,
        .mass => p.mass = v,
        .mdot => p.mdot = v,
        .distance => p.distance = v,
        .fcol => p.fcol = v,
        .norm => p.norm = v,
        .r_in => p.r_in = v,
    }
}

test "Jacobian with returning radiation matches finite differences" {
    var p = baseParams();
    p.returning_radiation = true;
    p.limb_darkening = true;
    const opts: Options = .{ .image = test_opts.image, .returning = .{ .n_radii = 24, .n_psi = 24, .n_chi = 12 } };
    const free = FreeSet.initMany(&.{ .eta, .a, .incl, .mdot, .r_in });
    var flux: [4]f64 = undefined;
    var jac: [4 * 5]f64 = undefined;
    try evaluate(testing.allocator, p, free, &test_edges, &flux, &jac, opts);

    // With the returning radiation switched off the flux must be lower.
    var p0 = p;
    p0.returning_radiation = false;
    var flux0: [4]f64 = undefined;
    try evaluate(testing.allocator, p0, .initEmpty(), &test_edges, &flux0, null, opts);
    for (flux, flux0) |f, f0| try testing.expect(f > f0);

    var it = free.iterator();
    var k: usize = 0;
    while (it.next()) |par| : (k += 1) {
        const rel: f64 = switch (par) {
            .a, .incl, .r_in => 3e-4,
            else => 1e-5,
        };
        const h = rel * @max(1.0, @abs(p.get(par)));
        var pp = p;
        var pm = p;
        setParam(&pp, par, p.get(par) + h);
        setParam(&pm, par, p.get(par) - h);
        var fp: [4]f64 = undefined;
        var fm: [4]f64 = undefined;
        try evaluate(testing.allocator, pp, .initEmpty(), &test_edges, &fp, null, opts);
        try evaluate(testing.allocator, pm, .initEmpty(), &test_edges, &fm, null, opts);
        for (0..4) |b| {
            const fd = (fp[b] - fm[b]) / (2 * h);
            const tol = 1e-4 * @max(@abs(fd), 1e-2 * flux[b]);
            testing.expectApproxEqAbs(fd, jac[b * 5 + k], tol) catch |err| {
                std.debug.print("returning: parameter {s}, bin {d}: dual {e} fd {e}\n", .{ @tagName(par), b, jac[b * 5 + k], fd });
                return err;
            };
        }
    }
}

test "spin derivative with returning radiation and r_in at the ISCO" {
    var p = baseParams();
    p.r_in = null;
    p.a = 0.95;
    p.returning_radiation = true;
    const opts: Options = .{ .image = test_opts.image, .returning = .{ .n_radii = 24, .n_psi = 24, .n_chi = 12 } };
    var flux: [4]f64 = undefined;
    var jac: [4]f64 = undefined;
    try evaluate(testing.allocator, p, FreeSet.initOne(.a), &test_edges, &flux, &jac, opts);
    const h = 3e-4;
    var pp = p;
    var pm = p;
    pp.a += h;
    pm.a -= h;
    var fp: [4]f64 = undefined;
    var fm: [4]f64 = undefined;
    try evaluate(testing.allocator, pp, .initEmpty(), &test_edges, &fp, null, opts);
    try evaluate(testing.allocator, pm, .initEmpty(), &test_edges, &fm, null, opts);
    for (0..4) |b| {
        const fd = (fp[b] - fm[b]) / (2 * h);
        // The discretised kernel is not exactly smooth in a: rays switch
        // between landing inside and outside r_in, or being captured, away
        // from the resolved inner edge. Finite differences with h and h/2
        // differ by ~1e-3, which sets the tolerance here.
        testing.expectApproxEqRel(fd, jac[b], 5e-3) catch |err| {
            std.debug.print("bin {d}: dual {e} fd {e}\n", .{ b, jac[b], fd });
            return err;
        };
    }
}

test "Jacobian columns do not depend on which other parameters are free" {
    const p = baseParams();
    var flux: [4]f64 = undefined;
    var full: [4 * 9]f64 = undefined;
    try evaluate(testing.allocator, p, .initFull(), &test_edges, &flux, &full, test_opts);
    const subset = FreeSet.initMany(&.{ .a, .fcol, .r_in });
    var part: [4 * 3]f64 = undefined;
    try evaluate(testing.allocator, p, subset, &test_edges, &flux, &part, test_opts);
    const cols = [_]usize{ @intFromEnum(Parameter.a), @intFromEnum(Parameter.fcol), @intFromEnum(Parameter.r_in) };
    for (0..4) |b| for (cols, 0..) |c, k| {
        try testing.expectApproxEqRel(full[b * 9 + c], part[b * 3 + k], 1e-12);
    };
}

test "cached model reuses the ray tracing and gives identical results" {
    var p = baseParams();
    p.returning_radiation = true;
    const opts: Options = .{ .image = test_opts.image, .returning = .{ .n_radii = 16, .n_psi = 16, .n_chi = 8 }, .n_threads = 3 };
    var model = Model.init(testing.allocator, opts);
    defer model.deinit();
    const free = FreeSet.initMany(&.{ .a, .mdot });
    var flux: [4]f64 = undefined;
    var jac: [8]f64 = undefined;
    var ref: [4]f64 = undefined;
    var ref_jac: [8]f64 = undefined;
    for ([_]f64{ 1.5, 2.0, 2.5 }) |mdot| {
        p.mdot = mdot;
        try model.evaluate(p, free, &test_edges, &flux, &jac);
        var serial = opts;
        serial.n_threads = 1;
        try evaluate(testing.allocator, p, free, &test_edges, &ref, &ref_jac, serial);
        try testing.expectEqualSlices(f64, &ref, &flux);
        try testing.expectEqualSlices(f64, &ref_jac, &jac);
    }
    try testing.expectEqual(@as(usize, 2), model.cache.geometry_hits);
    try testing.expectEqual(@as(usize, 2), model.cache.kernel_hits);
    // A value-only call uses the plain-number cache entries.
    try model.evaluate(p, .initEmpty(), &test_edges, &flux, null);
    try testing.expectApproxEqRel(ref[0], flux[0], 1e-12);
    p.a = 0.7;
    try model.evaluate(p, free, &test_edges, &flux, &jac);
    try testing.expectEqual(@as(usize, 2), model.cache.geometry_hits);
}

test "input validation" {
    var p = baseParams();
    p.r_in = null;
    var flux: [4]f64 = undefined;
    var jac: [4]f64 = undefined;
    try testing.expectError(Error.FreeInnerRadiusWithoutValue, evaluate(testing.allocator, p, FreeSet.initOne(.r_in), &test_edges, &flux, &jac, test_opts));
    try testing.expectError(Error.JacobianSizeMismatch, evaluate(testing.allocator, p, FreeSet.initMany(&.{ .a, .incl }), &test_edges, &flux, &jac, test_opts));
    p.a = 0.0;
    try testing.expectError(image.Error.SpinOutOfRange, evaluate(testing.allocator, p, .initEmpty(), &test_edges, &flux, null, test_opts));
    p.a = 0.5;
    p.r_in = 1.0;
    try testing.expectError(spectrum.Error.InnerRadiusBelowIsco, evaluate(testing.allocator, p, .initEmpty(), &test_edges, &flux, null, test_opts));
}
