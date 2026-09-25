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
    std.debug.assert(flux.len + 1 == edges.len);
    if (free.contains(.r_in) and params.r_in == null) return Error.FreeInnerRadiusWithoutValue;
    const n_free = if (jacobian != null) free.count() else 0;
    if (jacobian) |j| if (j.len != flux.len * n_free) return Error.JacobianSizeMismatch;

    inline for (0..@typeInfo(Parameter).@"enum".fields.len + 1) |n| {
        if (n == n_free) return evaluateN(n, allocator, params, free, edges, flux, jacobian, opts);
    }
    unreachable;
}

fn evaluateN(
    comptime N: usize,
    allocator: std.mem.Allocator,
    params: Params,
    free: FreeSet,
    edges: []const f64,
    flux: []f64,
    jacobian: ?[]f64,
    opts: Options,
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
    const spec = try spectrum.Spectrum(T).init(allocator, p, opts.image);
    defer spec.deinit();
    spec.binned(edges, out, opts.n_energy);

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
