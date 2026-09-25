//! C ABI for kerrzbb. See include/kerrzbb.h for the documented interface.

const std = @import("std");
const kerrbb = @import("kerrbb.zig");
const image = @import("image.zig");

const Parameter = kerrbb.Parameter;
const n_parameters = @typeInfo(Parameter).@"enum".fields.len;

pub const Params = extern struct {
    eta: f64,
    a: f64,
    incl: f64,
    mass: f64,
    mdot: f64,
    distance: f64,
    fcol: f64,
    norm: f64,
    r_in: f64,
    use_r_in: c_int,
    limb_darkening: c_int,
    returning_radiation: c_int,
};

pub const Options = extern struct {
    n_theta: usize,
    n_rho: usize,
    n_outer: usize,
    n_energy: usize,
    r_break: f64,
    r_out: f64,
    observer_distance: f64,
    n_radii: usize,
    n_psi: usize,
    n_chi: usize,
    r_max: f64,
    n_threads: usize,
    energy_grid: c_int,
    grid_step: f64,
};

pub const Status = enum(c_int) {
    success = 0,
    out_of_memory = 1,
    spin_out_of_range = 2,
    inclination_out_of_range = 3,
    inner_radius_below_isco = 4,
    invalid_parameter = 5,
    free_inner_radius_without_value = 6,
    /// Reserved (formerly: returning radiation not implemented).
    reserved_7 = 7,
    contour_failed = 8,
    invalid_argument = 9,
    singular_system = 10,
};

fn statusFromError(err: kerrbb.Error) Status {
    return switch (err) {
        error.OutOfMemory => .out_of_memory,
        error.SpinOutOfRange => .spin_out_of_range,
        error.InclinationOutOfRange => .inclination_out_of_range,
        error.InnerRadiusBelowIsco => .inner_radius_below_isco,
        error.InvalidParameter => .invalid_parameter,
        error.FreeInnerRadiusWithoutValue => .free_inner_radius_without_value,
        error.SingularSystem => .singular_system,
        error.ContourNotBracketed, error.ContourNotConverged => .contour_failed,
        error.JacobianSizeMismatch => .invalid_argument,
    };
}

fn toOptions(o: Options) kerrbb.Options {
    return .{
        .image = .{
            .n_theta = o.n_theta,
            .n_rho = o.n_rho,
            .n_outer = o.n_outer,
            .r_break = o.r_break,
            .r_out = o.r_out,
            .observer_distance = o.observer_distance,
        },
        .returning = .{
            .n_radii = o.n_radii,
            .n_psi = o.n_psi,
            .n_chi = o.n_chi,
            .r_max = o.r_max,
        },
        .n_energy = o.n_energy,
        .n_threads = o.n_threads,
        .energy_grid = o.energy_grid != 0,
        .grid_step = o.grid_step,
    };
}

fn validOptions(o: Options) bool {
    return o.n_theta >= 4 and o.n_rho >= 2 and o.n_outer >= 2 and o.n_energy >= 1 and
        o.n_energy <= 32 and o.n_radii >= 3 and o.n_psi >= 2 and o.n_chi >= 1 and
        o.r_break > 0 and o.r_out > o.r_break and o.observer_distance > o.r_out and
        o.r_max > 0 and o.grid_step > 0;
}

export fn kzbb_default_options() Options {
    const d: kerrbb.Options = .{};
    return .{
        .n_theta = d.image.n_theta,
        .n_rho = d.image.n_rho,
        .n_outer = d.image.n_outer,
        .n_energy = d.n_energy,
        .r_break = d.image.r_break,
        .r_out = d.image.r_out,
        .observer_distance = d.image.observer_distance,
        .n_radii = d.returning.n_radii,
        .n_psi = d.returning.n_psi,
        .n_chi = d.returning.n_chi,
        .r_max = d.returning.r_max,
        .n_threads = d.n_threads,
        .energy_grid = @intFromBool(d.energy_grid),
        .grid_step = d.grid_step,
    };
}

export fn kzbb_free_count(free_mask: u32) c_int {
    return @popCount(free_mask & ((1 << n_parameters) - 1));
}

const Prepared = struct {
    params: kerrbb.Params,
    free: kerrbb.FreeSet,
    edges: []const f64,
    flux: []f64,
    jacobian: ?[]f64,
};

fn prepare(
    params: ?*const Params,
    free_mask: u32,
    edges: ?[*]const f64,
    n_bins: usize,
    flux: ?[*]f64,
    jacobian: ?[*]f64,
) ?Prepared {
    const p = params orelse return null;
    const e = edges orelse return null;
    const f = flux orelse return null;
    if (n_bins == 0) return null;
    if (free_mask >> n_parameters != 0) return null;
    var free: kerrbb.FreeSet = .initEmpty();
    for (std.enums.values(Parameter)) |par| {
        if (free_mask & (@as(u32, 1) << @intCast(@intFromEnum(par))) != 0) free.insert(par);
    }
    const n_free = free.count();
    if (n_free > 0 and jacobian == null) return null;
    return .{
        .params = .{
            .eta = p.eta,
            .a = p.a,
            .incl = p.incl,
            .mass = p.mass,
            .mdot = p.mdot,
            .distance = p.distance,
            .fcol = p.fcol,
            .norm = p.norm,
            .r_in = if (p.use_r_in != 0) p.r_in else null,
            .limb_darkening = p.limb_darkening != 0,
            .returning_radiation = p.returning_radiation != 0,
        },
        .free = free,
        .edges = e[0 .. n_bins + 1],
        .flux = f[0..n_bins],
        .jacobian = if (n_free > 0) jacobian.?[0 .. n_bins * n_free] else null,
    };
}

export fn kzbb_evaluate(
    params: ?*const Params,
    free_mask: u32,
    edges: ?[*]const f64,
    n_bins: usize,
    flux: ?[*]f64,
    jacobian: ?[*]f64,
    options: ?*const Options,
) c_int {
    const q = prepare(params, free_mask, edges, n_bins, flux, jacobian) orelse
        return @intFromEnum(Status.invalid_argument);
    if (options) |o| if (!validOptions(o.*)) return @intFromEnum(Status.invalid_argument);
    const opts = if (options) |o| toOptions(o.*) else kerrbb.Options{};
    kerrbb.evaluate(std.heap.c_allocator, q.params, q.free, q.edges, q.flux, q.jacobian, opts) catch |err|
        return @intFromEnum(statusFromError(err));
    return @intFromEnum(Status.success);
}

/// Opaque model handle with a cache of the ray tracing.
pub const Model = kerrbb.Model;

export fn kzbb_model_create(options: ?*const Options) ?*Model {
    if (options) |o| if (!validOptions(o.*)) return null;
    const opts = if (options) |o| toOptions(o.*) else kerrbb.Options{};
    const m = std.heap.c_allocator.create(Model) catch return null;
    m.* = .init(std.heap.c_allocator, opts);
    return m;
}

export fn kzbb_model_destroy(model: ?*Model) void {
    const m = model orelse return;
    m.deinit();
    std.heap.c_allocator.destroy(m);
}

export fn kzbb_model_evaluate(
    model: ?*Model,
    params: ?*const Params,
    free_mask: u32,
    edges: ?[*]const f64,
    n_bins: usize,
    flux: ?[*]f64,
    jacobian: ?[*]f64,
) c_int {
    const m = model orelse return @intFromEnum(Status.invalid_argument);
    const q = prepare(params, free_mask, edges, n_bins, flux, jacobian) orelse
        return @intFromEnum(Status.invalid_argument);
    m.evaluate(q.params, q.free, q.edges, q.flux, q.jacobian) catch |err|
        return @intFromEnum(statusFromError(err));
    return @intFromEnum(Status.success);
}

export fn kzbb_status_string(code: c_int) [*:0]const u8 {
    const status = std.meta.intToEnum(Status, code) catch return "unknown status code";
    return switch (status) {
        .success => "success",
        .out_of_memory => "out of memory",
        .spin_out_of_range => "spin outside 1e-3 <= |a| <= 0.9999",
        .inclination_out_of_range => "inclination outside 0 < i <= 89 degrees",
        .inner_radius_below_isco => "r_in is below the marginally stable orbit",
        .invalid_parameter => "invalid parameter (eta < 0, or a non-positive mass, accretion rate, distance or fcol)",
        .free_inner_radius_without_value => "r_in is free but use_r_in is 0",
        .reserved_7 => "unused status code",
        .contour_failed => "ray tracing failed to locate a disc contour (usually |a| < ~0.01 near edge-on)",
        .invalid_argument => "invalid argument (null pointer, empty energy grid, or bad free mask)",
        .singular_system => "the returning-radiation linear system is singular",
    };
}

export fn kzbb_version() [*:0]const u8 {
    return "0.0.1";
}

// ---------------------------------------------------------------------------
// Tests

const testing = std.testing;

test "C API evaluates and reports errors" {
    var p: Params = .{
        .eta = 0, .a = 0.9, .incl = 45, .mass = 10, .mdot = 1, .distance = 10,
        .fcol = 1.7, .norm = 1, .r_in = 0, .use_r_in = 0, .limb_darkening = 0, .returning_radiation = 0,
    };
    var opts = kzbb_default_options();
    opts.n_theta = 48;
    opts.n_rho = 32;
    opts.n_outer = 16;
    const edges = [_]f64{ 0.5, 1.0, 5.0 };
    var flux: [2]f64 = undefined;
    var jac: [4]f64 = undefined;
    const mask: u32 = (1 << @intFromEnum(Parameter.a)) | (1 << @intFromEnum(Parameter.fcol));
    try testing.expectEqual(@as(c_int, 2), kzbb_free_count(mask));
    try testing.expectEqual(@as(c_int, 0), kzbb_evaluate(&p, mask, &edges, 2, &flux, &jac, &opts));
    try testing.expect(flux[0] > 0 and flux[1] > 0);

    var ref: [2]f64 = undefined;
    var ref_jac: [4]f64 = undefined;
    try kerrbb.evaluate(testing.allocator, .{ .a = 0.9, .incl = 45, .mass = 10, .mdot = 1, .distance = 10 }, kerrbb.FreeSet.initMany(&.{ .a, .fcol }), &edges, &ref, &ref_jac, toOptions(opts));
    try testing.expectEqualSlices(f64, &ref, &flux);
    try testing.expectEqualSlices(f64, &ref_jac, &jac);

    const model = kzbb_model_create(&opts) orelse return error.ModelCreateFailed;
    defer kzbb_model_destroy(model);
    var flux_m: [2]f64 = undefined;
    var jac_m: [4]f64 = undefined;
    try testing.expectEqual(@as(c_int, 0), kzbb_model_evaluate(model, &p, mask, &edges, 2, &flux_m, &jac_m));
    try testing.expectEqualSlices(f64, &flux, &flux_m);
    try testing.expectEqualSlices(f64, &jac, &jac_m);

    var bad = opts;
    bad.n_energy = 0;
    try testing.expect(kzbb_model_create(&bad) == null);

    p.a = 0;
    try testing.expectEqual(@intFromEnum(Status.spin_out_of_range), kzbb_evaluate(&p, 0, &edges, 2, &flux, null, &opts));
    try testing.expectEqual(@intFromEnum(Status.invalid_argument), kzbb_evaluate(&p, mask, &edges, 2, &flux, null, &opts));
    try testing.expectEqual(@intFromEnum(Status.invalid_argument), kzbb_evaluate(&p, 1 << 20, &edges, 2, &flux, null, &opts));
}
