//! Compare kerrzbb with reference kerrbb spectra, either binned fluxes from
//! XSPEC (validation/kerrbb_reference.py: "edges" + "flux") or differential
//! spectra read from kerrbb's table at grid nodes
//! (validation/kerrbb_table_extract.py: "energies" + "density").
//!
//!     zig build compare -- <reference.json> [output.json]
//!
//! Prints, for each case, the ratio of summed photon fluxes and the largest
//! pointwise deviation over points holding at least 1e-3 of the peak. With an
//! output path, also writes the kerrzbb spectra next to the reference ones.

const std = @import("std");
const kbb = @import("kerrzbb");
const D0 = kbb.kerrz.DualNumber(f64, 0);

const Case = struct {
    eta: f64,
    a: f64,
    incl_deg: f64,
    mass: f64,
    mdot: f64,
    distance: f64,
    fcol: f64,
    rflag: i64,
    lflag: i64,
    norm: f64,
    edges: ?[]f64 = null,
    flux: ?[]f64 = null,
    energies: ?[]f64 = null,
    density: ?[]f64 = null,
};

const Reference = struct {
    source: []const u8 = "",
    cases: []Case,
};

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 2) {
        std.debug.print("usage: compare <reference.json> [output.json]\n", .{});
        return error.MissingArgument;
    }

    const text = try std.fs.cwd().readFileAlloc(allocator, args[1], 1 << 28);
    defer allocator.free(text);
    const parsed = try std.json.parseFromSlice(Reference, allocator, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    var results: std.ArrayList([]f64) = .empty;
    defer {
        for (results.items) |r| allocator.free(r);
        results.deinit(allocator);
    }

    std.debug.print("{s:>8} {s:>5} {s:>5} {s:>2} {s:>2} {s:>10} {s:>10} {s:>10}\n", .{ "a", "i", "eta", "rf", "ld", "mean", "max|dev|", "at E" });
    var sum_worst: f64 = 0;
    for (parsed.value.cases) |c| {
        const ref = c.flux orelse c.density orelse return error.MissingReference;
        const ours = try allocator.alloc(f64, ref.len);
        try results.append(allocator, ours);
        @memset(ours, std.math.nan(f64));
        const ld: u8 = @intCast(c.lflag);
        const p: kbb.spectrum.Params(D0) = .{
            .eta = .promote(c.eta),
            .a = .promote(c.a),
            .incl = .promote(std.math.degreesToRadians(c.incl_deg)),
            .mass = .promote(c.mass),
            .mdot = .promote(c.mdot),
            .distance = .promote(c.distance),
            .fcol = .promote(c.fcol),
            .norm = .promote(c.norm),
            .limb_darkening = c.lflag > 0,
            .returning_radiation = c.rflag > 0,
        };
        const spec = kbb.spectrum.Spectrum(D0).init(allocator, p, .{}) catch |err| {
            std.debug.print("{d:>8.4} {d:>5.1} {d:>5.2} {d:>2}: {s}\n", .{ c.a, c.incl_deg, c.eta, ld, @errorName(err) });
            continue;
        };
        defer spec.deinit();

        // Point energies for reporting.
        const energies = try allocator.alloc(f64, ref.len);
        defer allocator.free(energies);
        if (c.density) |_| {
            const e = c.energies orelse return error.MissingEnergies;
            for (ours, e, energies) |*o, E, *eo| {
                o.* = spec.density(E).x;
                eo.* = E;
            }
        } else {
            const edges = c.edges orelse return error.MissingEdges;
            const out = try allocator.alloc(D0, ref.len);
            defer allocator.free(out);
            try spec.binned(edges, out, 4);
            for (ours, out, energies, 0..) |*o, v, *eo, b| {
                o.* = v.x;
                eo.* = @sqrt(edges[b] * edges[b + 1]);
            }
        }

        var peak: f64 = 0;
        for (ref) |f| peak = @max(peak, f);
        var sum_ours: f64 = 0;
        var sum_ref: f64 = 0;
        var worst: f64 = 0;
        var worst_e: f64 = 0;
        for (ours, ref, energies) |o, f, E| {
            if (f < 1e-3 * peak) continue;
            // Weight by E so that log-spaced points sum to a photon flux.
            sum_ours += o * E;
            sum_ref += f * E;
            const dev = o / f - 1;
            if (@abs(dev) > @abs(worst)) {
                worst = dev;
                worst_e = E;
            }
        }
        sum_worst = @max(sum_worst, @abs(worst));
        std.debug.print("{d:>8.4} {d:>5.1} {d:>5.2} {d:>2} {d:>2} {d:>10.5} {d:>10.5} {d:>10.3}\n", .{ c.a, c.incl_deg, c.eta, @as(u8, @intCast(c.rflag)), ld, sum_ours / sum_ref, worst, worst_e });
    }
    std.debug.print("largest deviation over all cases: {d:.5}\n", .{sum_worst});

    if (args.len >= 3) {
        const file = try std.fs.cwd().createFile(args[2], .{});
        defer file.close();
        var buf: [4096]u8 = undefined;
        var writer = file.writer(&buf);
        const w = &writer.interface;
        try w.writeAll("{\"cases\": [");
        for (parsed.value.cases, results.items, 0..) |c, ours, k| {
            if (k > 0) try w.writeAll(",");
            try w.print("{{\"a\": {d}, \"incl_deg\": {d}, \"eta\": {d}, \"rflag\": {d}, \"lflag\": {d}, ", .{ c.a, c.incl_deg, c.eta, c.rflag, c.lflag });
            if (c.edges) |e| {
                try w.writeAll("\"edges\": ");
                try std.json.Stringify.value(e, .{}, w);
            } else {
                try w.writeAll("\"energies\": ");
                try std.json.Stringify.value(c.energies, .{}, w);
            }
            try w.writeAll(", \"reference\": ");
            try std.json.Stringify.value(c.flux orelse c.density, .{}, w);
            try w.writeAll(", \"kerrzbb\": ");
            try std.json.Stringify.value(ours, .{}, w);
            try w.writeAll("}");
        }
        try w.writeAll("]}\n");
        try w.flush();
    }
}
