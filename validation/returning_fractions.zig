//! Fate of the photons emitted by the disc, by forward ray tracing from the
//! emission point: returning to the disc (r >= r_in), captured (horizon or
//! plunging region), or escaping. This is independent of the backward kernel
//! in src/returning.zig and checks it through iota_ret, the returned fraction
//! of the emitted energy-at-infinity. Compare with Li et al. (2005) Fig. 2:
//! (iota_ret, iota_BH, iota_esc) = (1.7%, 0.66%, 97.6%) at a = 0 and
//! (27%, 4%, 69%) at a = 0.9999, for eta = 0 and isotropic emission.
//!
//!     zig build returning-fractions

const std = @import("std");
const kbb = @import("kerrzbb");
const kerrz = kbb.kerrz;
const D0 = kerrz.DualNumber(f64, 0);

const Fates = struct { ret: f64 = 0, bh: f64 = 0, esc: f64 = 0 };

/// Energy-at-infinity weighted fates of photons emitted isotropically from r.
fn fatesAt(metric: kerrz.KerrMetric(D0), a: f64, r: f64, r_in: f64, n_mu: usize, n_phi: usize) Fates {
    const orb = kbb.disc.Orbit(D0).init(.promote(r), .promote(a));
    const r2 = r * r;
    const delta = r2 - 2 * r + a * a;
    const kerr_a = r2 * r2 + a * a * r * (r + 2);
    const chi = @sqrt(r2 * delta / kerr_a);
    const gamma = chi / orb.energyFactor().x;
    const b = gamma * @sqrt(kerr_a) / r;
    const x: kerrz.FourVector(D0) = .{ .t = .zero, .r = .promote(r), .th = .promote(std.math.pi / 2.0), .ph = .zero };

    var f: Fates = .{};
    // Midpoint rule in mu = cos(theta) and phi; weight (E_inf / E_local) mu dmu dphi / pi.
    for (0..n_mu) |im| {
        const mu = (@as(f64, @floatFromInt(im)) + 0.5) / @as(f64, @floatFromInt(n_mu));
        const s = @sqrt(1 - mu * mu);
        for (0..n_phi) |ip| {
            const phi = 2 * std.math.pi * (@as(f64, @floatFromInt(ip)) + 0.5) / @as(f64, @floatFromInt(n_phi));
            const nr = s * @cos(phi);
            const nphi = s * @sin(phi);
            const E = orb.energy.x + b * orb.omega.x * nphi;
            const L = orb.ang_mom.x + b * nphi;
            const Q = r2 * mu * mu;
            var geod = kerrz.NullGeodesic(D0).fromConstantsOfMotion(x, .promote(E), .promote(L), .promote(Q));
            geod.theta_sign = -1;
            geod.radial_sign = if (nr >= 0) 1 else -1;
            const res = geod.traceToAngle(metric, .promote(std.math.pi / 2.0), .{});
            const w = E * mu / (@as(f64, @floatFromInt(n_mu)) * @as(f64, @floatFromInt(n_phi))) * 2.0;
            if (res.status == .infinity) {
                f.esc += w;
            } else if (res.status == .event_horizon or !std.math.isFinite(res.r.x) or res.r.x < r_in) {
                f.bh += w;
            } else {
                f.ret += w;
            }
        }
    }
    return f;
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var xg: [24]f64 = undefined;
    var wg: [24]f64 = undefined;
    kbb.quadrature.gaussLegendre(&xg, &wg);

    for ([_]f64{ 0.001, 0.5, 0.9, 0.998, 0.9999 }) |a_val| {
        const a: D0 = .promote(a_val);
        const r_in = kbb.disc.iscoRadius(D0, a);
        const metric = kerrz.KerrMetric(D0).init(.one, a);
        const kernel = try kbb.returning.buildKernel(D0, allocator, a, r_in, false, .{});
        defer kernel.deinit();
        const profile = try kbb.returning.solve(D0, allocator, kernel, .zero);
        defer profile.deinit();

        // Integrate in log r from r_in to 1e5 (beyond, photons essentially all escape).
        const panels = 12;
        const t0 = @log(r_in.x);
        const width = (@log(1e5) - t0) / panels;
        var tot: Fates = .{};
        var emitted: f64 = 0;
        for (0..panels) |p| {
            for (xg, wg) |xi, wi| {
                const r = @exp(t0 + width * (@as(f64, @floatFromInt(p)) + 0.5 * (xi + 1)));
                const fout = profile.flux(.promote(r)).x;
                const dr = 0.5 * width * wi * r;
                const f = fatesAt(metric, a_val, r, r_in.x, 64, 128);
                tot.ret += f.ret * fout * r * dr;
                tot.bh += f.bh * fout * r * dr;
                tot.esc += f.esc * fout * r * dr;
                emitted += kbb.disc.Orbit(D0).init(.promote(r), a).energy.x * fout * r * dr;
            }
        }
        const sum = tot.ret + tot.bh + tot.esc;
        std.debug.print(
            "a = {d:.4}: iota_ret {d:.4}  iota_BH {d:.4}  iota_esc {d:.4}  (sum of fates / emitted = {d:.5})\n",
            .{ a_val, tot.ret / sum, tot.bh / sum, tot.esc / sum, sum / emitted },
        );
    }
}
