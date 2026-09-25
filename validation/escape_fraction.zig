//! Fraction of the disc emission (no returning radiation, eta = 0, isotropic)
//! that reaches infinity in the primary image, integrated over all observer
//! directions. Compare with Li et al. (2005) §3.1 / Fig. 2: iota_esc = 97.6%
//! at a = 0 and 69% at a = 0.9999 (those include returning radiation in the
//! emitted flux and all image orders, so agreement is approximate).

const std = @import("std");
const kbb = @import("kerrzbb");
const D0 = kbb.kerrz.DualNumber(f64, 0);

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var x: [24]f64 = undefined;
    var w: [24]f64 = undefined;
    kbb.quadrature.gaussLegendre(&x, &w);

    const opts: kbb.image.Options = .{ .n_theta = 96, .n_rho = 64, .n_outer = 32 };

    for ([_]f64{ 0.1, 0.5, 0.9, 0.998 }) |a_val| {
        const a: D0 = .promote(a_val);
        const r_in = kbb.disc.iscoRadius(D0, a);
        const eps_in = 1 - kbb.disc.Orbit(D0).init(r_in, a).energy.x;

        // Integrate mu = cos(i) over [mu_min, 1]; the band below mu_min
        // (i > 89 deg) is added assuming the flux there equals its value
        // at the lowest node.
        const mu_min = @cos(kbb.image.max_inclination);
        var total: f64 = 0;
        var flux_low: f64 = 0;
        for (x, w) |xi, wi| {
            const mu = mu_min + 0.5 * (xi + 1) * (1 - mu_min);
            const incl = std.math.acos(mu);
            const samples = try kbb.image.traceImage(D0, allocator, a, .promote(incl), r_in, opts);
            defer allocator.free(samples);
            var flux: f64 = 0;
            for (samples) |s| {
                if (s.weight.x == 0) continue;
                const f = kbb.disc.fluxNoReturn(D0, s.r, a, r_in, .zero).x;
                flux += s.weight.x * std.math.pow(f64, s.g.x, 4) * f;
            }
            total += 0.5 * (1 - mu_min) * wi * flux;
            if (xi == x[0]) flux_low = flux;
        }
        total += mu_min * flux_low;
        const iota = total / (2.0 * std.math.pi / 3.0 * eps_in);
        std.debug.print("a = {d:.3}: iota_esc (primary image) = {d:.4}\n", .{ a_val, iota });
    }
}
