//! Gauss-Legendre quadrature rules.

const std = @import("std");

/// Fill `nodes` and `weights` with the n-point Gauss-Legendre rule on [-1, 1],
/// where n = nodes.len. Nodes are returned in ascending order.
pub fn gaussLegendre(nodes: []f64, weights: []f64) void {
    std.debug.assert(nodes.len == weights.len);
    const n = nodes.len;
    const nf: f64 = @floatFromInt(n);
    const m = (n + 1) / 2;
    for (0..m) |i| {
        const fi: f64 = @floatFromInt(i);
        // Initial guess (Tricomi), then Newton iterations on P_n.
        var x = @cos(std.math.pi * (fi + 0.75) / (nf + 0.5));
        var dp: f64 = 0;
        for (0..100) |_| {
            var p0: f64 = 1;
            var p1: f64 = x;
            for (2..n + 1) |k| {
                const kf: f64 = @floatFromInt(k);
                const p2 = ((2 * kf - 1) * x * p1 - (kf - 1) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            if (n == 1) p0 = 1;
            // P_n = p1, P_{n-1} = p0
            dp = nf * (x * p1 - p0) / (x * x - 1);
            const dx = p1 / dp;
            x -= dx;
            if (@abs(dx) < 1e-15) break;
        }
        const w = 2 / ((1 - x * x) * dp * dp);
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
}

test "gauss-legendre integrates polynomials exactly" {
    var x: [7]f64 = undefined;
    var w: [7]f64 = undefined;
    gaussLegendre(&x, &w);
    var sum_w: f64 = 0;
    var sum_x12: f64 = 0;
    for (x, w) |xi, wi| {
        sum_w += wi;
        sum_x12 += wi * std.math.pow(f64, xi, 12);
    }
    try std.testing.expectApproxEqAbs(2.0, sum_w, 1e-14);
    try std.testing.expectApproxEqAbs(2.0 / 13.0, sum_x12, 1e-14);
    try std.testing.expect(x[0] < x[1]);
}
