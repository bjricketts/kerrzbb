//! A minimal parallel-for over an index range, using plain threads.

const std = @import("std");

pub const max_threads = 64;

/// Number of threads to use for `requested` (0 means one per CPU).
pub fn resolve(requested: usize) usize {
    if (requested != 0) return @min(requested, max_threads);
    const n = std.Thread.getCpuCount() catch 1;
    return @max(1, @min(n, max_threads));
}

/// Call `func(ctx, start, end)` on contiguous chunks of [0, count), on up to
/// `n_threads` threads (0 means one per CPU). The chunks must write disjoint
/// data. Returns the first error from any chunk. If a thread cannot be
/// spawned, its chunk runs on the calling thread.
pub fn forRange(
    comptime E: type,
    n_threads: usize,
    count: usize,
    ctx: anytype,
    comptime func: fn (@TypeOf(ctx), usize, usize) E!void,
) E!void {
    const n = @min(resolve(n_threads), count);
    if (n <= 1) return func(ctx, 0, count);

    const Ctx = @TypeOf(ctx);
    const Worker = struct {
        fn run(c: Ctx, start: usize, end: usize, result: *(E!void)) void {
            result.* = func(c, start, end);
        }
    };

    var results: [max_threads](E!void) = undefined;
    var threads: [max_threads]?std.Thread = .{null} ** max_threads;
    for (0..n) |k| results[k] = {};

    for (1..n) |k| {
        const start = count * k / n;
        const end = count * (k + 1) / n;
        threads[k] = std.Thread.spawn(.{}, Worker.run, .{ ctx, start, end, &results[k] }) catch null;
    }
    // Chunk 0, and any chunk whose thread failed to spawn, run here.
    results[0] = func(ctx, 0, count / n);
    for (1..n) |k| {
        if (threads[k] == null) results[k] = func(ctx, count * k / n, count * (k + 1) / n);
    }
    for (1..n) |k| if (threads[k]) |t| t.join();
    for (0..n) |k| try results[k];
}

test "forRange covers the range once and propagates errors" {
    const Ctx = struct {
        hits: []std.atomic.Value(u32),
        fail_at: usize,
        fn f(self: @This(), start: usize, end: usize) error{Boom}!void {
            for (start..end) |i| {
                if (i == self.fail_at) return error.Boom;
                _ = self.hits[i].fetchAdd(1, .monotonic);
            }
        }
    };
    var hits: [1000]std.atomic.Value(u32) = undefined;
    for (&hits) |*h| h.* = .init(0);
    try forRange(error{Boom}, 7, hits.len, Ctx{ .hits = &hits, .fail_at = std.math.maxInt(usize) }, Ctx.f);
    for (hits) |h| try std.testing.expectEqual(@as(u32, 1), h.load(.monotonic));
    try std.testing.expectError(error.Boom, forRange(error{Boom}, 4, hits.len, Ctx{ .hits = &hits, .fail_at = 900 }, Ctx.f));
}
