//! kerrzbb: a multi-temperature blackbody model for a thin accretion disc
//! around a Kerr black hole (Li et al. 2005, KERRBB), ray traced on the fly
//! with kerrz and differentiable through kerrz's dual numbers.

/// Re-export of the kerrz dependency (dual numbers, metric, geodesics).
pub const kerrz = @import("kerrz");
pub const constants = @import("constants.zig");
pub const disc = @import("disc.zig");
pub const quadrature = @import("quadrature.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
