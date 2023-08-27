pipeline: *const Pipeline,
uniform: *const anyopaque,
render_pass: *const Renderer.Pass,
renderer: *Renderer,

out_triangles: std.BoundedArray(geometry_processor.OutTriangle, 4096) = .{},

const std = @import("std");
const geometry_processor = @import("geometry_processor.zig");
const Pipeline = @import("../Pipeline.zig");
const Renderer = @import("../Renderer.zig");
const RasterUnit = @This();
