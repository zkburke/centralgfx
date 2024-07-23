const triangle_queue_size = 4096 * 10;

pipeline: *const Pipeline = undefined,
uniform: *const anyopaque = undefined,
render_pass: *const Renderer.Pass = undefined,
renderer: *Renderer = undefined,
command_processor_thread: if (!@import("builtin").single_threaded) std.Thread else void = undefined,

scissor: struct {
    offset: @Vector(2, i32) = .{ 0, 0 },
    extent: @Vector(2, i32) = .{ 0, 0 },
} = .{},

///vertex = fma(point, pxyz, oxyz) = point * pxyz + oxyz;
viewport_transform: struct {
    ///Factor that is multiplied by each post-clip vertex
    pxyz: @Vector(4, f32) = .{ 0, 0, 0, 1 },
    ///Factor that is added by each post-clip vertex
    oxyz: @Vector(4, f32) = .{ 0, 0, 0, 0 },
} = .{},

depth_min: f32 = 0,
depth_max: f32 = 1,

sort_triangles: bool = false,

out_triangles: std.BoundedArray(geometry_processor.OutTriangle, triangle_queue_size) = .{},

///stored in x, y, z, w form
triangle_positions: std.BoundedArray(@Vector(4, f32), triangle_queue_size) = .{},
///stored in x0, x1, x2 form: x = {x0, x1, x2} * barycentric
triangle_interpolators: std.BoundedArray(@Vector(3, f32), triangle_queue_size) = .{},
triangle_flats: std.BoundedArray(u32, triangle_queue_size) = .{},

triangle_layout: struct {
    interpolator_count: u8 = 0,
    flat_count: u8 = 0,
} = .{},

cull_mode: CommandBuffer.FaceCullMode = .back,

cull_xor: u8 = 0x00, //xor with cull internal status
cull_and: u8 = 0xff, //and with cull internal status
cull_or: u8 = 0x00, //or with cull internal status

command_queue: struct {
    command_buffers: [256]*const CommandBuffer = undefined,
    submition_semaphores: [256]*std.Thread.Semaphore = undefined,

    completion_goal: u32 = 0,
    completion_count: u32 = 0,
    next_entry_to_read: u32 = 0,
    next_entry_to_write: u32 = 0,
    semaphore: std.Thread.Semaphore = .{ .permits = 1 },
} = .{},

pub fn init(self: *RasterUnit, renderer: *Renderer) !void {
    if (!@import("builtin").single_threaded) {
        const command_processor_thread = try std.Thread.spawn(.{}, command_processor.init, .{self});

        self.command_processor_thread = command_processor_thread;
    }

    self.renderer = renderer;
    self.out_triangles = .{};
}

pub fn deinit(self: *RasterUnit) void {
    defer self.* = undefined;

    if (!@import("builtin").single_threaded) self.command_processor_thread.detach();
}

const std = @import("std");
const command_processor = @import("command_processor.zig");
const geometry_processor = @import("geometry_processor.zig");
const Pipeline = @import("../Pipeline.zig");
const Renderer = @import("../Renderer.zig");
const CommandBuffer = @import("../CommandBuffer.zig");
const RasterUnit = @This();
