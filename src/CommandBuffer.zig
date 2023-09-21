pub const Command = union(enum) {
    begin_raster_pass: BeginRasterPassCommand,
    end_raster_pass: EndRasterPassCommand,
    set_pipeline: SetPipelineCommand,
    set_scissor: SetScissorCommand,
    set_viewport: SetViewportCommand,
    draw: DrawCommand,
};

pub const BeginRasterPassCommand = struct {
    pass: *const Renderer.Pass,
    render_area: struct { offset: @Vector(2, i32), extent: @Vector(2, i32) },
};

pub const EndRasterPassCommand = struct {};

pub const SetPipelineCommand = struct {
    pipeline: *const Pipeline,
};

pub const SetScissorCommand = struct {
    scissor: struct {
        offset: @Vector(2, i32),
        extent: @Vector(2, i32),
    },
};

pub const SetViewportCommand = struct {
    viewport: struct {
        px: f32,
        py: f32,
        pz: f32,
        ox: f32,
        oy: f32,
        oz: f32,
        depth_min: f32,
        depth_max: f32,
    },
};

pub const DrawCommand = struct {
    uniform: *const anyopaque,
    vertex_offset: u32,
    vertex_count: u32,
};

pub const Status = enum(u8) {
    recording,
    pending,
    executing,
};

allocator: std.mem.Allocator,
commands: std.ArrayListUnmanaged(Command) = .{},
status: Status = .recording,

pub fn init() void {}

pub fn deinit(self: *CommandBuffer) void {
    defer self.* = undefined;

    self.commands.deinit(self.allocator);
}

///Begin recording commands
pub fn begin(self: *CommandBuffer) void {
    @atomicStore(Status, &self.status, .recording, std.builtin.AtomicOrder.Unordered);

    self.commands.clearRetainingCapacity();
}

///End recording commands
pub fn end(self: *CommandBuffer) void {
    @atomicStore(Status, &self.status, .pending, std.builtin.AtomicOrder.Unordered);
}

pub fn beginRasterPass(
    self: *CommandBuffer,
    render_pass: *Renderer.Pass,
    render_area: struct { offset: @Vector(2, i32), extent: @Vector(2, i32) },
    fragment_to_physical_space_ratio: struct {
        ///Non-zero, power of two rational scale
        numerator: i16 = 1,
        denominator: i16 = 1,
    },
) void {
    self.commands.append(self.allocator, .{
        .begin_raster_pass = .{
            .pass = render_pass,
            .render_area = .{
                .offset = (render_area.offset * @as(@Vector(2, i32), @splat(fragment_to_physical_space_ratio.numerator))) / @as(@Vector(2, i32), @splat(fragment_to_physical_space_ratio.numerator)),
                .extent = (render_area.extent * @as(@Vector(2, i32), @splat(fragment_to_physical_space_ratio.numerator))) / @as(@Vector(2, i32), @splat(fragment_to_physical_space_ratio.numerator)),
                // .offset = render_area.offset,
                // .extent = render_area.extent,
            },
        },
    }) catch unreachable;
}

pub fn endRasterPass(self: *CommandBuffer) void {
    self.commands.append(self.allocator, .{
        .end_raster_pass = .{},
    }) catch unreachable;
}

pub fn setPipeline(self: *CommandBuffer, pipeline: *const Pipeline) void {
    self.commands.append(self.allocator, .{
        .set_pipeline = .{
            .pipeline = pipeline,
        },
    }) catch unreachable;
}

pub fn setScissor(self: *CommandBuffer, scissor: struct {
    offset: @Vector(2, i32),
    extent: @Vector(2, i32),
}) void {
    self.commands.append(self.allocator, .{
        .set_scissor = .{
            .scissor = .{
                .offset = scissor.offset,
                .extent = scissor.extent,
            },
        },
    }) catch unreachable;
}

pub fn setViewport(self: *CommandBuffer, viewport: struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    depth_min: f32,
    depth_max: f32,
}) void {
    const px = viewport.width / 2;
    const py = viewport.height / 2;

    const pz = 0.5 * (viewport.depth_max - viewport.depth_min);
    const oz = 0.5 * (viewport.depth_min + viewport.depth_max);

    self.commands.append(self.allocator, .{
        .set_viewport = .{
            .viewport = .{
                .ox = viewport.x + px,
                .oy = viewport.y + py,
                .px = px,
                .py = py,
                .oz = oz,
                .pz = pz,
                .depth_min = viewport.depth_min,
                .depth_max = viewport.depth_max,
            },
        },
    }) catch unreachable;
}

pub fn draw(self: *CommandBuffer, uniform: *const anyopaque, vertex_offset: u32, vertex_count: u32) void {
    self.commands.append(self.allocator, .{ .draw = .{
        .uniform = uniform,
        .vertex_offset = vertex_offset,
        .vertex_count = vertex_count,
    } }) catch unreachable;
}

const CommandBuffer = @This();
const std = @import("std");
const Pipeline = @import("Pipeline.zig");
const Renderer = @import("Renderer.zig");
