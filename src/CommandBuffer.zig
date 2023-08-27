pub const Command = union(enum) {
    begin_raster_pass: BeginRasterPassCommand,
    end_raster_pass: EndRasterPassCommand,
    bind_pipeline: BindPipelineCommand,
    draw: DrawCommand,
};

pub const BeginRasterPassCommand = struct {
    pass: *const Renderer.Pass,
};

pub const EndRasterPassCommand = struct {};

pub const BindPipelineCommand = struct {
    pipeline: *const Pipeline,
};

pub const DrawCommand = struct {
    uniform: *const anyopaque,
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

pub fn beginRasterPass(self: *CommandBuffer, render_pass: *Renderer.Pass) void {
    self.commands.append(self.allocator, .{
        .begin_raster_pass = .{
            .pass = render_pass,
        },
    }) catch unreachable;
}

pub fn endRasterPass(self: *CommandBuffer) void {
    self.commands.append(self.allocator, .{
        .end_raster_pass = .{},
    }) catch unreachable;
}

pub fn bindPipeline(self: *CommandBuffer, pipeline: *const Pipeline) void {
    self.commands.append(self.allocator, .{
        .bind_pipeline = .{
            .pipeline = pipeline,
        },
    }) catch unreachable;
}

pub fn draw(self: *CommandBuffer, uniform: *const anyopaque, vertex_count: u32) void {
    self.commands.append(self.allocator, .{ .draw = .{
        .uniform = uniform,
        .vertex_count = vertex_count,
    } }) catch unreachable;
}

const CommandBuffer = @This();
const std = @import("std");
const Pipeline = @import("Pipeline.zig");

const Renderer = @import("Renderer.zig");
