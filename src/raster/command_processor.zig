pub fn init() void {}

pub fn submit(raster_unit: *RasterUnit, command_buffer: *CommandBuffer) void {
    @atomicStore(CommandBuffer.Status, &command_buffer.status, .executing, std.builtin.AtomicOrder.Unordered);

    for (command_buffer.commands.items) |command| {
        switch (command) {
            .begin_raster_pass => |begin_raster_pass| {
                raster_unit.render_pass = begin_raster_pass.pass;
            },
            .end_raster_pass => {
                raster_unit.out_triangles.len = 0;
            },
            .bind_pipeline => |bind_pipeline| {
                raster_unit.pipeline = bind_pipeline.pipeline;
            },
            .draw => |draw| {
                raster_unit.uniform = draw.uniform;

                geometry_processor.pipelineDrawTriangles(raster_unit, draw.vertex_count);

                for (raster_unit.out_triangles.buffer[0..raster_unit.out_triangles.len]) |out_triangle| {
                    raster_processor.pipelineRasteriseTriangle(
                        raster_unit,
                        out_triangle.positions,
                        out_triangle.interpolators,
                    );
                }
            },
        }
    }
}

const std = @import("std");
const CommandBuffer = @import("../CommandBuffer.zig");
const Pipeline = @import("../Pipeline.zig");
const RasterUnit = @import("RasterUnit.zig");
const geometry_processor = @import("geometry_processor.zig");
const raster_processor = @import("raster_processor.zig");

const Renderer = @import("../Renderer.zig");
