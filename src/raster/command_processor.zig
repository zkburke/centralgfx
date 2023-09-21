pub fn init(raster_unit: *RasterUnit) void {
    while (true) {
        var should_sleep: bool = false;

        const original_next_entry_to_read = raster_unit.command_queue.next_entry_to_read;
        const new_entry_to_read: u32 = @intCast((raster_unit.command_queue.next_entry_to_read + 1) % raster_unit.command_queue.command_buffers.len);

        //if there is work to complete
        if (original_next_entry_to_read != raster_unit.command_queue.next_entry_to_write) {
            const index = @cmpxchgStrong(
                u32,
                &raster_unit.command_queue.next_entry_to_read,
                new_entry_to_read,
                original_next_entry_to_read,
                std.builtin.AtomicOrder.Acquire,
                std.builtin.AtomicOrder.Acquire,
            );

            if (index == original_next_entry_to_read) {
                const command_buffer = raster_unit.command_queue.command_buffers[index.?];
                const render_finished = raster_unit.command_queue.submition_semaphores[index.?];

                executeWorkload(raster_unit, command_buffer, render_finished);

                _ = @atomicRmw(
                    u32,
                    &raster_unit.command_queue.completion_count,
                    .Add,
                    1,
                    std.builtin.AtomicOrder.Acquire,
                );
            }
        } else {
            should_sleep = true;
        }

        if (should_sleep) {
            raster_unit.command_queue.semaphore.wait();
        }
    }
}

fn executeWorkload(
    raster_unit: *RasterUnit,
    command_buffer: *const CommandBuffer,
    render_finished: *std.Thread.Semaphore,
) void {
    render_finished.wait();

    for (command_buffer.commands.items) |command| {
        switch (command) {
            .begin_raster_pass => |begin_raster_pass| {
                raster_unit.render_pass = begin_raster_pass.pass;
            },
            .end_raster_pass => {
                raster_unit.out_triangles.len = 0;
            },
            .set_pipeline => |bind_pipeline| {
                raster_unit.pipeline = bind_pipeline.pipeline;
            },
            .set_scissor => |set_scissor| {
                raster_unit.scissor = .{
                    .offset = set_scissor.scissor.offset,
                    .extent = set_scissor.scissor.offset + set_scissor.scissor.extent,
                };
            },
            .set_viewport => |set_viewport| {
                raster_unit.viewport_transform.pxyz = .{ set_viewport.viewport.px, set_viewport.viewport.py, set_viewport.viewport.pz, 1 };
                raster_unit.viewport_transform.oxyz = .{ set_viewport.viewport.ox, set_viewport.viewport.oy, set_viewport.viewport.oz, 0 };

                raster_unit.depth_min = set_viewport.viewport.depth_min;
                raster_unit.depth_max = set_viewport.viewport.depth_max;
            },
            .draw => |draw| {
                raster_unit.uniform = draw.uniform;

                geometry_processor.pipelineDrawTriangles(
                    raster_unit,
                    draw.vertex_offset,
                    draw.vertex_count,
                );

                const S = struct {
                    pub fn lessThan(_: void, lhs: geometry_processor.OutTriangle, rhs: geometry_processor.OutTriangle) bool {
                        const nearest_z_a = @max(lhs.positions[0][2], @max(lhs.positions[1][2], lhs.positions[2][2]));
                        const nearest_z_b = @max(rhs.positions[0][2], @max(rhs.positions[1][2], rhs.positions[2][2]));

                        return nearest_z_a < nearest_z_b;
                    }
                };

                if (raster_unit.sort_triangles) std.sort.insertion(
                    geometry_processor.OutTriangle,
                    raster_unit.out_triangles.buffer[0..raster_unit.out_triangles.len],
                    {},
                    S.lessThan,
                );

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

    render_finished.post();
}

pub fn submit(
    raster_unit: *RasterUnit,
    command_buffer: *CommandBuffer,
    render_finished: *std.Thread.Semaphore,
) void {
    @atomicStore(CommandBuffer.Status, &command_buffer.status, .executing, std.builtin.AtomicOrder.Unordered);

    if (!@import("builtin").single_threaded) {
        const new_next_entry_to_write: u32 = @intCast((raster_unit.command_queue.next_entry_to_write + 1) % raster_unit.command_queue.command_buffers.len);

        std.debug.assert(new_next_entry_to_write != raster_unit.command_queue.next_entry_to_read);

        raster_unit.command_queue.command_buffers[raster_unit.command_queue.next_entry_to_write] = command_buffer;
        raster_unit.command_queue.submition_semaphores[raster_unit.command_queue.next_entry_to_write] = render_finished;

        raster_unit.command_queue.completion_goal += 1;

        std.atomic.compilerFence(std.builtin.AtomicOrder.Acquire);

        raster_unit.command_queue.next_entry_to_write = new_next_entry_to_write;

        raster_unit.command_queue.semaphore.post();
    }

    if (@import("builtin").single_threaded) {
        executeWorkload(raster_unit, command_buffer, render_finished);
    }
}

///Single producer, single consumer (SpSc)
///A one way queue based on a fixed sized ring buffer
pub fn UnorderedOneWayRing(comptime capacity: comptime_int, comptime T: type) type {
    std.debug.assert(std.mem.isAligned(capacity, 2));

    return struct {
        buffer: [capacity]T align(64),

        count: usize align(64),
        tail: usize align(64),
        head: usize align(64),

        pub fn enqueue(self: *@This(), value: T) void {
            self.enqueueSlice(&.{value});
        }

        pub inline fn enqueueSlice(self: *@This(), values: []const T) void {
            if (self.isFull()) {
                @panic("Queue full");
            }

            @memcpy(self.buffer[self.tail .. self.tail + values.len], values);

            self.count += values.len;
            self.tail = (self.tail + values.len) % capacity;
        }

        pub inline fn dequeue(self: *@This()) ?T {
            if (self.isEmpty()) return null;

            const result = self.buffer[self.head];

            self.head = (self.head + 1) % capacity;
            self.count -= 1;

            return result;
        }

        pub inline fn isFull(self: @This()) bool {
            return self.count == capacity;
        }

        pub inline fn isEmpty(self: @This()) bool {
            return self.head == self.tail;
        }
    };
}

const std = @import("std");
const CommandBuffer = @import("../CommandBuffer.zig");
const Pipeline = @import("../Pipeline.zig");
const RasterUnit = @import("RasterUnit.zig");
const geometry_processor = @import("geometry_processor.zig");
const raster_processor = @import("raster_processor.zig");
const Renderer = @import("../Renderer.zig");
