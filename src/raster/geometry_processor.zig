pub const OutTriangle = struct {
    positions: [3]@Vector(4, f32),
    interpolators: [3]TestPipelineFragmentInput,
};

pub fn pipelineDrawTriangles(
    raster_unit: *RasterUnit,
    vertex_offset: usize,
    vertex_count: usize,
) void {
    for (0..vertex_count / 3) |index| {
        pipelineDrawTriangle(raster_unit, vertex_offset, index);
    }
}

fn vectorDotGeneric3D(comptime N: comptime_int, comptime T: type, a: @Vector(N, T), b: @Vector(N, T)) T {
    return @reduce(.Add, a * b);
}

fn vectorDot(a: @Vector(3, f32), b: @Vector(3, f32)) f32 {
    return @reduce(.Add, a * b);
}

fn interpolateVertex(
    comptime Interpolator: type,
    position0: @Vector(4, f32),
    position1: @Vector(4, f32),
    interpolator0: Interpolator,
    interpolator1: Interpolator,
    t: f32,
) struct { position: @Vector(4, f32), interpolator: Interpolator } {
    const interpolant: @Vector(2, f32) = .{ 1 - t, t };

    var position: @Vector(4, f32) = .{ 0, 0, 0, 0 };

    position[0] = position0[0] * (1 - t) + position1[0] * t;
    position[1] = position0[1] * (1 - t) + position1[1] * t;
    position[2] = position0[2] * (1 - t) + position1[2] * t;

    inline for (0..4) |n| {
        _ = n;
        // position[n] = vectorDotGeneric3D(2, f32, .{ position0[n], position1[n] }, interpolant);
    }

    var interpolator: Interpolator = undefined;

    inline for (std.meta.fields(Interpolator)) |field| {
        const vector_dimensions = @typeInfo(field.type).Vector.len;
        const Component = std.meta.Child(field.type);

        for (0..vector_dimensions) |n| {
            @field(interpolator, field.name)[n] = vectorDotGeneric3D(2, Component, .{
                @field(interpolator0, field.name)[n],
                @field(interpolator1, field.name)[n],
            }, interpolant);
        }
    }

    return .{
        .position = position,
        .interpolator = interpolator,
    };
}

fn twoTriangleArea(triangle: [3]@Vector(2, f32)) f32 {
    const p1 = triangle[1] - triangle[0];
    const p2 = triangle[2] - triangle[0];

    return vectorCross2D(p1, p2);
}

fn vectorCross2D(a: @Vector(2, f32), b: @Vector(2, f32)) f32 {
    return a[0] * b[1] - b[0] * a[1];
}

///TODO: Placeholder until we have general purpose (type erased) interpolants
pub const TestPipelineFragmentInput = struct {
    color: @Vector(4, f32),
    uv: @Vector(2, f32),
    normal: @Vector(3, f32),
    position_world_space: @Vector(3, f32),
};

fn pipelineDrawTriangle(
    raster_unit: *RasterUnit,
    vertex_offset: usize,
    triangle_index: usize,
) void {
    @setFloatMode(.optimized);

    var fragment_input_0: TestPipelineFragmentInput = undefined;
    var fragment_input_1: TestPipelineFragmentInput = undefined;
    var fragment_input_2: TestPipelineFragmentInput = undefined;

    var triangle: [3]@Vector(4, f32) = undefined;

    triangle[0] = raster_unit.pipeline.vertexShader(raster_unit.uniform, vertex_offset + triangle_index * 3 + 0, &fragment_input_0);
    triangle[1] = raster_unit.pipeline.vertexShader(raster_unit.uniform, vertex_offset + triangle_index * 3 + 1, &fragment_input_1);
    triangle[2] = raster_unit.pipeline.vertexShader(raster_unit.uniform, vertex_offset + triangle_index * 3 + 2, &fragment_input_2);

    {
        const pa: @Vector(2, f32) = .{ triangle[0][0] / triangle[0][3], triangle[0][1] / triangle[0][3] };
        const pb: @Vector(2, f32) = .{ triangle[1][0] / triangle[1][3], triangle[1][1] / triangle[1][3] };
        const pc: @Vector(2, f32) = .{ triangle[2][0] / triangle[2][3], triangle[2][1] / triangle[2][3] };

        const eb = pb - pa;
        const ec = pc - pa;

        var culled = eb[0] * ec[1] <= eb[1] * ec[0];

        //Could be done using bit manipulation instead of branching
        switch (raster_unit.cull_mode) {
            .none => culled = false,
            .back => culled = culled,
            .front => culled = !culled,
            .front_and_back => culled = true,
        }

        if (culled) {
            return;
        }
    }

    const clip_mins: [3]@Vector(4, f32) = .{
        .{ -1, -1, 0, 0 },
        .{ -1, -1, 0, 0 },
        .{ -1, -1, 0, 0 },
    };

    const clip_maxs: [3]@Vector(4, f32) = .{
        .{ 1, 1, 1, std.math.floatMax(f32) },
        .{ 1, 1, 1, std.math.floatMax(f32) },
        .{ 1, 1, 1, std.math.floatMax(f32) },
    };

    //Pre clip cull
    if ((@reduce(.And, triangle[0] > clip_maxs[0]) and
        @reduce(.And, triangle[1] > clip_maxs[1]) and
        @reduce(.And, triangle[2] > clip_maxs[2])) or
        (@reduce(.And, triangle[0] < clip_mins[0]) and
        @reduce(.And, triangle[1] < clip_mins[1]) and
        @reduce(.And, triangle[2] < clip_mins[2])))
    {
        return;
    }

    var should_clip: bool = false;

    for (triangle) |point| {
        if (point[3] <= 0) {
            should_clip = true;
        }
    }

    if (!should_clip) {
        emitTriangle(
            raster_unit,
            triangle,
            .{
                fragment_input_0,
                fragment_input_1,
                fragment_input_2,
            },
        );

        return;
    } else {
        return;
    }

    var clipper = TriangleClipper(TestPipelineFragmentInput).init(
        &triangle,
        &.{ fragment_input_0, fragment_input_1, fragment_input_2 },
        0,
        1,
        2,
    );

    const clipping_planes: [2]@Vector(4, f32) = .{
        .{ 0, 0, -1, 1 },
        .{ 0, 0, 1, 1 },
    };

    for (clipping_planes) |plane| {
        clipper.clipToPlaneDeIndexed(plane);
    }

    if (clipper.vertex_positions.len < 3) return;

    for (0..clipper.vertex_positions.len / 3) |clip_triangle_index| {
        const position_0 = clipper.vertex_positions[clip_triangle_index * 3];
        const position_1 = clipper.vertex_positions[clip_triangle_index * 3 + 1];
        const position_2 = clipper.vertex_positions[clip_triangle_index * 3 + 2];

        const interpolator_0 = clipper.vertex_interpolators[clip_triangle_index * 3];
        const interpolator_1 = clipper.vertex_interpolators[clip_triangle_index * 3 + 1];
        const interpolator_2 = clipper.vertex_interpolators[clip_triangle_index * 3 + 2];

        emitTriangle(
            raster_unit,
            .{
                position_0,
                position_1,
                position_2,
            },
            .{
                interpolator_0,
                interpolator_1,
                interpolator_2,
            },
        );
    }
}

inline fn perspectiveDivide(triangle: [3]@Vector(4, f32)) [3]@Vector(4, f32) {
    var result = triangle;

    for (&result) |*point| {
        const w_vector: @Vector(4, f32) = .{ point[3], point[3], point[3], 1 };

        const reciprocal_w = simd.reciprocal(w_vector);
        point.* *= reciprocal_w;
    }

    return result;
}

///Working set for vertex processing
var vtx_positions: [256]@Vector(4, f32) = undefined;
var vtx_interpolators: [256]TestPipelineFragmentInput = undefined;
var vtx_indices: [512]u8 = undefined;
var vtx_indices_count: u32 = 0;

///Emits a triangle for the rasteriser
fn emitTriangle(
    raster_unit: *RasterUnit,
    triangle: [3]@Vector(4, f32),
    interpolators: [3]TestPipelineFragmentInput,
) void {
    var points: [3]@Vector(4, f32) = triangle;

    points = perspectiveDivide(points);

    for (&points) |*point| {
        point.* = @mulAdd(
            @Vector(4, f32),
            point.*,
            raster_unit.viewport_transform.pxyz,
            raster_unit.viewport_transform.oxyz,
        );
    }

    _ = raster_unit.raster_processor_state.work_count.fetchAdd(1, .release);

    _ = raster_unit.out_triangle_queue.tryPush(.{
        .positions = points,
        .interpolators = interpolators,
    });
}

pub fn TriangleClipper(comptime Interpolator: type) type {
    return struct {
        index_in_buffer: [512]u8 = undefined,
        index_out_buffer: [512]u8 = undefined,
        vertex_position_buffer: [512]@Vector(4, f32) = undefined,
        vertex_interpolator_buffer: [512]Interpolator = undefined,

        indices_in: []u8,
        indices_out: []u8,

        vertex_positions: []@Vector(4, f32),
        vertex_interpolators: []Interpolator,

        pub fn init(
            vertex_positions: []const @Vector(4, f32),
            vertex_interpolators: []const Interpolator,
            index1: u8,
            index2: u8,
            index3: u8,
        ) @This() {
            var self = @This(){
                .indices_in = &.{},
                .indices_out = &.{},
                .vertex_positions = &.{},
                .vertex_interpolators = &.{},
            };

            @memset(&self.index_in_buffer, 0);

            self.index_in_buffer[0] = index1;
            self.index_in_buffer[1] = index2;
            self.index_in_buffer[2] = index3;

            self.indices_in = self.index_in_buffer[0..3];

            @memcpy(self.vertex_position_buffer[0..vertex_positions.len], vertex_positions);
            @memcpy(self.vertex_interpolator_buffer[0..vertex_interpolators.len], vertex_interpolators);

            self.vertex_positions = self.vertex_position_buffer[0..vertex_positions.len];
            self.vertex_interpolators = self.vertex_interpolator_buffer[0..vertex_interpolators.len];

            return self;
        }

        pub fn clipToPlane(self: *@This(), plane: @Vector(4, f32)) void {
            if (self.indices_in.len < 3) return;

            self.indices_out.ptr = &self.index_out_buffer;
            self.indices_out.len = 0;

            var index_previous = self.indices_in[0];

            self.indices_in.len += 1;
            self.indices_in[self.indices_in.len - 1] = index_previous;

            const previous_vertex_position = self.vertex_positions[index_previous];

            var previous_dotp = vectorDotProduct(4, f32, plane, previous_vertex_position);

            for (1..self.indices_in.len) |i| {
                const index = self.indices_in[i];

                const vertex_position = self.vertex_positions[index];

                const dotp = vectorDotProduct(4, f32, plane, vertex_position);

                const distance_to_previous = previous_dotp;
                const distance_to_current = dotp;

                const keep = distance_to_previous >= 0;
                _ = keep;

                //If the edge between these points crosses the plane, generate an intersection point
                if ((distance_to_previous < 0 and distance_to_current > 0) or
                    (distance_to_previous > 0 and distance_to_current < 0))
                {
                    const factor = distance_to_previous / (distance_to_previous - distance_to_current);

                    const t = factor;

                    const vertex_out = interpolateVertex(
                        Interpolator,
                        self.vertex_positions[index_previous],
                        self.vertex_positions[index],
                        self.vertex_interpolators[index_previous],
                        self.vertex_interpolators[index],
                        t,
                    );

                    self.vertex_positions.len += 1;
                    self.vertex_positions[self.vertex_positions.len - 1] = vertex_out.position;
                    self.vertex_interpolators.len += 1;
                    self.vertex_interpolators[self.vertex_interpolators.len - 1] = vertex_out.interpolator;

                    self.indices_out.len += 1;
                    self.indices_out[self.indices_out.len - 1] = @intCast(self.vertex_positions.len - 1);
                } else {}

                if (previous_dotp >= 0) {
                    self.indices_out.len += 1;
                    self.indices_out[self.indices_out.len - 1] = index_previous;
                }

                // if (std.math.sign(dotp) != std.math.sign(previous_dotp)) {
                // if (distance_to_current <= 0) {
                //     const t = if (dotp < 0) previous_dotp / (previous_dotp - dotp) else -previous_dotp / (dotp - previous_dotp);

                //     const vertex_out = interpolateVertex(
                //         Interpolator,
                //         self.vertex_positions[index_previous],
                //         self.vertex_positions[index],
                //         self.vertex_interpolators[index_previous],
                //         self.vertex_interpolators[index],
                //         t,
                //     );

                //     self.vertex_positions.len += 1;
                //     self.vertex_positions[self.vertex_positions.len - 1] = vertex_out.position;
                //     self.vertex_interpolators.len += 1;
                //     self.vertex_interpolators[self.vertex_interpolators.len - 1] = vertex_out.interpolator;

                //     self.indices_out.len += 1;
                //     self.indices_out[self.indices_out.len - 1] = @intCast(self.vertex_positions.len - 1);
                // }

                index_previous = index;
                previous_dotp = dotp;
            }

            std.mem.swap([]u8, &self.indices_in, &self.indices_out);
        }

        pub fn clipToPlane2(self: *@This(), plane: @Vector(4, f32)) void {
            if (self.indices_in.len < 3) return;

            var keep_first: bool = true;

            var current_index: usize = 0;

            for (0..self.indices_in.len) |_| {
                var next_index = current_index + 1;

                if (next_index == self.indices_in.len) next_index = 0;

                const current_vertex_position = self.vertex_positions[self.indices_in[current_index]];
                const next_vertex_position = self.vertex_positions[self.indices_in[next_index]];

                const outside_current = vectorDotProduct(4, f32, plane, current_vertex_position);
                const outside_next = vectorDotProduct(4, f32, plane, next_vertex_position);

                var keep = outside_current >= 0;

                if (current_index == 0) {
                    //keep the first index?
                    keep_first = keep;
                    keep = true;
                }

                //if this edge crosses the plane, generate an intersection
                if ((outside_current < 0 and outside_next > 0) or
                    (outside_current > 0 and outside_next < 0))
                {
                    const factor = outside_current / (outside_current - outside_next);

                    const t = factor;

                    const vertex_out = interpolateVertex(
                        Interpolator,
                        self.vertex_positions[self.indices_in[current_index]],
                        self.vertex_positions[self.indices_in[next_index]],
                        self.vertex_interpolators[self.indices_in[current_index]],
                        self.vertex_interpolators[self.indices_in[next_index]],
                        t,
                    );

                    self.vertex_positions = self.vertex_position_buffer[0 .. self.vertex_positions.len + 1];
                    self.vertex_positions[self.vertex_positions.len - 1] = vertex_out.position;

                    self.vertex_interpolators = self.vertex_interpolator_buffer[0 .. self.vertex_interpolators.len + 1];
                    self.vertex_interpolators[self.vertex_interpolators.len - 1] = vertex_out.interpolator;

                    if (keep) {
                        self.indices_in = self.index_in_buffer[0 .. self.indices_in.len + 1];
                        self.indices_in[self.indices_in.len - 1] = @intCast(self.vertex_positions.len - 1);
                    } else {
                        self.indices_in[current_index] = @intCast(self.vertex_positions.len - 1);
                        current_index += 1;
                    }
                } else {
                    if (keep) {
                        current_index += 1;
                    } else {
                        //erase the point (don't generate it?)
                        const remove_index = current_index;

                        if (remove_index != 0) {
                            for (remove_index..self.indices_in.len - 1) |i| {
                                const next = i + 1;

                                self.indices_in[i] = self.indices_in[next];
                            }

                            self.indices_in = self.index_in_buffer[0 .. self.indices_in.len - 1];

                            current_index -= 1;
                        }
                    }
                }
            }

            if (!keep_first) {
                if (self.indices_in.len > 1) {
                    self.indices_in = self.index_in_buffer[1 .. self.indices_in.len - 1];
                }
            }
        }

        pub fn clipToPlaneDeIndexed(self: *@This(), plane: @Vector(4, f32)) void {
            if (self.vertex_positions.len < 3) return;

            var keep_first: bool = true;

            var current_index: usize = 0;

            while (current_index < self.vertex_positions.len) {
                var next_index = current_index + 1;

                if (next_index == self.vertex_positions.len) next_index = 0;

                const current_vertex_position = self.vertex_positions[current_index];
                const next_vertex_position = self.vertex_positions[next_index];

                const outside_current = vectorDotProduct(4, f32, plane, current_vertex_position);
                const outside_next = vectorDotProduct(4, f32, plane, next_vertex_position);

                var keep = outside_current >= 0;

                if (current_index == 0) {
                    //keep the first index?
                    keep_first = keep;
                    keep = true;
                }

                //if this edge crosses the plane, generate an intersection
                if ((outside_current < 0 and outside_next > 0) or
                    (outside_current > 0 and outside_next < 0))
                {
                    // const factor = outside_current / (outside_current - outside_next);
                    // const factor = distance_to_previous / (distance_to_previous - distance_to_current);
                    const t = if (outside_next < 0) outside_current / (outside_current - outside_next) else -outside_current / (outside_next - outside_current);

                    // const t = factor;

                    const vertex_out = interpolateVertex(
                        Interpolator,
                        self.vertex_positions[current_index],
                        self.vertex_positions[next_index],
                        self.vertex_interpolators[current_index],
                        self.vertex_interpolators[next_index],
                        @abs(t),
                    );

                    if (keep) {
                        if (self.vertex_positions.len + 1 > self.vertex_position_buffer.len) {
                            return;
                        }

                        self.vertex_positions = self.vertex_position_buffer[0 .. self.vertex_positions.len + 1];
                        self.vertex_positions[self.vertex_positions.len - 1] = vertex_out.position;

                        self.vertex_interpolators = self.vertex_interpolator_buffer[0 .. self.vertex_interpolators.len + 1];
                        self.vertex_interpolators[self.vertex_interpolators.len - 1] = vertex_out.interpolator;

                        current_index = @intCast(self.vertex_positions.len - 1);
                    } else {
                        self.vertex_positions[current_index] = vertex_out.position;
                        self.vertex_interpolators[current_index] = vertex_out.interpolator;

                        current_index += 1;
                    }
                } else {
                    if (keep) {
                        current_index += 1;
                    } else {
                        //erase the point (don't generate it?)
                        const remove_index = current_index;

                        if (remove_index != 0) {
                            for (remove_index..self.vertex_positions.len - 1) |i| {
                                const next = i + 1;

                                self.vertex_positions[i] = self.vertex_positions[next];
                                self.vertex_interpolators[i] = self.vertex_interpolators[next];
                            }

                            self.vertex_positions = self.vertex_position_buffer[0 .. self.vertex_positions.len - 1];
                            self.vertex_interpolators = self.vertex_interpolator_buffer[0 .. self.vertex_interpolators.len - 1];

                            // current_index -= 1;
                        }
                    }
                }
            }

            if (!keep_first) {
                if (self.vertex_positions.len > 1) {
                    self.vertex_positions = self.vertex_positions[1 .. self.vertex_positions.len - 1];
                    self.vertex_interpolators = self.vertex_interpolators[1 .. self.vertex_interpolators.len - 1];
                }
            }
        }
    };
}

fn matrixVectorProduct(
    comptime N: comptime_int,
    comptime T: type,
    matrix: @Vector(N * N, T),
    vector: @Vector(N, T),
) @Vector(N, T) {
    var result: @Vector(N, f32) = undefined;

    inline for (0..N) |n| {
        var dot_b: @Vector(N, T) = undefined;

        inline for (0..N) |n_prime| {
            dot_b[n_prime] = matrix[n * N + n_prime];
        }

        result[n] = vectorDotProduct(N, T, vector, dot_b);
    }

    return result;
}

fn vectorDotProduct(comptime N: comptime_int, comptime T: type, a: @Vector(N, T), b: @Vector(N, T)) T {
    return @reduce(.Add, a * b);
}

const std = @import("std");
const Renderer = @import("../Renderer.zig");
const Pipeline = @import("../Pipeline.zig");
const RasterUnit = @import("RasterUnit.zig");
const simd = @import("simd.zig");
