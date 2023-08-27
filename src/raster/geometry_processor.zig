pub const OutTriangle = struct {
    positions: [3]@Vector(4, f32),
    interpolators: [3]TestPipelineFragmentInput,
};

pub fn pipelineDrawTriangles(
    raster_unit: *RasterUnit,
    vertex_count: usize,
) void {
    for (0..vertex_count / 3) |index| {
        pipelineDrawTriangle(raster_unit, index);
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

    // position[0] = position0[0] * (1 - t) + position1[0] * t;

    inline for (0..4) |n| {
        position[n] = vectorDotGeneric3D(2, f32, .{ position0[n], position1[n] }, interpolant);
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

fn clipTriangleToPlane(
    comptime Interpolator: type,
    comptime max_vertex_out: comptime_int,
    comptime max_index_out: comptime_int,
    vertex_positions_in: [3]@Vector(4, f32),
    vertex_interpolators_in: [3]Interpolator,
    plane_a: f32,
    plane_b: f32,
    plane_c: f32,
    plane_d: f32,
) struct {
    indices_out: [max_index_out]u8,
    vertex_positions_out: [max_vertex_out]@Vector(4, f32),
    vertex_interpolators_out: [max_vertex_out]Interpolator,
    index_count: u8,
} {
    var vertex_positions_out: [max_vertex_out]@Vector(4, f32) = undefined;
    var vertex_interpolators_out: [max_vertex_out]Interpolator = undefined;
    var indices_out: [max_index_out]u8 = undefined;

    for (vertex_positions_in, 0..) |position, i| {
        vertex_positions_out[i] = .{
            position[0],
            position[1],
            position[2],
            position[3],
        };
    }

    for (vertex_interpolators_in, 0..) |vertex_interpolator, i| {
        vertex_interpolators_out[i] = vertex_interpolator;
    }

    for (0..3) |i| {
        indices_out[i] = @intCast(i);
    }

    var index_count: u8 = 3;
    var vertex_count: usize = 3;

    var previous_index = indices_out[0];

    indices_out[index_count] = previous_index;
    index_count += 1;

    const previous_vertex_position = vertex_positions_out[previous_index];

    var previous_dotp =
        plane_a * previous_vertex_position[0] +
        plane_b * previous_vertex_position[1] +
        plane_c * previous_vertex_position[2] +
        plane_d * previous_vertex_position[3];

    for (1..3) |i| {
        const index: u8 = @intCast(i);

        const vertex_position = &vertex_positions_out[index];

        const dotp =
            plane_a * vertex_position[0] +
            plane_b * vertex_position[1] +
            plane_c * vertex_position[2] +
            plane_d * vertex_position[3];

        if (previous_dotp >= 0) {
            indices_out[index_count] = previous_index;
            index_count += 1;
        }

        if (std.math.sign(dotp) != std.math.sign(previous_dotp)) {
            const t = if (dotp < 0) previous_dotp / (previous_dotp - dotp) else -previous_dotp / (dotp - previous_dotp);

            const vertex_out = interpolateVertex(
                Interpolator,
                vertex_positions_out[previous_index],
                vertex_positions_out[index],
                vertex_interpolators_out[previous_index],
                vertex_interpolators_out[index],
                t,
            );

            vertex_positions_out[vertex_count] = vertex_out.position;
            vertex_interpolators_out[vertex_count] = vertex_out.interpolator;

            vertex_count += 1;

            indices_out[index_count] = @intCast(vertex_count - 1);

            index_count += 1;
        }

        previous_index = index;
        previous_dotp = dotp;
    }

    return .{
        .indices_out = indices_out,
        .vertex_positions_out = vertex_positions_out,
        .vertex_interpolators_out = vertex_interpolators_out,
        .index_count = index_count,
    };
}

fn clipTriangle(
    comptime Interpolator: type,
    comptime max_vertex_out: comptime_int,
    comptime max_index_out: comptime_int,
    vertex_positions_in: [3]@Vector(4, f32),
    vertex_interpolators_in: [3]Interpolator,
) struct {
    indices_out: [max_index_out]u8,
    vertex_positions_out: [max_vertex_out]@Vector(4, f32),
    vertex_interpolators_out: [max_vertex_out]Interpolator,
    index_count: u8,
} {
    var indices_out: [max_index_out]u8 = undefined;
    var index_count: u8 = 3;
    var vertex_positions_out: [max_vertex_out]@Vector(4, f32) = undefined;
    var vertex_interpolators_out: [max_vertex_out]Interpolator = undefined;
    var vertex_count: usize = 3;

    const clipping_planes: [6][4]f32 = .{
        .{ -1, 0, 0, 1 },
        .{ 1, 0, 0, 1 },
        .{ 0, -1, 0, 1 },
        .{ 0, 1, 0, 1 },
        .{ 0, 0, -1, 1 },
        .{ 0, 0, 1, 1 },
    };

    for (vertex_positions_in, 0..) |position, i| {
        vertex_positions_out[i] = position;
    }

    for (vertex_interpolators_in, 0..) |vertex_interpolator, i| {
        vertex_interpolators_out[i] = vertex_interpolator;
    }

    for (0..3) |i| {
        indices_out[i] = @intCast(i);
    }

    var previous_index_count: usize = 3;

    for (clipping_planes) |clipping_plane| {
        for (0..previous_index_count / 3) |triangle_index| {
            const index_0 = indices_out[triangle_index * 3 + 0];
            const index_1 = indices_out[triangle_index * 3 + 1];
            const index_2 = indices_out[triangle_index * 3 + 2];

            const clip_result = clipTriangleToPlane(
                Interpolator,
                8,
                16,
                .{
                    vertex_positions_out[index_0],
                    vertex_positions_out[index_1],
                    vertex_positions_out[index_2],
                },
                .{
                    vertex_interpolators_out[index_0],
                    vertex_interpolators_out[index_1],
                    vertex_interpolators_out[index_2],
                },
                clipping_plane[0],
                clipping_plane[1],
                clipping_plane[2],
                clipping_plane[3],
            );

            previous_index_count = clip_result.index_count;

            for (0..clip_result.index_count) |i| {
                const index = clip_result.indices_out[i];

                vertex_positions_out[vertex_count + index] = clip_result.vertex_positions_out[index];
                vertex_interpolators_out[vertex_count + index] = clip_result.vertex_interpolators_out[index];

                vertex_count += 1;
            }
        }
    }

    return .{
        .indices_out = indices_out,
        .vertex_positions_out = vertex_positions_out,
        .vertex_interpolators_out = vertex_interpolators_out,
        .index_count = index_count,
    };
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
    triangle_index: usize,
) void {
    @setFloatMode(.Optimized);

    var fragment_input_0: TestPipelineFragmentInput = undefined;
    var fragment_input_1: TestPipelineFragmentInput = undefined;
    var fragment_input_2: TestPipelineFragmentInput = undefined;

    var triangle: [3]@Vector(4, f32) = undefined;

    triangle[0] = raster_unit.pipeline.vertexShader(raster_unit.uniform, triangle_index * 3 + 0, &fragment_input_0);
    triangle[1] = raster_unit.pipeline.vertexShader(raster_unit.uniform, triangle_index * 3 + 1, &fragment_input_1);
    triangle[2] = raster_unit.pipeline.vertexShader(raster_unit.uniform, triangle_index * 3 + 2, &fragment_input_2);

    //Correct for upside down meshes
    //TODO: use a 2x2 clip transform matrix
    for (&triangle) |*point| {
        point[1] = 1 - point[1];
    }

    const clip_volume_min = @Vector(4, f32){ -1, -1, 0, 0 };
    const clip_volume_max = @Vector(4, f32){ 1, 1, 1, 1 };

    for (&triangle) |*point| {
        point[0] /= point[3];
        point[1] /= point[3];
        point[2] /= point[3];
    }

    // Pre clip cull
    if ((@reduce(.And, triangle[0] > clip_volume_max) and
        @reduce(.And, triangle[1] > clip_volume_max) and
        @reduce(.And, triangle[2] > clip_volume_max)) or
        (@reduce(.And, triangle[0] < clip_volume_min) and
        @reduce(.And, triangle[1] < clip_volume_min) and
        @reduce(.And, triangle[2] < clip_volume_min)))
    {
        return;
    }

    const two_triangle_area = twoTriangleArea(.{
        .{ triangle[0][0], triangle[0][1] },
        .{ triangle[1][0], triangle[1][1] },
        .{ triangle[2][0], triangle[2][1] },
    });

    //backface and contribution cull
    if (two_triangle_area >= 0) {
        return;
    }

    //True if the triangle is entirely contained in the clip volume
    const entire_fit = ((@reduce(.And, triangle[0] <= clip_volume_max) and
        @reduce(.And, triangle[1] <= clip_volume_max) and
        @reduce(.And, triangle[2] <= clip_volume_max)) or
        (@reduce(.And, triangle[0] >= clip_volume_min) and
        @reduce(.And, triangle[1] >= clip_volume_min) and
        @reduce(.And, triangle[2] >= clip_volume_min)));

    if (entire_fit) {
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
    }

    const clip_result = clipTriangle(
        TestPipelineFragmentInput,
        500,
        500,
        .{ triangle[0], triangle[1], triangle[2] },
        .{ fragment_input_0, fragment_input_1, fragment_input_2 },
    );

    for (0..clip_result.index_count / 3) |clip_triangle_index| {
        const index_0 = clip_result.indices_out[clip_triangle_index * 3];
        const index_1 = clip_result.indices_out[clip_triangle_index * 3 + 1];
        const index_2 = clip_result.indices_out[clip_triangle_index * 3 + 2];

        //emit triangle
        emitTriangle(
            raster_unit,
            .{
                @min(@max(clip_result.vertex_positions_out[index_0], clip_volume_min), clip_volume_max),
                @min(@max(clip_result.vertex_positions_out[index_1], clip_volume_min), clip_volume_max),
                @min(@max(clip_result.vertex_positions_out[index_2], clip_volume_min), clip_volume_max),
            },
            .{
                clip_result.vertex_interpolators_out[index_0],
                clip_result.vertex_interpolators_out[index_1],
                clip_result.vertex_interpolators_out[index_2],
            },
        );
    }
}

///Emits a triangle for the rasteriser
fn emitTriangle(
    raster_unit: *RasterUnit,
    triangle: [3]@Vector(4, f32),
    interpolators: [3]TestPipelineFragmentInput,
) void {
    raster_unit.out_triangles.append(.{
        .positions = triangle,
        .interpolators = interpolators,
    }) catch unreachable;
}

const std = @import("std");
const Renderer = @import("../Renderer.zig");
const Pipeline = @import("../Pipeline.zig");
const RasterUnit = @import("RasterUnit.zig");
