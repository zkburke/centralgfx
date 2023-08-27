///Draws a triangle using rasterisation, supplying a compile time known pipeline
///Of operation functions
///Assumes that points is entirely contained within the clip volume
pub fn pipelineRasteriseTriangle(
    raster_unit: *RasterUnit,
    points: [3]@Vector(4, f32),
    fragment_inputs: [3]geometry_processor.TestPipelineFragmentInput,
) void {
    const Interpolators = geometry_processor.TestPipelineFragmentInput;

    const Edge = struct {
        p0: @Vector(3, isize),
        p1: @Vector(3, isize),

        interpolators0: Interpolators,
        interpolators1: Interpolators,

        pub fn init(
            p0: @Vector(3, isize),
            p1: @Vector(3, isize),
            interpolators0: Interpolators,
            interpolators1: Interpolators,
        ) @This() {
            if (p0[1] < p1[1]) {
                return .{
                    .p0 = p0,
                    .p1 = p1,
                    .interpolators0 = interpolators0,
                    .interpolators1 = interpolators1,
                };
            } else {
                return .{
                    .p0 = p1,
                    .p1 = p0,
                    .interpolators0 = interpolators1,
                    .interpolators1 = interpolators0,
                };
            }
        }
    };

    const Span = struct {
        x0: isize,
        x1: isize,

        interpolators0: Interpolators,
        interpolators1: Interpolators,

        pub fn init(
            x0: isize,
            x1: isize,
            interpolators0: Interpolators,
            interpolators1: Interpolators,
        ) @This() {
            if (x0 < x1) {
                return .{
                    .x0 = x0,
                    .x1 = x1,
                    .interpolators0 = interpolators0,
                    .interpolators1 = interpolators1,
                };
            } else {
                return .{
                    .x0 = x1,
                    .x1 = x0,
                    .interpolators0 = interpolators1,
                    .interpolators1 = interpolators0,
                };
            }
        }

        fn drawEdges(
            _raster_unit: *RasterUnit,
            triangle: [3]@Vector(4, f32),
            triangle_attributes: [3]geometry_processor.TestPipelineFragmentInput,
            screen_area: f32,
            inverse_screen_area: f32,
            left: Edge,
            right: Edge,
        ) void {
            const x_diff: u64 = std.math.absCast(right.p1[0] - right.p0[0]) + std.math.absCast(left.p1[0] - left.p0[0]);

            if (x_diff == 0) return;

            const left_y_diff = left.p1[1] - left.p0[1];

            if (left_y_diff == 0) return;
            // std.debug.assert(left_y_diff != 0);

            const right_y_diff = right.p1[1] - right.p0[1];

            if (right_y_diff == 0) return;
            // std.debug.assert(right_y_diff != 0);

            const left_x_diff = left.p1[0] - left.p0[0];
            const right_x_diff = right.p1[0] - right.p0[0];

            var factor0: f32 = @as(f32, @floatFromInt(right.p0[1] - left.p0[1])) / @as(f32, @floatFromInt(left_y_diff));
            var factor1: f32 = 0;

            const factor_step_0 = 1 / @as(f32, @floatFromInt(left_y_diff));
            const factor_step_1 = 1 / @as(f32, @floatFromInt(right_y_diff));

            if (false) {
                const tile_x: usize = @intCast(std.math.clamp(
                    @as(isize, @intCast(@divFloor(left.p0[0], 4))),
                    0,
                    @as(isize, @intCast(_raster_unit.render_pass.color_image.width / 4)),
                ));
                const tile_y: usize = @intCast(std.math.clamp(
                    @as(isize, @intCast(@divFloor(left.p0[1], 4))),
                    0,
                    @as(isize, @intCast(_raster_unit.render_pass.color_image.height / 4)),
                ));

                const tile_count_x: usize = @intCast(std.math.divCeil(usize, _raster_unit.color_image.width, 4) catch unreachable);
                const tile_count_y: usize = @intCast(std.math.divCeil(isize, right_y_diff, 4) catch unreachable);

                for (tile_y..tile_y + tile_count_y) |tile_y_rel| {
                    const tile_begin_x = tile_x;
                    const tile_begin_y = tile_y_rel;

                    const tile_pointer = _raster_unit.color_image.texel_buffer.ptr + ((tile_begin_x + tile_begin_y * tile_count_x) * (4 * 4));

                    @prefetch(tile_pointer, std.builtin.PrefetchOptions{ .rw = .write, .locality = 3 });
                }
            }

            var pixel_y: isize = @max(right.p0[1], 0);

            while (pixel_y < @min(right.p1[1], @as(isize, @intCast(_raster_unit.render_pass.color_image.height)))) : (pixel_y += 1) {
                defer {
                    factor0 += factor_step_0;
                    factor1 += factor_step_1;
                }

                drawSpan(
                    _raster_unit,
                    @This().init(
                        left.p0[0] + @as(isize, @intFromFloat(@ceil(@as(f32, @floatFromInt(left_x_diff)) * factor0))),
                        right.p0[0] + @as(isize, @intFromFloat(@ceil(@as(f32, @floatFromInt(right_x_diff)) * factor1))),
                        undefined,
                        undefined,
                    ),
                    triangle,
                    triangle_attributes,
                    screen_area,
                    inverse_screen_area,
                    pixel_y,
                );
            }
        }

        fn drawSpan(
            _raster_unit: *RasterUnit,
            span: @This(),
            triangle: [3]@Vector(4, f32),
            triangle_attributes: [3]geometry_processor.TestPipelineFragmentInput,
            screen_area: f32,
            inverse_screen_area: f32,
            pixel_y: isize,
        ) void {
            _ = screen_area;
            const xdiff = span.x1 - span.x0;

            const factor_step = 1 / @as(f32, @floatFromInt(xdiff));

            var factor: f32 = 0;

            var pixel_x: isize = @max(span.x0, 0);

            while (pixel_x < span.x1) : (pixel_x += 1) {
                defer {
                    factor += factor_step;
                }

                defer switch (_raster_unit.pipeline.polygon_fill_mode) {
                    .line => {
                        if (pixel_x == span.x0) {
                            pixel_x = span.x1 - 1;
                        }
                    },
                    .fill => {},
                };

                var point = @Vector(3, f32){
                    ((@as(f32, @floatFromInt(pixel_x)) / @as(f32, @floatFromInt(_raster_unit.render_pass.color_image.width))) * 2) - 1,
                    ((@as(f32, @floatFromInt(pixel_y)) / @as(f32, @floatFromInt(_raster_unit.render_pass.color_image.height))) * 2) - 1,
                    0,
                };

                var barycentrics = calculateBarycentrics2DOptimized(
                    .{
                        .{ triangle[0][0], triangle[0][1] },
                        .{ triangle[1][0], triangle[1][1] },
                        .{ triangle[2][0], triangle[2][1] },
                    },
                    inverse_screen_area,
                    .{ point[0], point[1] },
                );

                barycentrics[0] = barycentrics[0] / triangle[0][3];
                barycentrics[1] = barycentrics[1] / triangle[1][3];
                barycentrics[2] = barycentrics[2] / triangle[2][3];

                barycentrics = barycentrics / @as(@Vector(3, f32), @splat(barycentrics[0] + barycentrics[1] + barycentrics[2]));

                point[2] =
                    triangle[0][2] * barycentrics[0] +
                    triangle[1][2] * barycentrics[1] +
                    triangle[2][2] * barycentrics[2];

                const fragment_index = @as(usize, @intCast(pixel_x)) + @as(usize, @intCast(pixel_y)) * _raster_unit.render_pass.color_image.width;

                const depth = &_raster_unit.render_pass.depth_buffer[fragment_index];

                if (point[2] > depth.*) {
                    continue;
                }

                depth.* = point[2];

                var fragment_input: geometry_processor.TestPipelineFragmentInput = undefined;

                inline for (std.meta.fields(Interpolators)) |field| {
                    const vector_dimensions = @typeInfo(field.type).Vector.len;

                    inline for (0..vector_dimensions) |dimension| {
                        const Component = std.meta.Child(field.type);

                        const interpolator_lane: @Vector(3, Component) = .{
                            @field(triangle_attributes[0], field.name)[dimension],
                            @field(triangle_attributes[1], field.name)[dimension],
                            @field(triangle_attributes[2], field.name)[dimension],
                        };

                        @field(fragment_input, field.name)[dimension] = vectorDotGeneric3D(3, Component, interpolator_lane, barycentrics);
                    }
                }

                const pixel = _raster_unit.render_pass.color_image.texelFetch(.{ @intCast(pixel_x), @intCast(pixel_y) });

                var fragment_color: @Vector(4, f32) = undefined;

                var _fragment_input = fragment_input;

                fragment_color = _raster_unit.pipeline.fragmentShader(_raster_unit.uniform, &_fragment_input);

                pixel.* = Image.Color.fromNormalized(fragment_color);
            }
        }
    };

    const view_scale = @Vector(4, f32){
        @as(f32, @floatFromInt(raster_unit.render_pass.color_image.width)),
        @as(f32, @floatFromInt(raster_unit.render_pass.color_image.height)),
        1,
        1,
    };

    const points_2d: [3]@Vector(2, f32) = .{
        .{ points[0][0], points[0][1] },
        .{ points[1][0], points[1][1] },
        .{ points[2][0], points[2][1] },
    };

    const screen_area = @fabs(vectorCross2D(points_2d[1] - points_2d[0], points_2d[2] - points_2d[0]));
    const inverse_screen_area = 1 / screen_area;

    const p0_orig = @ceil((points[0] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale);
    const p1_orig = @ceil((points[1] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale);
    const p2_orig = @ceil((points[2] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale);

    const p0: @Vector(3, isize) = .{ @intFromFloat(p0_orig[0]), @intFromFloat(p0_orig[1]), @intFromFloat(p0_orig[2]) };
    const p1: @Vector(3, isize) = .{ @intFromFloat(p1_orig[0]), @intFromFloat(p1_orig[1]), @intFromFloat(p1_orig[2]) };
    const p2: @Vector(3, isize) = .{ @intFromFloat(p2_orig[0]), @intFromFloat(p2_orig[1]), @intFromFloat(p2_orig[2]) };

    const edges: [3]Edge = .{
        Edge.init(p0, p1, fragment_inputs[0], fragment_inputs[1]),
        Edge.init(p1, p2, fragment_inputs[1], fragment_inputs[2]),
        Edge.init(p2, p0, fragment_inputs[2], fragment_inputs[0]),
    };

    var max_length: isize = 0;
    var long_edge_index: usize = 0;

    for (edges, 0..) |edge, i| {
        const length = edge.p1[1] - edge.p0[1];
        if (length > max_length) {
            max_length = std.math.absInt(length) catch unreachable;
            long_edge_index = i;
        }
    }

    const short_edge_1: usize = (long_edge_index + 1) % 3;
    const short_edge_2: usize = (long_edge_index + 2) % 3;

    Span.drawEdges(
        raster_unit,
        points,
        fragment_inputs,
        screen_area,
        inverse_screen_area,
        edges[long_edge_index],
        edges[short_edge_1],
    );
    Span.drawEdges(
        raster_unit,
        points,
        fragment_inputs,
        screen_area,
        inverse_screen_area,
        edges[long_edge_index],
        edges[short_edge_2],
    );
}

inline fn calculateBarycentrics2DOptimized(
    triangle: [3]@Vector(2, f32),
    triangle_area_inverse: f32,
    point: @Vector(2, f32),
) @Vector(3, f32) {
    @setRuntimeSafety(false);

    const pb = triangle[1] - point;
    const cp = triangle[2] - point;
    const pa = triangle[0] - point;

    const one_over_area = triangle_area_inverse;

    const areas = @fabs(@Vector(3, f32){
        pb[0] * cp[1],
        cp[0] * pa[1],
        0,
    } - @Vector(3, f32){
        cp[0] * pb[1],
        pa[0] * cp[1],
        0,
    });

    var bary = areas * @as(@Vector(3, f32), @splat(one_over_area));

    bary[2] = 1 - bary[0] - bary[1];

    bary = @max(bary, @Vector(3, f32){ 0, 0, 0 });
    bary = @min(bary, @Vector(3, f32){ 1, 1, 1 });

    return bary;
}

fn vectorCross2D(a: @Vector(2, f32), b: @Vector(2, f32)) f32 {
    return a[0] * b[1] - b[0] * a[1];
}

fn vectorDotGeneric3D(comptime N: comptime_int, comptime T: type, a: @Vector(N, T), b: @Vector(N, T)) T {
    return @reduce(.Add, a * b);
}

const RasterUnit = @import("RasterUnit.zig");
const std = @import("std");
const geometry_processor = @import("geometry_processor.zig");
const Image = @import("../Image.zig");
