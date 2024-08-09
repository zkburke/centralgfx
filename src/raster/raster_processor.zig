fn Fixed(comptime T: type, comptime integer_bits: comptime_int, fractional_bits: comptime_int) type {
    _ = T;
    const UnderlyingInteger = std.meta.Int(.Unsigned, integer_bits + fractional_bits);

    return packed struct(UnderlyingInteger) {
        bits: Int,

        pub const Int = UnderlyingInteger;

        pub const IntegerBits = std.meta.Int(.Signed, integer_bits);
        pub const FractionalBits = std.meta.Int(.Unsigned, fractional_bits);

        pub const Repr = packed struct(Int) {
            int: IntegerBits,
            fract: FractionalBits,
        };

        pub inline fn fromFloat(value: f32) void {
            _ = value;
        }

        pub inline fn negate(a: @This()) @This() {
            var repr: Repr = @bitCast(a);

            repr.int = -repr.int;

            return @bitCast(repr);
        }

        pub inline fn reciprocal(a: @This()) @This() {
            _ = a;
            //y = 1 / x
        }

        pub inline fn add(a: @This(), b: @This()) @This() {
            return .{ .bits = a.bits + b.bits };
        }

        pub inline fn sub(a: @This(), b: @This()) @This() {
            return .{ .bits = a.bits - b.bits };
        }

        pub inline fn mul(a: @This(), b: @This()) @This() {
            return .{ .bits = a.bits * b.bits };
        }

        pub inline fn div(a: @This(), b: @This()) @This() {
            return .{ .bits = a.bits / b.bits };
        }
    };
}

///Processor state
pub const State = struct {
    has_work: std.Thread.Condition = .{},
    has_work_mutex: std.Thread.Mutex = .{},
    should_kill: std.atomic.Value(bool) = .{ .raw = false },
    input_queue: queue.AtomicQueue(geometry_processor.OutTriangle, 10 * 1024) = .{},
    work_count: std.atomic.Value(i32) = .{ .raw = 0 },
};

const queue = @import("../queue.zig");

pub fn threadMain(
    raster_unit: *RasterUnit,
    state: *State,
) void {
    @setRuntimeSafety(false);

    while (!state.should_kill.load(.monotonic)) {
        const work = raster_unit.out_triangle_queue.tryPop() orelse {
            continue;
        };

        const triangle: geometry_processor.OutTriangle = work;

        pipelineRasteriseTriangle(
            raster_unit,
            triangle.positions,
            triangle.interpolators,
        );

        _ = state.work_count.fetchSub(1, .release);
    }
}

///Draws a triangle using rasterisation, supplying a pipeline
///Of operation functions
pub fn pipelineRasteriseTriangle(
    raster_unit: *RasterUnit,
    points: [3]@Vector(4, f32),
    fragment_inputs: [3]geometry_processor.TestPipelineFragmentInput,
) void {
    @setRuntimeSafety(false);

    const Edge = struct {
        p0: @Vector(3, isize),
        p1: @Vector(3, isize),

        pub fn init(
            p0: @Vector(3, isize),
            p1: @Vector(3, isize),
        ) @This() {
            if (p0[1] < p1[1]) {
                return .{
                    .p0 = p0,
                    .p1 = p1,
                };
            } else {
                return .{
                    .p0 = p1,
                    .p1 = p0,
                };
            }
        }
    };

    const points_2d: [3]@Vector(2, f32) = .{
        .{ points[0][0], points[0][1] },
        .{ points[1][0], points[1][1] },
        .{ points[2][0], points[2][1] },
    };

    const screen_area = @abs(vectorCross2D(points_2d[1] - points_2d[0], points_2d[2] - points_2d[0]));
    const inverse_screen_area = 1 / screen_area;

    const p0_orig = @ceil(points[0]);
    const p1_orig = @ceil(points[1]);
    const p2_orig = @ceil(points[2]);

    const p0: @Vector(3, isize) = .{ @intFromFloat(p0_orig[0]), @intFromFloat(p0_orig[1]), @intFromFloat(p0_orig[2]) };
    const p1: @Vector(3, isize) = .{ @intFromFloat(p1_orig[0]), @intFromFloat(p1_orig[1]), @intFromFloat(p1_orig[2]) };
    const p2: @Vector(3, isize) = .{ @intFromFloat(p2_orig[0]), @intFromFloat(p2_orig[1]), @intFromFloat(p2_orig[2]) };

    const edges: [3]Edge = .{
        Edge.init(p0, p1),
        Edge.init(p1, p2),
        Edge.init(p2, p0),
    };

    var max_length: isize = 0;
    var long_edge_index: usize = 0;

    for (edges, 0..) |edge, i| {
        const length = edge.p1[1] - edge.p0[1];
        if (length > max_length) {
            max_length = length;
            long_edge_index = i;
        }
    }

    const short_edge_1: usize = (long_edge_index + 1) % 3;
    const short_edge_2: usize = (long_edge_index + 2) % 3;

    drawEdges(
        raster_unit,
        points,
        fragment_inputs,
        screen_area,
        inverse_screen_area,
        edges[long_edge_index].p0,
        edges[long_edge_index].p1,
        edges[short_edge_1].p0,
        edges[short_edge_1].p1,
        Image.Color{ .r = 255, .g = 0, .b = 0, .a = 255 },
    );
    drawEdges(
        raster_unit,
        points,
        fragment_inputs,
        screen_area,
        inverse_screen_area,
        edges[long_edge_index].p0,
        edges[long_edge_index].p1,
        edges[short_edge_2].p0,
        edges[short_edge_2].p1,
        Image.Color{ .r = 0, .g = 255, .b = 0, .a = 255 },
    );
}

inline fn drawEdges(
    raster_unit: *RasterUnit,
    triangle: [3]@Vector(4, f32),
    triangle_attributes: [3]geometry_processor.TestPipelineFragmentInput,
    screen_area: f32,
    inverse_screen_area: f32,
    left_p0: @Vector(3, isize),
    left_p1: @Vector(3, isize),
    right_p0: @Vector(3, isize),
    right_p1: @Vector(3, isize),
    color: Image.Color,
) void {
    @setRuntimeSafety(false);

    const x_diff: u64 = @abs(right_p1[0] - right_p0[0]) + @abs(left_p1[0] - left_p0[0]);

    if (x_diff == 0) return;

    const left_y_diff = left_p1[1] - left_p0[1];

    if (left_y_diff == 0) return;

    const right_y_diff = right_p1[1] - right_p0[1];

    if (right_y_diff == 0) return;

    const left_x_diff = left_p1[0] - left_p0[0];
    const right_x_diff = right_p1[0] - right_p0[0];

    var factor0: f32 = @as(f32, @floatFromInt(right_p0[1] - left_p0[1])) / @as(f32, @floatFromInt(left_y_diff));
    var factor1: f32 = 0;

    const factor_step_0 = 1 / @as(f32, @floatFromInt(left_y_diff));
    const factor_step_1 = 1 / @as(f32, @floatFromInt(right_y_diff));

    var pixel_y: isize = @max(right_p0[1], raster_unit.scissor.offset[1]);

    var skip_step: f32 = 0;

    if (right_p0[1] < raster_unit.scissor.offset[1]) {
        skip_step = @abs(@as(f32, @floatFromInt(raster_unit.scissor.offset[1] - right_p0[1])));
    }

    factor0 = @mulAdd(f32, factor_step_0, skip_step, factor0);
    factor1 = @mulAdd(f32, factor_step_1, skip_step, factor1);

    const sub_pixels = 16;

    var x0_fixed: isize = left_p0[0] * sub_pixels;
    var x1_fixed: isize = right_p0[0] * sub_pixels;

    // if (right_p0[1] - left_p0[1] == 0) return;

    const x0_increment: isize = @divFloor(left_x_diff * sub_pixels, left_y_diff);
    const x1_increment: isize = @divFloor(right_x_diff * sub_pixels, right_y_diff);
    // const x0_increment: isize = @divFloor(left_x_diff * sub_pixels, left_y_diff * (right_p0[1] - left_p0[1]));

    const skip_step_fixed: isize = @max(raster_unit.scissor.offset[1] - right_p0[1], 0);

    x0_fixed += x0_increment * skip_step_fixed;
    x1_fixed += x1_increment * skip_step_fixed;

    x0_fixed += x0_increment * (right_p0[1] - left_p0[1]);

    while (pixel_y < @min(right_p1[1], raster_unit.scissor.extent[1])) : (pixel_y += 1) {
        defer {
            factor0 += factor_step_0;
            factor1 += factor_step_1;

            x0_fixed += x0_increment;
            x1_fixed += x1_increment;
        }

        const x0 = left_p0[0] + @as(isize, @intFromFloat(@ceil(@as(f32, @floatFromInt(left_x_diff)) * factor0)));
        const x1 = right_p0[0] + @as(isize, @intFromFloat(@ceil(@as(f32, @floatFromInt(right_x_diff)) * factor1)));

        // const x0: isize = @divTrunc(x0_fixed, 16);
        // const x1: isize = @divTrunc(x1_fixed, 16);

        // const x0: isize = x0_fixed >> 4;
        // const x1: isize = x1_fixed >> 4;

        const x_min = @min(x0, x1);
        const x_max = @max(x0, x1);

        const use_simd = true;
        if (use_simd) {
            drawSpanExperimental(
                raster_unit,
                x_min,
                x_max,
                triangle_attributes[0],
                triangle_attributes[1],
                triangle,
                triangle_attributes,
                screen_area,
                inverse_screen_area,
                pixel_y,
            );
        } else {
            drawSpan(
                raster_unit,
                x_min,
                x_max,
                triangle,
                triangle_attributes,
                screen_area,
                inverse_screen_area,
                pixel_y,
                color,
            );
        }
    }
}

inline fn drawSpan(
    raster_unit: *RasterUnit,
    x0: isize,
    x1: isize,
    triangle: [3]@Vector(4, f32),
    triangle_attributes: [3]geometry_processor.TestPipelineFragmentInput,
    screen_area: f32,
    inverse_screen_area: f32,
    pixel_y: isize,
    color: Image.Color,
) void {
    @setRuntimeSafety(false);

    _ = color;
    _ = screen_area;

    const reciprocal_w = simd.reciprocalVec3(.{
        triangle[0][3],
        triangle[1][3],
        triangle[2][3],
    });

    const pixel_increment: isize = switch (raster_unit.pipeline.polygon_fill_mode) {
        .line => @max(@as(isize, @intCast(@abs(x1 - x0 - 1))), 1),
        .fill => 1,
    };

    var pixel_x: isize = @max(x0, raster_unit.scissor.offset[0]);

    while (pixel_x < @min(x1, raster_unit.scissor.extent[0])) : (pixel_x += pixel_increment) {
        const point = @Vector(2, f32){
            @floatFromInt(pixel_x),
            @floatFromInt(pixel_y),
        };

        var barycentrics = calculateBarycentrics2DOptimized(
            .{
                .{ triangle[0][0], triangle[0][1] },
                .{ triangle[1][0], triangle[1][1] },
                .{ triangle[2][0], triangle[2][1] },
            },
            inverse_screen_area,
            point,
        );

        const current_depth =
            triangle[0][2] * barycentrics[0] +
            triangle[1][2] * barycentrics[1] +
            triangle[2][2] * barycentrics[2];

        //Correct for perspective
        barycentrics = barycentrics * reciprocal_w;

        const inverse_barycentric_sum = simd.reciprocalVec3(@as(@Vector(3, f32), @splat(barycentrics[0] + barycentrics[1] + barycentrics[2])));

        barycentrics = barycentrics * inverse_barycentric_sum;

        const fragment_index = @as(usize, @intCast(pixel_x)) + @as(usize, @intCast(pixel_y)) * raster_unit.render_pass.color_image.width;

        if (fragment_index >= raster_unit.render_pass.depth_buffer.len) continue;

        const depth = &raster_unit.render_pass.depth_buffer[fragment_index];

        if (current_depth <= raster_unit.depth_min or current_depth > raster_unit.depth_max) {
            continue;
        }

        if (current_depth > depth.*) {
            continue;
        }

        depth.* = current_depth;

        var fragment_input: geometry_processor.TestPipelineFragmentInput = undefined;

        inline for (std.meta.fields(geometry_processor.TestPipelineFragmentInput)) |field| {
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

        const pixel = raster_unit.render_pass.color_image.texelFetch(.{ .x = @intCast(pixel_x), .y = @intCast(pixel_y) });

        var fragment_color: @Vector(4, f32) = undefined;

        var _fragment_input = fragment_input;

        fragment_color = raster_unit.pipeline.fragmentShader(raster_unit.uniform, &_fragment_input);

        const previous_color = pixel.toNormalized();
        _ = previous_color;

        // pixel.* = Image.Color.fromNormalized(.{ previous_color[0] + 0.1, 0, 0, 1 });
        pixel.* = Image.Color.fromNormalized(fragment_color);
        // pixel.* = color;
    }
}

inline fn drawSpanExperimental(
    raster_unit: *RasterUnit,
    x0: isize,
    x1: isize,
    interpolators0: geometry_processor.TestPipelineFragmentInput,
    interpolators1: geometry_processor.TestPipelineFragmentInput,
    triangle: [3]@Vector(4, f32),
    triangle_attributes: [3]geometry_processor.TestPipelineFragmentInput,
    screen_area: f32,
    inverse_screen_area: f32,
    pixel_y: isize,
) void {
    _ = interpolators1;
    _ = interpolators0;
    _ = screen_area;

    const start_x = @max(x0, raster_unit.scissor.offset[0]);
    const end_x = @min(x1, raster_unit.scissor.extent[0] - 1);

    //Why the fuck is this ever able to be negative?
    if (end_x - start_x <= 0) {
        return;
    }

    const span_width: usize = @intCast(end_x - start_x);

    const span_start: usize = @intCast(start_x);

    //The warp size is the largest vector size that fits in the largest vector register size
    //for the platform that allows for 32 bit scalars for each "thread" of execution per physical register
    const warp_size = 8;
    //Should use divCeil, but for now I'm using divFloor so we don't have to mask anything
    const warp_count = @divTrunc(span_width, warp_size) + 1;

    const last_warp_count: usize = @intCast(@rem(span_width, warp_size));

    const reciprocal_w = simd.reciprocalVec3(.{
        triangle[0][3],
        triangle[1][3],
        triangle[2][3],
    });

    const fragment_y: @Vector(warp_size, u32) = @splat(@intCast(pixel_y));
    const point_y: WarpRegister(f32) = @floatFromInt(fragment_y);

    var last_mask: @Vector(warp_size, bool) = @splat(false);

    for (0..last_warp_count) |i| {
        last_mask[i] = true;
    }

    for (0..warp_count) |warp_index| {
        var fragment_write_mask: @Vector(warp_size, bool) = @splat(true);

        fragment_write_mask = @select(
            bool,
            @as(@Vector(warp_size, bool), @splat(warp_index != warp_count - 1)),
            fragment_write_mask,
            last_mask,
        );

        const warp_scan_start: u32 = @as(u32, @intCast(span_start)) + @as(u32, @truncate(warp_index * 8));

        const fragment_x: @Vector(warp_size, u32) = .{
            warp_scan_start + 0,
            warp_scan_start + 1,
            warp_scan_start + 2,
            warp_scan_start + 3,
            warp_scan_start + 4,
            warp_scan_start + 5,
            warp_scan_start + 6,
            warp_scan_start + 7,
        };

        const point_x: WarpRegister(f32) = @floatFromInt(fragment_x);

        var point_z: WarpRegister(f32) = @splat(0);

        var barycentric_x: WarpRegister(f32) = undefined;
        var barycentric_y: WarpRegister(f32) = undefined;
        var barycentric_z: WarpRegister(f32) = undefined;

        //Compute barycentrics
        {
            const pb_x = @as(WarpRegister(f32), @splat(triangle[1][0])) - point_x;
            const pb_y = @as(WarpRegister(f32), @splat(triangle[1][1])) - point_y;

            const cp_x = @as(WarpRegister(f32), @splat(triangle[2][0])) - point_x;
            const cp_y = @as(WarpRegister(f32), @splat(triangle[2][1])) - point_y;

            const pa_x = @as(WarpRegister(f32), @splat(triangle[0][0])) - point_x;
            const pa_y = @as(WarpRegister(f32), @splat(triangle[0][1])) - point_y;

            const one_over_area = inverse_screen_area;

            const areas_x = @abs(pb_x * cp_y - cp_x * pb_y);
            const areas_y = @abs(cp_x * pa_y - pa_x * cp_y);

            barycentric_x = areas_x * @as(WarpRegister(f32), @splat(one_over_area));
            barycentric_y = areas_y * @as(WarpRegister(f32), @splat(one_over_area));
            barycentric_z = @as(WarpRegister(f32), @splat(1)) - (barycentric_x + barycentric_y);

            barycentric_x = @max(barycentric_x, @as(WarpRegister(f32), @splat(0)));
            barycentric_y = @max(barycentric_y, @as(WarpRegister(f32), @splat(0)));
            barycentric_z = @max(barycentric_z, @as(WarpRegister(f32), @splat(0)));

            barycentric_x = @min(barycentric_x, @as(WarpRegister(f32), @splat(1)));
            barycentric_y = @min(barycentric_y, @as(WarpRegister(f32), @splat(1)));
            barycentric_z = @min(barycentric_z, @as(WarpRegister(f32), @splat(1)));
        }

        barycentric_x = barycentric_x * @as(WarpRegister(f32), @splat(reciprocal_w[0]));
        barycentric_y = barycentric_y * @as(WarpRegister(f32), @splat(reciprocal_w[1]));
        barycentric_z = barycentric_z * @as(WarpRegister(f32), @splat(reciprocal_w[2]));

        const barycentric_sum = barycentric_x + barycentric_y + barycentric_z;

        const reciprocal_barycentric_sum = @as(WarpRegister(f32), @splat(1)) / barycentric_sum;

        //Correct barycentrics to fall within 0-1
        barycentric_x = barycentric_x * reciprocal_barycentric_sum;
        barycentric_y = barycentric_y * reciprocal_barycentric_sum;
        barycentric_z = barycentric_z * reciprocal_barycentric_sum;

        //Compute (interpolate) z
        {
            const a_z = @as(WarpRegister(f32), @splat(triangle[0][2]));
            const b_z = @as(WarpRegister(f32), @splat(triangle[1][2]));
            const c_z = @as(WarpRegister(f32), @splat(triangle[2][2]));

            point_z = @mulAdd(WarpRegister(f32), a_z, barycentric_x, point_z);
            point_z = @mulAdd(WarpRegister(f32), b_z, barycentric_y, point_z);
            point_z = @mulAdd(WarpRegister(f32), c_z, barycentric_z, point_z);
        }

        //Depth test
        //Depth tests can be done by loading 8 depths from the zbuffer at a time
        //We can then mask out the execution of the warps using a comparison
        //If the entire mask is zero then we can discard the whole warp
        const fragment_index = @as(usize, @intCast(warp_scan_start)) + @as(usize, @intCast(pixel_y)) * raster_unit.render_pass.color_image.width;

        const fragment_index_vector: @Vector(warp_size, u32) = .{
            @truncate(@min(fragment_index + 0, raster_unit.render_pass.depth_buffer.len - 1)),
            @truncate(@min(fragment_index + 1, raster_unit.render_pass.depth_buffer.len - 1)),
            @truncate(@min(fragment_index + 2, raster_unit.render_pass.depth_buffer.len - 1)),
            @truncate(@min(fragment_index + 3, raster_unit.render_pass.depth_buffer.len - 1)),
            @truncate(@min(fragment_index + 4, raster_unit.render_pass.depth_buffer.len - 1)),
            @truncate(@min(fragment_index + 5, raster_unit.render_pass.depth_buffer.len - 1)),
            @truncate(@min(fragment_index + 6, raster_unit.render_pass.depth_buffer.len - 1)),
            @truncate(@min(fragment_index + 7, raster_unit.render_pass.depth_buffer.len - 1)),
        };

        const previous_z: WarpRegister(f32) = .{
            raster_unit.render_pass.depth_buffer[fragment_index_vector[0]],
            raster_unit.render_pass.depth_buffer[fragment_index_vector[1]],
            raster_unit.render_pass.depth_buffer[fragment_index_vector[2]],
            raster_unit.render_pass.depth_buffer[fragment_index_vector[3]],
            raster_unit.render_pass.depth_buffer[fragment_index_vector[4]],
            raster_unit.render_pass.depth_buffer[fragment_index_vector[5]],
            raster_unit.render_pass.depth_buffer[fragment_index_vector[6]],
            raster_unit.render_pass.depth_buffer[fragment_index_vector[7]],
        };

        //Mask where true(1) means depth test succeed and 0 means depth test fail
        const z_mask = point_z < previous_z;

        const use_depth_testing = true;

        if (use_depth_testing) {
            const mask_int = @intFromBool(fragment_write_mask);
            const z_mask_int = @intFromBool(z_mask);
            const depth_min: WarpRegister(f32) = @splat(raster_unit.depth_min);
            const depth_max: WarpRegister(f32) = @splat(raster_unit.depth_max);

            const range_check_min = @intFromBool(point_z > depth_min);
            const range_check_max = @intFromBool(point_z < depth_max);

            //Why the fuck do I have to go through hoops to and a bool vector
            const out_mask = mask_int & z_mask_int & range_check_min & range_check_max;
            const vector_one: @Vector(8, u1) = @splat(1);

            fragment_write_mask = out_mask == vector_one;
        }

        const debug_depth_reject = false;

        //If all invocations fail, skip
        if (!@reduce(.Or, fragment_write_mask)) {
            if (debug_depth_reject) {
                inline for (0..warp_size) |pixel_index| {
                    const pixel_out_start: [*]Image.Color = @ptrCast(raster_unit.render_pass.color_image.texelFetch(.{
                        .x = @intCast(warp_scan_start + pixel_index),
                        .y = @intCast(pixel_y),
                    }));

                    if (!z_mask[pixel_index]) {
                        pixel_out_start[0] = Image.Color{ .r = 255, .g = 0, .b = 0, .a = 255 };
                        pixel_out_start[0].r +|= 5;
                        pixel_out_start[0].g = 0;
                        pixel_out_start[0].a = 255;
                    }
                }
            }
            continue;
        }

        if (debug_depth_reject) {
            inline for (0..warp_size) |pixel_index| {
                const pixel_out_start: [*]Image.Color = @ptrCast(raster_unit.render_pass.color_image.texelFetch(.{
                    .x = @intCast(warp_scan_start + pixel_index),
                    .y = @intCast(pixel_y),
                }));

                const depth_out = @select(f32, fragment_write_mask, point_z, previous_z);

                if (fragment_write_mask[pixel_index]) {
                    pixel_out_start[0].g +|= 5;
                    pixel_out_start[0].a = 255;

                    if (fragment_index <= raster_unit.render_pass.depth_buffer.len) {
                        const depth_out_buffer: *align(1) WarpRegister(f32) = @ptrCast(&raster_unit.render_pass.depth_buffer[fragment_index]);
                        depth_out_buffer[pixel_index] = depth_out[pixel_index];
                    }
                }
            }
            continue;
        }

        var r: WarpRegister(f32) = @splat(0);
        var g: WarpRegister(f32) = @splat(0);
        var b: WarpRegister(f32) = @splat(0);
        var a: WarpRegister(f32) = @splat(0);

        var u: WarpRegister(f32) = @splat(0);
        var v: WarpRegister(f32) = @splat(0);

        var normal: Vec3(f32) = Vec3(f32).init(.{ 0, 0, 0 });
        var position_world_space: Vec3(f32) = Vec3(f32).init(.{ 0, 0, 0 });

        //interpolate attributes
        {
            const ua = @as(WarpRegister(f32), @splat(triangle_attributes[0].uv[0]));
            const ub = @as(WarpRegister(f32), @splat(triangle_attributes[1].uv[0]));
            const uc = @as(WarpRegister(f32), @splat(triangle_attributes[2].uv[0]));

            const va = @as(WarpRegister(f32), @splat(triangle_attributes[0].uv[1]));
            const vb = @as(WarpRegister(f32), @splat(triangle_attributes[1].uv[1]));
            const vc = @as(WarpRegister(f32), @splat(triangle_attributes[2].uv[1]));

            u = @mulAdd(WarpRegister(f32), ua, barycentric_x, u);
            u = @mulAdd(WarpRegister(f32), ub, barycentric_y, u);
            u = @mulAdd(WarpRegister(f32), uc, barycentric_z, u);

            v = @mulAdd(WarpRegister(f32), va, barycentric_x, v);
            v = @mulAdd(WarpRegister(f32), vb, barycentric_y, v);
            v = @mulAdd(WarpRegister(f32), vc, barycentric_z, v);

            u = @min(@abs(u), @as(WarpRegister(f32), @splat(1)));
            v = @min(@abs(v), @as(WarpRegister(f32), @splat(1)));

            const ra = @as(WarpRegister(f32), @splat(triangle_attributes[0].color[0]));
            const rb = @as(WarpRegister(f32), @splat(triangle_attributes[1].color[0]));
            const rc = @as(WarpRegister(f32), @splat(triangle_attributes[2].color[0]));

            const ga = @as(WarpRegister(f32), @splat(triangle_attributes[0].color[1]));
            const gb = @as(WarpRegister(f32), @splat(triangle_attributes[1].color[1]));
            const gc = @as(WarpRegister(f32), @splat(triangle_attributes[2].color[1]));

            const ba = @as(WarpRegister(f32), @splat(triangle_attributes[0].color[2]));
            const bb = @as(WarpRegister(f32), @splat(triangle_attributes[1].color[2]));
            const bc = @as(WarpRegister(f32), @splat(triangle_attributes[2].color[2]));

            const aa = @as(WarpRegister(f32), @splat(triangle_attributes[0].color[3]));
            const ab = @as(WarpRegister(f32), @splat(triangle_attributes[1].color[3]));
            const ac = @as(WarpRegister(f32), @splat(triangle_attributes[2].color[3]));

            r = @mulAdd(WarpRegister(f32), ra, barycentric_x, r);
            r = @mulAdd(WarpRegister(f32), rb, barycentric_y, r);
            r = @mulAdd(WarpRegister(f32), rc, barycentric_z, r);

            g = @mulAdd(WarpRegister(f32), ga, barycentric_x, g);
            g = @mulAdd(WarpRegister(f32), gb, barycentric_y, g);
            g = @mulAdd(WarpRegister(f32), gc, barycentric_z, g);

            b = @mulAdd(WarpRegister(f32), ba, barycentric_x, b);
            b = @mulAdd(WarpRegister(f32), bb, barycentric_y, b);
            b = @mulAdd(WarpRegister(f32), bc, barycentric_z, b);

            a = @mulAdd(WarpRegister(f32), aa, barycentric_x, a);
            a = @mulAdd(WarpRegister(f32), ab, barycentric_y, a);
            a = @mulAdd(WarpRegister(f32), ac, barycentric_z, a);

            {
                inline for (0..3) |comp_idx| {
                    const value_a = @as(WarpRegister(f32), @splat(triangle_attributes[0].normal[comp_idx]));
                    const value_b = @as(WarpRegister(f32), @splat(triangle_attributes[1].normal[comp_idx]));
                    const value_c = @as(WarpRegister(f32), @splat(triangle_attributes[2].normal[comp_idx]));

                    normal.set(comp_idx, @mulAdd(WarpRegister(f32), value_a, barycentric_x, normal.get(comp_idx)));
                    normal.set(comp_idx, @mulAdd(WarpRegister(f32), value_b, barycentric_y, normal.get(comp_idx)));
                    normal.set(comp_idx, @mulAdd(WarpRegister(f32), value_c, barycentric_z, normal.get(comp_idx)));
                }
            }

            {
                inline for (0..3) |comp_idx| {
                    const value_a = @as(WarpRegister(f32), @splat(triangle_attributes[0].position_world_space[comp_idx]));
                    const value_b = @as(WarpRegister(f32), @splat(triangle_attributes[1].position_world_space[comp_idx]));
                    const value_c = @as(WarpRegister(f32), @splat(triangle_attributes[2].position_world_space[comp_idx]));

                    position_world_space.set(comp_idx, @mulAdd(WarpRegister(f32), value_a, barycentric_x, position_world_space.get(comp_idx)));
                    position_world_space.set(comp_idx, @mulAdd(WarpRegister(f32), value_b, barycentric_y, position_world_space.get(comp_idx)));
                    position_world_space.set(comp_idx, @mulAdd(WarpRegister(f32), value_c, barycentric_z, position_world_space.get(comp_idx)));
                }
            }
        }

        const uniforms: *const @import("root").TestPipelineUniformInput = @ptrCast(@alignCast(raster_unit.uniform));

        const pixels: @Vector(8, u32) = if (uniforms.texture != null) shader8xTest(
            uniforms,
            @ptrCast(uniforms.texture.?.texel_buffer.ptr),
            @intCast(uniforms.texture.?.width),
            @intCast(uniforms.texture.?.height),
            normal,
            position_world_space,
            u,
            v,
        ) else @splat(0);

        const fetched_pixels_unpacked = unpackUnorm4xf32(pixels);

        const fragments = packUnorm4xf32(
            r * fetched_pixels_unpacked.x,
            g * fetched_pixels_unpacked.y,
            b * fetched_pixels_unpacked.z,
            a * fetched_pixels_unpacked.w,
        );

        const depth_out_buffer: *align(1) WarpRegister(f32) = @ptrCast(&raster_unit.render_pass.depth_buffer[fragment_index]);
        const fragment_out_buffer: *align(1) @Vector(warp_size, u32) = @ptrCast(&raster_unit.render_pass.color_image.texel_buffer.ptr[fragment_index]);

        maskedStore(f32, depth_out_buffer, point_z, fragment_write_mask);
        maskedStore(u32, fragment_out_buffer, fragments, fragment_write_mask);
    }
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

    const areas = @abs(@Vector(3, f32){
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

    // bary = @max(bary, @Vector(3, f32){ 0, 0, 0 });
    // bary = @min(bary, @Vector(3, f32){ 1, 1, 1 });

    return bary;
}

fn vectorCross2D(a: @Vector(2, f32), b: @Vector(2, f32)) f32 {
    return a[0] * b[1] - b[0] * a[1];
}

fn vectorDotGeneric3D(comptime N: comptime_int, comptime T: type, a: @Vector(N, T), b: @Vector(N, T)) T {
    return @reduce(.Add, a * b);
}

inline fn gather(
    base: [*]align(16) const u32,
    address: @Vector(8, u32),
) @Vector(8, u32) {
    const mask: @Vector(8, u32) = @splat(std.math.maxInt(u32));
    return asm (
        \\vgatherdps %[mask], (%[base], %[address], 4), %[ret]
        : [ret] "=v" (-> @Vector(8, u32)),
        : [address] "{ymm15}" (address),
          [base] "r" (base),
          [mask] "{ymm14}" (mask),
    );
}

inline fn scatter(base: [*]u32, address: @Vector(8, u32), values: @Vector(8, u32)) void {
    inline for (0..8) |i| {
        base[address[i]] = values[i];
    }
}

//Stores values into destination based on the value of the predicate
inline fn maskedStore(
    comptime T: type,
    dest: *align(1) @Vector(8, T),
    values: @Vector(8, T),
    predicate: @Vector(8, bool),
) void {
    const shift_amnt: @Vector(8, u32) = @splat(31);
    const mask: @Vector(8, u32) = @as(@Vector(8, u32), @intFromBool(predicate)) << shift_amnt;

    asm volatile (
        \\vmaskmovps %[values], %[mask], (%[dest])
        :
        : [dest] "r" (dest),
          [values] "v" (values),
          [mask] "v" (mask),
    );
}

inline fn packUnorm4xf32(
    x: @Vector(8, f32),
    y: @Vector(8, f32),
    z: @Vector(8, f32),
    w: @Vector(8, f32),
) @Vector(8, u32) {
    const x_scaled = x * @as(@Vector(8, f32), @splat(255));
    const y_scaled = y * @as(@Vector(8, f32), @splat(255));
    const z_scaled = z * @as(@Vector(8, f32), @splat(255));
    const w_scaled = w * @as(@Vector(8, f32), @splat(255));

    const x_int: @Vector(8, u32) = @intFromFloat(x_scaled);
    const y_int: @Vector(8, u32) = @intFromFloat(y_scaled);
    const z_int: @Vector(8, u32) = @intFromFloat(z_scaled);
    const w_int: @Vector(8, u32) = @intFromFloat(w_scaled);

    var result: @Vector(8, u32) = undefined;

    result = @intCast(w_int << @as(@Vector(8, u5), @splat(24)));
    result |= @intCast(z_int << @as(@Vector(8, u5), @splat(16)));
    result |= @intCast(y_int << @as(@Vector(8, u5), @splat(8)));
    result |= @intCast(x_int);

    return result;
}

inline fn unpackUnorm4xf32(value: @Vector(8, u32)) Vec4(f32) {
    const w_int: @Vector(8, u32) = @intCast((value & @as(@Vector(8, u32), @splat(0xff000000))) >> @as(@Vector(8, u5), @splat(24)));
    const z_int: @Vector(8, u32) = @intCast((value & @as(@Vector(8, u32), @splat(0x00ff0000))) >> @as(@Vector(8, u5), @splat(16)));
    const y_int: @Vector(8, u32) = @intCast((value & @as(@Vector(8, u32), @splat(0x0000ff00))) >> @as(@Vector(8, u5), @splat(8)));
    const x_int: @Vector(8, u32) = @intCast(value & @as(@Vector(8, u32), @splat(0x000000ff)));

    const x: @Vector(8, f32) = @floatFromInt(x_int);
    const y: @Vector(8, f32) = @floatFromInt(y_int);
    const z: @Vector(8, f32) = @floatFromInt(z_int);
    const w: @Vector(8, f32) = @floatFromInt(w_int);

    const x_scaled = x / @as(@Vector(8, f32), @splat(255));
    const y_scaled = y / @as(@Vector(8, f32), @splat(255));
    const z_scaled = z / @as(@Vector(8, f32), @splat(255));
    const w_scaled = w / @as(@Vector(8, f32), @splat(255));

    return .{
        .x = x_scaled,
        .y = y_scaled,
        .z = z_scaled,
        .w = w_scaled,
    };
}

///Pointer to an optimally tiled (8x8) image
const ImageDescriptor = packed struct(u128) {
    base: [*]align(16) const u32,
    width: u32,
    height: u32,
};

pub fn WarpRegister(comptime T: type) type {
    const warp_size = 8;

    return @Vector(warp_size, T);
}

pub fn Vec2(comptime T: type) type {
    return struct {
        x: WarpRegister(T),
        y: WarpRegister(T),
    };
}

pub fn Vec3(comptime T: type) type {
    return struct {
        x: WarpRegister(T),
        y: WarpRegister(T),
        z: WarpRegister(T),

        pub inline fn get(self: @This(), comptime index: u32) WarpRegister(f32) {
            return switch (index) {
                0 => self.x,
                1 => self.y,
                2 => self.z,
                else => @compileError("Index out of bounds!"),
            };
        }

        pub inline fn set(self: *@This(), comptime index: u32, value: WarpRegister(f32)) void {
            return switch (index) {
                0 => self.x = value,
                1 => self.y = value,
                2 => self.z = value,
                else => @compileError("Index out of bounds!"),
            };
        }

        pub inline fn init(
            values: [3]f32,
        ) @This() {
            return .{
                .x = @splat(values[0]),
                .y = @splat(values[1]),
                .z = @splat(values[2]),
            };
        }

        pub inline fn neg(self: @This()) @This() {
            return .{
                .x = -self.x,
                .y = -self.y,
                .z = -self.z,
            };
        }

        pub inline fn norm(self: @This()) @This() {
            const one: WarpRegister(f32) = @splat(1);
            const inverse_mag = one / self.mag();
            return self.scale(inverse_mag);
        }

        pub inline fn mag(self: @This()) WarpRegister(f32) {
            const squared = self.scalarProduct(self);

            return @sqrt(squared);
        }

        pub inline fn distance(left: @This(), right: @This()) WarpRegister(f32) {
            return mag(left.add(right.neg()));
        }

        pub inline fn add(left: @This(), right: @This()) @This() {
            return .{
                .x = left.x + right.x,
                .y = left.y + right.y,
                .z = left.z + right.z,
            };
        }

        pub inline fn hadamardProduct(left: @This(), right: @This()) @This() {
            return .{
                .x = left.x * right.x,
                .y = left.y * right.y,
                .z = left.z * right.z,
            };
        }

        pub inline fn scalarProduct(left: @This(), right: @This()) WarpRegister(T) {
            const xx = left.x * right.x;
            const yy = left.y * right.y;
            const zz = left.z * right.z;

            return xx + yy + zz;
        }

        pub inline fn scale(left: @This(), right: WarpRegister(T)) @This() {
            return .{
                .x = left.x * right,
                .y = left.y * right,
                .z = left.z * right,
            };
        }
    };
}

pub fn Vec4(comptime T: type) type {
    return struct {
        x: WarpRegister(T),
        y: WarpRegister(T),
        z: WarpRegister(T),
        w: WarpRegister(T),

        pub inline fn neg(self: @This()) @This() {
            return .{
                .x = -self.x,
                .y = -self.y,
                .z = -self.z,
                .w = -self.w,
            };
        }

        pub inline fn add(left: @This(), right: @This()) @This() {
            return .{
                .x = left.x + right.x,
                .y = left.y + right.y,
                .z = left.z + right.z,
                .w = left.w + right.w,
            };
        }

        pub inline fn hadamardProduct(left: @This(), right: @This()) @This() {
            return .{
                .x = left.x * right.x,
                .y = left.y * right.y,
                .z = left.z * right.z,
                .w = left.w * right.w,
            };
        }

        pub inline fn scalarProduct(left: @This(), right: @This()) WarpRegister(T) {
            const xx = left.x * right.x;
            const yy = left.y * right.y;
            const zz = left.z * right.z;
            const ww = left.w * right.w;

            return xx + yy + zz + ww;
        }

        pub inline fn scale(left: @This(), right: WarpRegister(T)) @This() {
            return .{
                .x = left.x * right,
                .y = left.y * right,
                .z = left.z * right,
                .w = left.w * right,
            };
        }
    };
}

pub fn Mat4x4(comptime T: type) type {
    return struct {
        elements: [16]WarpRegister(T),
    };
}

inline fn imageTexelFetch(
    descriptor: ImageDescriptor,
    x: WarpRegister(u32),
    y: WarpRegister(u32),
) WarpRegister(u32) {
    //idx = y * width + x = fma(y, width, x); (linear mode)
    var texel_address: @Vector(8, u32) = undefined;

    //Could use a shift if image widths were restricted to be power of two size
    texel_address = y * @as(@Vector(8, u32), @splat(descriptor.width)) + x;

    const tile_width = 8;
    const tile_height = tile_width;

    //texel fetch
    const tile_count_x: @Vector(8, u32) = @splat(descriptor.width / tile_width);

    const tile_x = @divFloor(x, @as(@Vector(8, u32), @splat(tile_width)));
    const tile_y = @divFloor(y, @as(@Vector(8, u32), @splat(tile_height)));

    const tile_begin_x = tile_x * @as(@Vector(8, u32), @splat(tile_width));
    const tile_begin_y = tile_y * @as(@Vector(8, u32), @splat(tile_height));

    const tile_pointer: @Vector(8, u32) = ((tile_x + tile_y * tile_count_x) * @as(@Vector(8, u32), @splat(tile_width * tile_height)));

    //x, y relative to tile
    const local_tile_x = x - tile_begin_x;
    const local_tile_y = y - tile_begin_y;

    texel_address = tile_pointer + local_tile_y * @as(@Vector(8, u32), @splat(tile_width)) + local_tile_x;

    const texel = gather(descriptor.base, texel_address);

    return texel;
}

///Nearest neighbour texture sampling. Does not handle wrapping
inline fn imageTexelSampleNearest(
    descriptor: ImageDescriptor,
    u: WarpRegister(f32),
    v: WarpRegister(f32),
) WarpRegister(u32) {
    const width_float: @Vector(8, f32) = @splat(@floatFromInt(descriptor.width));
    const height_float: @Vector(8, f32) = @splat(@floatFromInt(descriptor.height));

    var texel_x: @Vector(8, u32) = undefined;
    var texel_y: @Vector(8, u32) = undefined;

    texel_x = @intFromFloat(u * width_float);
    texel_y = @intFromFloat(v * height_float);

    const texel = imageTexelFetch(
        descriptor,
        texel_x,
        texel_y,
    );

    return texel;
}

///Bilinear texture sampling. Does not handle wrapping
inline fn imageTexelSampleLinear4x(
    descriptor: ImageDescriptor,
    u: WarpRegister(f32),
    v: WarpRegister(f32),
) Vec4(f32) {
    const scaled_u = u * @as(WarpRegister(f32), @splat(@floatFromInt(descriptor.width)));
    const scaled_v = v * @as(WarpRegister(f32), @splat(@floatFromInt(descriptor.height)));

    const texel_point_x = @floor(scaled_u);
    const texel_point_y = @floor(scaled_v);
    const texel_point_offset_x = scaled_u - texel_point_x;
    const texel_point_offset_y = scaled_v - texel_point_y;
    _ = texel_point_offset_y; // autofix
    _ = texel_point_offset_x; // autofix

    const texel_0 = imageTexelFetch(descriptor, u, v);
    _ = texel_0; // autofix
}

///Execute 8 fragment shaders at a time
///Returns a 8 vector of u32 rgba values
fn shader8xTest(
    uniforms: *const @import("root").TestPipelineUniformInput,
    img: [*]align(16) const u32,
    ///Power of two image width
    img_width_po2: u32,
    ///Power of two image height
    img_height_po2: u32,
    normal: Vec3(f32),
    position_world_space: Vec3(f32),
    u: WarpRegister(f32),
    v: WarpRegister(f32),
) WarpRegister(u32) {
    const texel = imageTexelSampleNearest(
        .{ .base = img, .width = img_width_po2, .height = img_height_po2 },
        u,
        v,
    );

    var color: Vec4(f32) = unpackUnorm4xf32(texel);

    const enable_lighting = true;

    if (enable_lighting) {
        var light_contribution = Vec3(f32).init(.{ 0.1, 0.1, 0.1 });

        for (uniforms.lights) |light| {
            const light_pos: Vec3(f32) = .{
                .x = @splat(light.position[0]),
                .y = @splat(light.position[1]),
                .z = @splat(light.position[2]),
            };
            const light_color = Vec3(f32).init(light.color);
            const light_dir = Vec3(f32).norm(light_pos.add(position_world_space.neg()));
            const light_radius: WarpRegister(f32) = @splat(0.01);

            const distance_to_light = Vec3(f32).distance(light_pos, position_world_space);

            const attentuation = @as(WarpRegister(f32), @splat(1)) / @max(distance_to_light * distance_to_light, light_radius * light_radius);

            const light_intensity = @max(Vec3(f32).scalarProduct(normal, light_dir) * @as(WarpRegister(f32), @splat(light.intensity)), @as(WarpRegister(f32), @splat(0)));
            const attenuated_light_intensity = light_intensity * attentuation;

            const local_contribution = light_color.hadamardProduct(Vec3(f32){ .x = attenuated_light_intensity, .y = attenuated_light_intensity, .z = attenuated_light_intensity });

            light_contribution = light_contribution.add(local_contribution);
        }

        color.x *= light_contribution.x;
        color.y *= light_contribution.y;
        color.z *= light_contribution.z;
    }

    color.x = std.math.clamp(color.x, @as(WarpRegister(f32), @splat(0)), @as(WarpRegister(f32), @splat(1)));
    color.y = std.math.clamp(color.y, @as(WarpRegister(f32), @splat(0)), @as(WarpRegister(f32), @splat(1)));
    color.z = std.math.clamp(color.z, @as(WarpRegister(f32), @splat(0)), @as(WarpRegister(f32), @splat(1)));

    return packUnorm4xf32(color.x, color.y, color.z, color.w);
}

const RasterUnit = @import("RasterUnit.zig");
const std = @import("std");
const geometry_processor = @import("geometry_processor.zig");
const Image = @import("../Image.zig");
const simd = @import("simd.zig");
