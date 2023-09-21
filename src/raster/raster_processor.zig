///Draws a triangle using rasterisation, supplying a pipeline
///Of operation functions
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

    const view_scale = @Vector(4, f32){
        @as(f32, @floatFromInt(raster_unit.render_pass.color_image.width)),
        @as(f32, @floatFromInt(raster_unit.render_pass.color_image.height)),
        1,
        1,
    };
    _ = view_scale;

    const points_2d: [3]@Vector(2, f32) = .{
        .{ points[0][0], points[0][1] },
        .{ points[1][0], points[1][1] },
        .{ points[2][0], points[2][1] },
    };

    const screen_area = @fabs(vectorCross2D(points_2d[1] - points_2d[0], points_2d[2] - points_2d[0]));
    const inverse_screen_area = 1 / screen_area;

    // const p0_orig = @ceil((points[0] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale);
    // const p1_orig = @ceil((points[1] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale);
    // const p2_orig = @ceil((points[2] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale);

    const p0_orig = @ceil(points[0]);
    const p1_orig = @ceil(points[1]);
    const p2_orig = @ceil(points[2]);

    //std.log.info("points[0] = {}", .{points[0]});
    //std.log.info("points[1] = {}", .{points[1]});
    //std.log.info("points[2] = {}", .{points[2]});

    //std.log.info("p0_orig = {}", .{p0_orig});
    //std.log.info("p1_orig = {}", .{p1_orig});
    //std.log.info("p2_orig = {}", .{p2_orig});

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

    drawEdges(
        raster_unit,
        points,
        fragment_inputs,
        screen_area,
        inverse_screen_area,
        edges[long_edge_index].p0,
        edges[long_edge_index].p1,
        edges[long_edge_index].interpolators0,
        edges[long_edge_index].interpolators1,
        edges[short_edge_1].p0,
        edges[short_edge_1].p1,
        edges[short_edge_1].interpolators0,
        edges[short_edge_1].interpolators1,
    );
    drawEdges(
        raster_unit,
        points,
        fragment_inputs,
        screen_area,
        inverse_screen_area,
        edges[long_edge_index].p0,
        edges[long_edge_index].p1,
        edges[long_edge_index].interpolators0,
        edges[long_edge_index].interpolators1,
        edges[short_edge_2].p0,
        edges[short_edge_2].p1,
        edges[short_edge_2].interpolators0,
        edges[short_edge_2].interpolators1,
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
    left_interpolators0: geometry_processor.TestPipelineFragmentInput,
    left_interpolators1: geometry_processor.TestPipelineFragmentInput,
    right_p0: @Vector(3, isize),
    right_p1: @Vector(3, isize),
    right_interpolators0: geometry_processor.TestPipelineFragmentInput,
    right_interpolators1: geometry_processor.TestPipelineFragmentInput,
) void {
    _ = right_interpolators1;
    _ = right_interpolators0;
    _ = left_interpolators1;
    _ = left_interpolators0;
    const x_diff: u64 = std.math.absCast(right_p1[0] - right_p0[0]) + std.math.absCast(left_p1[0] - left_p0[0]);

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
        skip_step = @fabs(@as(f32, @floatFromInt(raster_unit.scissor.offset[1] - right_p0[1])));
    }

    factor0 = @mulAdd(f32, factor_step_0, skip_step, factor0);
    factor1 = @mulAdd(f32, factor_step_1, skip_step, factor1);

    while (pixel_y < @min(right_p1[1], raster_unit.scissor.extent[1])) : (pixel_y += 1) {
        defer {
            factor0 += factor_step_0;
            factor1 += factor_step_1;
        }

        //start and end of the span
        var x0 = left_p0[0] + @as(isize, @intFromFloat(@ceil(@as(f32, @floatFromInt(left_x_diff)) * factor0)));
        var x1 = right_p0[0] + @as(isize, @intFromFloat(@ceil(@as(f32, @floatFromInt(right_x_diff)) * factor1)));

        if (x1 < x0) {
            std.mem.swap(isize, &x0, &x1);
        }

        drawSpan(
            raster_unit,
            x0,
            x1,
            undefined,
            undefined,
            triangle,
            triangle_attributes,
            screen_area,
            inverse_screen_area,
            pixel_y,
        );
    }
}

inline fn drawSpan(
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

    const reciprocal_w = simd.reciprocalVec3(.{
        triangle[0][3],
        triangle[1][3],
        triangle[2][3],
    });

    const xdiff = x1 - x0;

    const factor_step = 1 / @as(f32, @floatFromInt(xdiff));

    var factor: f32 = 0;

    const pixel_increment: isize = switch (raster_unit.pipeline.polygon_fill_mode) {
        .line => @max(@as(isize, @intCast(std.math.absCast(x1 - x0 - 1))), 1),
        .fill => 1,
    };

    var pixel_x: isize = @max(x0, raster_unit.scissor.offset[0]);

    while (pixel_x < @min(x1, raster_unit.scissor.extent[0])) : (pixel_x += pixel_increment) {
        defer {
            factor += factor_step;
        }

        var point = @Vector(2, f32){
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

        const pixel = raster_unit.render_pass.color_image.texelFetch(.{ @intCast(pixel_x), @intCast(pixel_y) });

        var fragment_color: @Vector(4, f32) = undefined;

        var _fragment_input = fragment_input;

        fragment_color = raster_unit.pipeline.fragmentShader(raster_unit.uniform, &_fragment_input);

        const previous_color = pixel.toNormalized();
        _ = previous_color;

        // pixel.* = Image.Color.fromNormalized(.{ previous_color[0] + 0.1, 0, 0, 1 });
        pixel.* = Image.Color.fromNormalized(fragment_color);
    }
}

fn drawSpanExperimental(
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
    // var pixel_x: isize = @max(x0, raster_unit.scissor.offset[0]);
    // var pixel_x: isize = x0;

    const pixel_increment: isize = switch (raster_unit.pipeline.polygon_fill_mode) {
        .line => @max(@as(isize, @intCast(std.math.absCast(x1 - x0 - 1))), 1),
        .fill => 1,
    };
    _ = pixel_increment;

    const start_x = @max(x0, raster_unit.scissor.offset[0]);
    _ = start_x;
    const end_x = @min(x1, raster_unit.scissor.extent[0]);
    _ = end_x;

    const span_width = x1 - x0;

    //The warp size is the largest vector size that fits in the largest vector register size
    //for the platform that allows for 32 bit scalars for each "thread" of execution per physical register
    const warp_size = 8;
    //Should use divCeil, but for now I'm using divFloor so we don't have to mask anything
    const warp_count = std.math.absCast(std.math.divCeil(isize, span_width, warp_size) catch unreachable) + 1;

    const last_warp_count: usize = @intCast(@rem(span_width, warp_size));

    const reciprocal_w = simd.reciprocalVec3(.{
        triangle[0][3],
        triangle[1][3],
        triangle[2][3],
    });

    const fragment_y: @Vector(warp_size, u32) = @splat(@intCast(pixel_y));
    const point_y: @Vector(warp_size, f32) = @floatFromInt(fragment_y);

    for (0..warp_count) |warp_index| {
        var fragment_write_mask: @Vector(warp_size, bool) = @splat(true);

        if (warp_index == warp_count - 1) {
            for (last_warp_count..last_warp_count + 8 - last_warp_count) |i| {
                fragment_write_mask[i] = false;
            }
        }

        const warp_scan_start: u32 = @as(u32, @intCast(x0)) + @as(u32, @intCast(warp_index * 8));

        // const pixel_out_start: [*]Image.Color = @ptrCast(raster_unit.render_pass.color_image.texelFetch(.{
        //     .x = @intCast(warp_scan_start),
        //     .y = @intCast(pixel_y),
        // }));

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

        const point_x: @Vector(warp_size, f32) = @floatFromInt(fragment_x);

        var point_z: @Vector(warp_size, f32) = @splat(0);

        var barycentric_x: @Vector(warp_size, f32) = undefined;
        var barycentric_y: @Vector(warp_size, f32) = undefined;
        var barycentric_z: @Vector(warp_size, f32) = undefined;

        //Compute barycentrics
        {
            const pb_x = @as(@Vector(warp_size, f32), @splat(triangle[1][0])) - point_x;
            const pb_y = @as(@Vector(warp_size, f32), @splat(triangle[1][1])) - point_y;

            const cp_x = @as(@Vector(warp_size, f32), @splat(triangle[2][0])) - point_x;
            const cp_y = @as(@Vector(warp_size, f32), @splat(triangle[2][1])) - point_y;

            const pa_x = @as(@Vector(warp_size, f32), @splat(triangle[0][0])) - point_x;
            const pa_y = @as(@Vector(warp_size, f32), @splat(triangle[0][1])) - point_y;

            const one_over_area = inverse_screen_area;

            const areas_x = @fabs(pb_x * cp_y - cp_x * pb_y);
            const areas_y = @fabs(cp_x * pa_y - pa_x * cp_y);

            barycentric_x = areas_x * @as(@Vector(warp_size, f32), @splat(one_over_area));
            barycentric_y = areas_y * @as(@Vector(warp_size, f32), @splat(one_over_area));
            barycentric_z = @as(@Vector(warp_size, f32), @splat(1)) - (barycentric_x + barycentric_y);

            barycentric_x = @max(barycentric_x, @as(@Vector(warp_size, f32), @splat(0)));
            barycentric_y = @max(barycentric_y, @as(@Vector(warp_size, f32), @splat(0)));
            barycentric_z = @max(barycentric_z, @as(@Vector(warp_size, f32), @splat(0)));

            barycentric_x = @min(barycentric_x, @as(@Vector(warp_size, f32), @splat(1)));
            barycentric_y = @min(barycentric_y, @as(@Vector(warp_size, f32), @splat(1)));
            barycentric_z = @min(barycentric_z, @as(@Vector(warp_size, f32), @splat(1)));
        }

        barycentric_x = barycentric_x * @as(@Vector(warp_size, f32), @splat(reciprocal_w[0]));
        barycentric_y = barycentric_y * @as(@Vector(warp_size, f32), @splat(reciprocal_w[1]));
        barycentric_z = barycentric_z * @as(@Vector(warp_size, f32), @splat(reciprocal_w[2]));

        const barycentric_sum = barycentric_x + barycentric_y + barycentric_z;

        const reciprocal_barycentric_sum = @as(@Vector(warp_size, f32), @splat(1)) / barycentric_sum;

        //Correct barycentrics to fall within 0-1
        barycentric_x = barycentric_x * reciprocal_barycentric_sum;
        barycentric_y = barycentric_y * reciprocal_barycentric_sum;
        barycentric_z = barycentric_z * reciprocal_barycentric_sum;

        //Compute (interpolate) z
        {
            const a_z = @as(@Vector(warp_size, f32), @splat(triangle[0][2]));
            const b_z = @as(@Vector(warp_size, f32), @splat(triangle[1][2]));
            const c_z = @as(@Vector(warp_size, f32), @splat(triangle[2][2]));

            point_z = @mulAdd(@Vector(warp_size, f32), a_z, barycentric_x, point_z);
            point_z = @mulAdd(@Vector(warp_size, f32), b_z, barycentric_y, point_z);
            point_z = @mulAdd(@Vector(warp_size, f32), c_z, barycentric_z, point_z);
        }

        //Depth test
        //Depth tests can be done by loading 8 depths from the zbuffer at a time
        //We can then mask out the execution of the warps using a comparison
        //If the entire mask is zero then we can discard the whole warp
        const fragment_index = @as(usize, @intCast(warp_scan_start)) + @as(usize, @intCast(pixel_y)) * raster_unit.render_pass.color_image.width;

        if (fragment_index + 8 >= raster_unit.render_pass.depth_buffer.len) {
            return;
        }

        const previous_z: @Vector(warp_size, f32) = .{
            raster_unit.render_pass.depth_buffer[fragment_index],
            raster_unit.render_pass.depth_buffer[fragment_index + 1],
            raster_unit.render_pass.depth_buffer[fragment_index + 2],
            raster_unit.render_pass.depth_buffer[fragment_index + 3],
            raster_unit.render_pass.depth_buffer[fragment_index + 4],
            raster_unit.render_pass.depth_buffer[fragment_index + 5],
            raster_unit.render_pass.depth_buffer[fragment_index + 6],
            raster_unit.render_pass.depth_buffer[fragment_index + 7],
        };

        //Mask where true(1) means depth test succeed and 0 means depth test fail
        const z_mask = point_z <= previous_z;

        fragment_write_mask = z_mask;

        //If all invocations fail, skip
        if (!@reduce(.And, fragment_write_mask)) {
            // inline for (0..warp_size) |pixel_index| {
            //     const pixel_out_start: [*]Image.Color = @ptrCast(raster_unit.render_pass.color_image.texelFetch(.{
            //         .x = @intCast(warp_scan_start + pixel_index),
            //         .y = @intCast(pixel_y),
            //     }));

            //     pixel_out_start[0] = Image.Color{ .r = 255, .g = 0, .b = 0, .a = 255 };
            // }

            continue;
        }

        //Fragment shader start

        //warp sizes: 1, 2, 4, 8, 16
        //u0->u8  = 64 bit uniform registers, uniform to a warp, readonly
        //
        //b0->b127 = 16 bit virtual scalar registers
        //s0->s63 = 16 bit virtual scalar registers
        //r0->r31 = 32 bit virtual scalar registers
        //w0->w15 = 64 bit virtual scalar registers
        //v0->v7  = 128 bit virtual vector registers

        //load128 r0,
        //load64
        //load32 r0, u0 + r0;
        //load16 r0, u0 + r0;
        //load8 r0, u0 + r0;
        //load_addr w0, u0 + r0;
        //load32 r1, w0 + 0;
        //load32 r1, w0 + 4;
        //load32 r1, w0 + 8;
        //iadd r0, r1, r2;
        //isub r0, r1, r2;
        //idiv r1, 1, r1;
        //frcp r1, r0;
        //fadd r1, 1, r1;
        //fsub r1, 1, r1;
        //fdiv r1, 1, r1;
        //fmadd r1, r0, r1, r2;
        //store32 r0, 0x00;
        //texel_sample r1, u3, u4, 0.5, 0.5;
        //texel_fetch r1, u3, 50, 50;
        //texel_addr r1, u3, 50, 50;
        //unpack_unorm4f32 r2, r1;
        //ret

        var r: @Vector(warp_size, f32) = @splat(0);
        var g: @Vector(warp_size, f32) = @splat(0);
        var b: @Vector(warp_size, f32) = @splat(0);
        var a: @Vector(warp_size, f32) = @splat(0);

        var u: @Vector(warp_size, f32) = @splat(0);
        var v: @Vector(warp_size, f32) = @splat(0);

        //interpolate attributes
        {
            const ua = @as(@Vector(warp_size, f32), @splat(triangle_attributes[0].uv[0]));
            const ub = @as(@Vector(warp_size, f32), @splat(triangle_attributes[1].uv[0]));
            const uc = @as(@Vector(warp_size, f32), @splat(triangle_attributes[2].uv[0]));

            const va = @as(@Vector(warp_size, f32), @splat(triangle_attributes[0].uv[1]));
            const vb = @as(@Vector(warp_size, f32), @splat(triangle_attributes[1].uv[1]));
            const vc = @as(@Vector(warp_size, f32), @splat(triangle_attributes[2].uv[1]));

            u = @mulAdd(@Vector(warp_size, f32), ua, barycentric_x, u);
            u = @mulAdd(@Vector(warp_size, f32), ub, barycentric_y, u);
            u = @mulAdd(@Vector(warp_size, f32), uc, barycentric_z, u);

            v = @mulAdd(@Vector(warp_size, f32), va, barycentric_x, v);
            v = @mulAdd(@Vector(warp_size, f32), vb, barycentric_y, v);
            v = @mulAdd(@Vector(warp_size, f32), vc, barycentric_z, v);

            u = @min(@fabs(u), @as(@Vector(warp_size, f32), @splat(1)));
            v = @min(@fabs(v), @as(@Vector(warp_size, f32), @splat(1)));

            const ra = @as(@Vector(warp_size, f32), @splat(triangle_attributes[0].color[0]));
            const rb = @as(@Vector(warp_size, f32), @splat(triangle_attributes[1].color[0]));
            const rc = @as(@Vector(warp_size, f32), @splat(triangle_attributes[2].color[0]));

            const ga = @as(@Vector(warp_size, f32), @splat(triangle_attributes[0].color[1]));
            const gb = @as(@Vector(warp_size, f32), @splat(triangle_attributes[1].color[1]));
            const gc = @as(@Vector(warp_size, f32), @splat(triangle_attributes[2].color[1]));

            const ba = @as(@Vector(warp_size, f32), @splat(triangle_attributes[0].color[2]));
            const bb = @as(@Vector(warp_size, f32), @splat(triangle_attributes[1].color[2]));
            const bc = @as(@Vector(warp_size, f32), @splat(triangle_attributes[2].color[2]));

            const aa = @as(@Vector(warp_size, f32), @splat(triangle_attributes[0].color[3]));
            const ab = @as(@Vector(warp_size, f32), @splat(triangle_attributes[1].color[3]));
            const ac = @as(@Vector(warp_size, f32), @splat(triangle_attributes[2].color[3]));

            r = @mulAdd(@Vector(warp_size, f32), ra, barycentric_x, r);
            r = @mulAdd(@Vector(warp_size, f32), rb, barycentric_y, r);
            r = @mulAdd(@Vector(warp_size, f32), rc, barycentric_z, r);

            g = @mulAdd(@Vector(warp_size, f32), ga, barycentric_x, g);
            g = @mulAdd(@Vector(warp_size, f32), gb, barycentric_y, g);
            g = @mulAdd(@Vector(warp_size, f32), gc, barycentric_z, g);

            b = @mulAdd(@Vector(warp_size, f32), ba, barycentric_x, b);
            b = @mulAdd(@Vector(warp_size, f32), bb, barycentric_y, b);
            b = @mulAdd(@Vector(warp_size, f32), bc, barycentric_z, b);

            a = @mulAdd(@Vector(warp_size, f32), aa, barycentric_x, a);
            a = @mulAdd(@Vector(warp_size, f32), ab, barycentric_y, a);
            a = @mulAdd(@Vector(warp_size, f32), ac, barycentric_z, a);
        }

        const uniforms: *const @import("root").TestPipelineUniformInput = @ptrCast(@alignCast(raster_unit.uniform));

        const pixels = shader8xTest(
            @ptrCast(uniforms.texture.?.texel_buffer.ptr),
            @intCast(uniforms.texture.?.width),
            @intCast(uniforms.texture.?.height),
            u,
            v,
        );

        const depth_out = @select(f32, fragment_write_mask, point_z, previous_z);

        var fragments = packUnorm4xf32(a, b, g, r);

        inline for (0..warp_size) |pixel_index| {
            const pixel_out_start: [*]Image.Color = @ptrCast(raster_unit.render_pass.color_image.texelFetch(.{
                .x = @intCast(warp_scan_start + pixel_index),
                .y = @intCast(pixel_y),
            }));

            raster_unit.render_pass.depth_buffer[fragment_index + pixel_index] = depth_out[pixel_index];

            if (fragment_write_mask[pixel_index]) {
                pixel_out_start[0] = @bitCast(pixels[pixel_index]);

                const fragment: Image.Color = @bitCast(pixels[pixel_index]);

                pixel_out_start[0] = Image.Color.fromNormalized(.{
                    r[pixel_index] * fragment.toNormalized()[0],
                    g[pixel_index] * fragment.toNormalized()[1],
                    b[pixel_index] * fragment.toNormalized()[2],
                    a[pixel_index] * fragment.toNormalized()[3],
                });
                pixel_out_start[0] = @bitCast(fragments[pixel_index]);
            } else {
                // pixel_out_start[0] = Image.Color{ .r = 255, .g = 0, .b = 0, .a = 255 };
            }

            // pixel_out_start[0] = Image.Color{ .r = 255, .g = 0, .b = 0, .a = 255 };
            // pixel_out_start[0] = Image.Color.fromNormalized(.{ point_z[pixel_index] * 10, point_z[pixel_index] * 10, point_z[pixel_index] * 10, 1 });
        }
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

inline fn gather(base: [*]const u32, address: @Vector(8, u32)) @Vector(8, u32) {
    var result: @Vector(8, u32) = undefined;

    inline for (0..8) |i| {
        result[i] = base[address[i]];
    }

    return result;
}

inline fn scatter(base: [*]u32, address: @Vector(8, u32), values: @Vector(8, u32)) void {
    inline for (0..8) |i| {
        base[address[i]] = values[i];
    }
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

    result = @intCast(x_int << @as(@Vector(8, u5), @splat(24)));
    result |= @intCast(y_int << @as(@Vector(8, u5), @splat(16)));
    result |= @intCast(z_int << @as(@Vector(8, u5), @splat(8)));
    result |= @intCast(w_int);

    return result;
}

///Execute 8 fragment shaders at a time
///Returns a 8 vector of u32 rgba values
fn shader8xTest(
    img: [*]const u32,
    ///Power of two image width
    img_width_po2: u32,
    ///Power of two image height
    img_height_po2: u32,
    u: @Vector(8, f32),
    v: @Vector(8, f32),
) @Vector(8, u32) {
    const tile_width = Image.tile_width;
    const tile_height = Image.tile_height;

    std.debug.assert(@reduce(.And, u >= @as(@Vector(8, f32), @splat(0))) and
        @reduce(.And, u <= @as(@Vector(8, f32), @splat(1))));

    std.debug.assert(@reduce(.And, v >= @as(@Vector(8, f32), @splat(0))) and
        @reduce(.And, v <= @as(@Vector(8, f32), @splat(1))));

    const width_float: @Vector(8, f32) = @splat(@floatFromInt(img_width_po2));
    const height_float: @Vector(8, f32) = @splat(@floatFromInt(img_height_po2));

    var texel_x: @Vector(8, u32) = undefined;
    var texel_y: @Vector(8, u32) = undefined;

    texel_x = @intFromFloat(u * width_float);
    texel_y = @intFromFloat(v * height_float);

    //idx = y * width + x = fma(y, width, x); (linear mode)
    var texel_address: @Vector(8, u32) = undefined;

    //Could use a shift if image widths were restricted to be power of two size
    texel_address = texel_y * @as(@Vector(8, u32), @splat(img_width_po2)) + texel_x;
    // texel_address = (texel_y << @as(@Vector(8, u32), @splat(img_width_po2))) + texel_x;

    //texel fetch
    const tile_count_x: @Vector(8, u32) = @splat(@intCast(std.math.divCeil(usize, img_width_po2, tile_width) catch unreachable));

    const tile_x = @divFloor(texel_x, @as(@Vector(8, u32), @splat(tile_width)));
    const tile_y = @divFloor(texel_y, @as(@Vector(8, u32), @splat(tile_height)));

    const tile_begin_x = tile_x * @as(@Vector(8, u32), @splat(tile_width));
    const tile_begin_y = tile_y * @as(@Vector(8, u32), @splat(tile_height));

    const tile_pointer: @Vector(8, u32) = ((tile_x + tile_y * tile_count_x) * @as(@Vector(8, u32), @splat(tile_width * tile_height)));

    //x, y relative to tile
    const x = texel_x - tile_begin_x;
    const y = texel_y - tile_begin_y;

    texel_address = tile_pointer + y * @as(@Vector(8, u32), @splat(tile_width)) + x;

    const texel = gather(img, texel_address);

    return texel;
}

const RasterUnit = @import("RasterUnit.zig");
const std = @import("std");
const geometry_processor = @import("geometry_processor.zig");
const Image = @import("../Image.zig");
const simd = @import("simd.zig");
