const std = @import("std");
const c = @import("c_bindings.zig");
const Image = @import("Image.zig");

const Renderer = @This();

window: ?*c.SDL_Window = null,
sdl_renderer: ?*c.SDL_Renderer = null,
sdl_texture: ?*c.SDL_Texture = null,
swapchain_image: []Image.Color,

pub fn init(
    self: *Renderer,
    allocator: std.mem.Allocator,
    window_width: usize,
    window_height: usize,
    surface_width: usize,
    surface_height: usize,
    title: []const u8,
) !void {
    if (c.SDL_Init(c.SDL_INIT_VIDEO) != 0) {
        return error.FailedToCreateWindow;
    }

    self.window = c.SDL_CreateWindow(
        title.ptr,
        c.SDL_WINDOWPOS_CENTERED,
        c.SDL_WINDOWPOS_CENTERED,
        @as(c_int, @intCast(window_width)),
        @as(c_int, @intCast(window_height)),
        c.SDL_WINDOW_RESIZABLE,
    );

    self.sdl_renderer = c.SDL_CreateRenderer(self.window, -1, c.SDL_RENDERER_ACCELERATED);
    self.sdl_texture = c.SDL_CreateTexture(self.sdl_renderer, c.SDL_PIXELFORMAT_ABGR8888, c.SDL_TEXTUREACCESS_STREAMING, @as(c_int, @intCast(surface_width)), @as(c_int, @intCast(surface_height)));

    self.swapchain_image = try allocator.alloc(Image.Color, surface_width * surface_height);
}

pub fn deinit(self: Renderer, allocator: std.mem.Allocator) void {
    defer c.SDL_DestroyWindow(self.window);
    defer c.SDL_DestroyRenderer(self.sdl_renderer);
    defer c.SDL_DestroyTexture(self.sdl_texture);
    defer allocator.free(self.swapchain_image);
}

pub fn shouldWindowClose(self: Renderer) bool {
    _ = self;

    var sdl_event: c.SDL_Event = undefined;

    _ = c.SDL_PollEvent(&sdl_event);

    return sdl_event.type == c.SDL_QUIT;
}

pub const Pass = struct {
    color_image: Image,
    depth_buffer: []f32,
};

fn allTrue(value: @Vector(2, bool)) bool {
    return @reduce(.And, value == @as(@Vector(2, bool), @splat(true)));
}

pub fn drawLine(self: Renderer, pass: Pass, a: @Vector(2, f32), b: @Vector(2, f32)) void {
    _ = self;

    const clip_minimum: @Vector(2, f32) = @splat(@as(f32, -1));
    const clip_maximum: @Vector(2, f32) = @splat(@as(f32, 1));

    if (!(allTrue(a >= clip_minimum) and allTrue(a < clip_maximum) and
        allTrue(b >= clip_minimum) and allTrue(b < clip_maximum)) and false)
    {
        return;
    }

    var p0 = a;
    var p1 = b;

    if (p0[0] > p1[0]) {
        const temp = p0;
        p0 = p1;
        p1 = temp;
    }

    const viewport_scale = @Vector(2, f32){ @as(f32, @floatFromInt(pass.color_image.width)), @as(f32, @floatFromInt(pass.color_image.height)) };

    const target_p0 = (p0 + @as(@Vector(2, f32), @splat(1))) / @as(@Vector(2, f32), @splat(2)) * viewport_scale;
    const target_p1 = (p1 + @as(@Vector(2, f32), @splat(1))) / @as(@Vector(2, f32), @splat(2)) * viewport_scale;

    const displacement = target_p1 - target_p0;
    const gradient = displacement[1] / displacement[0];

    const b1 = target_p0[1] - gradient * target_p0[0];

    var start: isize = 0;
    var end = @as(isize, @intFromFloat(displacement[0]));

    while (start < end) : (start += 1) {
        const x = @as(isize, @intFromFloat(target_p0[0])) + start;
        const y = @as(isize, @intFromFloat(@round(gradient * @as(f32, @floatFromInt(x)) + b1)));

        if (x < 0 or x >= pass.color_image.width or
            y < 0 or y >= pass.color_image.height)
        {
            continue;
        }

        const index = @as(usize, @intCast(x)) + pass.color_image.width * @as(usize, @intCast(y));

        pass.color_image.pixels[index] = .{ .r = 255, .g = 255, .b = 255, .a = 255 };
    }
}

fn lerp(a: f32, b: f32, ratio: f32) f32 {
    return (a * (1 - ratio)) + (b * ratio);
}

fn vectorLerp(comptime T: type, comptime n: usize, p0: @Vector(n, T), p1: @Vector(n, T), p2: @Vector(n, T)) @Vector(n, T) {
    return @mulAdd(@TypeOf(p0), p0, @as(@Vector(n, T), @splat(1)) - p2, p1 * p2);
}

pub const PrimitiveType = enum {
    triangle,
    line,
    point,
};

pub fn Pipeline(
    comptime UniformInputType: type,
    comptime FragmentInputType: type,
    comptime clusterShaderFn: anytype,
    comptime vertexShaderFn: anytype,
    comptime fragmentShaderFn: anytype,
) type {
    _ = clusterShaderFn;
    const VertexShaderFn = fn (uniform: UniformInputType, vertex_index: usize) struct { @Vector(4, f32), FragmentInputType };
    const FragmentShaderFn = fn (uniform: UniformInputType, input: FragmentInputType, position: @Vector(3, f32), pixel: *Image.Color) void;

    return struct {
        pub const UniformInput = UniformInputType;
        pub const FragmentInput = FragmentInputType;

        pub const vertexShader: VertexShaderFn = vertexShaderFn;
        pub const fragmentShader: FragmentShaderFn = fragmentShaderFn;
    };
}

pub fn pipelineDraw(
    self: Renderer,
    pass: Pass,
    primitive_type: PrimitiveType,
    uniform: anytype,
    vertex_count: usize,
    comptime pipeline: type,
) void {
    _ = pipeline;
    _ = vertex_count;
    _ = uniform;
    _ = primitive_type;
    _ = pass;
    _ = self;
}

fn transformClipToFragmentSpace(width: usize, height: usize, point: @Vector(3, f32)) @Vector(2, usize) {
    _ = height;
    _ = width;
    _ = point;
}

fn transformFragmentToClipSpace(width: usize, height: usize, point: @Vector(2, usize)) @Vector(3, f32) {
    _ = height;
    _ = width;
    _ = point;
}

pub fn pipelineDrawTriangles(
    self: Renderer,
    pass: Pass,
    uniform: anytype,
    triangle_count: usize,
    comptime pipeline: anytype,
) void {
    for (0..triangle_count) |index| {
        self.pipelineDrawTriangle(pass, uniform, index, pipeline);
    }
}

const ClippingPlane = struct {
    normal: @Vector(3, f32),
    d: f32 = 0,
};

fn vectorDotGeneric3D(comptime N: comptime_int, comptime T: type, a: @Vector(N, T), b: @Vector(N, T)) T {
    return @reduce(.Add, a * b);
}

fn vectorDot(a: @Vector(3, f32), b: @Vector(3, f32)) f32 {
    return @reduce(.Add, a * b);
}

fn signedDistance(plane: ClippingPlane, vertex: @Vector(4, f32)) f32 {
    return vectorDot(plane.normal, .{ vertex[0], vertex[1], vertex[2] }) + plane.d;
}

fn planeLineIntersection(plane: ClippingPlane, line: [2]@Vector(4, f32)) @Vector(4, f32) {
    const u = line[1] - line[0];
    const dot = vectorDot(plane.normal, .{ u[0], u[1], u[2] });

    if (@fabs(dot) > 1e-6) {
        const w = line[0] - @as(@Vector(4, f32), @splat(plane.d));
        const frac = -vectorDot(plane.normal, .{ w[0], w[1], w[2] }) / dot;

        return line[0] + u * @as(@Vector(4, f32), @splat(frac));
    }

    return .{ 0, 0, 0, 1 };
}

const TriangleClipResult = union(enum) {
    ///The triangle is discarded
    discarded,
    ///A single triangle is produced
    single: [3]@Vector(4, f32),
    ///Two triangles are produced
    double: [2][3]@Vector(4, f32),
};

fn clipTriangleAgainstPlane(plane: ClippingPlane, triangle: [3]@Vector(4, f32)) TriangleClipResult {
    const S = struct {

        //returns the index of the only positive param, or null
        pub fn onlyPositive(x: f32, y: f32, z: f32) ?usize {
            if (x > 0 and (y <= 0 and z <= 0)) {
                return 0;
            }

            if (y > 0 and (x <= 0 and z <= 0)) {
                return 1;
            }

            if (z > 0 and (x <= 0 and y <= 0)) {
                return 2;
            }

            return null;
        }

        //returns the index of the only negative param, or null
        pub fn onlyNegative(x: f32, y: f32, z: f32) ?usize {
            if (x < 0 and (y >= 0 and z >= 0)) {
                return 0;
            }

            if (y < 0 and (x >= 0 and z >= 0)) {
                return 1;
            }

            if (z < 0 and (x >= 0 and y >= 0)) {
                return 2;
            }

            return null;
        }
    };

    const d0 = signedDistance(plane, triangle[0]);
    const d1 = signedDistance(plane, triangle[1]);
    const d2 = signedDistance(plane, triangle[2]);

    if (d0 > 0 and d1 > 0 and d2 > 0) {
        return .{ .single = triangle };
    } else if (d0 < 0 and d1 < 0 and d2 < 0) {
        return .discarded;
    } else if (S.onlyPositive(d0, d1, d2)) |a_index| {
        const a = triangle[a_index]; //positive vertex
        const b_prime = planeLineIntersection(plane, .{ triangle[0], triangle[1] });
        const c_prime = planeLineIntersection(plane, .{ triangle[0], triangle[2] });

        return .{ .single = .{ a, b_prime, c_prime } };
    } else if (S.onlyNegative(d0, d1, d2)) |c_index| {
        const _c = triangle[c_index];
        const a_prime = planeLineIntersection(plane, .{ triangle[0], _c });
        const b_prime = planeLineIntersection(plane, .{ triangle[1], _c });

        return .{ .double = .{
            .{ triangle[0], triangle[1], a_prime },
            .{ a_prime, triangle[1], b_prime },
        } };
    }

    return .{ .single = triangle };
}

///Clips the triangle against all planes and returns the clipped triangles
///The upper bound for the number of triangles produced is equal to the number of intersecting,
///or touching planes. If a triangle is clipped in the corner of a 3D clip volume,
///A 5 sided polygon can be produced. Hence why [plane_count]TriangleClipResult is returned
fn clipTriangleAgainstPlanes(
    comptime plane_count: comptime_int,
    planes: [plane_count]ClippingPlane,
    triangle: [3]@Vector(4, f32),
) [plane_count * 2]TriangleClipResult {
    const two_triangle_area = twoTriangleArea3D(triangle);

    const clip_volume_min = @Vector(4, f32){ -1, -1, 0, 0 };
    _ = clip_volume_min;
    const clip_volume_max = @Vector(4, f32){ 1, 1, 1, 1 };
    _ = clip_volume_max;

    var result: [plane_count * 2]TriangleClipResult = [_]TriangleClipResult{.discarded} ** (plane_count * 2);

    //Pre clip cull
    // if ((@reduce(.Or, triangle[0] > clip_volume_max) and
    //     @reduce(.Or, triangle[1] > clip_volume_max) and
    //     @reduce(.Or, triangle[2] > clip_volume_max)) or
    //     (@reduce(.Or, triangle[0] < clip_volume_min) and
    //     @reduce(.Or, triangle[1] < clip_volume_min) and
    //     @reduce(.Or, triangle[2] < clip_volume_min)))
    // {
    //     return result;
    // }

    //Pre clip cull
    if (two_triangle_area < 0) {
        return result;
    }

    var result_count: usize = 1;

    result[0] = clipTriangleAgainstPlane(planes[0], triangle);

    for (planes[1..], 1..) |plane, index| {
        const previous_result = result[index - 1];

        switch (previous_result) {
            .discarded => break,
            .single => |result_triangle| {
                result[result_count] = clipTriangleAgainstPlane(plane, result_triangle);
                result_count += 1;
            },
            .double => |result_triangles| {
                for (result_triangles) |result_triangle| {
                    result[result_count] = clipTriangleAgainstPlane(plane, result_triangle);
                    result_count += 1;
                }
            },
        }
    }

    return result;
}

fn clipTriangleNew(
    triangle: [3]@Vector(3, f32),
    comptime Interpolator: type,
    triangle_interpolators: [3]Interpolator,
) struct {
    vertices: [16 * 3]@Vector(3, f32),
    interpolators: [16 * 3]Interpolator,
    vertex_count: u32,
} {
    _ = triangle_interpolators;
    _ = triangle;
}

fn calculateBarycentrics2D(triangle: [3]@Vector(2, f32), point: @Vector(2, f32)) @Vector(3, f32) {
    const area_abc = @fabs(vectorCross2D(triangle[1] - triangle[0], triangle[2] - triangle[0]));

    const area_pbc = @fabs(vectorCross2D(triangle[1] - point, triangle[2] - point));
    const area_pca = @fabs(vectorCross2D(triangle[2] - point, triangle[0] - point));

    var bary: @Vector(3, f32) = .{
        area_pbc / area_abc,
        area_pca / area_abc,
        0,
    };

    bary[2] = 1 - bary[0] - bary[1];

    //TODO: I think the degenerate barys (generating barys of > 1) needs to be sovled properly?
    bary[0] = std.math.clamp(bary[0], 0, 1);
    bary[1] = std.math.clamp(bary[1], 0, 1);
    bary[2] = std.math.clamp(bary[2], 0, 1);

    return bary;
}

fn pipelineDrawTriangle(
    self: Renderer,
    pass: Pass,
    uniform: anytype,
    triangle_index: usize,
    comptime pipeline: anytype,
) void {
    const fragment_input_0 = pipeline.vertexShader(uniform, triangle_index * 3 + 0);
    const fragment_input_1 = pipeline.vertexShader(uniform, triangle_index * 3 + 1);
    const fragment_input_2 = pipeline.vertexShader(uniform, triangle_index * 3 + 2);

    var triangle: [3]@Vector(4, f32) = .{ fragment_input_0[0], fragment_input_1[0], fragment_input_2[0] };

    for (&triangle) |*point| {
        point[0] /= point[3];
        point[1] /= point[3];
        point[2] /= point[3];
    }

    const two_triangle_area = twoTriangleArea(.{
        .{ triangle[0][0], triangle[0][1] },
        .{ triangle[1][0], triangle[1][1] },
        .{ triangle[2][0], triangle[2][1] },
    });

    //backface cull
    if (false and two_triangle_area < 0) {
        return;
    }

    self.pipelineRasteriseTriangle(
        pipeline,
        pass,
        uniform,
        triangle,
        .{ fragment_input_0[1], fragment_input_1[1], fragment_input_2[1] },
    );

    if (false) {
        const clip_results = clipTriangleAgainstPlanes(5, .{
            //near
            .{
                .normal = .{ 0, 0, 1 },
                .d = 0,
            },
            //left
            .{
                .normal = .{ 1 / @sqrt(@as(f32, 2)), 0, 1 / @sqrt(@as(f32, 2)) },
                .d = 0,
            },
            //right
            .{
                .normal = .{ -1 / @sqrt(@as(f32, 2)), 0, @sqrt(@as(f32, 2)) },
                .d = 0,
            },
            //top
            .{
                .normal = .{ 0, 1 / @sqrt(@as(f32, 2)), 1 / @sqrt(@as(f32, 2)) },
                .d = 0,
            },
            //bottom
            .{
                .normal = .{ 0, -1 / @sqrt(@as(f32, 2)), @sqrt(@as(f32, 2)) },
                .d = 0,
            },
        }, .{ fragment_input_0[0], fragment_input_1[0], fragment_input_2[0] });

        for (clip_results) |clip_result| {
            switch (clip_result) {
                .discarded => break,
                .single => |points| {
                    std.debug.assert(points[0][3] > 0);
                    std.debug.assert(points[1][3] > 0);
                    std.debug.assert(points[2][3] > 0);

                    self.pipelineRasteriseTriangle(
                        pipeline,
                        pass,
                        uniform,
                        // points,
                        .{
                            .{ points[0][0] / points[0][3], points[0][1] / points[0][3], points[0][2] / points[0][3], points[0][3] },
                            .{ points[1][0] / points[1][3], points[1][1] / points[1][3], points[1][2] / points[1][3], points[1][3] },
                            .{ points[2][0] / points[2][3], points[2][1] / points[2][3], points[2][2] / points[2][3], points[2][3] },
                        },
                        .{
                            fragment_input_0[1],
                            fragment_input_1[1],
                            fragment_input_2[1],
                        },
                    );
                },
                .double => |triangles| {
                    for (triangles) |points| {
                        self.pipelineRasteriseTriangle(
                            pipeline,
                            pass,
                            uniform,
                            // points,
                            .{
                                .{ points[0][0] / points[0][3], points[0][1] / points[0][3], points[0][2] / points[0][3], points[0][3] },
                                .{ points[1][0] / points[1][3], points[1][1] / points[1][3], points[1][2] / points[1][3], points[1][3] },
                                .{ points[2][0] / points[2][3], points[2][1] / points[2][3], points[2][2] / points[2][3], points[2][3] },
                            },
                            .{
                                fragment_input_0[1],
                                fragment_input_1[1],
                                fragment_input_2[1],
                            },
                        );
                    }
                },
            }
        }
    }
}

///Draws a triangle using rasterisation, supplying a compile time known pipeline
///Of operation functions
///Assumes that points is entirely contained within the clip volume
fn pipelineRasteriseTriangle(
    self: Renderer,
    comptime pipeline: anytype,
    pass: Pass,
    uniform: anytype,
    points: [3]@Vector(4, f32),
    fragment_inputs: [3]pipeline.FragmentInput,
) void {
    _ = self;

    const Interpolators = pipeline.FragmentInput;

    const Edge = struct {
        p0: @Vector(3, isize),
        p1: @Vector(3, isize),
        z0: f32,
        z1: f32,

        interpolators0: Interpolators,
        interpolators1: Interpolators,

        pub fn init(
            p0: @Vector(3, isize),
            p1: @Vector(3, isize),
            z0: f32,
            z1: f32,
            interpolators0: Interpolators,
            interpolators1: Interpolators,
        ) @This() {
            if (p0[1] < p1[1]) {
                return .{
                    .p0 = p0,
                    .p1 = p1,
                    .z0 = z0,
                    .z1 = z1,
                    .interpolators0 = interpolators0,
                    .interpolators1 = interpolators1,
                };
            } else {
                return .{
                    .p0 = p1,
                    .p1 = p0,
                    .z0 = z1,
                    .z1 = z0,
                    .interpolators0 = interpolators1,
                    .interpolators1 = interpolators0,
                };
            }
        }
    };

    const Span = struct {
        x0: isize,
        x1: isize,
        z0: f32,
        z1: f32,

        interpolators0: Interpolators,
        interpolators1: Interpolators,

        pub fn init(
            x0: isize,
            x1: isize,
            z0: f32,
            z1: f32,
            interpolators0: Interpolators,
            interpolators1: Interpolators,
        ) @This() {
            if (x0 < x1) {
                return .{
                    .x0 = x0,
                    .x1 = x1,
                    .z0 = z0,
                    .z1 = z1,
                    .interpolators0 = interpolators0,
                    .interpolators1 = interpolators1,
                };
            } else {
                return .{
                    .x0 = x1,
                    .x1 = x0,
                    .z0 = z1,
                    .z1 = z0,
                    .interpolators0 = interpolators1,
                    .interpolators1 = interpolators0,
                };
            }
        }

        fn drawEdges(
            render_pass: Pass,
            _uniform: anytype,
            triangle: [3]@Vector(4, f32),
            triangle_attributes: [3]pipeline.FragmentInput,
            twice_triangle_area: f32,
            left: Edge,
            right: Edge,
        ) void {
            const left_y_diff = left.p1[1] - left.p0[1];

            if (left_y_diff == 0) return;

            const right_y_diff = right.p1[1] - right.p0[1];

            if (right_y_diff == 0) return;

            const left_x_diff = left.p1[0] - left.p0[0];
            const right_x_diff = right.p1[0] - right.p0[0];

            const left_z_diff = left.z1 - left.z0;
            const right_z_diff = right.z1 - right.z0;

            var left_interpolator_diffs: Interpolators = undefined;
            var right_interpolator_diffs: Interpolators = undefined;

            inline for (std.meta.fields(Interpolators)) |field| {
                @field(left_interpolator_diffs, field.name) =
                    @field(left.interpolators1, field.name) -
                    @field(left.interpolators0, field.name);

                @field(right_interpolator_diffs, field.name) =
                    @field(right.interpolators1, field.name) -
                    @field(right.interpolators0, field.name);
            }

            var factor0: f32 = @as(f32, @floatFromInt(right.p0[1] - left.p0[1])) / @as(f32, @floatFromInt(left_y_diff));
            var factor1: f32 = 0;

            const factor_step_0 = 1 / @as(f32, @floatFromInt(left_y_diff));
            const factor_step_1 = 1 / @as(f32, @floatFromInt(right_y_diff));

            var z_factor0: f32 = (right.z0 - left.z0) / left_z_diff;
            var z_factor1: f32 = 0;

            const z_factor_step0 = 1 / left_z_diff;
            const z_factor_step1 = 1 / right_z_diff;

            var pixel_y: isize = right.p0[1];

            while (pixel_y < right.p1[1]) : (pixel_y += 1) {
                defer {
                    factor0 += factor_step_0;
                    factor1 += factor_step_1;

                    z_factor0 += z_factor_step0;
                    z_factor1 += z_factor_step1;
                }

                var span_interpolators0: Interpolators = undefined;
                var span_interpolators1: Interpolators = undefined;

                inline for (std.meta.fields(Interpolators)) |field| {
                    const vector_factor0 = @as(field.type, @splat(factor0));
                    const vector_factor1 = @as(field.type, @splat(factor1));

                    @field(span_interpolators0, field.name) =
                        @field(left.interpolators0, field.name) + @field(left_interpolator_diffs, field.name) * vector_factor0;

                    @field(span_interpolators1, field.name) =
                        @field(right.interpolators0, field.name) + @field(right_interpolator_diffs, field.name) * vector_factor1;
                }

                if (pixel_y < 0 or pixel_y >= render_pass.color_image.height) {
                    continue;
                }

                drawSpan(
                    render_pass,
                    _uniform,
                    @This().init(
                        left.p0[0] + @as(isize, @intFromFloat(@ceil(@as(f32, @floatFromInt(left_x_diff)) * factor0))),
                        right.p0[0] + @as(isize, @intFromFloat(@ceil(@as(f32, @floatFromInt(right_x_diff)) * factor1))),
                        left.z0 + left_z_diff * z_factor0,
                        right.z0 + right_z_diff * z_factor1,
                        span_interpolators0,
                        span_interpolators1,
                    ),
                    triangle,
                    triangle_attributes,
                    twice_triangle_area,
                    pixel_y,
                );
            }
        }

        fn drawSpan(
            render_pass: Pass,
            _uniform: anytype,
            span: @This(),
            triangle: [3]@Vector(4, f32),
            triangle_attributes: [3]pipeline.FragmentInput,
            twice_triangle_area: f32,
            pixel_y: isize,
        ) void {
            _ = twice_triangle_area;

            const xdiff = span.x1 - span.x0;

            if (xdiff == 0) return;

            const zdiff = span.z1 - span.z0;

            var interpolator_diffs: Interpolators = undefined;

            inline for (std.meta.fields(Interpolators)) |field| {
                @field(interpolator_diffs, field.name) =
                    @field(span.interpolators1, field.name) - @field(span.interpolators0, field.name);
            }

            const factor_step = 1 / @as(f32, @floatFromInt(xdiff));
            const factor_step1 = 1 / zdiff;

            var factor: f32 = 0;
            var factor1: f32 = 0;

            var pixel_x: isize = span.x0;

            while (pixel_x < span.x1) : (pixel_x += 1) {
                defer {
                    factor += factor_step;
                    factor1 += factor_step1;
                }

                //not needed if clipping is perfect
                //x, y can be usize if clipping is perfect too
                if (pixel_x < 0 or
                    pixel_x >= render_pass.color_image.width)
                {
                    continue;
                }

                var point = @Vector(3, f32){
                    ((@as(f32, @floatFromInt(pixel_x)) / @as(f32, @floatFromInt(render_pass.color_image.width))) * 2) - 1,
                    ((@as(f32, @floatFromInt(pixel_y)) / @as(f32, @floatFromInt(render_pass.color_image.height))) * 2) - 1,
                    span.z0 + zdiff * factor1,
                };

                var barycentrics = calculateBarycentrics2D(.{
                    .{ triangle[0][0], triangle[0][1] },
                    .{ triangle[1][0], triangle[1][1] },
                    .{ triangle[2][0], triangle[2][1] },
                }, .{ point[0], point[1] });

                barycentrics[0] = barycentrics[0] / triangle[0][3];
                barycentrics[1] = barycentrics[1] / triangle[1][3];
                barycentrics[2] = barycentrics[2] / triangle[2][3];

                barycentrics = barycentrics / @as(@Vector(3, f32), @splat(barycentrics[0] + barycentrics[1] + barycentrics[2]));

                // std.debug.assert(barycentrics[0] < 1);
                // std.debug.assert(barycentrics[1] < 1);
                // std.debug.assert(barycentrics[2] < 1);

                point[2] =
                    triangle[0][2] * barycentrics[0] +
                    triangle[1][2] * barycentrics[1] +
                    triangle[2][2] * barycentrics[2];

                const fragment_index = @as(usize, @intCast(pixel_x)) + @as(usize, @intCast(pixel_y)) * render_pass.color_image.width;

                const depth = &render_pass.depth_buffer[fragment_index];

                if (point[2] > depth.*) {
                    continue;
                }

                depth.* = point[2];

                var fragment_input: pipeline.FragmentInput = undefined;

                inline for (std.meta.fields(Interpolators)) |field| {
                    const factor_vector = @as(field.type, @splat(factor));
                    _ = factor_vector;
                    const vector_dimensions = @typeInfo(field.type).Vector.len;

                    // @field(fragment_input, field.name) =
                    //     @field(span.interpolators0, field.name) + (@field(interpolator_diffs, field.name) * factor_vector);

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

                const pixel = render_pass.color_image.texelFetch(.{ @intCast(pixel_x), @intCast(pixel_y) });

                @call(.always_inline, pipeline.fragmentShader, .{
                    _uniform,
                    fragment_input,
                    point,
                    pixel,
                });

                // pixel.* = Image.Color.fromNormalized(.{ point[2] * 10, point[2] * 10, point[2] * 10, 1 });
            }
        }
    };

    const view_scale = @Vector(4, f32){
        @as(f32, @floatFromInt(pass.color_image.width)),
        @as(f32, @floatFromInt(pass.color_image.height)),
        1,
        1,
    };

    //twice the area of the triangle
    const signed_edge0 = points[1] - points[0];
    const signed_edge1 = points[2] - points[1];

    const double_area = vectorLength(vectorCross3D(.{ signed_edge0[0], signed_edge0[1], signed_edge0[2] }, .{ signed_edge1[0], signed_edge1[1], signed_edge1[2] }));

    const p0_orig = @ceil((points[0] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale);
    const p1_orig = @ceil((points[1] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale);
    const p2_orig = @ceil((points[2] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale);

    const p0: @Vector(3, isize) = .{ @intFromFloat(p0_orig[0]), @intFromFloat(p0_orig[1]), @intFromFloat(p0_orig[2]) };
    const p1: @Vector(3, isize) = .{ @intFromFloat(p1_orig[0]), @intFromFloat(p1_orig[1]), @intFromFloat(p1_orig[2]) };
    const p2: @Vector(3, isize) = .{ @intFromFloat(p2_orig[0]), @intFromFloat(p2_orig[1]), @intFromFloat(p2_orig[2]) };

    const edges: [3]Edge = .{
        Edge.init(p0, p1, points[0][2], points[1][2], fragment_inputs[0], fragment_inputs[1]),
        Edge.init(p1, p2, points[1][2], points[2][2], fragment_inputs[1], fragment_inputs[2]),
        Edge.init(p2, p0, points[2][2], points[0][2], fragment_inputs[2], fragment_inputs[0]),
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
        pass,
        uniform,
        points,
        fragment_inputs,
        double_area,
        edges[long_edge_index],
        edges[short_edge_1],
    );
    Span.drawEdges(
        pass,
        uniform,
        points,
        fragment_inputs,
        double_area,
        edges[long_edge_index],
        edges[short_edge_2],
    );
}

pub fn pipelineDrawLine(
    self: Renderer,
    pass: Pass,
    uniform: anytype,
    vertex_count: usize,
    comptime pipeline: type,
) void {
    _ = self;

    var vertex_index: usize = 0;

    while (vertex_index < vertex_count) : (vertex_index += 2) {
        const inv0 = @call(.always_inline, pipeline.vertexShader, .{ uniform, vertex_index });
        const inv1 = @call(.always_inline, pipeline.vertexShader, .{ uniform, vertex_index + 1 });

        const fragment_input0: pipeline.FragmentInput = inv0[1];
        const fragment_input1: pipeline.FragmentInput = inv1[1];

        var p0 = inv0[0];
        var p1 = inv1[0];

        if (p0[0] > p1[0]) {
            const temp = p0;
            p0 = p1;
            p1 = temp;
        }

        const viewport_scale = @Vector(4, f32){
            @as(f32, @floatFromInt(pass.color_image.width)),
            @as(f32, @floatFromInt(pass.color_image.height)),
            1,
            1,
        };

        //target_p0/1 is in pixel space (0..width), (0..height)
        const target_p0 = (p0 + @as(@Vector(4, f32), @splat(1))) / @as(@Vector(4, f32), @splat(2)) * viewport_scale;
        const target_p1 = (p1 + @as(@Vector(4, f32), @splat(1))) / @as(@Vector(4, f32), @splat(2)) * viewport_scale;

        const displacement = target_p1 - target_p0;
        const gradient = displacement[1] / displacement[0];

        const b1 = target_p0[1] - gradient * target_p0[0];

        var start_x: isize = 0;
        const end_x = @as(isize, @intFromFloat(displacement[0]));

        while (start_x < end_x) : (start_x += 1) {
            const x = @as(f32, @floatFromInt(start_x)) + target_p0[0];
            const y = @round(gradient * x + b1);

            if (x < 0 or x >= @as(f32, @floatFromInt(pass.color_image.width)) or
                y < 0 or y >= @as(f32, @floatFromInt(pass.color_image.height)))
            {
                continue;
            }

            const index = @as(usize, @intFromFloat(x)) + pass.color_image.width * @as(usize, @intFromFloat(y));

            const line_ratio_x = @as(f32, @floatFromInt(start_x)) / @as(f32, @floatFromInt(end_x));
            const line_ratio_y = line_ratio_x;
            const line_ratio_z = line_ratio_y;

            const interpolant = @Vector(4, f32){
                line_ratio_x,
                line_ratio_y,
                line_ratio_z,
                1,
            };

            const interpolated_point = vectorLerp(
                f32,
                4,
                p0,
                p1,
                interpolant,
            );

            var fragment_input: pipeline.FragmentInput = undefined;

            inline for (std.meta.fields(pipeline.FragmentInput)) |field| {
                if (field.is_comptime) continue;

                const field_type_info = @typeInfo(field.type);

                switch (field_type_info) {
                    .Float => {
                        @field(fragment_input, field.name) = lerp(
                            @field(fragment_input0, field.name),
                            @field(fragment_input1, field.name),
                            line_ratio_x,
                        );
                    },
                    .Vector => |vector_info| {
                        @field(fragment_input, field.name) = vectorLerp(
                            vector_info.child,
                            vector_info.len,
                            @field(fragment_input0, field.name),
                            @field(fragment_input1, field.name),
                            @splat(line_ratio_x),
                        );
                    },
                    else => {},
                }
            }

            const depth_index = index;

            if (pass.depth_buffer[depth_index] <= interpolated_point[2]) {
                continue;
            }

            pass.depth_buffer[depth_index] = interpolated_point[2];

            const pixel = pass.color_image.texelFetch(.{ @intFromFloat(x), @intFromFloat(y) });

            @call(.always_inline, pipeline.fragmentShader, .{
                uniform,
                fragment_input,
                .{ interpolated_point[0], interpolated_point[1], interpolated_point[2] },
                pixel,
            });
        }
    }
}

pub fn drawLinedTriangle(self: Renderer, pass: Pass, points: [3][2]f32) void {
    self.drawLine(pass, points[0], points[1]);
    self.drawLine(pass, points[1], points[2]);
    self.drawLine(pass, points[2], points[0]);
}

pub fn drawLinedTriangles(self: Renderer, pass: Pass, triangles: [][3][2]f32) void {
    for (triangles) |triangle| {
        self.drawLinedTriangle(pass, triangle);
    }
}

fn twoTriangleArea(triangle: [3]@Vector(2, f32)) f32 {
    const p1 = triangle[1] - triangle[0];
    const p2 = triangle[2] - triangle[0];

    return vectorCross2D(p1, p2);
}

fn twoTriangleArea3D(triangle: [3]@Vector(4, f32)) f32 {
    const p1 = triangle[1] - triangle[0];
    const p2 = triangle[2] - triangle[0];

    return vectorLength(vectorCross3D(.{ p1[0], p1[1], p1[2] }, .{ p2[0], p2[1], p2[2] }));
}

fn vectorCross2D(a: @Vector(2, f32), b: @Vector(2, f32)) f32 {
    return a[0] * b[1] - b[0] * a[1];
}

fn vectorCross3D(a: @Vector(3, f32), b: @Vector(3, f32)) @Vector(3, f32) {
    const result_x = (a[1] * b[2]) - (a[2] * b[1]);
    const result_y = (a[2] * b[0]) - (a[0] * b[2]);
    const result_z = (a[0] * b[1]) - (a[1] * b[0]);
    return .{ result_x, result_y, result_z };
}

fn vectorLength(a: @Vector(3, f32)) f32 {
    return @sqrt(vectorDot(a, a));
}

pub fn drawTriangle(
    self: Renderer,
    pass: Pass,
    points: [3]@Vector(3, f32),
) void {
    _ = self;
    const Edge = struct {
        p0: @Vector(3, isize),
        p1: @Vector(3, isize),

        color0: @Vector(4, f32),
        color1: @Vector(4, f32),

        pub fn init(
            p0: @Vector(3, isize),
            p1: @Vector(3, isize),
            color0: @Vector(4, f32),
            color1: @Vector(4, f32),
        ) @This() {
            if (p0[1] < p1[1]) {
                return .{
                    .p0 = p0,
                    .p1 = p1,
                    .color0 = color0,
                    .color1 = color1,
                };
            } else {
                return .{
                    .p0 = p1,
                    .p1 = p0,
                    .color0 = color1,
                    .color1 = color0,
                };
            }
        }
    };

    const Span = struct {
        x0: isize,
        x1: isize,
        z0: f32,
        z1: f32,
        color0: @Vector(4, f32),
        color1: @Vector(4, f32),

        pub fn init(
            x0: isize,
            x1: isize,
            z0: f32,
            z1: f32,
            color0: @Vector(4, f32),
            color1: @Vector(4, f32),
        ) @This() {
            if (x0 < x1) {
                return .{
                    .x0 = x0,
                    .x1 = x1,
                    .z0 = z0,
                    .z1 = z1,
                    .color0 = color0,
                    .color1 = color1,
                };
            } else {
                return .{
                    .x0 = x1,
                    .x1 = x0,
                    .z0 = z1,
                    .z1 = z0,
                    .color0 = color1,
                    .color1 = color0,
                };
            }
        }

        fn drawScanLines(render_pass: Pass, left: Edge, right: Edge) void {
            const left_y_diff = left.p1[1] - left.p0[1];

            if (left_y_diff == 0) return;

            const right_y_diff = right.p1[1] - right.p0[1];

            if (right_y_diff == 0) return;

            const left_x_diff = left.p1[0] - left.p0[0];
            const right_x_diff = right.p1[0] - right.p0[0];

            const left_color_diff = left.color1 - left.color0;
            const right_color_diff = right.color1 - right.color0;

            var factor0: f32 = @as(f32, @floatFromInt(right.p0[1] - left.p0[1])) / @as(f32, @floatFromInt(left_y_diff));
            const factor_step_0 = 1 / @as(f32, @floatFromInt(left_y_diff));
            var factor1: f32 = 0;
            const factor_step_1 = 1 / @as(f32, @floatFromInt(right_y_diff));
            var factor2: f32 = 0;
            _ = factor2;
            const factor_step_2 = 1 / @as(f32, @floatFromInt(right_y_diff));
            _ = factor_step_2;

            var y: isize = right.p0[1];

            while (y < right.p1[1]) : (y += 1) {
                drawSpan(
                    render_pass,
                    @This().init(
                        left.p0[0] + @as(isize, @intFromFloat(@trunc(@as(f32, @floatFromInt(left_x_diff)) * factor0))),
                        right.p0[0] + @as(isize, @intFromFloat(@trunc(@as(f32, @floatFromInt(right_x_diff)) * factor1))),
                        0,
                        0,
                        left.color0 + left_color_diff * @as(@Vector(4, f32), @splat(factor0)),
                        right.color0 + right_color_diff * @as(@Vector(4, f32), @splat(factor1)),
                    ),
                    @intCast(y),
                );

                factor0 += factor_step_0;
                factor1 += factor_step_1;
            }
        }

        fn drawSpan(render_pass: Pass, span: @This(), y: usize) void {
            const xdiff = span.x1 - span.x0;
            const zdiff = span.x1 - span.x0;
            _ = zdiff;

            if (xdiff == 0) return;

            const colordiff = span.color1 - span.color0;

            var factor: f32 = 0;
            const factor_step = 1 / @as(f32, @floatFromInt(xdiff));

            var x: isize = span.x0;

            while (x < span.x1) : (x += 1) {
                if (!(x < 0 or
                    y < 0 or
                    x >= render_pass.color_image.width or
                    y >= render_pass.color_image.height))
                {
                    render_pass.color_image.setPixel(
                        .{ .x = @intCast(x), .y = @intCast(y) },
                        Image.Color.fromNormalized(span.color0 + (colordiff * @as(@Vector(4, f32), @splat(factor)))),
                    );
                }

                factor += factor_step;
            }
        }
    };

    const colors = [3]@Vector(4, f32){
        .{ 1, 0, 0, 1 },
        .{ 0, 1, 0, 1 },
        .{ 0, 0, 1, 1 },
    };

    const view_scale = @Vector(3, f32){
        @as(f32, @floatFromInt(pass.color_image.width)),
        @as(f32, @floatFromInt(pass.color_image.height)),
        0,
    };

    const p0_orig = (points[0] + @Vector(3, f32){ 1, 1, 1 }) / @Vector(3, f32){ 2, 2, 2 } * view_scale;
    const p1_orig = (points[1] + @Vector(3, f32){ 1, 1, 1 }) / @Vector(3, f32){ 2, 2, 2 } * view_scale;
    const p2_orig = (points[2] + @Vector(3, f32){ 1, 1, 1 }) / @Vector(3, f32){ 2, 2, 2 } * view_scale;

    const p0 = @Vector(3, isize){ @intFromFloat(p0_orig[0]), @intFromFloat(p0_orig[1]), @intFromFloat(p0_orig[2]) };
    const p1 = @Vector(3, isize){ @intFromFloat(p1_orig[0]), @intFromFloat(p1_orig[1]), @intFromFloat(p1_orig[2]) };
    const p2 = @Vector(3, isize){ @intFromFloat(p2_orig[0]), @intFromFloat(p2_orig[1]), @intFromFloat(p2_orig[2]) };

    const edges: [3]Edge = .{
        Edge.init(p0, p1, colors[0], colors[1]),
        Edge.init(p1, p2, colors[1], colors[2]),
        Edge.init(p2, p0, colors[2], colors[0]),
    };

    var max_length: usize = 0;
    var long_edge_index: usize = 0;

    for (edges, 0..) |edge, i| {
        const length = edge.p1[1] - edge.p0[1];
        if (length > max_length) {
            max_length = std.math.absCast(length);
            long_edge_index = i;
        }
    }

    const short_edge_1: usize = (long_edge_index + 1) % 3;
    const short_edge_2: usize = (long_edge_index + 2) % 3;

    Span.drawScanLines(pass, edges[long_edge_index], edges[short_edge_1]);
    Span.drawScanLines(pass, edges[long_edge_index], edges[short_edge_2]);
}

pub fn presentImage(self: Renderer, image: Image) void {
    for (0..image.height) |y| {
        for (0..image.width) |x| {
            self.swapchain_image[x + y * image.width] = image.texelFetch(.{ x, y }).*;
        }
    }

    _ = c.SDL_UpdateTexture(self.sdl_texture, null, self.swapchain_image.ptr, @as(c_int, @intCast(image.width * @sizeOf(Image.Color))));
    _ = c.SDL_RenderCopy(self.sdl_renderer, self.sdl_texture, null, null);
    _ = c.SDL_RenderPresent(self.sdl_renderer);
}
