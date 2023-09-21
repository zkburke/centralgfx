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

    while (c.SDL_PollEvent(&sdl_event) != 0) {
        switch (sdl_event.type) {
            c.SDL_QUIT => return true,
            c.SDL_WINDOWEVENT => {
                switch (sdl_event.window.type) {
                    c.SDL_WINDOWEVENT_CLOSE => return true,
                    else => {},
                }
            },
            else => {},
        }
    }

    return false;
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

pub fn Pipeline(
    comptime UniformInputType: type,
    comptime FragmentInputType: type,
    comptime clusterShaderFn: anytype,
    comptime vertexShaderFn: anytype,
    comptime fragmentShaderFn: anytype,
    comptime _polygon_fill_mode: RuntimePipeline.PolygonFillMode,
) type {
    _ = clusterShaderFn;
    const VertexShaderFn = fn (
        uniform: UniformInputType,
        vertex_index: usize,
    ) struct { @Vector(4, f32), FragmentInputType };
    const FragmentShaderFn = fn (uniform: UniformInputType, input: FragmentInputType, position: @Vector(3, f32)) @Vector(4, f32);

    return struct {
        pub const UniformInput = UniformInputType;
        pub const FragmentInput = FragmentInputType;

        pub const vertexShader: VertexShaderFn = vertexShaderFn;
        pub const fragmentShader: FragmentShaderFn = fragmentShaderFn;

        pub const polygon_fill_mode: RuntimePipeline.PolygonFillMode = _polygon_fill_mode;

        pub fn runtimeVertexShader(
            uniform: *const anyopaque,
            vertex_index: usize,
            fragment_input: *anyopaque,
        ) @Vector(4, f32) {
            const uniform_read: *const UniformInputType = @ptrCast(@alignCast(uniform));
            const fragment_input_write: *FragmentInputType = @ptrCast(@alignCast(fragment_input));

            const output = vertexShader(uniform_read.*, vertex_index);

            fragment_input_write.* = output[1];

            return output[0];
        }

        pub fn fragmentShaderFnRuntime(
            uniform: *const anyopaque,
            fragment_input: *const anyopaque,
        ) @Vector(4, f32) {
            const uniform_read: *const UniformInputType = @ptrCast(@alignCast(uniform));
            const fragment_input_read: *const FragmentInputType = @ptrCast(@alignCast(fragment_input));

            return fragmentShader(
                uniform_read.*,
                fragment_input_read.*,
                undefined,
            );
        }

        pub const runtime: RuntimePipeline = .{
            .polygon_fill_mode = polygon_fill_mode,
            .vertexShader = &runtimeVertexShader,
            .fragmentShader = &fragmentShaderFnRuntime,
        };
    };
}

const RuntimePipeline = @import("Pipeline.zig");

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

fn calculateBarycentrics2D(triangle: [3]@Vector(2, f32), point: @Vector(2, f32)) @Vector(3, f32) {
    const area_abc = @fabs(vectorCross2D(triangle[1] - triangle[0], triangle[2] - triangle[0]));

    const area_pbc = @fabs(vectorCross2D(triangle[1] - point, triangle[2] - point));
    const area_pca = @fabs(vectorCross2D(triangle[2] - point, triangle[0] - point));

    const one_over_area = 1 / area_abc;

    var bary: @Vector(3, f32) = .{
        area_pbc * one_over_area,
        area_pca * one_over_area,
        0,
    };

    bary[2] = 1 - bary[0] - bary[1];

    //TODO: I think the degenerate barys (generating barys of > 1) needs to be sovled properly?
    bary[0] = std.math.clamp(bary[0], 0, 1);
    bary[1] = std.math.clamp(bary[1], 0, 1);
    bary[2] = std.math.clamp(bary[2], 0, 1);

    return bary;
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
        // position[n] = position0[n] * (1 - t) + position1[n] * t;
        position[n] = vectorDotProduct(2, f32, .{ position0[n], position1[n] }, interpolant);
    }

    var interpolator: Interpolator = undefined;

    inline for (std.meta.fields(Interpolator)) |field| {
        const vector_dimensions = @typeInfo(field.type).Vector.len;
        const Component = std.meta.Child(field.type);

        for (0..vector_dimensions) |n| {
            @field(interpolator, field.name)[n] = vectorDotProduct(2, Component, .{
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

pub fn TriangleClipper(comptime Interpolator: type) type {
    return struct {
        index_in_buffer: [512]u8 = undefined,
        index_out_buffer: [512]u8 = undefined,
        vertex_position_buffer: [256]@Vector(4, f32) = undefined,
        vertex_interpolator_buffer: [256]Interpolator = undefined,

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
            self.indices_out.ptr = &self.index_out_buffer;
            self.indices_out.len = 0;

            if (self.indices_in.len == 0) return;

            var index_previous = self.indices_in[0];

            self.indices_in.len += 1;
            self.indices_in[self.indices_in.len - 1] = index_previous;

            var previous_vertex_position = self.vertex_positions[index_previous];

            var previous_dotp = vectorDotProduct(4, f32, plane, previous_vertex_position);

            for (1..self.indices_in.len) |i| {
                const index = self.indices_in[i];

                const vertex_position = self.vertex_positions[index];

                const dotp = vectorDotProduct(4, f32, plane, vertex_position);

                if (previous_dotp >= 0) {
                    self.indices_out.len += 1;
                    self.indices_out[self.indices_out.len - 1] = index_previous;
                }

                if (std.math.sign(dotp) != std.math.sign(previous_dotp)) {
                    const t = if (dotp < 0) previous_dotp / (previous_dotp - dotp) else -previous_dotp / (dotp - previous_dotp);

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
                }

                index_previous = index;
                previous_dotp = dotp;
            }

            std.mem.swap([]u8, &self.indices_in, &self.indices_out);
        }
    };
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

    for (1..4) |i| {
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
    var index_offset: usize = 0;

    for (clipping_planes) |clipping_plane| {
        for (0..previous_index_count / 3) |triangle_index| {
            const index_0 = indices_out[index_offset + (triangle_index * 3 + 0)];
            const index_1 = indices_out[index_offset + (triangle_index * 3 + 1)];
            const index_2 = indices_out[index_offset + (triangle_index * 3 + 2)];

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

                vertex_positions_out[vertex_count] = clip_result.vertex_positions_out[index];
                vertex_interpolators_out[vertex_count] = clip_result.vertex_interpolators_out[index];

                indices_out[index_count] = @intCast(vertex_count);
                index_count += 1;

                vertex_count += 1;
            }

            index_offset += 3;
        }
    }

    return .{
        .indices_out = indices_out,
        .vertex_positions_out = vertex_positions_out,
        .vertex_interpolators_out = vertex_interpolators_out,
        .index_count = index_count,
    };
}

fn pipelineDrawTriangle(
    self: Renderer,
    pass: Pass,
    uniform: anytype,
    triangle_index: usize,
    comptime pipeline: anytype,
) void {
    @setFloatMode(.Optimized);

    var pipeline_runtime: RuntimePipeline = pipeline.runtime;

    const use_runtime: bool = true;

    var fragment_input_0: pipeline.FragmentInput = undefined;
    var fragment_input_1: pipeline.FragmentInput = undefined;
    var fragment_input_2: pipeline.FragmentInput = undefined;

    var triangle: [3]@Vector(4, f32) = undefined;

    if (use_runtime) {
        var _uniform: pipeline.UniformInput = uniform;

        triangle[0] = pipeline_runtime.vertexShader(&_uniform, triangle_index * 3 + 0, &fragment_input_0);
        triangle[1] = pipeline_runtime.vertexShader(&_uniform, triangle_index * 3 + 1, &fragment_input_1);
        triangle[2] = pipeline_runtime.vertexShader(&_uniform, triangle_index * 3 + 2, &fragment_input_2);
    } else {
        const vertex_result_0 = @call(.always_inline, pipeline.vertexShader, .{ uniform, triangle_index * 3 + 0 });
        const vertex_result_1 = @call(.always_inline, pipeline.vertexShader, .{ uniform, triangle_index * 3 + 1 });
        const vertex_result_2 = @call(.always_inline, pipeline.vertexShader, .{ uniform, triangle_index * 3 + 2 });

        triangle[0] = vertex_result_0[0];
        triangle[1] = vertex_result_1[0];
        triangle[2] = vertex_result_2[0];

        fragment_input_0 = vertex_result_0[1];
        fragment_input_1 = vertex_result_1[1];
        fragment_input_2 = vertex_result_2[1];
    }

    const clip_volume_min = @Vector(4, f32){ -1, -1, 0, std.math.floatMin(f32) };
    const clip_volume_max = @Vector(4, f32){ 1, 1, 1, std.math.floatMax(f32) };

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
        .{ triangle[0][0] * (1 / triangle[0][3]), triangle[0][1] * (1 / triangle[0][3]) },
        .{ triangle[1][0] * (1 / triangle[1][3]), triangle[1][1] * (1 / triangle[1][3]) },
        .{ triangle[2][0] * (1 / triangle[2][3]), triangle[2][1] * (1 / triangle[2][3]) },
    });

    //backface and contribution cull
    if (two_triangle_area <= 0) {
        return;
    }

    //True if the triangle is entirely contained in the clip volume
    const entire_fit = ((@reduce(.And, triangle[0] <= clip_volume_max) and
        @reduce(.And, triangle[1] <= clip_volume_max) and
        @reduce(.And, triangle[2] <= clip_volume_max)) or
        (@reduce(.And, triangle[0] >= clip_volume_min) and
        @reduce(.And, triangle[1] >= clip_volume_min) and
        @reduce(.And, triangle[2] >= clip_volume_min)));

    //2x2 transform matrix
    const clip_transform: @Vector(2 * 2, f32) = .{
        1, 0,
        0, -1,
    };

    if (entire_fit) {
        for (&triangle) |*point| {
            const inverse_w = 1 / point[3];
            point[0] *= inverse_w;
            point[1] *= inverse_w;
            point[2] *= inverse_w;

            const transformed = matrixVectorProduct(2, f32, clip_transform, .{ point[0], point[1] });

            point.* = .{ transformed[0], transformed[1], point[2], point[3] };
        }

        self.pipelineRasteriseTriangle(
            pipeline,
            pass,
            uniform,
            triangle,
            .{
                fragment_input_0,
                fragment_input_1,
                fragment_input_2,
            },
        );

        return;
    }

    var clipper = TriangleClipper(pipeline.FragmentInput).init(
        &triangle,
        &.{ fragment_input_0, fragment_input_1, fragment_input_2 },
        0,
        1,
        2,
    );

    const clipping_planes: [6]@Vector(4, f32) = .{
        .{ -1, 0, 0, 1 },
        .{ 1, 0, 0, 1 },
        .{ 0, -1, 0, 1 },
        .{ 0, 1, 0, 1 },
        .{ 0, 0, -1, 1 },
        .{ 0, 0, 1, 1 },
    };

    for (clipping_planes) |plane| {
        clipper.clipToPlane(plane);
    }

    for (clipper.vertex_positions) |*point| {
        const inverse_w = 1 / point[3];
        point[0] *= inverse_w;
        point[1] *= inverse_w;
        point[2] *= inverse_w;

        const transformed = matrixVectorProduct(2, f32, clip_transform, .{ point[0], point[1] });

        point.* = .{ transformed[0], transformed[1], point[2], point[3] };
    }

    for (0..clipper.indices_out.len / 3) |clip_triangle_index| {
        const index_0 = clipper.indices_out[clip_triangle_index * 3];
        const index_1 = clipper.indices_out[clip_triangle_index * 3 + 1];
        const index_2 = clipper.indices_out[clip_triangle_index * 3 + 2];

        self.pipelineRasteriseTriangle(
            pipeline,
            pass,
            uniform,
            .{
                clipper.vertex_positions[index_0],
                clipper.vertex_positions[index_1],
                clipper.vertex_positions[index_2],
            },
            .{
                clipper.vertex_interpolators[index_0],
                clipper.vertex_interpolators[index_1],
                clipper.vertex_interpolators[index_2],
            },
        );
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
            render_pass: Pass,
            _uniform: anytype,
            triangle: [3]@Vector(4, f32),
            triangle_attributes: [3]pipeline.FragmentInput,
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

            var pixel_y: isize = @max(right.p0[1], 0);

            while (pixel_y < @min(right.p1[1], @as(isize, @intCast(render_pass.color_image.height)))) : (pixel_y += 1) {
                defer {
                    factor0 += factor_step_0;
                    factor1 += factor_step_1;
                }

                drawSpan(
                    render_pass,
                    _uniform,
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
            render_pass: Pass,
            _uniform: anytype,
            span: @This(),
            triangle: [3]@Vector(4, f32),
            triangle_attributes: [3]pipeline.FragmentInput,
            screen_area: f32,
            inverse_screen_area: f32,
            pixel_y: isize,
        ) void {
            _ = screen_area;
            const xdiff = span.x1 - span.x0;

            const factor_step = 1 / @as(f32, @floatFromInt(xdiff));

            var factor: f32 = 0;

            var pixel_x: isize = @max(span.x0, 0);

            const pixel_increment: isize = switch (pipeline.polygon_fill_mode) {
                .line => @max(@as(isize, @intCast(std.math.absCast(span.x1 - span.x0 - 1))), 1),
                .fill => 1,
            };

            while (pixel_x < span.x1) : (pixel_x += pixel_increment) {
                defer {
                    factor += factor_step;
                }

                if (pixel_x < 0 or pixel_x >= render_pass.color_image.width) continue;

                var point = @Vector(3, f32){
                    ((@as(f32, @floatFromInt(pixel_x)) / @as(f32, @floatFromInt(render_pass.color_image.width))) * 2) - 1,
                    ((@as(f32, @floatFromInt(pixel_y)) / @as(f32, @floatFromInt(render_pass.color_image.height))) * 2) - 1,
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

                const fragment_index = @as(usize, @intCast(pixel_x)) + @as(usize, @intCast(pixel_y)) * render_pass.color_image.width;

                const depth = &render_pass.depth_buffer[fragment_index];

                if (point[2] > depth.*) {
                    continue;
                }

                depth.* = point[2];

                var fragment_input: pipeline.FragmentInput = undefined;

                inline for (std.meta.fields(Interpolators)) |field| {
                    const vector_dimensions = @typeInfo(field.type).Vector.len;

                    inline for (0..vector_dimensions) |dimension| {
                        const Component = std.meta.Child(field.type);

                        const interpolator_lane: @Vector(3, Component) = .{
                            @field(triangle_attributes[0], field.name)[dimension],
                            @field(triangle_attributes[1], field.name)[dimension],
                            @field(triangle_attributes[2], field.name)[dimension],
                        };

                        @field(fragment_input, field.name)[dimension] = vectorDotProduct(3, Component, interpolator_lane, barycentrics);
                    }
                }

                const pixel = render_pass.color_image.texelFetch(.{ @intCast(pixel_x), @intCast(pixel_y) });

                var fragment_color: @Vector(4, f32) = undefined;

                const use_runtime: bool = true;

                if (use_runtime) {
                    var __uniform: pipeline.UniformInput = _uniform;
                    var _fragment_input = fragment_input;

                    fragment_color = pipeline.runtime.fragmentShader(&__uniform, &_fragment_input);
                } else {
                    fragment_color = @call(.always_inline, pipeline.fragmentShader, .{
                        _uniform,
                        fragment_input,
                        point,
                    });
                }

                pixel.* = Image.Color.fromNormalized(fragment_color);
            }
        }
    };

    const view_scale = @Vector(4, f32){
        @as(f32, @floatFromInt(pass.color_image.width)),
        @as(f32, @floatFromInt(pass.color_image.height)),
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

    const p0_orig: @Vector(4, isize) = @intFromFloat(@ceil((points[0] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale));
    const p1_orig: @Vector(4, isize) = @intFromFloat(@ceil((points[1] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale));
    const p2_orig: @Vector(4, isize) = @intFromFloat(@ceil((points[2] + @Vector(4, f32){ 1, 1, 1, 1 }) / @Vector(4, f32){ 2, 2, 2, 2 } * view_scale));

    const p0: @Vector(3, isize) = .{ p0_orig[0], p0_orig[1], p0_orig[2] };
    const p1: @Vector(3, isize) = .{ p1_orig[0], p1_orig[1], p1_orig[2] };
    const p2: @Vector(3, isize) = .{ p2_orig[0], p2_orig[1], p2_orig[2] };

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
        pass,
        uniform,
        points,
        fragment_inputs,
        screen_area,
        inverse_screen_area,
        edges[long_edge_index],
        edges[short_edge_1],
    );
    Span.drawEdges(
        pass,
        uniform,
        points,
        fragment_inputs,
        screen_area,
        inverse_screen_area,
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
