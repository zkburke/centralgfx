const std = @import("std");
const c = @import("c_bindings.zig");
const Image = @import("Image.zig");

const Renderer = @This();

window: ?*c.SDL_Window = null,
sdl_renderer: ?*c.SDL_Renderer = null,
sdl_texture: ?*c.SDL_Texture = null,

pub fn init(self: *Renderer, window_width: usize, window_height: usize, surface_width: usize, surface_height: usize, title: []const u8) !void {
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
}

pub fn deinit(self: Renderer) void {
    defer c.SDL_DestroyWindow(self.window);
    defer c.SDL_DestroyRenderer(self.sdl_renderer);
    defer c.SDL_DestroyTexture(self.sdl_texture);
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
    const VertexShaderFn = fn (uniform: UniformInputType, vertex_index: usize) struct { @Vector(2, f32), FragmentInputType };
    const FragmentShaderFn = fn (uniform: UniformInputType, input: FragmentInputType, position: @Vector(2, f32), pixel: *Image.Color) void;

    return struct {
        pub const UniformInput = UniformInputType;
        pub const FragmentInput = FragmentInputType;

        pub const vertexShader: VertexShaderFn = vertexShaderFn;
        pub const fragmentShader: FragmentShaderFn = fragmentShaderFn;
    };
}

// pub fn pipelineDraw(
//     self: Renderer,
//     pass: Pass,
//     primitive_type: PrimitiveType,
//     uniform: anytype,
//     vertex_count: usize,
//     FragmentInput: type,
//     comptime vertexShader: anytype,
//     comptime fragmentShader: anytype,
//     ) void
// {

// }

///Draws a triangle using rasterisation, supplying a compile time known pipeline
///Of operation functions
pub fn drawTrianglePipeline(
    self: Renderer,
    pass: Pass,
    uniform: anytype,
    vertex_count: usize,
    comptime pipeline: anytype,
    comptime FragmentInput: type,
    comptime vertexShader: anytype,
    comptime fragmentShader: anytype,
) void {
    _ = pipeline;
    _ = self;
    _ = pass;
    _ = uniform;
    _ = vertex_count;
    _ = FragmentInput;
    _ = vertexShader;
    _ = fragmentShader;
}

pub fn drawLinePipeline(
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

        const viewport_scale = @Vector(2, f32){ @as(f32, @floatFromInt(pass.color_image.width)), @as(f32, @floatFromInt(pass.color_image.height)) };

        const target_p0 = (p0 + @as(@Vector(2, f32), @splat(1))) / @as(@Vector(2, f32), @splat(2)) * viewport_scale;
        const target_p1 = (p1 + @as(@Vector(2, f32), @splat(1))) / @as(@Vector(2, f32), @splat(2)) * viewport_scale;

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

            const interpolated_point = vectorLerp(f32, 2, p0, p1, @Vector(2, f32){ line_ratio_x, line_ratio_y });

            var fragment_input: pipeline.FragmentInput = undefined;

            inline for (std.meta.fields(pipeline.FragmentInput)) |field| {
                if (field.is_comptime) continue;

                const field_type_info = @typeInfo(field.type);

                switch (field_type_info) {
                    .Float => {
                        @field(fragment_input, field.name) = lerp(@field(fragment_input0, field.name), @field(fragment_input1, field.name), line_ratio_x);
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

            @call(.always_inline, pipeline.fragmentShader, .{ uniform, fragment_input, interpolated_point, &pass.color_image.pixels[index] });
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

    return p1[0] * p2[1] - p2[0] * p1[1];
}

pub fn drawTriangle(self: Renderer, pass: Pass, points: [3]@Vector(2, f32)) void {
    var view_scale = @Vector(2, f32){
        @as(f32, @floatFromInt(pass.color_image.width)),
        @as(f32, @floatFromInt(pass.color_image.height)),
    };

    var min = (points[0] + @Vector(2, f32){ 1, 1 }) / @Vector(2, f32){ 2, 2 } * view_scale;
    var mid = (points[1] + @Vector(2, f32){ 1, 1 }) / @Vector(2, f32){ 2, 2 } * view_scale;
    var max = (points[2] + @Vector(2, f32){ 1, 1 }) / @Vector(2, f32){ 2, 2 } * view_scale;

    if (max[1] < mid[1]) {
        const temp = max;
        max = mid;
        mid = temp;
    }

    if (mid[1] < min[1]) {
        const temp = mid;
        mid = min;
        min = temp;
    }

    if (max[1] < mid[1]) {
        const temp = max;
        max = mid;
        mid = temp;
    }

    self.scanTriangle(pass, min, mid, max, twoTriangleArea(.{ min, max, mid }) >= 0);
}

pub fn drawBoundedTriangle(self: Renderer, pass: Pass, points: [3]@Vector(2, f32)) void {
    _ = self;

    const Edge = struct {
        a: f32,
        b: f32,
        c: f32,
        tie: bool,

        pub fn init(vertices: [2]@Vector(2, f32)) @This() {
            var this: @This() = undefined;

            this.a = vertices[0][1] - vertices[1][1];
            this.b = vertices[1][0] - vertices[0][0];
            this.c = (this.a * (vertices[0][0] + vertices[1][0]) + this.b * (vertices[0][1] + vertices[1][1])) / 2;
            this.tie = if (this.a != 0) this.a > 0 else this.b > 0;

            return this;
        }

        pub fn evaluate(this: @This(), x: f32, y: f32) f32 {
            return this.a * x + this.b * y + this.c;
        }

        pub fn isInside(this: @This(), x: f32, y: f32) bool {
            const val = this.evaluate(x, y);

            return val >= 0 and this.tie;
        }
    };

    var min_x = @min(@min(points[0][0], points[1][0]), points[2][0]);
    var min_y = @min(@min(points[0][1], points[1][1]), points[2][1]);
    var max_x = @max(@max(points[0][0], points[1][0]), points[2][0]);
    var max_y = @max(@max(points[0][1], points[1][1]), points[2][1]);

    min_x = 0.1;
    min_y = 0.1;
    max_x = 0.9;
    max_y = 0.9;

    //clip
    // min_x = @maximum(min_x, -1);
    // max_x = @minimum(max_x, 1);
    // min_y = @maximum(min_y, -1);
    // max_y = @minimum(max_y, 1);

    // min_x += 1;
    // max_x += 1;
    // min_y += 1;
    // max_y += 1;

    // min_x /= 2;
    // max_x /= 2;
    // min_y /= 2;
    // max_y /= 2;

    const e0 = Edge.init(.{ points[1], points[2] });
    const e1 = Edge.init(.{ points[2], points[0] });
    const e2 = Edge.init(.{ points[0], points[1] });

    const area = 0.5 * (e0.c + e1.c + e2.c);

    if (area < 0 and false) {
        return;
    }

    var y: usize = @as(usize, @intFromFloat(min_y * @as(f32, @floatFromInt(pass.color_image.height))));

    while (y < @as(usize, @intFromFloat(max_y * @as(f32, @floatFromInt(pass.color_image.height))))) : (y += 1) {
        var pixels = pass.color_image.pixels.ptr + y * pass.color_image.width;

        var x: usize = @as(usize, @intFromFloat(min_x * @as(f32, @floatFromInt(pass.color_image.width))));

        while (x < @as(usize, @intFromFloat(max_x * @as(f32, @floatFromInt(pass.color_image.width))))) : ({
            x += 1;
            pixels += 1;
        }) {
            const xf = @as(f32, @floatFromInt(x));
            const yf = @as(f32, @floatFromInt(y));

            if (e0.isInside(xf, yf) and e1.isInside(xf, yf) and e2.isInside(xf, yf)) {
                pixels[0] = .{ .r = 0, .g = 255, .b = 0, .a = 255 };
            }
        }

        pixels[0] = .{ .r = 255, .g = 0, .b = 0, .a = 255 };
    }
}

fn scanTriangle(
    self: Renderer,
    pass: Pass,
    min: @Vector(2, f32),
    mid: @Vector(2, f32),
    max: @Vector(2, f32),
    handedness: bool,
) void {
    _ = self;

    const Edge = struct {
        start_y: f32 = 0,
        end_y: f32 = 0,
        step_x: f32 = 0,
        x: f32 = 0,
        color: @Vector(4, f32),

        fn init(start: @Vector(2, f32), end: @Vector(2, f32), color: @Vector(4, f32)) @This() {
            const start_y = @ceil(start[1]);
            const end_y = @ceil(end[1]);
            const dist_x = end[0] - start[0];
            const dist_y = end[1] - start[1];
            const step_x = dist_x / dist_y;
            const prestep_y = start_y - start[1];

            return .{
                .start_y = start_y,
                .end_y = end_y,
                .step_x = step_x,
                .x = start[1] + prestep_y * step_x,
                .color = color,
            };
        }

        fn step(edge: *@This()) void {
            edge.x += edge.step_x;
        }

        fn drawScanLine(render_pass: Pass, left: @This(), right: @This(), y: usize) void {
            const min_x = @as(i32, @intFromFloat(@max(@ceil(left.x), 0)));
            const max_x = @as(i32, @intFromFloat(@min(@ceil(right.x), @as(f32, @floatFromInt(render_pass.color_image.width)))));

            var x = min_x;

            const min_color = left.color;
            const max_color = right.color;

            var interpolant: f32 = 0;
            var interpolation_step = 1 / @as(f32, @floatFromInt(max_x - min_x));

            while (x < max_x) : ({
                x += 1;
                interpolant += interpolation_step;
            }) {
                const color = vectorLerp(f32, 4, min_color, max_color, @splat(interpolant));

                render_pass.color_image.setPixel(
                    .{
                        .x = @as(u32, @intCast(x)),
                        .y = y,
                    },
                    Image.Color.fromNormalized(color),
                );
            }
        }
    };

    const colors = [3]@Vector(4, f32){
        .{ 1, 0, 0, 1 },
        .{ 0, 1, 0, 1 },
        .{ 0, 0, 1, 1 },
    };

    const top_bottom = Edge.init(min, max, colors[0]);
    const top_middle = Edge.init(min, mid, colors[2]);
    const middle_bottom = Edge.init(mid, max, colors[1]);

    var left = top_bottom;
    var right = top_middle;

    if (handedness) {
        const temp = left;
        left = right;
        right = temp;
    }

    var start_y = top_middle.start_y;
    var end_y = @min(top_middle.end_y, @as(f32, @floatFromInt(pass.color_image.height)));

    var y = start_y;

    while (y < end_y) : (y += 1) {
        Edge.drawScanLine(pass, left, right, @as(usize, @intFromFloat(@ceil(y))));

        left.step();
        right.step();
    }

    left = top_bottom;
    right = middle_bottom;

    if (true) return;

    if (handedness) {
        const temp = left;
        left = right;
        right = temp;
    }

    start_y = top_middle.start_y;
    end_y = @min(middle_bottom.end_y, @as(f32, @floatFromInt(pass.color_image.height)));

    y = start_y;

    while (y < end_y) : (y += 1) {
        Edge.drawScanLine(pass, left, right, @as(usize, @intFromFloat(@ceil(y))));

        left.step();
        right.step();
    }
}

pub fn drawCircle(self: Renderer, pass: Pass, position: [2]f32, radius: f32) void {
    _ = self;
    _ = radius;

    const target_position = [_]usize{
        @as(usize, @intFromFloat(((position[0] + 1) / 2) * pass.color_image.width)),
        @as(usize, @intFromFloat(((position[1] + 1) / 2) * pass.color_image.height)),
    };

    if (target_position[0] >= pass.color_image.width or target_position[1] >= pass.color_image.height) {
        return;
    }

    var y: usize = 0;

    while (y < pass.color_image.height) : (y += 1) {
        var x: usize = 0;

        while (x < pass.color_image.width) : (x += 1) {}
    }
}

pub fn presentImage(self: Renderer, image: Image) void {
    _ = c.SDL_UpdateTexture(self.sdl_texture, null, image.pixels.ptr, @as(c_int, @intCast(image.width * @sizeOf(Image.Color))));
    _ = c.SDL_RenderCopy(self.sdl_renderer, self.sdl_texture, null, null);
    _ = c.SDL_RenderPresent(self.sdl_renderer);
}
