const std = @import("std");
const c = @import("c_bindings.zig");
const zigimg = @import("zigimg");
const Image = @import("Image.zig");
const Renderer = @import("Renderer.zig");
const RayTracer = @import("RayTracer.zig");

const cog_png = @embedFile("res/wood_floor.png");

const MeshCluster = struct {
    pub const vertex_count = 128;

    vertex_begin: u32,
};

const TestPipelineClusterOutput = struct {};

const TestVertex = struct {
    position: @Vector(3, f32),
    color: @Vector(4, f32),
    uv: @Vector(2, f32),
};

const TestPipelineUniformInput = struct {
    texture: ?Image,
    vertices: []const TestVertex,
};

const TestPipelineFragmentInput = struct {
    color: @Vector(4, f32),
    uv: @Vector(2, f32),
};

fn testClusterShader(
    uniform: TestPipelineUniformInput,
    cluster: MeshCluster,
) ?TestPipelineClusterOutput {
    _ = uniform;
    _ = cluster;

    //return null for clulled
    return null;
}

fn testVertexShader(
    uniform: TestPipelineUniformInput,
    vertex_index: usize,
) struct { @Vector(3, f32), TestPipelineFragmentInput } {
    const vertex = uniform.vertices[vertex_index];

    const output: TestPipelineFragmentInput = .{
        .color = vertex.color,
        .uv = vertex.uv,
    };

    return .{ .{
        vertex.position[0] / @fabs(1.0 + vertex.position[2]),
        vertex.position[1] / @fabs(1.0 + vertex.position[2]),
        vertex.position[2],
    }, output };
}

fn testFragmentShader(
    uniform: TestPipelineUniformInput,
    input: TestPipelineFragmentInput,
    position: @Vector(3, f32),
    pixel: *Image.Color,
) void {
    _ = uniform;
    _ = position;

    var color: @Vector(4, f32) = .{ 1, 1, 1, 1 };

    color *= input.color;

    // if (uniform.texture) |texture|
    // color *= texture.affineSample(input.uv).toNormalized();

    pixel.* = Image.Color.fromNormalized(color);
}

pub const TestPipeline = Renderer.Pipeline(
    TestPipelineUniformInput,
    TestPipelineFragmentInput,
    testClusterShader,
    testVertexShader,
    testFragmentShader,
);

pub fn main() !void {
    var general_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = general_allocator.deinit();

    const allocator = general_allocator.allocator();

    if (c.SDL_Init(c.SDL_INIT_VIDEO | c.SDL_INIT_AUDIO) != 0) {
        return error.FailedToCreateWindow;
    }

    const window_width = 640;
    const window_height = 480;

    const surface_width = 640 / 8;
    const surface_height = 480 / 8;

    var renderer: Renderer = undefined;

    try renderer.init(allocator, window_width, window_height, surface_width, surface_height, "CentralGfx");
    defer renderer.deinit(allocator);

    var render_target = try Image.init(allocator, surface_width, surface_height);
    defer render_target.deinit(allocator);

    const depth_target = try allocator.alloc(f32, surface_width * surface_height);
    defer allocator.free(depth_target);

    var cog_image_loaded = try zigimg.Image.fromMemory(allocator, cog_png);
    defer cog_image_loaded.deinit();

    const cog_image_data = cog_image_loaded.rawBytes();

    var cog_image = try Image.initFromLinear(
        allocator,
        @as([*]Image.Color, @constCast(@ptrCast(cog_image_data)))[0..@as(usize, @intCast(cog_image_loaded.width * cog_image_loaded.height))],
        cog_image_loaded.width,
        cog_image_loaded.height,
    );
    defer cog_image.deinit(allocator);

    const enable_raster_pass = true;
    var enable_ray_pass = false;
    _ = enable_ray_pass;

    // var default_prng = std.rand.DefaultPrng.init(@intCast(u64, std.time.timestamp()));

    const core_count = try std.Thread.getCpuCount();

    const ray_threads = try allocator.alloc(std.Thread, core_count);
    defer allocator.free(ray_threads);

    // const thread_pixel_count = render_target.pixels.len / core_count;
    const thread_pixel_height = render_target.height / core_count;
    _ = thread_pixel_height;

    var thread_slice_index: usize = 0;
    _ = thread_slice_index;
    var pixel_offset: @Vector(2, usize) = .{ 0, 0 };
    _ = pixel_offset;

    const spheres = [_]RayTracer.Sphere{
        .{
            .radius = 0.5,
            .position = .{ 0, 0, -1.5 },
            .material_index = 0,
        },
        .{ .radius = 0.5, .position = .{ -1, 0, -1 }, .material_index = 1 },
        .{ .radius = 0.5, .position = .{ 1, 0, -1 }, .material_index = 3 },
        // .{ .radius = 100, .position = .{ 0, -100.5, -1 }, .material_index = 2 },
        // .{ .radius = 4, .position = .{ 0, 10, 3 }, .material_index = 1 },
        // .{ .radius = 4, .position = .{ 0, 10, 3 }, .material_index = 1 },
    };

    const planes = [_]RayTracer.Plane{
        .{
            .position = .{ 0, -0.5, 0 },
            .normal = .{ 0, 1, 0 },
            .material_index = 2,
        },
        // .{ .position = .{ 0, 0, 10 }, .normal = .{ 0, 0, -1 }, .material_index = 1, },
    };

    const materials = [_]RayTracer.Material{
        .{ .metal = .{ .albedo = .{
            .r = 255,
            .g = 255,
            .b = 255,
            .a = 255,
        }, .roughness = 0.25 } },
        .{ .lambertion = .{ .albedo = .{
            .r = 0,
            .g = 0,
            .b = 255,
            .a = 255,
        } } },
        .{ .lambertion = .{ .albedo = .{
            .r = 255,
            .g = 0,
            .b = 0,
            .a = 255,
        } } },
        .{ .dieletric = .{ .refractive_index = 1.5 } },
    };

    const lights = [_]RayTracer.Light{
        .{ .position = .{ 0, 0, 0 }, .color = .{ 1, 1, 1 } },
    };

    const scene = RayTracer.Scene{
        .spheres = &spheres,
        .materials = &materials,
        .planes = &planes,
        .lights = &lights,
    };
    _ = scene;

    while (!renderer.shouldWindowClose()) {
        const frame_start_time = std.time.microTimestamp();
        const time_s: f32 = @as(f32, @floatFromInt(c.SDL_GetTicks())) / 1000;

        if (true) {
            @memset(render_target.texel_buffer, Image.Color.fromNormalized(.{ 0.25, 0.25, 1, 1 }));
            @memset(depth_target, 1);

            const render_pass = Renderer.Pass{
                .color_image = render_target,
                .depth_buffer = depth_target,
            };

            if (enable_raster_pass) {
                var tris = [_][3][3]f32{
                    .{
                        .{ -0.5, 0.5, 0 },
                        .{ 0.5, 0.5, 0 },
                        .{ 0.0, -0.5, 0 },
                    },
                    .{
                        .{ -0.5, -0.5, 0 },
                        .{ 0.5, -0.5, 0 },
                        .{ 0.0, 0.5, 0 },
                    },
                };

                for (&tris) |*tri| for (tri[0..3]) |*vertex| {
                    vertex[0] += @sin(time_s);
                };

                for (tris) |tri| {
                    renderer.drawTriangle(
                        render_pass,
                        .{
                            .{ tri[0][0] + @sin(time_s), tri[0][1], tri[0][2] },
                            .{ tri[1][0] + @sin(time_s), tri[1][1], tri[1][2] },
                            tri[2],
                        },
                    );
                }

                var line_vertices = [_]TestVertex{ .{
                    .position = .{ -1.0 / 2.0, -1.0 / 2.0, 0 },
                    .color = .{ 1, 1, 1, 1 },
                    .uv = .{ 0, 0 },
                }, .{
                    .position = .{ 1.0 / 2.0, 1.0 / 2.0, 0 },
                    .color = .{ 1, 1, 1, 1 },
                    .uv = .{ 1, 1 },
                }, .{
                    .position = .{ (-1.0 / 2.0) + 0.25, (-1.0 / 2.0), 0 },
                    .color = .{ 1, 0, 0, 1 },
                    .uv = .{ 0, 0 },
                }, .{
                    .position = .{ (1.0 / 2.0) + 0.25, (1.0 / 2.0), 0 },
                    .color = .{ 0, 1, 0, 1 },
                    .uv = .{ 1, 1 },
                } };

                const uniforms = TestPipelineUniformInput{
                    .texture = cog_image,
                    .vertices = &line_vertices,
                };

                renderer.pipelineDrawLine(render_pass, uniforms, line_vertices.len, TestPipeline);

                var triangle = [3]@Vector(3, f32){
                    .{ -0.5, 0.5, 0 },
                    .{ 0.5, 0.5, 0 },
                    .{ 0, -0.5, 0 },
                };

                var triangle_vertices = [_]TestVertex{
                    .{
                        .position = .{ -0.5, 0.5, 1 },
                        .color = .{ 1, 0, 0, 1 },
                        .uv = .{ 0, 0 },
                    },
                    .{
                        .position = .{ 0.5, 0.5, 0.5 },
                        .color = .{ 0, 1, 0, 1 },
                        .uv = .{ 1, 0 },
                    },
                    .{
                        .position = .{ 0, -0.5, 0 },
                        .color = .{ 0, 0, 1, 1 },
                        .uv = .{ 0.5, 1.0 },
                    },
                };

                triangle_vertices[1].position[0] += @sin(time_s);
                triangle_vertices[1].position[2] += @sin(time_s);

                for (&triangle) |*vertex| {
                    vertex.* = (vertex.* + @as(@Vector(3, f32), @splat(@as(f32, 1)))) / @as(@Vector(3, f32), @splat(@as(f32, 2)));
                }

                renderer.drawTriangle(render_pass, triangle);

                renderer.pipelineDrawTriangles(
                    render_pass,
                    .{
                        .texture = cog_image,
                        .vertices = &triangle_vertices,
                    },
                    1,
                    TestPipeline,
                );
            }

            // if (enable_ray_pass) {
            //     const render_start = std.time.milliTimestamp();

            //     // const thread_args = .{ render_target, pixel_offset, Vec(2, usize) { render_target.width, pixel_offset[1] + thread_pixel_height }};

            //     // const thread = try std.Thread.spawn(.{}, RayTracer.traceRays, thread_args);
            //     // defer thread.join();

            //     // spheres[0].position[0] = @sin(time);

            //     RayTracer.traceRays(scene, render_target, pixel_offset, .{ render_target.width, pixel_offset[1] + thread_pixel_height });
            //     pixel_offset += @Vector(2, usize){ 0, thread_pixel_height };

            //     std.log.err("Rendered slice {} ({}-{}): time = {}ms", .{ thread_slice_index, pixel_offset[1], thread_pixel_height, std.time.milliTimestamp() - render_start });

            //     thread_slice_index += 1;

            //     if (thread_slice_index == ray_threads.len) {
            //         thread_slice_index = 0;
            //         pixel_offset = .{ 0, 0 };
            //         enable_ray_pass = false;
            //     }

            //     // for (ray_threads) |thread|
            //     // {
            //     //     _ = thread;
            //     // }
            // }
        }

        const frame_end_time = std.time.microTimestamp();
        const frame_time = frame_end_time - frame_start_time;

        std.log.err("frame_time: {d:.2}ms", .{@as(f32, @floatFromInt(frame_time)) / 1000});

        renderer.presentImage(render_target);
    }
}
