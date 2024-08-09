const std = @import("std");
const c = @import("c_bindings.zig");
const zigimg = @import("zigimg");
const zalgebra = @import("zalgebra");
const zgltf = @import("zgltf");
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
    normal: @Vector(3, f32) = .{ 0, 1, 0 },
};

pub const PointLight = struct {
    position: @Vector(3, f32),
    color: @Vector(3, f32),
    intensity: f32,
};

pub const TestPipelineUniformInput = struct {
    texture: ?Image,
    vertices: []const TestVertex,
    indices: []const u32,
    lights: []const PointLight = &.{
        .{
            .position = .{ -2, 3, 0 },
            .color = .{ 1, 1, 1 },
            .intensity = 10,
        },
    },
    transform: zalgebra.Mat4,
    transform_geo: struct {
        ///Vector
        position: @Vector(3, f32) = .{ 0, 0, 0 },
        ///Bivector
        scale: @Vector(3, f32) = .{ 1, 1, 1 },
        ///scalar + Bivector (Rotor)
        rotor: @Vector(4, f32) = .{ 1, 0, 0, 0 },
    },
    view_projection: zalgebra.Mat4,
    base_vertex: u32 = 0,
};

const TestPipelineFragmentInput = struct {
    color: @Vector(4, f32),
    uv: @Vector(2, f32),
    normal: @Vector(3, f32),
    position_world_space: @Vector(3, f32),
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

fn testVertexIndexShader(uniform: TestPipelineUniformInput, vertex_index: usize) usize {
    const index = uniform.indices[uniform.base_vertex + vertex_index];

    return index;
}

fn testVertexPositionShader(uniform: TestPipelineUniformInput, vertex_index: usize) @Vector(4, f32) {
    _ = vertex_index;
    _ = uniform;
}

fn testVertexInterpolatorShader(uniform: TestPipelineFragmentInput, vertex_index: f32, position: @Vector(4, f32)) TestPipelineFragmentInput {
    _ = position;
    _ = vertex_index;
    _ = uniform;
}

fn vectorDotProduct(a: @Vector(3, f32), b: @Vector(3, f32)) f32 {
    return a[0] * b[0] + a[1] * b[1];
}

///Wedge product
fn vectorWedge(a: @Vector(3, f32), b: @Vector(3, f32)) @Vector(3, f32) {
    return .{
        a[0] * b[1] - a[1] * b[0],
        a[0] * b[2] - a[2] * b[0],
        a[1] * b[2] - a[2] * b[1],
    };
}

//R = a + xe0 + ye1 + ze2
const Rotor = struct {
    a: f32,
    b01: f32,
    b02: f32,
    b12: f32,

    pub fn fromVectors(v0: @Vector(3, f32), v1: @Vector(3, f32)) Rotor {
        const a = 1 + vectorDotProduct(v0, v1);

        const minusb = vectorWedge(v1, v0);

        return .{
            .a = a,
            .b01 = minusb[0],
            .b02 = minusb[1],
            .b12 = minusb[2],
        };
    }

    pub fn productRotor(p: Rotor, q: Rotor) Rotor {
        var r: Rotor = .{
            .a = 0,
            .b01 = 0,
            .b02 = 0,
            .b12 = 0,
        };

        r.a = p.a * q.a - p.b01 * q.b01 - p.b02 * q.b02 - p.b12 * q.b12;
        r.b01 = p.b01 * q.a + p.a * q.b01 + p.b12 * q.b02 - p.b02 * q.b12;
        r.b02 = p.b02 * q.a + p.a * q.b02 - p.b12 * q.b01 + p.b01 * q.b12;
        r.b12 = p.b12 * q.a + p.a * q.b12 + p.b02 * q.b01 - p.b01 * q.b02;
    }

    pub fn rotateVector(p: Rotor, v: @Vector(3, f32)) @Vector(3, f32) {
        const q = @Vector(3, f32){
            p.a * v[0] + v[1] * p.b01 + v[2] * p.b02,
            p.a * v[1] - v[0] * p.b01 + v[2] * p.b12,
            p.a * v[2] - v[0] * p.b02 - v[1] * p.b12,
        };

        const q012 = v[0] * p.b12 - v[1] * p.b02 + v[2] * p.b01;

        const r = @Vector(3, f32){
            p.a * q[0] + q[1] * p.b01 + q[2] * p.b02 + q012 * p.b12,
            p.a * q[1] - q[0] * p.b01 - q012 * p.b02 + q[2] * p.b12,
            p.a * q[2] + q012 * p.b01 - q[0] * p.b02 - q[1] * p.b12,
        };

        return r;
    }

    pub fn rotateRotor(p: Rotor, q: Rotor) Rotor {
        return p.productRotor(q).productRotor(p.reverse());
    }

    pub fn reverse(self: Rotor) Rotor {
        return .{
            .a = self.a,
            .b01 = -self.b01,
            .b02 = -self.b02,
            .b12 = -self.b12,
        };
    }
};

fn testVertexShader(
    uniform: TestPipelineUniformInput,
    vertex_index: usize,
) struct { @Vector(4, f32), TestPipelineFragmentInput } {
    const index = uniform.indices[vertex_index];
    const vertex = uniform.vertices[uniform.base_vertex + index];

    const use_geo = false;

    var world_space_position = uniform.transform.mulByVec4(.{ .data = .{ vertex.position[0], vertex.position[1], vertex.position[2], 1 } });

    if (use_geo) {
        const rotated_vertex = Rotor.rotateVector(.{
            .a = uniform.transform_geo.rotor[0],
            .b01 = uniform.transform_geo.rotor[1],
            .b02 = uniform.transform_geo.rotor[2],
            .b12 = uniform.transform_geo.rotor[3],
        }, vertex.position);

        const world_space_position_v3 = uniform.transform_geo.position + rotated_vertex * uniform.transform_geo.scale;

        world_space_position.data = .{
            world_space_position_v3[0],
            world_space_position_v3[1],
            world_space_position_v3[2],
            1,
        };
    }

    var triangle_color: f32 = @floatFromInt(vertex_index / 3);

    triangle_color /= @floatFromInt((uniform.indices.len / 3) * 2);

    var color: @Vector(4, f32) = @splat(1);

    color[(vertex_index / 3) % 3] = triangle_color;
    color[(vertex_index / 3 + 1) % 3] = triangle_color + 0.25;
    // color = @splat(1);

    const output: TestPipelineFragmentInput = .{
        .color = color,
        .uv = .{ vertex.uv[0], vertex.uv[1] },
        .normal = vertex.normal,
        .position_world_space = .{ world_space_position.data[0], world_space_position.data[1], world_space_position.data[2] },
    };

    const position = uniform.view_projection.mulByVec4(world_space_position);

    return .{ .{ position.data[0], position.data[1], position.data[2], position.data[3] }, output };
}

fn testFragmentShader(
    uniform: TestPipelineUniformInput,
    input: TestPipelineFragmentInput,
    position: @Vector(3, f32),
) @Vector(4, f32) {
    _ = position;
    var color: @Vector(4, f32) = .{ 1, 1, 1, 1 };

    // color *= input.color;

    if (uniform.texture) |texture| {
        // color *= texture.sampleBilinear(input.uv, .point, .black).toNormalized();
        color *= texture.sample(input.uv, .point, .black).toNormalized();
    }

    const enable_lighting = true;

    if (enable_lighting) {
        var light_contribution: @Vector(3, f32) = .{ 0.1, 0.1, 0.1 };

        for (uniform.lights) |light| {
            const light_pos = light.position;
            const light_dir = zalgebra.Vec3.norm(.{ .data = light_pos - input.position_world_space }).data;
            const light_radius = 0.01;

            const distance_to_light = zalgebra.Vec3.distance(.{ .data = light_pos }, .{ .data = input.position_world_space });

            const attentuation = 1 / @max(distance_to_light * distance_to_light, light_radius * light_radius);

            const light_intensity = @max(zalgebra.Vec3.dot(.{ .data = input.normal }, .{ .data = light_dir }), 0);

            light_contribution += light.color * @as(@Vector(3, f32), @splat(light_intensity * light.intensity * attentuation));
        }

        color *= @Vector(4, f32){ light_contribution[0], light_contribution[1], light_contribution[2], 1 };
    }

    // color[0] = input.uv[0];
    // color[1] = input.uv[1];
    // color[2] = 0;
    // color[3] = 1;

    color[0] = std.math.clamp(color[0], 0, 1);
    color[1] = std.math.clamp(color[1], 0, 1);
    color[2] = std.math.clamp(color[2], 0, 1);
    color[3] = std.math.clamp(color[3], 0, 1);

    const draw_depth_factor = 1;
    _ = draw_depth_factor;

    return color;
}

pub const TestPipeline = Renderer.Pipeline(
    TestPipelineUniformInput,
    TestPipelineFragmentInput,
    testClusterShader,
    testVertexShader,
    testFragmentShader,
    .fill,
);

pub const Mesh = struct {
    vertices: []TestVertex,
    indices: []u32,
    sub_meshes: []SubMesh,
    textures: []Image = &.{},

    pub const SubMesh = extern struct {
        vertex_offset: u32,
        vertex_count: u32,
        index_offset: u32,
        index_count: u32,
        material_index: u32,
        transform: [4][4]f32,
        bounding_min: [3]f32,
        bounding_max: [3]f32,

        albedo_texture_index: u32,
    };
};

pub fn loadMesh(allocator: std.mem.Allocator, file_path: []const u8) !Mesh {
    const file_directory_name = std.fs.path.dirname(file_path) orelse unreachable;

    var file_directory = try std.fs.cwd().openDir(file_directory_name, .{});
    defer file_directory.close();

    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const file_data = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(file_data);

    var gltf = zgltf.init(allocator);
    defer gltf.deinit();

    try gltf.parse(file_data);

    const buffer_file_datas = try allocator.alloc([]const u8, gltf.data.buffers.items.len);
    defer allocator.free(buffer_file_datas);

    for (gltf.data.buffers.items, 0..) |buffer, i| {
        if (buffer.uri == null) continue;

        std.log.info("{s}", .{buffer.uri.?});

        const bin_file = try file_directory.openFile(buffer.uri.?, .{});
        defer bin_file.close();

        const bin_file_data = try bin_file.readToEndAlloc(allocator, std.math.maxInt(usize));

        buffer_file_datas[i] = bin_file_data;
    }

    defer for (buffer_file_datas) |buffer_file_data| {
        allocator.free(buffer_file_data);
    };

    const texture_count = gltf.data.textures.items.len;

    var textures = try std.ArrayList(Image).initCapacity(allocator, texture_count);
    errdefer textures.deinit();

    for (gltf.data.images.items) |image| {
        if (image.uri == null) continue;

        const path = image.uri.?;

        std.log.info("file path: {s}", .{path});

        const image_file = try file_directory.openFile(path, .{});
        defer image_file.close();

        const raw_data = try image_file.readToEndAlloc(allocator, std.math.maxInt(u64));
        defer allocator.free(raw_data);

        var texture_image_loaded = try zigimg.Image.fromMemory(allocator, raw_data);
        defer texture_image_loaded.deinit();

        var linear_data: []Image.Color = undefined;

        switch (texture_image_loaded.pixels) {
            .rgba32 => |data| {
                linear_data = @as([*]Image.Color, @alignCast(@ptrCast(data)))[0..@as(usize, @intCast(texture_image_loaded.width * texture_image_loaded.height))];
            },
            .rgb24 => |data| {
                linear_data = try allocator.alloc(Image.Color, texture_image_loaded.width * texture_image_loaded.height);

                for (data, 0..) |pixel, i| {
                    linear_data[i] = Image.Color{ .r = pixel.r, .g = pixel.g, .b = pixel.b, .a = 255 };
                }
            },
            .grayscale1 => |data| {
                linear_data = try allocator.alloc(Image.Color, texture_image_loaded.width * texture_image_loaded.height);

                for (data, 0..) |pixel, i| {
                    linear_data[i] = Image.Color{ .r = pixel.value, .g = pixel.value, .b = pixel.value, .a = 255 };
                }
            },
            .grayscale8 => |data| {
                linear_data = try allocator.alloc(Image.Color, texture_image_loaded.width * texture_image_loaded.height);

                for (data, 0..) |pixel, i| {
                    linear_data[i] = Image.Color{ .r = pixel.value, .g = pixel.value, .b = pixel.value, .a = 255 };
                }
            },
            else => {
                std.log.info("format: {s}", .{@tagName(texture_image_loaded.pixels)});

                @panic("Format?");
            },
        }

        var texture_image = try Image.initFromLinear(
            allocator,
            linear_data,
            texture_image_loaded.width,
            texture_image_loaded.height,
        );
        errdefer texture_image.deinit(allocator);

        try textures.append(texture_image);
    }

    var model_vertices = std.ArrayList(TestVertex).init(allocator);
    defer model_vertices.deinit();

    var model_vertex_positions = std.ArrayList(@Vector(3, f32)).init(allocator);
    defer model_vertex_positions.deinit();

    var model_indices = std.ArrayList(u32).init(allocator);
    defer model_indices.deinit();

    var sub_meshes = std.ArrayList(Mesh.SubMesh).init(allocator);
    errdefer sub_meshes.deinit();

    for (gltf.data.nodes.items) |node| {
        if (node.mesh == null) continue;

        const transform_matrix = zgltf.getGlobalTransform(&gltf.data, node);

        const mesh: *zgltf.Mesh = &gltf.data.meshes.items[node.mesh.?];

        std.log.info("mesh.primitive_count = {}", .{mesh.primitives.items.len});

        for (mesh.primitives.items) |primitive| {
            const vertex_start = model_vertices.items.len;
            const index_start = model_indices.items.len;

            var bounding_min: @Vector(3, f32) = .{ std.math.floatMax(f32), std.math.floatMax(f32), std.math.floatMax(f32) };
            var bounding_max: @Vector(3, f32) = .{ std.math.floatMin(f32), std.math.floatMin(f32), std.math.floatMin(f32) };

            var vertex_count: usize = 0;

            var positions = std.ArrayList(f32).init(allocator);
            defer positions.deinit();

            var normals = std.ArrayList(f32).init(allocator);
            defer normals.deinit();

            var texture_coordinates = std.ArrayList(f32).init(allocator);
            defer texture_coordinates.deinit();

            var colors = std.ArrayList(f32).init(allocator);
            defer colors.deinit();

            for (primitive.attributes.items) |attribute| {
                switch (attribute) {
                    .position => |accessor_index| {
                        const accessor = gltf.data.accessors.items[accessor_index];
                        const buffer_view = gltf.data.buffer_views.items[accessor.buffer_view.?];
                        const buffer_data = buffer_file_datas[buffer_view.buffer];

                        vertex_count += @as(usize, @intCast(accessor.count));

                        try positions.ensureTotalCapacity(@as(usize, @intCast(accessor.count)));

                        gltf.getDataFromBufferView(f32, &positions, accessor, buffer_data);
                    },
                    .normal => |accessor_index| {
                        const accessor = gltf.data.accessors.items[accessor_index];
                        const buffer_view = gltf.data.buffer_views.items[accessor.buffer_view.?];
                        const buffer_data = buffer_file_datas[buffer_view.buffer];

                        try normals.ensureTotalCapacity(@as(usize, @intCast(accessor.count)));

                        gltf.getDataFromBufferView(f32, &normals, accessor, buffer_data);
                    },
                    .tangent => {},
                    .texcoord => |accessor_index| {
                        const accessor = gltf.data.accessors.items[accessor_index];
                        const buffer_view = gltf.data.buffer_views.items[accessor.buffer_view.?];
                        const buffer_data = buffer_file_datas[buffer_view.buffer];

                        try texture_coordinates.ensureTotalCapacity(@as(usize, @intCast(accessor.count)));

                        gltf.getDataFromBufferView(f32, &texture_coordinates, accessor, buffer_data);
                    },
                    .color => |accessor_index| {
                        const accessor = gltf.data.accessors.items[accessor_index];
                        const buffer_view = gltf.data.buffer_views.items[accessor.buffer_view.?];
                        const buffer_data = buffer_file_datas[buffer_view.buffer];

                        try colors.ensureTotalCapacity(@as(usize, @intCast(accessor.count)));

                        std.debug.assert(accessor.component_type == .float);

                        gltf.getDataFromBufferView(f32, &colors, accessor, buffer_data);
                    },
                    .joints => {},
                    .weights => {},
                }
            }

            std.debug.assert(vertex_count != 0);

            try model_vertices.ensureTotalCapacity(model_vertices.items.len + vertex_count);

            //Vertices
            {
                var position_index: usize = 0;
                var normal_index: usize = 0;
                var uv_index: usize = 0;
                var color_index: usize = 0;

                std.log.info("vertex position accessor.count = {}", .{vertex_count});

                while (position_index < vertex_count * 3) : ({
                    position_index += 3;
                    normal_index += 3;
                    uv_index += 2;
                    color_index += 4;
                }) {
                    const position_source_x = positions.items[position_index];
                    const position_source_y = positions.items[position_index + 1];
                    const position_source_z = positions.items[position_index + 2];

                    const position_vector = @Vector(3, f32){ position_source_x, position_source_y, position_source_z };
                    const uv: @Vector(2, f32) = .{ texture_coordinates.items[uv_index], texture_coordinates.items[uv_index + 1] };
                    const color: @Vector(4, f32) = if (colors.items.len != 0) .{ colors.items[color_index], colors.items[color_index + 1], colors.items[color_index + 2], 1 } else @splat(1);
                    const normal: @Vector(3, f32) = .{ normals.items[normal_index], normals.items[normal_index + 1], normals.items[normal_index + 2] };
                    _ = color;

                    const position_transformed = (zalgebra.Mat4{ .data = transform_matrix }).mulByVec4(.{ .data = .{
                        position_vector[0],
                        position_vector[1],
                        position_vector[2],
                        1,
                    } });

                    const triangle_index_color = @as(f32, @floatFromInt(position_index / 6)) / @as(f32, @floatFromInt(vertex_count));
                    _ = triangle_index_color;

                    try model_vertex_positions.append(position_vector);
                    try model_vertices.append(.{
                        .position = .{ position_transformed.data[0], position_transformed.data[1], position_transformed.data[2] },
                        .uv = uv,
                        .color = .{ 1, 1, 1, 1 },
                        .normal = normal,
                    });

                    bounding_min = @min(bounding_min, position_vector);
                    bounding_max = @max(bounding_max, position_vector);
                }
            }

            //Indices
            {
                const index_accessor = gltf.data.accessors.items[primitive.indices.?];
                const buffer_view = gltf.data.buffer_views.items[index_accessor.buffer_view.?];
                const buffer_data = buffer_file_datas[buffer_view.buffer];

                switch (index_accessor.component_type) {
                    .byte => unreachable,
                    .unsigned_byte => unreachable,
                    .short => unreachable,
                    .unsigned_short => {
                        var indices_u16 = std.ArrayList(u16).init(allocator);
                        defer indices_u16.deinit();

                        gltf.getDataFromBufferView(u16, &indices_u16, index_accessor, buffer_data);

                        try model_indices.ensureTotalCapacity(indices_u16.items.len);

                        for (indices_u16.items) |index_u16| {
                            try model_indices.append(@as(u32, index_u16));
                        }
                    },
                    .unsigned_integer => {
                        gltf.getDataFromBufferView(u32, &model_indices, index_accessor, buffer_data);
                    },
                    .float => unreachable,
                }
            }

            const has_material = primitive.material != null;

            var material_index: u32 = 0;

            //material
            if (has_material) {
                const material = gltf.data.materials.items[primitive.material.?];

                const pbr = material.metallic_roughness;

                const has_albedo_texture = pbr.base_color_texture != null;

                const has_roughness_texture = pbr.metallic_roughness_texture != null;

                var albedo_index: ?u32 = null;

                if (has_albedo_texture) {
                    const albedo_texture = gltf.data.textures.items[pbr.base_color_texture.?.index];

                    albedo_index = @as(u32, @intCast(albedo_texture.source.?)) + 0;
                }

                var roughness_index: ?u32 = null;

                if (has_roughness_texture) {
                    const roughness_texture = gltf.data.textures.items[pbr.metallic_roughness_texture.?.index];

                    roughness_index = @as(u32, @intCast(roughness_texture.source.?)) + 1;
                }

                material_index = albedo_index orelse 0;

                std.log.info("Material {any}", .{pbr});
            }

            try sub_meshes.append(.{
                .vertex_offset = @as(u32, @intCast(vertex_start)),
                .vertex_count = @as(u32, @intCast(model_vertices.items.len - vertex_start)),
                .index_offset = @as(u32, @intCast(index_start)),
                .index_count = @as(u32, @intCast(model_indices.items.len - index_start)),
                .material_index = material_index,
                .transform = transform_matrix,
                .bounding_min = bounding_min,
                .bounding_max = bounding_max,
                .albedo_texture_index = material_index,
            });
        }
    }

    const mesh: Mesh = .{
        .vertices = try model_vertices.toOwnedSlice(),
        .indices = try model_indices.toOwnedSlice(),
        .sub_meshes = try sub_meshes.toOwnedSlice(),
        .textures = try textures.toOwnedSlice(),
    };

    return mesh;
}

fn freeMesh(mesh: *Mesh, allocator: std.mem.Allocator) void {
    allocator.free(mesh.vertices);
    allocator.free(mesh.indices);
    allocator.free(mesh.sub_meshes);
    mesh.* = undefined;
}

var previous_keyboard_state: ?[]const u8 = null;

fn getKeyDown(key: c.SDL_Scancode) bool {
    var key_count: c_int = 0;

    const keys = c.SDL_GetKeyboardState(&key_count)[0..@intCast(key_count)];

    return keys[key] == 1;
}

///Returns true the first time the key is detected as down
fn getKeyPressed(key: c.SDL_Scancode) bool {
    var key_count: c_int = 0;

    const keys = c.SDL_GetKeyboardState(&key_count)[0..@intCast(key_count)];

    if (previous_keyboard_state == null) {
        return keys[key] == 1;
    }

    return keys[key] == 1 and previous_keyboard_state.?[key] == 0;
}

const RasterUnit = @import("raster/RasterUnit.zig");
const geometry_processor = @import("raster/geometry_processor.zig");
const CommandBuffer = @import("CommandBuffer.zig");

pub fn main() !void {
    var general_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = general_allocator.deinit();

    const allocator = general_allocator.allocator();

    if (c.SDL_Init(c.SDL_INIT_VIDEO | c.SDL_INIT_AUDIO) != 0) {
        return error.FailedToCreateWindow;
    }

    defer if (previous_keyboard_state != null) {
        allocator.free(previous_keyboard_state.?);
    };

    const window_width = 1600;
    const window_height = 900;

    const surface_width = if (@import("builtin").mode == .ReleaseFast) 1080 / 1 else 1080 / 4;
    const surface_height = if (@import("builtin").mode == .ReleaseFast) 1920 / 1 else 1920 / 4;

    var renderer: Renderer = undefined;

    var raster_unit: RasterUnit = .{};

    try raster_unit.init(&renderer);
    defer raster_unit.deinit();

    try renderer.init(allocator, window_width, window_height, surface_width, surface_height, "CentralGfx");
    defer renderer.deinit(allocator);

    var render_target = try Image.init(allocator, surface_width, surface_height);
    defer render_target.deinit(allocator);

    // const depth_target = try allocator.alloc(f32, surface_width * surface_height);
    const depth_target = try allocator.alignedAlloc(f32, @alignOf(@Vector(8, f32)), surface_width * surface_height);
    defer allocator.free(depth_target);

    var cog_image_loaded = try zigimg.Image.fromMemory(allocator, cog_png);
    defer cog_image_loaded.deinit();

    const cog_image_data = cog_image_loaded.rawBytes();

    var cog_image = try Image.initFromLinear(
        allocator,
        @as([*]Image.Color, @constCast(@alignCast(@ptrCast(cog_image_data))))[0..@as(usize, @intCast(cog_image_loaded.width * cog_image_loaded.height))],
        cog_image_loaded.width,
        cog_image_loaded.height,
    );
    defer cog_image.deinit(allocator);

    var mesh = try loadMesh(allocator, "src/res/light_test.gltf");
    defer freeMesh(&mesh, allocator);

    var shambler_mesh = try loadMesh(allocator, "src/res/shambler/scene.gltf");
    defer freeMesh(&shambler_mesh, allocator);

    const enable_raster_pass = true;
    const enable_ray_pass = false;
    _ = enable_ray_pass;

    // var default_prng = std.rand.DefaultPrng.init(@intCast(u64, std.time.timestamp()));

    const core_count = try std.Thread.getCpuCount();

    const ray_threads = try allocator.alloc(std.Thread, core_count);
    defer allocator.free(ray_threads);

    // const thread_pixel_count = render_target.pixels.len / core_count;
    const thread_pixel_height = render_target.height / core_count;
    _ = thread_pixel_height;

    const thread_slice_index: usize = 0;
    _ = thread_slice_index;
    const pixel_offset: @Vector(2, usize) = .{ 0, 0 };
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

    var last_mouse_x: f32 = 0;
    var last_mouse_y: f32 = 0;

    var yaw: f32 = 420.7;
    var pitch: f32 = -15.5;
    const roll: f32 = 0;
    _ = roll;

    var camera_enable: bool = false;

    var camera_front: @Vector(3, f32) = .{ 0.46, -0.26, 0.84 };
    var camera_target: @Vector(3, f32) = .{ -3.5, 2.17, -5.13 };
    var camera_translation: @Vector(3, f32) = .{ -4, 2.4, -5.9 };

    var frame_time: f32 = 0.016;

    var command_buffer = CommandBuffer{
        .allocator = allocator,
    };
    defer command_buffer.deinit();

    var test_pipeline = TestPipeline.runtime;

    var cull_mode: CommandBuffer.FaceCullMode = .back;

    while (!renderer.shouldWindowClose()) {
        const frame_start_time = std.time.microTimestamp();
        const time_s: f32 = @as(f32, @floatFromInt(c.SDL_GetTicks())) / 1000;

        if (true) {
            @memset(render_target.texel_buffer, Image.Color.fromNormalized(.{ 0.25, 0.25, 0.25, 1 }));
            @memset(depth_target, 1);

            var render_pass = Renderer.Pass{
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
                    if (false) renderer.drawTriangle(
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
                    .indices = &.{ 0, 1, 2, 3 },
                    .transform = zalgebra.Mat4.identity(),
                    .transform_geo = .{},
                    .view_projection = zalgebra.Mat4.identity(),
                };
                _ = uniforms;

                // renderer.pipelineDrawLine(render_pass, uniforms, line_vertices.len, TestPipeline);

                var triangle = [3]@Vector(3, f32){
                    .{ -0.5, 0.5, 0 },
                    .{ 0.5, 0.5, 0 },
                    .{ 0, -0.5, 0 },
                };

                const triangle_vertices = [_]TestVertex{
                    .{
                        .position = .{ -0.5, 0.5, 0 },
                        .color = .{ 1, 0, 0, 1 },
                        .uv = .{ 0, 0 },
                    },
                    .{
                        .position = .{ 0.5, 0.5, 0 },
                        .color = .{ 0, 1, 0, 1 },
                        .uv = .{ 1, 0 },
                    },
                    .{
                        .position = .{ 0, -0.5, 0 },
                        .color = .{ 0, 0, 1, 1 },
                        .uv = .{ 0.5, 1.0 },
                    },
                };

                {
                    var cursor_pos_x: c_int = 0;
                    var cursor_pos_y: c_int = 0;

                    _ = c.SDL_GetMouseState(&cursor_pos_x, &cursor_pos_y);

                    const x_position = @as(f32, @floatFromInt(cursor_pos_x));
                    const y_position = @as(f32, @floatFromInt(cursor_pos_y));

                    const x_offset = x_position - last_mouse_x;
                    const y_offset = last_mouse_y - y_position;

                    last_mouse_x = x_position;
                    last_mouse_y = y_position;

                    if (camera_enable) {
                        const sensitivity = 0.1;
                        var camera_speed: @Vector(3, f32) = @splat(10 * frame_time);

                        if (getKeyDown(c.SDL_SCANCODE_LCTRL)) {
                            camera_speed *= @splat(2);
                        }

                        yaw += x_offset * sensitivity;
                        pitch += y_offset * sensitivity;

                        pitch = std.math.clamp(pitch, -89, 89);

                        camera_front = zalgebra.Vec3.norm(.{ .data = .{
                            @cos(zalgebra.toRadians(yaw)) * @cos(zalgebra.toRadians(-pitch)),
                            @sin(zalgebra.toRadians(pitch)),
                            @sin(zalgebra.toRadians(yaw)) * @cos(zalgebra.toRadians(-pitch)),
                        } }).data;

                        if (getKeyDown(c.SDL_SCANCODE_W)) {
                            camera_translation += camera_speed * camera_front;
                        } else if (getKeyDown(c.SDL_SCANCODE_S)) {
                            camera_translation -= camera_speed * camera_front;
                        }

                        if (getKeyDown(c.SDL_SCANCODE_A)) {
                            camera_translation -= zalgebra.Vec3.norm(zalgebra.Vec3.cross(.{ .data = camera_front }, .{ .data = .{ 0, 1, 0 } })).mul(.{ .data = camera_speed }).data;
                        } else if (getKeyDown(c.SDL_SCANCODE_D)) {
                            camera_translation += zalgebra.Vec3.norm(zalgebra.Vec3.cross(.{ .data = camera_front }, .{ .data = .{ 0, 1, 0 } })).mul(.{ .data = camera_speed }).data;
                        }

                        if (getKeyDown(c.SDL_SCANCODE_SPACE)) {
                            camera_translation[1] += camera_speed[1];
                        } else if (getKeyDown(c.SDL_SCANCODE_LSHIFT)) {
                            camera_translation[1] -= camera_speed[1];
                        }
                    }

                    camera_target = camera_translation + camera_front;
                }

                if (getKeyPressed(c.SDL_SCANCODE_TAB)) {
                    camera_enable = !camera_enable;

                    c.SDL_SetWindowGrab(renderer.window, @intFromBool(camera_enable));
                    _ = c.SDL_CaptureMouse(@intFromBool(camera_enable));
                    _ = c.SDL_SetRelativeMouseMode(@intFromBool(camera_enable));
                }

                if (getKeyPressed(c.SDL_SCANCODE_L)) {
                    test_pipeline.polygon_fill_mode = switch (test_pipeline.polygon_fill_mode) {
                        .line => .fill,
                        .fill => .line,
                    };
                }

                if (getKeyPressed(c.SDL_SCANCODE_F1)) {
                    raster_unit.sort_triangles = !raster_unit.sort_triangles;
                }

                if (getKeyPressed(c.SDL_SCANCODE_F2)) {
                    cull_mode = switch (cull_mode) {
                        .none => .front,
                        .front => .back,
                        .back => .front_and_back,
                        .front_and_back => .none,
                    };
                }

                var triangle_matrix: zalgebra.Mat4 = zalgebra.Mat4.fromTranslate(.{ .data = .{ 0, 0, 1 + @sin(time_s) * 0 * 0 } });

                triangle_matrix = triangle_matrix.mul(zalgebra.Mat4.fromEulerAngles(.{ .data = .{ 0, time_s * 0, 0 } }));

                const view_matrix = zalgebra.Mat4.lookAt(
                    .{ .data = camera_translation },
                    .{ .data = camera_target },
                    .{ .data = .{ 0, 1, 0 } },
                );

                var window_size_x: c_int = 0;
                var window_size_y: c_int = 0;

                c.SDL_GetWindowSize(renderer.window, &window_size_x, &window_size_y);

                const projection = zalgebra.Mat4.perspective(
                    45,
                    @as(f32, @floatFromInt(window_size_x)) / @as(f32, @floatFromInt(window_size_y)),
                    0.1,
                    1000,
                );

                const view_projection = projection.mul(view_matrix);

                for (&triangle) |*vertex| {
                    vertex.* = (vertex.* + @as(@Vector(3, f32), @splat(@as(f32, 1)))) / @as(@Vector(3, f32), @splat(@as(f32, 2)));
                }

                if (false) renderer.pipelineDrawTriangles(
                    render_pass,
                    .{
                        .texture = cog_image,
                        .vertices = &triangle_vertices,
                        .indices = &.{ 0, 1, 2 },
                        .transform = zalgebra.Mat4.fromTranslate(.{ .data = .{ 4, 1, 4 } }),
                        .view_projection = view_projection,
                    },
                    1,
                    TestPipeline,
                );

                if (false) renderer.pipelineDrawTriangles(
                    render_pass,
                    .{
                        .texture = cog_image,
                        .vertices = mesh.vertices,
                        .indices = mesh.indices,
                        .transform = triangle_matrix,
                        .view_projection = view_projection,
                    },
                    mesh.indices.len / 3,
                    TestPipeline,
                );

                command_buffer.begin();

                command_buffer.beginRasterPass(&render_pass, .{
                    .offset = .{ 0, 0 },
                    .extent = .{ surface_width, surface_height },
                }, .{
                    .numerator = 1,
                    .denominator = 1,
                });

                command_buffer.setPipeline(&test_pipeline);
                command_buffer.setScissor(.{
                    .offset = .{ 0, 0 },
                    .extent = .{ surface_width, surface_height },
                });
                command_buffer.setViewport(.{
                    .x = 0,
                    .y = @floatFromInt(surface_height),
                    .width = @floatFromInt(surface_width),
                    .height = -@as(f32, @floatFromInt(surface_height)),
                    .depth_min = 0,
                    .depth_max = 1,
                });
                command_buffer.setFaceCullMode(cull_mode);

                var point_lights: [4]PointLight = undefined;

                point_lights[0] = .{
                    .color = .{ 1, 1, 1 },
                    .position = .{ -3, @cos(time_s), 5 },
                    .intensity = 30 + @abs(@sin(time_s) * 20),
                };

                point_lights[1] = .{
                    .color = .{ 1, 0.4, 0.1 },
                    .position = .{ 3, @sin(time_s), -4 },
                    .intensity = 20 + @abs(@cos(time_s) * 30),
                };

                point_lights[2] = .{
                    .color = .{ 0, 0.3, 0.9 },
                    .position = .{ 3, 16, -4 },
                    .intensity = 300 + @abs(@cos(time_s) * 30),
                };

                point_lights[3] = .{
                    .color = .{ 0.9, 0.9, 0.9 },
                    .position = .{ 3 + @abs(@sin(time_s)) * 100, 35, -4 },
                    .intensity = 4000 + @abs(@cos(time_s) * 60),
                };

                var triangle_uniform: TestPipelineUniformInput = .{
                    .texture = cog_image,
                    .vertices = &triangle_vertices,
                    .indices = &.{ 0, 1, 2 },
                    .transform = zalgebra.Mat4.fromTranslate(.{ .data = .{ 4, 1, 4 } }),
                    .transform_geo = .{
                        .position = .{ 4, 1, 4 },
                    },
                    .view_projection = view_projection,
                };

                command_buffer.draw(&triangle_uniform, 0, 3);

                var mesh_uniforms: [256]TestPipelineUniformInput = undefined;
                var mesh_base: usize = 0;

                for (mesh.sub_meshes[0..], 0..) |sub_mesh, i| {
                    mesh_uniforms[i] = .{
                        .texture = if (mesh.textures.len != 0) mesh.textures[sub_mesh.albedo_texture_index] else null,
                        .vertices = mesh.vertices,
                        .indices = mesh.indices,
                        .transform = zalgebra.Mat4.identity(),
                        .transform_geo = .{},
                        .view_projection = zalgebra.Mat4.identity(),
                    };

                    mesh_uniforms[i].transform_geo.scale = .{ 0.1, 0.1, 0.1 };

                    mesh_uniforms[i].transform = zalgebra.Mat4.identity().rotate(
                        time_s * 10,
                        .{ .data = .{ 0, 1, 0 } },
                    ).scale(
                        .{ .data = .{ 0.1, 0.1, 0.1 } },
                    );
                    mesh_uniforms[i].view_projection = view_projection;
                    mesh_uniforms[i].base_vertex = sub_mesh.vertex_offset;
                    // mesh_uniforms[i].texture = mesh.textures[sub_mesh.albedo_texture_index];
                    mesh_uniforms[i].lights = &point_lights;

                    command_buffer.draw(
                        &mesh_uniforms[i],
                        sub_mesh.index_offset,
                        sub_mesh.index_count,
                    );
                }

                mesh_base += mesh.sub_meshes.len;

                for (0..50) |shambler_index| {
                    for (shambler_mesh.sub_meshes[0..], mesh.sub_meshes.len..) |sub_mesh, i| {
                        mesh_uniforms[mesh_base + shambler_index + i] = .{
                            .texture = shambler_mesh.textures[sub_mesh.albedo_texture_index],
                            .vertices = shambler_mesh.vertices,
                            .indices = shambler_mesh.indices,
                            .transform = zalgebra.Mat4.identity(),
                            .transform_geo = .{},
                            .view_projection = zalgebra.Mat4.identity(),
                        };

                        mesh_uniforms[mesh_base + shambler_index + i].transform_geo.position = .{ 30, 0, 50 };
                        mesh_uniforms[mesh_base + shambler_index + i].transform_geo.scale = .{ 0.1, 0.1, 0.1 };

                        mesh_uniforms[mesh_base + shambler_index + i].transform = zalgebra.Mat4.fromTranslate(.{ .data = .{ 30 + @as(f32, @floatFromInt(shambler_index * 70)), 0, 50 } }).rotate(
                            time_s * 10,
                            .{ .data = .{ 0, 1, 0 } },
                        ).scale(
                            .{ .data = .{ 0.1, 0.1, 0.1 } },
                        );
                        mesh_uniforms[mesh_base + shambler_index + i].view_projection = view_projection;
                        mesh_uniforms[mesh_base + shambler_index + i].base_vertex = sub_mesh.vertex_offset;
                        mesh_uniforms[mesh_base + shambler_index + i].texture = shambler_mesh.textures[sub_mesh.albedo_texture_index];
                        mesh_uniforms[mesh_base + shambler_index + i].lights = &point_lights;

                        command_buffer.draw(
                            &mesh_uniforms[mesh_base + shambler_index + i],
                            sub_mesh.index_offset,
                            sub_mesh.index_count,
                        );
                    }
                }

                command_buffer.endRasterPass();

                command_buffer.end();

                var semaphore: std.Thread.Semaphore = .{ .permits = 1 };

                @import("raster/command_processor.zig").submit(
                    &raster_unit,
                    &command_buffer,
                    &semaphore,
                );

                std.log.info("Presenting", .{});

                // semaphore.wait();
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
        const current_frame_time = frame_end_time - frame_start_time;

        if ((@as(f32, @floatFromInt(current_frame_time)) / 1000) > 1000) {
            // @panic("Render took too long!");
        }

        frame_time = (@as(f32, @floatFromInt(current_frame_time)) / (1000 * 1000));

        std.log.err("frame_time: {d:.2}ms", .{@as(f32, @floatFromInt(current_frame_time)) / 1000});

        renderer.presentImage(render_target);

        {
            var count: c_int = 0;
            if (previous_keyboard_state != null) allocator.free(previous_keyboard_state.?);

            previous_keyboard_state = c.SDL_GetKeyboardState(&count)[0..@intCast(count)];

            previous_keyboard_state = try allocator.dupe(u8, previous_keyboard_state.?);
        }
    }
}
