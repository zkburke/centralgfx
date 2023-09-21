polygon_fill_mode: PolygonFillMode,
vertexShader: VertexShaderFn,
fragmentShader: FragmentShaderFn,
triangle_layout: struct {
    interpolator_count: u8,
    flat_count: u8,
} = undefined,

pub const VertexShaderFn = *const fn (
    uniform: *const anyopaque,
    vertex_index: usize,
    fragment_input: *anyopaque,
    // interpolators: [*]@Vector(3, f32),
    // flats: [*]u32,
) @Vector(4, f32);

pub const FragmentShaderFn = *const fn (
    uniform: *const anyopaque,
    fragment_input: *const anyopaque,
) @Vector(4, f32);

pub const PrimitiveType = enum {
    triangle,
    line,
};

pub const PolygonFillMode = enum {
    fill,
    line,
};
