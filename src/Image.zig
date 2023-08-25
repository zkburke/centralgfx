const std = @import("std");
const Image = @This();

///Size of a page in bytes
const page_size: usize = std.mem.page_size;
///Size of a cache line in bytes
const cache_line_size: usize = 64;

pub const TileSize = enum {
    @"4x4",
    @"8x8",
    @"16x16",
    @"32x32",
    @"64x64",
};

pub const PixelFormat = enum {
    r8_uint,
    rg16_uint,
    rgba32_uint,
    rgba64_uint,
    r32_uint,
    r32_sfloat,
};

pub const ImageLayout = enum {
    ///Image is stored as a sequence of rows
    linear,
    ///Image is stored as a sequence of fixed sized, square tiles
    tiled,
};

pub const ImageFormat = struct {
    pixel_format: PixelFormat,
    layout: ImageLayout,
    tile_size: TileSize,
};

pub const Color = struct {
    r: u8 = 0,
    g: u8 = 0,
    b: u8 = 0,
    a: u8 = 0,

    pub fn fromNormalized(rgba: @Vector(4, f32)) Color {
        const scaled: @Vector(4, f32) = rgba * @as(@Vector(4, f32), @splat(@as(f32, std.math.maxInt(u8))));

        @setRuntimeSafety(false);

        return .{
            .r = @as(u8, @intFromFloat(scaled[0])),
            .g = @as(u8, @intFromFloat(scaled[1])),
            .b = @as(u8, @intFromFloat(scaled[2])),
            .a = @as(u8, @intFromFloat(scaled[3])),
        };
    }

    pub fn toNormalized(color: Color) @Vector(4, f32) {
        return @Vector(4, f32){
            @as(f32, @floatFromInt(color.r)),
            @as(f32, @floatFromInt(color.g)),
            @as(f32, @floatFromInt(color.b)),
            @as(f32, @floatFromInt(color.a)),
        } / @as(@Vector(4, f32), @splat(@as(f32, 255)));
    }

    pub fn blend(self: Color, other: Color) Color {
        const blend_ratio = 1 - (@as(f32, @floatFromInt(self.a)) / 255);
        // const blend_ratio = 1;

        const normalized0 = [3]f32{
            @as(f32, @floatFromInt(self.r)) / 255,
            @as(f32, @floatFromInt(self.g)) / 255,
            @as(f32, @floatFromInt(self.b)) / 255,
        };

        const normalized1 = [3]f32{
            @as(f32, @floatFromInt(other.r)) / 255,
            @as(f32, @floatFromInt(other.g)) / 255,
            @as(f32, @floatFromInt(other.b)) / 255,
        };

        return .{
            .r = @as(u8, @intFromFloat(((normalized0[0] * blend_ratio + normalized1[0])) * 255)),
            .g = @as(u8, @intFromFloat(((normalized0[1] * blend_ratio + normalized1[1])) * 255)),
            .b = @as(u8, @intFromFloat(((normalized0[2] * blend_ratio + normalized1[2])) * 255)),
            .a = self.a,
        };
    }
};

///tile width and height in pixels
const tile_width = 4;
const tile_height = tile_width;

texel_buffer: []Color align(page_size),
width: usize,
height: usize,

pub fn init(allocator: std.mem.Allocator, width: usize, height: usize) !Image {
    const tiles_x = std.math.divCeil(usize, width, tile_width) catch unreachable;
    const tiles_y = std.math.divCeil(usize, height, tile_height) catch unreachable;

    const texel_buffer_size = tiles_x * tile_width * tiles_y * tile_height;

    const texel_buffer = try allocator.alignedAlloc(Color, page_size, texel_buffer_size);
    errdefer allocator.free(texel_buffer);

    return .{
        .texel_buffer = texel_buffer,
        .width = width,
        .height = height,
    };
}

pub fn deinit(self: *Image, allocator: std.mem.Allocator) void {
    defer self.* = undefined;
    defer allocator.free(self.texel_buffer);
}

///Create and allocate a tiled image from a linear image
pub fn initFromLinear(
    allocator: std.mem.Allocator,
    image_pixels: []const Color,
    width: usize,
    height: usize,
) !Image {
    const image = try init(allocator, width, height);

    for (0..height) |y| {
        for (0..width) |x| {
            const pixel = image.texelFetch(.{ x, y });

            pixel.* = image_pixels[x + y * width];
        }
    }

    return image;
}

pub fn clear(self: Image, color: Color) void {
    @memset(self.texel_buffer, color);
}

pub fn copy(destination: Image, source: Image) void {
    @memcpy(destination.texel_buffer, source.texel_buffer);
}

pub fn texelFetch(self: Image, position: struct { x: usize, y: usize }) *Color {
    const tile_count_x = std.math.divCeil(usize, self.width, tile_width) catch unreachable;

    const tile_x = @divFloor(position.x, tile_width);
    const tile_y = @divFloor(position.y, tile_height);

    const tile_begin_x = tile_x * tile_width;
    const tile_begin_y = tile_y * tile_height;

    const tile_pointer: [*]Color = @ptrCast(&self.texel_buffer[(tile_x + tile_y * tile_count_x) * (tile_width * tile_height)]);

    //x, y relative to tile
    const x = position.x - tile_begin_x;
    const y = position.y - tile_begin_y;

    return &tile_pointer[x + y * tile_width];
}

pub fn setPixel(self: Image, position: struct { x: usize, y: usize }, color: Color) void {
    self.texelFetch(.{ position.x, position.y }).* = color;
}

pub fn affineSample(self: Image, uv: @Vector(2, f32)) Color {
    @setRuntimeSafety(false);

    const scaled_uv = uv * @Vector(2, f32){ @as(f32, @floatFromInt(self.width)), @as(f32, @floatFromInt(self.height)) };
    const x = @as(usize, @intFromFloat(scaled_uv[0]));
    const y = @as(usize, @intFromFloat(scaled_uv[1]));

    if (x >= self.width or y >= self.height) {
        return Color.fromNormalized(.{ 0, 0, 0, 0 });
    }

    return self.texelFetch(.{ x, y }).*;
}
