const std = @import("std");
const Image = @This();

pub const Color = struct {
    r: u8 = 0,
    g: u8 = 0,
    b: u8 = 0,
    a: u8 = 0,

    pub fn fromNormalized(rgba: @Vector(4, f32)) Color {
        const scaled: @Vector(4, f32) = rgba * @as(@Vector(4, f32), @splat(@as(f32, std.math.maxInt(u8))));

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

pixels: []Color,
width: usize,
height: usize,

pub fn setPixel(self: Image, position: struct { x: usize, y: usize }, color: Color) void {
    self.pixels[position.x + self.width * position.y] = color;
}

pub fn blit(position: struct { x: usize, y: usize }, source: Image, destination: Image) void {
    for (0..source.height) |y| {
        const destination_index = position.x + destination.width * (position.y + y);

        @memcpy(destination.pixels[destination_index..], source.pixels);
    }
}

pub fn blendedBlit(position: struct { x: usize, y: usize }, source: Image, destination: Image) void {
    var y: usize = 0;

    while (y < source.height and position.y + y < destination.height) : (y += 1) {
        var x: usize = 0;

        while (x < source.width and position.x + x < destination.width) : (x += 1) {
            const source_index = x + source.width * y;
            const destination_index = position.x + x + destination.width * (position.y + y);

            destination.pixels[destination_index] = destination.pixels[destination_index].blend(source.pixels[source_index]);
        }
    }
}

pub fn mappedBlit(position: struct { x: usize, y: usize }, scale: struct { x: usize, y: usize }, source: Image, destination: Image) void {
    var y: usize = 0;

    const true_width = source.width * scale.x;
    const true_height = source.height * scale.y;

    while (y < true_height and position.y + y < destination.height) : (y += 1) {
        var x: usize = 0;

        while (x < true_width and position.x + x < destination.width) : (x += 1) {
            const source_index = ((x % source.width) + source.width * (y % source.height));
            const destination_index = position.x + x + destination.width * (position.y + y);

            destination.pixels[destination_index] = destination.pixels[destination_index].blend(source.pixels[source_index]);
        }
    }
}

pub fn affineSample(self: Image, uv: @Vector(2, f32)) Color {
    const scaled_uv = uv * @Vector(2, f32){ @as(f32, @floatFromInt(self.width)), @as(f32, @floatFromInt(self.height)) };
    const x = @as(usize, @intFromFloat(scaled_uv[0]));
    const y = @as(usize, @intFromFloat(scaled_uv[1]));

    return self.pixels[x + self.width * y];
}
