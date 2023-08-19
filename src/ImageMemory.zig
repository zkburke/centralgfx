const std = @import("std");
const Image = @This();

pixels: []u8,
texel_width: usize,
texel_height: usize,
texel_depth: usize,
tile_width: usize,
tile_height: usize,
tile_depth: usize,
samples: usize,

pub fn init(allocator: std.mem.Allocator, width: usize, height: usize, depth: usize, samples: usize) !Image
{
    var self: Image = undefined;

    self.pixels = allocator.alloc(u8, width * height * depth * samples);
    self.width = width;
    self.height = height;
    self.depth = depth;
    self.samples = samples;

    return self;
}