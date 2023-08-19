const AudioMixer = @This();
const std = @import("std");
const c = @import("c_bindings.zig");

device: c.SDL_AudioDeviceID = 0,
audio_spec: c.SDL_AudioSpec = undefined,

fn audioCallback(user_data: ?*anyopaque, stream: [*c]u8, length: c_int) callconv(.C) void {
    _ = user_data;

    const float_buffer: []f32 = @as([*]f32, @ptrCast(@alignCast(stream)))[0..@as(usize, @intCast(@divTrunc(length, @sizeOf(f32))))];

    for (float_buffer) |*sample| {
        // sample.* = std.math.fabs(@sin(@intToFloat(f32, i) / (48000 * 10)));
        sample.* = 0;
    }
}

pub fn init(self: *AudioMixer) !void {
    const wanted_spec = c.SDL_AudioSpec{
        .freq = 48000,
        .format = c.AUDIO_F32,
        .channels = 2,
        .samples = 4096,
        .silence = 1,
        .padding = 0,
        .size = 0,
        .userdata = self,
        .callback = audioCallback,
    };

    self.device = c.SDL_OpenAudioDevice(null, 0, &wanted_spec, &self.audio_spec, c.SDL_AUDIO_ALLOW_FORMAT_CHANGE);

    if (self.device == 0) {
        std.log.info("SDL_Error: {s}", .{c.SDL_GetError()});

        return error.FailedToCreateDevice;
    }

    c.SDL_PauseAudioDevice(self.device, 0);
    // c.SDL_Delay(5000);
}

pub fn deinit(self: *AudioMixer) void {
    c.SDL_CloseAudioDevice(self.device);
}
