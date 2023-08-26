const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "centralgfx",
        .target = target,
        .optimize = mode,
        .root_source_file = std.build.LazyPath.relative("src/main.zig"),
    });

    exe.addSystemIncludePath(std.build.LazyPath.relative("SDL2"));
    exe.linkSystemLibrary("SDL2");
    exe.linkLibC();
    exe.addAnonymousModule("zigimg", .{
        .source_file = std.build.LazyPath.relative("lib/zigimg/zigimg.zig"),
    });
    exe.addAnonymousModule("zalgebra", .{
        .source_file = std.build.LazyPath.relative("lib/zalgebra/src/main.zig"),
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_tests = b.addTest(.{
        .root_source_file = std.build.FileSource.relative("lib/stb_image.zig"),
        .target = target,
        .optimize = mode,
    });

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&exe_tests.step);
}
