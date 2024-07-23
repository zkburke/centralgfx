const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardOptimizeOption(.{});

    const single_threaded = b.option(bool, "single-threaded", "Make the executable single threaded");
    _ = single_threaded;

    const exe = b.addExecutable(.{
        .name = "centralgfx",
        .target = target,
        .optimize = mode,
        .root_source_file = .{ .cwd_relative = "src/main.zig" },
    });

    exe.addSystemIncludePath(.{ .cwd_relative = "SDL2" });
    exe.linkSystemLibrary("SDL2");
    exe.linkLibC();
    exe.root_module.addAnonymousImport("zigimg", .{
        .root_source_file = .{ .cwd_relative = "lib/zigimg/zigimg.zig" },
    });
    exe.root_module.addAnonymousImport("zalgebra", .{
        .root_source_file = .{ .cwd_relative = "lib/zalgebra/src/main.zig" },
    });
    exe.root_module.addAnonymousImport("zgltf", .{
        .root_source_file = .{ .cwd_relative = "lib/zgltf/src/main.zig" },
    });

    // exe.single_threaded = true;

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_tests = b.addTest(.{
        .root_source_file = .{ .cwd_relative = "src/main.zig" },
        .target = target,
        .optimize = mode,
    });

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&exe_tests.step);
}
