const std = @import("std");
const Image = @import("Image.zig");

///Optimized 128 bit random
///We can't afford to be calling the std.rand.Random interface
const Random128 = struct {
    state: [2]u64,

    inline fn random128U64(self: *Random128) u64 {
        const s0 = self.state[0];
        var s1 = self.state[1];
        const r = s0 +% s1;

        s1 ^= s0;
        self.state[0] = std.math.rotl(u64, s0, @as(u8, 55)) ^ s1 ^ (s1 << 14);
        self.state[1] = std.math.rotl(u64, s1, @as(u8, 36));

        return r;
    }

    inline fn random128U32(self: *Random128) u32 {
        return @as(u32, @truncate(self.random128U64()));
    }

    inline fn random128F32(self: *Random128) f32 {
        const s = self.random128U32();
        const repr = 0x7f << 23 | (s >> 9);

        return @as(f32, @bitCast(repr)) - 1;
    }

    inline fn random128Vec3Int(self: *Random128) @Vector(3, u32) {
        return .{ self.random128U32(), self.random128U32(), self.random128U32() };
    }

    inline fn random128Vec3F32(self: *Random128) @Vector(3, f32) {
        const s = self.random128Vec3Int();
        const repr: @Vector(3, u32) = @as(@Vector(3, u32), @splat(0x7f << 23)) | (s >> @splat(@as(u5, 9)));

        return @as(@Vector(3, f32), @bitCast(repr)) - @as(@Vector(3, f32), @splat(@as(f32, 1)));
    }
};

inline fn dot(self: @Vector(3, f32), other: @Vector(3, f32)) f32 {
    return @reduce(.Add, self * other);
}

inline fn length(self: @Vector(3, f32)) f32 {
    return @sqrt(dot(self, self));
}

inline fn normalize(self: @Vector(3, f32)) @Vector(3, f32) {
    return self / @as(@Vector(3, f32), @splat(length(self)));
}

inline fn reflect(v: @Vector(3, f32), n: @Vector(3, f32)) @Vector(3, f32) {
    return v - @as(@Vector(3, f32), @splat(2)) * @as(@Vector(3, f32), @splat(dot(v, n))) * n;
}

inline fn refract(uvw: @Vector(3, f32), n: @Vector(3, f32), etai_over_etat: f32) @Vector(3, f32) {
    const cos_theta = @min(dot(-uvw, n), 1);
    const r_out_perp = @as(@Vector(3, f32), @splat(etai_over_etat)) * (uvw + @as(@Vector(3, f32), @splat(cos_theta)) * n);
    const r_out_parallel = -@as(@Vector(3, f32), @splat(@sqrt(@fabs(1 - dot(r_out_perp, r_out_perp))))) * n;

    return r_out_perp + r_out_parallel;
}

inline fn sphereSample(random: *Random128) @Vector(3, f32) {
    var point: @Vector(3, f32) = undefined;

    while (true) {
        point = @as(@Vector(3, f32), @splat(2)) * random.random128Vec3F32() - @Vector(3, f32){ 1, 1, 1 };

        if (dot(point, point) >= 1) break;
    }

    return point;
}

pub const Plane = struct {
    position: @Vector(3, f32),
    normal: @Vector(3, f32),
    material_index: u32,

    pub inline fn intersect(plane: Plane, ray_begin: @Vector(3, f32), ray_end: @Vector(3, f32)) f32 {
        const denom = dot(plane.normal, ray_end);

        if (@fabs(denom) <= 1e-4) //prevent div by 0
        {
            return -1;
        }

        // d = 10, whatever d is...
        const t = -(dot(plane.normal, plane.position - ray_begin) + 1) / dot(plane.normal, ray_end);

        if (t < 0) {
            return -1;
        }

        return t;
    }
};

pub const Sphere = struct {
    position: @Vector(3, f32),
    radius: f32,
    material_index: u32,

    pub inline fn intersect(sphere: Sphere, ray_begin: @Vector(3, f32), ray_end: @Vector(3, f32)) f32 {
        const oc = ray_begin - sphere.position; //center

        const a = dot(ray_end, ray_end);
        const half_b = dot(oc, ray_end);
        const c0 = dot(oc, oc) - sphere.radius * sphere.radius;
        const discriminant = half_b * half_b - a * c0;

        if (discriminant < 0) {
            return -1;
        }

        return (-half_b - @sqrt(discriminant)) / a;
    }

    pub fn createBoundingBox(self: Sphere) BoundingBox {
        return .{
            .minimum = .{self.position - @as(@Vector(3, f32), @splat(self.radius))},
            .maximum = .{self.position + @as(@Vector(3, f32), @splat(self.radius))},
        };
    }
};

pub const BoundingBox = struct {
    minimum: @Vector(3, f32),
    maximum: @Vector(3, f32),

    pub inline fn intersect(self: BoundingBox, ray_begin: @Vector(3, f32), ray_end: @Vector(3, f32), t_min: f32, t_max: f32) bool {
        //vector
        const inv_ds = @as(@Vector(3, f32), @splat(@as(f32, 1))) / ray_end;

        var t0s = (self.minimum - ray_begin) * inv_ds;
        var t1s = (self.maximum - ray_begin) * inv_ds;

        const is_inv_ds_negative = inv_ds < @as(@Vector(3, f32), @splat(0));

        //Vector swap
        {
            const new_t0s = @select(f32, is_inv_ds_negative, t1s, t0s);
            const new_t1s = @select(f32, is_inv_ds_negative, t0s, t1s);

            t0s = new_t0s;
            t1s = new_t1s;
        }

        const t0s_greater_than_min = t0s > self.minimum;
        const t1s_greater_than_max = t1s > self.maximum;

        const new_t_mins = @select(f32, t0s_greater_than_min, t0s, @as(@Vector(3, f32), @splat(t_min)));
        const new_t_maxs = @select(f32, t1s_greater_than_max, t1s, @as(@Vector(3, f32), @splat(t_max)));

        return !@reduce(.Or, new_t_maxs <= new_t_mins); // reverse of <=
    }
};

///Bounding Volume Heierarchy
pub const BVH = struct {
    pub const Node = struct {
        box: BoundingBox,
        left: usize,
        right: usize,
    };

    nodes: []Node,
};

pub const Material = union(enum) {
    pub const Lambertion = struct {
        albedo: Image.Color,
    };

    pub const Metal = struct {
        albedo: Image.Color,
        roughness: f32,
    };

    pub const Dieletric = struct {
        refractive_index: f32,
    };

    lambertion: Lambertion,
    metal: Metal,
    dieletric: Dieletric,
};

pub const Light = struct {
    position: @Vector(3, f32),
    color: @Vector(3, f32),
};

pub const Scene = struct {
    spheres: []const Sphere,
    planes: []const Plane,
    materials: []const Material,
    lights: []const Light,
    bvh: BVH = undefined,

    pub fn init() void {}
};

///Iterative path tracing
inline fn tracePath(scene: Scene, random: *Random128, init_ray_begin: @Vector(3, f32), init_ray_end: @Vector(3, f32)) @Vector(3, f32) {
    var ray_begin: @Vector(3, f32) = init_ray_begin;
    var ray_end: @Vector(3, f32) = init_ray_end;
    var color: @Vector(3, f32) = .{ 1, 1, 1 };

    const max_path_count: usize = 8;

    //Roulette termination?
    for (0..max_path_count) |i| {
        _ = i;
        var hit = false;
        var closest_t: f32 = std.math.floatMax(f32); //TODO: make programmable?
        var point: @Vector(3, f32) = undefined;
        var normal: @Vector(3, f32) = undefined;
        var material: Material = undefined;

        for (scene.spheres) |sphere| {
            const t = sphere.intersect(ray_begin, ray_end);

            if (t > 0 and t < closest_t) {
                hit = true;
                point = ray_begin + ray_end * @as(@Vector(3, f32), @splat(t));
                normal = point - sphere.position;
                closest_t = t;
                material = scene.materials[sphere.material_index];
            }
        }

        for (scene.planes) |plane| {
            const t = plane.intersect(ray_begin, ray_end);

            if (t > 0 and t < closest_t) {
                hit = true;
                point = ray_begin + ray_end * @as(@Vector(3, f32), @splat(t));
                normal = point - plane.position;
                closest_t = t;
                material = scene.materials[plane.material_index];
            }
        }

        if (hit) {
            switch (material) {
                .lambertion => |lambertion| {
                    ray_begin = point;
                    ray_end = normalize(normal) + sphereSample(random);

                    const albedo = Image.Color.toNormalized(lambertion.albedo);

                    color *= @Vector(3, f32){ albedo[0], albedo[1], albedo[2] } * @as(@Vector(3, f32), @splat(@as(f32, 0.5)));
                },
                .metal => |metal| {
                    ray_begin = point;
                    ray_end = reflect(normalize(ray_end), normalize(normal)) + (@as(@Vector(3, f32), @splat(metal.roughness)) * sphereSample(random));

                    const albedo = Image.Color.toNormalized(metal.albedo);

                    color *= @Vector(3, f32){ albedo[0], albedo[1], albedo[2] } * @as(@Vector(3, f32), @splat(0.5));
                },
                .dieletric => |dieletric| {
                    ray_begin = point;
                    ray_end = refract(normalize(ray_end), normalize(normal), 1 / dieletric.refractive_index);
                },
            }

            continue;
        } else {
            const ray_direction = normalize(ray_end);
            const t = 0.5 * (ray_direction[1] + 1);

            color *= @as(@Vector(3, f32), @splat(1 - t)) * @Vector(3, f32){ 1, 1, 1 } + @as(@Vector(3, f32), @splat(t)) * @Vector(3, f32){ 0.5, 0.7, 1 };

            break;
        }
    }

    return color;
}

pub fn traceRays(scene: Scene, target: Image, offset: @Vector(2, usize), size: @Vector(2, usize)) void {
    var random: Random128 = undefined;

    var split_max = std.rand.SplitMix64.init(@as(u64, @intCast(std.time.timestamp())));

    random.state[0] = split_max.next();
    random.state[1] = split_max.next();

    const viewport_height: f32 = 2;
    const viewport_width: f32 = (16 / 9) * viewport_height;
    const focal_length: f32 = 1;

    const ray_begin = @Vector(3, f32){ 0, 0, 0 };
    const horizontal = @Vector(3, f32){ viewport_width, 0, 0 };
    const vertical = @Vector(3, f32){ 0, viewport_height, 0 };
    const lower_left_corner = ray_begin - horizontal / @as(@Vector(3, f32), @splat(2)) - vertical / @as(@Vector(3, f32), @splat(2)) - @Vector(3, f32){ 0, 0, focal_length };

    var pixel = target.texel_buffer.ptr[offset[0] + target.width * offset[1] .. target.texel_buffer.len - 1].ptr;

    for (offset[1]..size[1]) |y| {
        for (offset[0]..size[0]) |x| {
            var color = @Vector(3, f32){ 0, 0, 0 };

            const sample_count = 16;
            const is_multisampled = sample_count > 1;

            var i: usize = 0;

            while (i < sample_count) : (i += 1) {
                const uv_offset: @Vector(2, f32) = .{ if (is_multisampled) random.random128F32() else 0, if (is_multisampled) random.random128F32() else 0 };

                const uv = (@Vector(2, f32){ @as(f32, @floatFromInt(x)), @as(f32, @floatFromInt(y)) } + uv_offset) /
                    @Vector(2, f32){ @as(f32, @floatFromInt(target.width)), @as(f32, @floatFromInt(target.height)) };

                const ray_end = lower_left_corner + horizontal * @as(@Vector(3, f32), @splat(uv[0])) + vertical * @as(@Vector(3, f32), @splat(1 - uv[1]));

                color += tracePath(scene, &random, ray_begin, ray_end);
            }

            if (is_multisampled) {
                color *= @as(@Vector(3, f32), @splat(1 / @as(f32, sample_count)));
            }

            color = @sqrt(color); //gamma correction (gamma = 2.2, approx = 2)

            pixel[0] = Image.Color.fromNormalized(.{ color[0], color[1], color[2], 1 });
            pixel += 1;
        }
    }
}
