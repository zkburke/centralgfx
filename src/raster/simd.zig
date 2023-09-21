pub inline fn reciprocal(x: @Vector(4, f32)) @Vector(4, f32) {
    const use_rcpps = false;

    if (use_rcpps) {
        return rcpps(x);
    } else {
        return @as(@Vector(4, f32), @splat(1)) / x;
    }
}

pub inline fn reciprocalVec3(x: @Vector(3, f32)) @Vector(3, f32) {
    const x_vec4: @Vector(4, f32) = .{ x[0], x[1], x[2], 1 };

    const reciprocal_x_vec4 = reciprocal(x_vec4);

    return .{ reciprocal_x_vec4[0], reciprocal_x_vec4[1], reciprocal_x_vec4[2] };
}

///Compute the reciprocal of x and return it
inline fn rcpps(x: @Vector(4, f32)) @Vector(4, f32) {
    return asm volatile (
        \\rcpps %[x], %[ret]
        : [ret] "={xmm1}" (-> @Vector(4, f32)),
        : [x] "{xmm0}" (x),
    );
}
