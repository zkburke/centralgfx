///Fixed sized multi producer multi consumer queue
///Based on https://github.com/dotnet/runtime/blob/main/src/libraries/System.Private.CoreLib/src/System/Collections/Concurrent/ConcurrentQueueSegment.cs
pub fn AtomicQueue(
    comptime T: type,
    comptime size: usize,
) type {
    return struct {
        items: [size]T = [_]T{undefined} ** size,
        sequence_numbers: [size]i32 align(std.atomic.cache_line) = blk: {
            var numbers: [size]i32 = undefined;

            //There must be a better way to initialize this
            @setEvalBranchQuota(size);

            for (0..size) |i| {
                numbers[i] = i;
            }

            break :blk numbers;
        },
        ///Aligned on cache line boundary to prevent false sharing
        //basically head
        front: std.atomic.Value(usize) align(std.atomic.cache_line) = .{ .raw = 0 },
        //basically tail
        back: std.atomic.Value(usize) align(std.atomic.cache_line) = .{ .raw = 0 },

        pub const index_mask: usize = size - 1;

        ///Returns true if the item was successfully enqueued, otherwise false
        pub fn tryPush(self: *@This(), item: T) bool {
            while (true) {
                const current_back = self.back.load(.monotonic);
                //Ring buffer indexing
                const i = current_back % size;

                const sequence_number = @atomicLoad(i32, &self.sequence_numbers[i], .acquire);

                const sequence_diff = sequence_number - @as(i32, @intCast(current_back));

                if (sequence_diff == 0) {
                    _ = self.back.cmpxchgWeak(current_back, current_back + 1, .acquire, .monotonic) orelse {
                        self.items[i] = item;

                        @atomicStore(i32, &self.sequence_numbers[i], @intCast(current_back + 1), .release);

                        return true;
                    };
                } else if (sequence_diff < 0) {
                    return false;
                } else {}
            }

            return false;
        }

        pub fn tryPop(self: *@This()) ?T {
            while (true) {
                const current_front = self.front.load(.acquire);
                //Ring indexing
                const i = current_front % size;

                const sequence_number = @atomicLoad(i32, &self.sequence_numbers[i], .acquire);

                const sequence_difference = sequence_number - @as(i32, @intCast(current_front + 1));

                if (sequence_difference == 0) {
                    _ = self.front.cmpxchgWeak(
                        current_front,
                        current_front + 1,
                        .acquire,
                        .monotonic,
                    ) orelse {
                        const item = self.items[i];
                        @atomicStore(
                            i32,
                            &self.sequence_numbers[i],
                            @as(i32, @intCast(current_front)) + @as(i32, size),
                            .release,
                        );
                        return item;
                    };
                } else if (sequence_difference < 0) {
                    const current_back = self.back.load(.acquire);

                    if (current_back - current_front <= 0) {
                        return null;
                    }

                    return null;

                    // std.Thread.yield() catch {};
                }
            }
        }

        fn prottytryPush(self: *@This(), item: T) bool {
            var slot_index: usize = undefined;
            var tail = @atomicLoad(usize, &self.back.raw, .monotonic);

            while (true) {
                slot_index = tail % size;

                const seq = @atomicLoad(usize, &self.sequence_numbers[slot_index], .acquire);
                const diff = @as(isize, @intCast(seq)) - @as(isize, @intCast(tail));

                if (diff < 0) {
                    return false;
                } else if (diff != 0) {
                    tail = @atomicLoad(usize, &self.back.raw, .monotonic);
                } else {
                    tail = @cmpxchgWeak(
                        usize,
                        &self.back.raw,
                        tail,
                        tail +% 1,
                        .monotonic,
                        .monotonic,
                    ) orelse break;
                }
            }

            self.items[slot_index] = item;

            @atomicStore(usize, &self.sequence_numbers[slot_index], tail +% 1, .release);

            return true;
        }

        fn prottytryPop(self: *@This()) ?T {
            var slot_index: usize = 0;

            var head = @atomicLoad(usize, &self.front.raw, .monotonic);

            while (true) {
                slot_index = head % size;

                const seq = @atomicLoad(usize, &self.sequence_numbers[slot_index], .acquire);
                const diff = @as(isize, @intCast(seq)) - @as(isize, @intCast(head + 1));

                if (diff < 0) {
                    return null;
                } else if (diff != 0) {
                    head = @atomicLoad(usize, &self.front.raw, .monotonic);
                } else {
                    head = @cmpxchgWeak(
                        usize,
                        &self.front.raw,
                        head,
                        head +% 1,
                        .monotonic,
                        .monotonic,
                    ) orelse break;
                }
            }

            const item = self.items[slot_index];

            @atomicStore(usize, &self.sequence_numbers[slot_index], head +% size, .release);

            return item;
        }
    };
}

test "Basic push/pop behaviour" {
    var queue: AtomicQueue(u32, 4) = .{};

    queue.push(1);
    queue.push(2);
    queue.push(3);
    queue.push(4);

    try std.testing.expect(queue.pop() == 1);
    try std.testing.expect(queue.pop() == 2);
    try std.testing.expect(queue.pop() == 3);
    try std.testing.expect(queue.pop() == 4);
    try std.testing.expect(queue.pop() == null);
}

test {
    _ = std.testing.refAllDecls(@This());
}

const std = @import("std");
