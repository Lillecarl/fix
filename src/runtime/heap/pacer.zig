//! Collection pacing: when the next collection runs, and whether it is major.
//!
//! One clock drives every decision: **allocation volume**, the bytes handed
//! out by the heap stores (fresh TLAB refills plus free-list reuse; see
//! `ObjectHeap.chargeAllocation`). Policy is a function of the program's own
//! heap shape, never of the host: the same evaluation collects at the same
//! points on a 4 GiB laptop and a 1 TiB builder.
//!
//! Phases:
//!   1. **Dormant.** No tracking, no barriers, bump-only allocation. Ends once
//!      volume reaches `arm_bytes`; evaluations below it pay nothing.
//!   2. **Armed.** Generational collection. After every collection the
//!      nursery (allocation volume until the next one) is
//!      `growth × old` clamped to `[min_nursery, max_nursery]`, where `old`
//!      estimates the tenured heap (the pinned pre-arming region plus
//!      promotions, reset to the measured live set at each major). Peak heap
//!      is therefore ≈ `old · (1 + growth)` — Go's GOGC rule applied to the
//!      nursery — and `max_nursery` bounds a minor's survivor trace, i.e.
//!      its pause.
//!   3. **Major.** Once `old` has grown past the last major's live set by
//!      `base · major_growth` (at least `min_nursery`), the next collection
//!      is a full one. Majors trace the whole live set, so this geometric
//!      gate bounds their total work by a constant multiple of the final live
//!      set. `major_growth` starts at `growth` and adapts to what majors
//!      actually reclaim: a major that frees under a quarter of the old
//!      generation doubles it (capped), one that frees half or more halves it
//!      back toward `growth`. Nix heaps mostly accrete, so low-yield majors —
//!      the long pauses — back off instead of recurring.
//!
//! An optional soft `limit_bytes` (`--gc-budget`) caps `old + nursery`, like
//! GOMEMLIMIT: tracking arms by half the limit and nurseries shrink to fit
//! under it. The nursery never drops below `min_nursery` and the major gate
//! ignores the limit, so a live set at or above the limit degrades to
//! frequent cheap minors rather than a livelock of full collections.
//! (Tightening the major gate near the limit was measured: on a NixOS eval
//! at a 512 MiB limit it doubled the full collections for 2% less RSS.)
//!
//! The pacer is plain data mutated only while the world is stopped (or before
//! evaluation starts). Mutators read `trigger` lock-free.

const std = @import("std");

pub const default_arm_bytes: u64 = 1 << 30;
pub const default_growth_percent: u32 = 100;
pub const default_min_nursery: u64 = 64 << 20;
pub const default_max_nursery: u64 = 1 << 30;
/// Ceiling for the adaptive major gate, as a percentage of the live base.
pub const max_major_growth_percent: u32 = 1600;

pub const Config = struct {
    /// Allocation volume after which tracking arms. Everything allocated
    /// before then is pinned for the heap's lifetime.
    arm_bytes: u64 = default_arm_bytes,
    /// Nursery size as a percentage of the old generation.
    growth_percent: u32 = default_growth_percent,
    min_nursery: u64 = default_min_nursery,
    max_nursery: u64 = default_max_nursery,
    /// Soft cap on `old + nursery`; null = none.
    limit_bytes: ?u64 = null,
    /// Validation override: arm at once and collect every `step_bytes` of
    /// allocation regardless of heap shape. 0 = off.
    step_bytes: u64 = 0,
};

pub const Pacer = struct {
    config: Config,
    /// Allocation volume at which mutators request the next collection
    /// (arming, while dormant). `maxInt` = never.
    trigger: std.atomic.Value(u64),
    /// Estimated tenured bytes: pinned region + promotions since last major.
    old_bytes: u64 = 0,
    /// Live bytes after the last major (the pinned region before the first).
    major_base: u64 = 0,
    /// Old-generation growth past `major_base` that gates the next major, as
    /// a percentage of `major_base`. Adapted by major yield.
    major_growth_percent: u32,

    pub fn init(config: Config) Pacer {
        var arm = config.arm_bytes;
        if (config.limit_bytes) |limit| arm = @min(arm, limit / 2);
        if (config.step_bytes > 0) arm = 0;
        return .{ .config = config, .trigger = .init(arm), .major_growth_percent = config.growth_percent };
    }

    /// A pacer that never requests a collection (`--gc-budget=0`).
    pub fn disabled() Pacer {
        return .{ .config = .{}, .trigger = .init(std.math.maxInt(u64)), .major_growth_percent = default_growth_percent };
    }

    pub fn isDisabled(self: *const Pacer) bool {
        return self.trigger.load(.monotonic) == std.math.maxInt(u64);
    }

    /// Tracking armed with `reserved` bytes already allocated. Those bytes
    /// can no longer be reclaimed by a minor; treat them as the old
    /// generation until a major measures it.
    pub fn onArm(self: *Pacer, allocated: u64, reserved: u64) void {
        self.old_bytes = reserved;
        self.major_base = reserved;
        self.schedule(allocated);
    }

    /// A minor promoted `promoted_bytes` into the old generation.
    pub fn onMinor(self: *Pacer, allocated: u64, promoted_bytes: u64) void {
        self.old_bytes +|= promoted_bytes;
        self.schedule(allocated);
    }

    /// A major left `live_bytes` reachable, all of it now old.
    pub fn onMajor(self: *Pacer, allocated: u64, live_bytes: u64) void {
        // Yield: the share of the estimated old generation the major freed.
        const freed = self.old_bytes -| live_bytes;
        if (freed *| 4 < self.old_bytes)
            self.major_growth_percent = @min(self.major_growth_percent *| 2, @max(max_major_growth_percent, self.config.growth_percent))
        else if (freed *| 2 >= self.old_bytes)
            self.major_growth_percent = @max(self.major_growth_percent / 2, self.config.growth_percent);
        self.old_bytes = live_bytes;
        self.major_base = live_bytes;
        self.schedule(allocated);
    }

    /// Should the next collection be major?
    pub fn wantMajor(self: *const Pacer) bool {
        const grown = self.old_bytes -| self.major_base;
        return grown >= @max(scale(self.major_base, self.major_growth_percent), self.config.min_nursery);
    }

    /// Bytes of allocation the next nursery may consume.
    pub fn nurseryBytes(self: *const Pacer) u64 {
        if (self.config.step_bytes > 0) return self.config.step_bytes;
        var nursery = std.math.clamp(self.growthOf(self.old_bytes), self.config.min_nursery, @max(self.config.min_nursery, self.config.max_nursery));
        if (self.config.limit_bytes) |limit|
            nursery = @max(self.config.min_nursery, @min(nursery, limit -| self.old_bytes));
        return nursery;
    }

    fn growthOf(self: *const Pacer, bytes: u64) u64 {
        return scale(bytes, self.config.growth_percent);
    }

    fn scale(bytes: u64, percent: u32) u64 {
        return std.math.cast(u64, @as(u128, bytes) * percent / 100) orelse std.math.maxInt(u64);
    }

    fn schedule(self: *Pacer, allocated: u64) void {
        self.trigger.store(allocated +| self.nurseryBytes(), .monotonic);
    }
};

const testing = std.testing;
const MiB: u64 = 1 << 20;
const GiB: u64 = 1 << 30;

test "dormant until arm_bytes; step mode arms at once" {
    const p = Pacer.init(.{});
    try testing.expectEqual(default_arm_bytes, p.trigger.load(.monotonic));
    const s = Pacer.init(.{ .step_bytes = 4 * MiB });
    try testing.expectEqual(@as(u64, 0), s.trigger.load(.monotonic));
    try testing.expect(Pacer.disabled().isDisabled());
}

test "nursery tracks the old generation, clamped" {
    var p = Pacer.init(.{});
    // Small pinned region: nursery floors at min_nursery.
    p.onArm(10 * MiB, 10 * MiB);
    try testing.expectEqual(10 * MiB + default_min_nursery, p.trigger.load(.monotonic));
    // Mid-size: nursery == old (growth 100%).
    p.onArm(400 * MiB, 400 * MiB);
    try testing.expectEqual(800 * MiB, p.trigger.load(.monotonic));
    // Huge old generation: nursery caps at max_nursery.
    p.onArm(8 * GiB, 8 * GiB);
    try testing.expectEqual(8 * GiB + default_max_nursery, p.trigger.load(.monotonic));
}

test "promotion grows the nursery and gates the major" {
    var p = Pacer.init(.{});
    p.onArm(512 * MiB, 512 * MiB);
    try testing.expect(!p.wantMajor());
    p.onMinor(GiB, 300 * MiB);
    try testing.expectEqual(GiB + 812 * MiB, p.trigger.load(.monotonic));
    try testing.expect(!p.wantMajor());
    p.onMinor(2 * GiB, 212 * MiB); // old 1024 MiB = 2 × base
    try testing.expect(p.wantMajor());
    p.onMajor(3 * GiB, 600 * MiB); // freed 424 of 1024 MiB: gate unchanged
    try testing.expect(!p.wantMajor());
    try testing.expectEqual(3 * GiB + 600 * MiB, p.trigger.load(.monotonic));
    try testing.expectEqual(default_growth_percent, p.major_growth_percent);
}

test "low-yield majors back the major gate off; productive ones restore it" {
    var p = Pacer.init(.{});
    p.onMajor(0, GiB);
    p.onMinor(0, GiB); // old 2 GiB
    try testing.expect(p.wantMajor());
    p.onMajor(0, 1800 * MiB); // freed ~12%
    try testing.expectEqual(2 * default_growth_percent, p.major_growth_percent);
    p.onMinor(0, 1800 * MiB); // old = 2 × base: no longer enough
    try testing.expect(!p.wantMajor());
    p.onMinor(0, 1800 * MiB); // old = 3 × base
    try testing.expect(p.wantMajor());
    // Repeated low yield saturates at the ceiling.
    for (0..8) |_| p.onMajor(0, p.old_bytes);
    try testing.expectEqual(max_major_growth_percent, p.major_growth_percent);
    // A productive major (freed ≥ half) halves the gate back.
    p.onMinor(0, 100 * GiB);
    p.onMajor(0, GiB);
    try testing.expectEqual(max_major_growth_percent / 2, p.major_growth_percent);
}

test "a tiny live set still waits a min_nursery of promotion before a major" {
    var p = Pacer.init(.{});
    p.onMajor(0, 0);
    p.onMinor(0, default_min_nursery - 1);
    try testing.expect(!p.wantMajor());
    p.onMinor(0, 1);
    try testing.expect(p.wantMajor());
}

test "soft limit arms early and shrinks the nursery, never below min_nursery" {
    var p = Pacer.init(.{ .limit_bytes = GiB });
    try testing.expectEqual(GiB / 2, p.trigger.load(.monotonic));
    p.onMajor(0, 900 * MiB);
    try testing.expectEqual(@as(u64, 124 * MiB), p.trigger.load(.monotonic));
    p.onMajor(0, 2 * GiB); // live above the limit
    try testing.expectEqual(default_min_nursery, p.trigger.load(.monotonic));
}

test "the soft limit never forces a major" {
    var p = Pacer.init(.{ .limit_bytes = GiB });
    p.onMajor(0, 4 * GiB); // live far above the limit
    p.onMinor(0, 2 * GiB);
    try testing.expect(!p.wantMajor());
    p.onMinor(0, 2 * GiB); // doubled: the ordinary gate
    try testing.expect(p.wantMajor());
}

test "step mode ignores heap shape" {
    var p = Pacer.init(.{ .step_bytes = 4 * MiB });
    p.onArm(0, 3 * GiB);
    try testing.expectEqual(4 * MiB, p.trigger.load(.monotonic));
    p.onMinor(100 * MiB, GiB);
    try testing.expectEqual(104 * MiB, p.trigger.load(.monotonic));
}
