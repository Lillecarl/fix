//! Engine-side GC coordination: root enumeration, stop-the-world marking,
//! sweeping, and collection metrics.
//!
//! The explicit `Context` below contains only the evaluator state GC owns or
//! scans, keeping this controller independent of the full `Engine` type.
//!

const std = @import("std");
const observ = @import("base").observ;
const gc = @import("runtime").gc;
const types = @import("runtime").types;
const ObjectHeap = @import("runtime").heap.ObjectHeap;
const heap_collector = @import("runtime").heap_collector;
const Pacing = @import("runtime").heap.pacing;
const thunk_mod = @import("runtime").thunk;
const future_mod = @import("runtime").future;
const vm_force = @import("../vm.zig").force;
const vm_access = @import("../vm.zig").access;
const memory_config = @import("../memory_config.zig");
const clock = @import("base").clock;
const SpinMutex = @import("base").sync.SpinMutex;
const Value = @import("runtime").value.Value;
const VM = @import("../vm/context.zig").VM;
const Frame = @import("../vm/context.zig").Frame;
const Worker = @import("workers/worker.zig").Worker;
const Scheduler = @import("workers/scheduler.zig").Scheduler;
const ChunkRegistry = @import("../bytecode.zig").ChunkRegistry;
const RealizationStore = @import("store").RealizationStore;
const ImportRegistry = @import("imports.zig").Registry;

const gc_observation: observ.SpanSpec = .{
    .category = "gc",
    .name = "collect",
    .begin_verb = "collecting",
    .finish_verb = "collected",
    .begin_level = std.math.maxInt(u8),
    .finish_level = std.math.maxInt(u8),
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    heap: *ObjectHeap,
    registry: *ChunkRegistry,
    scheduler: *Scheduler,
    realization: *RealizationStore,
    imports: *ImportRegistry,
    builtins_value: *?Value,
    env_map: ?*const std.process.Environ.Map,
    worker_count: u8,
    gc_budget_bytes: ?u64,
    tracer: *gc.Tracer,
    import_vms: *std.ArrayListUnmanaged(*VM),
    import_vms_mu: *SpinMutex,
    workers: []std.atomic.Value(?*Worker),
    chunks_scanned: *types.ChunkId,
    extra_roots: *std.ArrayListUnmanaged(Value),
    parallel_cap: u32,
    observer: observ.Observer,
};

/// Default cap on collection participants. Evaluation may still use every
/// worker. Overridden by `FIX_GC_PAR_CAP`.
pub const default_parallel_cap: u32 = 8;

/// GC: register a stack-local VM (the top-level entry's VM, a
/// nested-eval VM, or an import VM) so the collector scans its roots.
/// These VMs are NOT in any worker's `fibers` list, so without this their
/// operand stack / frames / force-chain / temp-roots are invisible and
/// their live objects get swept. Concurrent imports at --workers>1
/// interleave, hence the mutex + remove-by-value.
pub fn registerVm(ev: Context, vm: *VM) void {
    ev.import_vms_mu.lock();
    defer ev.import_vms_mu.unlock();
    // An unregistered VM is invisible to the root scan. Continuing would turn
    // allocation pressure into collection-time use-after-free.
    ev.import_vms.append(ev.allocator, vm) catch @panic("gc VM root registry exhausted");
}
pub fn unregisterVm(ev: Context, vm: *VM) void {
    ev.import_vms_mu.lock();
    defer ev.import_vms_mu.unlock();
    for (ev.import_vms.items, 0..) |ivm, i| {
        if (ivm == vm) {
            _ = ev.import_vms.swapRemove(i);
            break;
        }
    }
}

/// Body of the scheduler's parallel-mark hook: a parked peer helps drain
/// marker slot `worker_id` to termination. `GcCoordinator` owns the low-level
/// scheduler callback and dispatches here.
pub fn helpMark(ev: Context, worker_id: u8) void {
    _ = worker_id;
    const marker_count = @min(@as(u32, ev.worker_count), ev.parallel_cap);
    const slot = ev.heap.gcMarkSlotGrab();
    if (slot >= marker_count) return;
    // Capture the phase while joining the mark. Once this worker returns from
    // drainParallel, the collector may finish the serial major sweep and reset
    // collecting_major before this thread is scheduled again. Re-reading the
    // flag below would then make a helper from that major enter the minor-only
    // sweep loop and wait forever for a gate this collection never opens.
    const collecting_major = ev.heap.collection.collecting_major;
    // Join the mark. A minor then claims young-object lists from the shared
    // sweep queue; a major leaves every peer parked while the collector runs
    // the full sweep serially.
    ev.tracer.drainParallel(ev.heap, slot);
    if (!collecting_major)
        heap_collector.minorSweepClaimLoop(ev.heap, ev.tracer.mark_bits);
}

/// Run one stop-the-world generational collection at a safepoint.
pub fn collect(ev: Context, collector_id: u8) void {
    // The pacer's first trigger arms tracking; it schedules the first real
    // collection. The world is already stopped for publication.
    if (!ev.heap.collection.collect_enabled) {
        heap_collector.arm(ev.heap);
        return;
    }
    // A full mark is safe with multiple evaluator workers: the collection
    // coordinator has already parked every peer, and markRoots walks every
    // registered worker/fiber plus the scheduler and evaluator-owned roots.
    if (ev.heap.gcShouldMajor()) {
        collectMajor(ev, collector_id);
        return;
    }
    // Minor below: the marker slot is grabbed dynamically (capped participants),
    // so `collector_id` isn't needed past the major dispatch.
    var observation = ev.observer.begin(&gc_observation, .{ .subject = .{ .text = "minor" } });
    defer observation.cancel();
    // The young-gated mark starts from roots and the remembered set.
    const tr = ev.tracer;
    const t0 = nowNs();
    const SeedCtx = struct { tr: *gc.Tracer, heap: *ObjectHeap };
    const Seed = struct {
        fn cb(ctx: SeedCtx, source: types.ObjectId) void {
            ctx.tr.markRemsetSource(ctx.heap, source);
        }
    };
    if (ev.worker_count > 1) {
        const marker_count = @min(@as(u32, ev.worker_count), ev.parallel_cap);
        tr.resetParallelMinor(ev.heap.counts().objects, marker_count) catch {
            heap_collector.afterCollect(ev.heap, .skipped);
            return;
        };
        heap_collector.beginMinorSweep(ev.heap, ev.worker_count);
        const collector_slot = ev.heap.gcMarkSlotGrab(); // grabbed before gcOpenMark ⇒ slot 0
        // Seed roots + remset into the collector's own marker deque, open
        // the mark so parked peers help drain it, then drain alongside them.
        tr.beginSeeding(collector_slot);
        markRoots(ev, tr);
        heap_collector.forEachRemsetSource(ev.heap, SeedCtx{ .tr = tr, .heap = ev.heap }, Seed.cb);
        tr.endSeeding();
        ev.scheduler.gcOpenMark();
        tr.drainParallel(ev.heap, collector_slot); // returns at termination
        // Sweep cannot start until the mark is closed and verified.
        heap_collector.verifyMinorClosure(ev.heap, tr.mark_bits);
        heap_collector.openMinorSweep(ev.heap);
        heap_collector.minorSweepClaimLoop(ev.heap, tr.mark_bits);
        heap_collector.waitMinorSweepDone(ev.heap);
        ev.scheduler.gcCloseMark();
        tr.sumStats();
    } else {
        var remset_sources: u64 = 0;
        const CountSeedCtx = struct { tr: *gc.Tracer, heap: *ObjectHeap, n: *u64 };
        const CountSeed = struct {
            fn cb(ctx: CountSeedCtx, source: types.ObjectId) void {
                ctx.n.* += 1;
                ctx.tr.markRemsetSource(ctx.heap, source);
            }
        };
        tr.resetMinor(ev.heap.counts().objects) catch {
            heap_collector.afterCollect(ev.heap, .skipped);
            return;
        };
        const p0 = nowNs();
        markRoots(ev, tr);
        const p1 = nowNs();
        heap_collector.forEachRemsetSource(ev.heap, CountSeedCtx{ .tr = tr, .heap = ev.heap, .n = &remset_sources }, CountSeed.cb);
        const p2 = nowNs();
        tr.drainMinor(ev.heap);
        const p3 = nowNs();
        gc.recordMarkPhases(&ev.heap.collection.report, p0 - t0, p1 - p0, p2 - p1, p3 - p2, remset_sources);
    }
    const t1 = nowNs();
    // Parallel participants already drained the young lists.
    const st = if (ev.worker_count > 1)
        heap_collector.finishMinorSweep(ev.heap)
    else
        heap_collector.minorCollect(ev.heap, tr.mark_bits);
    const t2 = nowNs();
    heap_collector.remsetClear(ev.heap);
    // Young survivors are tenured: they grow the pacer's old generation.
    heap_collector.afterCollect(ev.heap, .{ .minor = tr.stats.bytes });
    gc.recordCollection(&ev.heap.collection.report, .minor, st.freed, tr.stats, ev.heap.totalReservedBytes());
    gc.recordTiming(&ev.heap.collection.report, .minor, t1 - t0, t2 - t1);
    const reserved = ev.heap.counts();
    gc.recordBreakdown(&ev.heap.collection.report, .{
        .obj_live = tr.stats.objects,
        .obj_reserved = reserved.objects,
        .val_live = tr.stats.values,
        .val_reserved = reserved.values,
        .attr_live = tr.stats.attrs,
        .attr_reserved = reserved.attrs,
        .attr_pos_live = tr.stats.attr_pos,
        .attr_pos_reserved = reserved.attr_positions,
    });
    observation.finish(.{ .metrics = &.{
        .{ .name = "freed", .value = .{ .unsigned = st.freed }, .unit = .items },
        .{ .name = "live", .value = .{ .unsigned = tr.stats.bytes }, .unit = .bytes },
    } });
}

/// GC: one stop-the-world MAJOR (full) collection at a safepoint.
/// Unlike `collect` (the young-gated minor), this marks the whole reachable
/// graph from roots and sweeps EVERY unmarked object — reclaiming the tenured
/// old-generation garbage a minor can't. Parked evaluator workers help with
/// the full mark; the collector performs the full sweep serially.
pub fn collectMajor(ev: Context, collector_id: u8) void {
    _ = collector_id; // marker slot grabbed dynamically, like the minor
    // Same lazy-arm as the minor: the first trigger arms tracking (everything
    // so far becomes untracked/old) rather than collecting.
    if (!ev.heap.collection.collect_enabled) {
        heap_collector.arm(ev.heap);
        return;
    }
    var observation = ev.observer.begin(&gc_observation, .{ .subject = .{ .text = "major" } });
    defer observation.cancel();
    const tr = ev.tracer;
    const t0 = nowNs();
    // Full mark from all roots. Rescan EVERY chunk's constants: the incremental
    // cursor (`gc_chunks_scanned`) only covers chunks compiled since the last
    // minor, but a non-gated mark must trace old referents too. (The remembered
    // set IS also seeded below — see the note before `forEachRemsetSource`.)
    ev.chunks_scanned.* = 0;
    const SeedCtx = struct { tr: *gc.Tracer, heap: *ObjectHeap };
    const Seed = struct {
        fn cb(ctx: SeedCtx, source: types.ObjectId) void {
            ctx.tr.markRemsetSource(ctx.heap, source);
        }
    };
    if (ev.worker_count > 1) {
        const marker_count = @min(@as(u32, ev.worker_count), ev.parallel_cap);
        tr.resetParallel(ev.heap.counts().objects, marker_count) catch {
            heap_collector.afterCollect(ev.heap, .skipped);
            return;
        };
        // The minor resets this dispenser in beginMinorSweep. A major has no
        // parallel sweep phase, so reset it directly before taking slot zero.
        ev.heap.collection.mark_slot.store(0, .release);
        // Published before gcOpenMark's release store; peers acquire that gate
        // before entering helpMark and consequently skip the minor sweep.
        ev.heap.collection.collecting_major = true;
        const collector_slot = ev.heap.gcMarkSlotGrab();
        tr.beginSeeding(collector_slot);
        markRoots(ev, tr);
        heap_collector.forEachRemsetSource(ev.heap, SeedCtx{ .tr = tr, .heap = ev.heap }, Seed.cb);
        seedPinnedRegion(ev.heap, tr);
        tr.endSeeding();
        ev.scheduler.gcOpenMark();
        tr.drainParallel(ev.heap, collector_slot);
        ev.scheduler.gcCloseMark();
        tr.sumStats();
    } else {
        tr.resetMajor(ev.heap.counts().objects) catch {
            heap_collector.afterCollect(ev.heap, .skipped);
            return;
        };
        markRoots(ev, tr);
        // Mutable edges recorded by the write barrier must seed the full mark
        // too; not every edge is recoverable by re-scanning its source object.
        heap_collector.forEachRemsetSource(ev.heap, SeedCtx{ .tr = tr, .heap = ev.heap }, Seed.cb);
        seedPinnedRegion(ev.heap, tr);
        tr.drain(ev.heap);
    }
    const st = heap_collector.sweep(ev.heap, tr.mark_bits); // serial full sweep
    // Minor sweeps coalesce only consecutive owners encountered together.
    // A full STW is the infrequent point where we can cheaply merge free
    // intervals accumulated across collections and worker shards.
    ev.heap.gcCoalesceFreeRanges();
    const t1 = nowNs();
    // Tenure every survivor, then clear the remembered set because no live
    // young objects remain. Swept slots stay
    // round-robined across worker shards; the allocation path work-steals across
    // shards, so a demand-concentrated workload still reuses the whole pool.
    ev.heap.gcMajorReconcile(tr.mark_bits);
    heap_collector.remsetClear(ev.heap);
    // Every helper finished its full-mark drain before the sweep began and has
    // skipped the minor phase. Restore the mode for the next collection.
    ev.heap.collection.collecting_major = false;
    const t2 = nowNs();
    // The live set is the pacer's new old generation and major base.
    heap_collector.afterCollect(ev.heap, .{ .major = tr.stats.bytes });
    gc.recordCollection(&ev.heap.collection.report, .major, st.objects_freed, tr.stats, ev.heap.totalReservedBytes());
    gc.recordTiming(&ev.heap.collection.report, .major, t1 - t0, t2 - t1);
    const reserved = ev.heap.counts();
    gc.recordBreakdown(&ev.heap.collection.report, .{
        .obj_live = tr.stats.objects,
        .obj_reserved = reserved.objects,
        .val_live = tr.stats.values,
        .val_reserved = reserved.values,
        .attr_live = tr.stats.attrs,
        .attr_reserved = reserved.attrs,
        .attr_pos_live = tr.stats.attr_pos,
        .attr_pos_reserved = reserved.attr_positions,
    });
    observation.finish(.{ .metrics = &.{
        .{ .name = "freed", .value = .{ .unsigned = st.objects_freed }, .unit = .items },
        .{ .name = "live", .value = .{ .unsigned = tr.stats.bytes }, .unit = .bytes },
    } });
}

pub const nowNs = clock.monotonicNs;

// --- collection policy ------------------------------------------------------
//
// The pacer (`runtime/heap/pacer.zig`) decides when to collect from the
// evaluation's own heap shape: allocation volume, tenured bytes, and the live
// set at the last major. Nothing here reads host memory, so a given
// evaluation collects at the same points on every machine. `--gc-budget` is
// an optional soft limit on top; the environment knobs are tuning and
// validation overrides.

/// Resolve the pacer configuration, or null when collection is disabled
/// (`--gc-budget=0`). Overrides:
///   FIX_GC_ARM=SIZE     allocation volume before tracking arms
///   FIX_GC_GROWTH=PCT   nursery size as a percentage of the old generation
///   FIX_GC_STEP_MB=N    validation: arm at once and collect every N MiB of
///                       allocation (wins over `--gc-budget=0`)
pub fn pacerConfig(ev: Context) ?Pacing.Config {
    var config: Pacing.Config = .{};
    if (envSize(ev, "FIX_GC_ARM")) |arm| config.arm_bytes = arm;
    if (envInt(ev, "FIX_GC_GROWTH")) |pct| config.growth_percent = std.math.cast(u32, pct) orelse config.growth_percent;
    if (envInt(ev, "FIX_GC_STEP_MB")) |mb| config.step_bytes = mb << 20;
    if (config.step_bytes > 0) return config;
    if (ev.gc_budget_bytes) |limit| {
        if (limit == 0) return null;
        config.limit_bytes = limit;
    }
    return config;
}

/// Keep precise transient roots from evaluation start so the first major can
/// reclaim the pre-arming region. Normal evaluation turns rooting on when the
/// pacer arms; `FIX_GC_ROOTS=1` keeps the earlier, more expensive behavior
/// available for detector and performance comparisons.
pub fn rootsFromStart(ev: Context) bool {
    if (ev.env_map) |em| if (em.get("FIX_GC_ROOTS")) |s|
        return !std.mem.eql(u8, s, "0");
    return false;
}

/// A `SIZE`-syntax env override (same grammar as `--gc-budget`), or null.
fn envSize(ev: Context, key: []const u8) ?u64 {
    if (ev.env_map) |em| if (em.get(key)) |s| return memory_config.parseSize(s);
    return null;
}

fn envInt(ev: Context, key: []const u8) ?u64 {
    if (ev.env_map) |em| if (em.get(key)) |s| return std.fmt.parseInt(u64, s, 10) catch null;
    return null;
}

/// Mark all GC roots into `tr` (without draining). See docs/gc.md.
pub fn markRoots(ev: Context, tr: *gc.Tracer) void {
    if (ev.builtins_value.*) |b| tr.markValue(ev.heap, b);
    // Caller-held external roots (repl scope bindings / last results —
    // values alive between evaluations with no VM holding them).
    for (ev.extra_roots.items) |v| tr.markValue(ev.heap, v);
    // Every worker's fibers' VM stack/frames/upvalues. At a stop-the-
    // world every live worker is parked at a safepoint, so its fiber
    // list is stable and its fibers' VMs hold that worker's roots. The
    // registry is published by each worker before it can allocate.
    for (ev.workers) |*slot| {
        const w = slot.load(.acquire) orelse continue;
        for (w.fibers.items) |f| {
            // Precise, for every fiber: operand stack + frames + upvalues +
            // in-flight force chain + builtin temp-roots (see force.zig).
            markVm(tr, ev.heap, &f.vm);
            // A task assigned to a fiber is out of the scheduler queue but
            // still a live reference. This covers only the pickTask →
            // slotEntry window: slotEntry clears `current_task` before
            // running, so each task runner must root its own ids for the
            // task's duration (rootKeep in runForceThunkTask and the
            // range/sweep runners).
            if (f.current_task) |task| switch (task) {
                .force_thunk => |id| tr.markObject(ev.heap, id),
                .force_list_range => |r| tr.markObject(ev.heap, r.list_id),
                .force_attrs_sweep => |id| tr.markObject(ev.heap, id),
                .force_attrs_range => |r| tr.markObject(ev.heap, r.attrs_id),
                .import_prefetch, .readdir_prefetch => {},
            };
        }
    }
    // Transient import VMs (not in any fiber list).
    ev.import_vms_mu.lock();
    for (ev.import_vms.items) |ivm| markVm(tr, ev.heap, ivm);
    ev.import_vms_mu.unlock();
    // Pending scheduler tasks reference thunks/lists that will be forced.
    ev.scheduler.gcMarkPendingTasks(tr, ev.heap);
    // Thread-local caches hold Values that can be the momentary sole
    // reference to a shared object during forcing — mark them.
    vm_force.gcMarkThunkMemo(tr, ev.heap);
    vm_access.gcMarkAttrCache(tr, ev.heap);
    // Chunk constants can hold heap references (e.g. a scoped-import's
    // ambient-scope attrset baked in via emitConstant). Chunks are never
    // GC'd, so their constants are permanent roots — but INCREMENTALLY:
    // only chunks compiled since the last minor can reference a still-young
    // object (earlier ones' referents were promoted at their first scan;
    // post-promotion young refs come via the remembered set). See
    // `gc_chunks_scanned`.
    const chunk_count = ev.registry.count();
    var cid: types.ChunkId = ev.chunks_scanned.*;
    while (cid < chunk_count) : (cid += 1) {
        const ch = ev.registry.get(cid) orelse continue;
        for (ch.constants) |c| tr.markValue(ev.heap, c);
    }
    ev.chunks_scanned.* = chunk_count;
    // Resolved import results.
    var it = ev.imports.entries.iterator();
    while (it.next()) |e| {
        const entry = e.value_ptr.*;
        if (@intFromEnum(entry.future.stateField(.monotonic)) == @intFromEnum(future_mod.FutureState.resolved))
            tr.markValue(ev.heap, entry.result);
    }
    var replay_it = ev.imports.replay_entries.iterator();
    while (replay_it.next()) |e| {
        const entry = e.value_ptr.*;
        if (@intFromEnum(entry.future.stateField(.monotonic)) == @intFromEnum(future_mod.FutureState.resolved))
            tr.markValue(ev.heap, entry.result);
    }
    // Lazy-derivation cache (Value bits keyed by attrs id). Only current-
    // token entries are live roots; stale ones (pre-GC id, now reused) are
    // dead and will miss on lookup, so don't retain them.
    const LazyRootContext = struct { tracer: *gc.Tracer, heap: *ObjectHeap };
    ev.realization.visitLiveLazyDerivations(
        ev.heap.token,
        LazyRootContext{ .tracer = tr, .heap = ev.heap },
        struct {
            fn mark(context: LazyRootContext, bits: u64) void {
                context.tracer.markValue(context.heap, .{ .bits = bits });
            }
        }.mark,
    );
}

/// The lazily-armed pre-arming region `[0, track_from)` is PINNED — never
/// swept, and readable forever through references older than root tracking
/// (frames, upvalues, other pinned objects) that the armed root set cannot
/// see. Its members are therefore permanent ROOTS for a full mark, not just
/// unsweepable slots: a pinned thunk resolved after arming holds the only
/// edge to its (sweepable, post-arming) result, and a major that fails to
/// trace FROM it demotes and frees that result while the pinned parent
/// lives on — a use-after-free read at the next access (wrong drvPaths /
/// InvalidObjectType under tight budgets at w>1). Minors are covered by the
/// remembered set; majors must seed the whole pinned region.
fn seedPinnedRegion(heap: *ObjectHeap, tr: *gc.Tracer) void {
    var id: types.ObjectId = 0;
    const pinned_end = heap.collection.track_from;
    while (id < pinned_end) : (id += 1) tr.markObject(heap, id);
}

pub fn markVm(tr: *gc.Tracer, heap: *ObjectHeap, vm: *VM) void {
    tr.markValue(heap, vm.builtins);
    tr.markValue(heap, vm.gc_roots.extra); // value in-flight at the safepoint
    for (vm.gc_roots.force_chain.items) |id| tr.markObject(heap, id); // in-flight force chain
    for (vm.gc_roots.temporary.items) |v| tr.markValue(heap, v); // native builtin roots
    for (vm.stack[0..vm.sp]) |v| tr.markValue(heap, v);
    for (vm.frames[0..vm.frames_len]) |*frame| markFrame(tr, heap, frame);
}

fn markFrame(tr: *gc.Tracer, heap: *ObjectHeap, frame: *const Frame) void {
    if (frame.upvalue_owner) |id| tr.markObject(heap, id);
    if (frame.upvalues) |ups| for (ups) |v| tr.markValue(heap, v);
}

test "a major roots the pinned pre-arming region (lazy arming)" {
    // Regression: under the lazy-arming budget mode, objects allocated
    // before arming are PINNED — never swept, and readable forever through
    // references the armed root set cannot see. A pinned thunk resolved
    // AFTER arming holds the only edge to its young result; the minor that
    // consumes the write barrier's remset entry promotes the result, and a
    // later major that does not seed the pinned region as roots demotes
    // (gcMajorReconcile clears old bits) and sweeps it — a use-after-free
    // read at the next access through the immortal pinned parent. This
    // produced silent wrong drvPaths / InvalidObjectType on tight-budget
    // nixpkgs runs.
    const allocator = std.testing.allocator;
    var heap = try ObjectHeap.init(allocator, 1);
    defer heap.deinit();
    // Lazy budget mode: nothing tracked or rooted until arming.
    heap_collector.configure(&heap, .{}, false);

    // Pinned parent: a thunk allocated before arming.
    const thunk_id = try heap.addBytecodeThunk(0, &.{});
    heap_collector.armTracking(&heap);
    try std.testing.expect(thunk_id < heap.collection.track_from);

    // Post-arming young result, reachable ONLY through the pinned thunk.
    const items = [_]Value{ Value.int(1), Value.int(2) };
    const young = try heap.addList(&items);
    const thunk = heap.getThunkAssumeValid(thunk_id);
    try std.testing.expect(thunk.tryForceSolo(1) == .claimed);
    thunk.resolveSolo(Value.list(young));
    heap.gcRecordEdge(thunk_id, Value.list(young));

    // Minor: consumes the remset entry, marks + promotes the result.
    var tr = gc.Tracer.init(allocator);
    defer tr.deinit();
    try tr.resetMinor(heap.counts().objects);
    tr.markRemsetSource(&heap, thunk_id);
    tr.drainMinor(&heap);
    try std.testing.expect(tr.isMarked(young));
    try std.testing.expectEqual(@as(u64, 0), heap_collector.minorCollect(&heap, tr.mark_bits).freed);
    heap_collector.remsetClear(&heap);
    try std.testing.expect(!heap.gcIsYoung(young)); // promoted

    // Control: a full mark from an EMPTY root set alone cannot see the
    // result — the pinned thunk is exactly the edge a major must seed.
    try tr.resetMajor(heap.counts().objects);
    tr.drain(&heap);
    try std.testing.expect(!tr.isMarked(young));

    // The fix: majors seed [0, track_from) as roots. The result survives
    // the sweep, stays tenured through reconcile, and remains readable.
    try tr.resetMajor(heap.counts().objects);
    seedPinnedRegion(&heap, &tr);
    tr.drain(&heap);
    try std.testing.expect(tr.isMarked(young));
    try std.testing.expectEqual(@as(u64, 0), heap_collector.sweep(&heap, tr.mark_bits).objects_freed);
    heap.gcMajorReconcile(tr.mark_bits);
    heap_collector.remsetClear(&heap);
    try std.testing.expect(!heap.gcIsYoung(young)); // re-tenured, not demoted

    // A follow-up young-gated minor (remset long gone) must not free it.
    try tr.resetMinor(heap.counts().objects);
    tr.drainMinor(&heap);
    try std.testing.expectEqual(@as(u64, 0), heap_collector.minorCollect(&heap, tr.mark_bits).freed);
    const read = try heap.getList(young);
    try std.testing.expectEqual(@as(usize, 2), read.len);
}

test "frame roots closure owner until its raw upvalue slice unwinds" {
    const allocator = std.testing.allocator;
    var heap = try ObjectHeap.init(allocator, 1);
    defer heap.deinit();
    heap_collector.configureEager(&heap, .{});

    const captures = [_]Value{
        Value.int(0), Value.int(1), Value.int(2), Value.int(3), Value.int(4),
        Value.int(5), Value.int(6), Value.int(7), Value.int(8), Value.int(9),
    };
    const owner = try heap.addClosure(0, &captures);
    const closure = try heap.getClosure(owner);
    const owned_range = switch (heap.get(owner).*) {
        .closure => |stored| stored.upvalues,
        else => unreachable,
    };
    const frame: Frame = .{
        .chunk_ptr = undefined,
        .chunk_id = 0,
        .ip = 0,
        .frame_base = 0,
        .local_count = 0,
        .upvalues = closure.upvalues,
        .upvalue_owner = owner,
    };

    var tr = gc.Tracer.init(allocator);
    defer tr.deinit();
    try tr.resetMajor(heap.counts().objects);
    markFrame(&tr, &heap, &frame);
    tr.drain(&heap);
    try std.testing.expect(tr.isMarked(owner));
    try std.testing.expectEqual(@as(u64, captures.len), tr.stats.values);
    try std.testing.expectEqual(@as(u64, 0), heap_collector.sweep(&heap, tr.mark_bits).objects_freed);

    // Once the frame is gone, the owner and its range are reclaimable.
    try tr.resetMajor(heap.counts().objects);
    tr.drain(&heap);
    try std.testing.expectEqual(@as(u64, 1), heap_collector.sweep(&heap, tr.mark_bits).objects_freed);
    const replacement = try heap.addList(&captures);
    const replacement_range = switch (heap.get(replacement).*) {
        .list => |range| range,
        else => unreachable,
    };
    try std.testing.expectEqual(owned_range.segment, replacement_range.segment);
    try std.testing.expectEqual(owned_range.offset, replacement_range.offset);
}
