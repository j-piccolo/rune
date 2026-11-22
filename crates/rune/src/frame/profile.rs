//! Opt-in headless phase timings. No clocks or allocator hooks in ordinary builds.
//! Commit and Layout include their nested phases; these overlapping times must not be summed.
#[derive(Clone, Copy)]
pub(crate) enum Phase { View, Commit, Layout, Text, Paint, Apply }

/// Phases timed, each with a wall and a CPU column.
#[cfg_attr(not(feature = "frame-profile"), allow(dead_code))]
pub const PHASES: usize = 6;

/// The stages of a headless frame after its update, each timed in this thread's CPU: readying GPU
/// resources, planning what to redraw, encoding the draw, submitting it, and waiting for the GPU.
/// The last two are the graphics API's and the driver's; the rest is this crate's and what it
/// asks of them as it records.
#[cfg_attr(not(feature = "frame-profile"), allow(dead_code))]
pub(crate) const STAGES: usize = 5;
#[cfg(feature = "frame-profile")]
const STAGE_NAMES: [&str; STAGES] = ["prep_cpu", "plan_cpu", "draw_cpu", "submit_cpu", "poll_cpu"];

/// CPU consumed by this thread, excluding time the scheduler runs other work. Debug builds
/// time app callbacks with it for `perf_warn`, and interaction timing its input frames.
/// On Windows it counts the thread's cycles (`QueryThreadCycleTime`), so short phases read
/// true; `GetThreadTimes` moves in 15.6 ms ticks.
pub(crate) fn cpu_ms() -> Option<f64> {
    #[cfg(any(unix, windows))]
    { Some(super::host_clock::cpu_ns(super::host_clock::thread_cpu()) / 1e6) }
    #[cfg(not(any(unix, windows)))]
    { None }
}

#[cfg(feature = "frame-profile")]
mod enabled {
    use super::{PHASES, Phase, cpu_ms};
    use crate::time::Instant;
    use std::cell::Cell;

    thread_local! {
        static TIMES: Cell<[f64; 2 * PHASES]> = const { Cell::new([0.0; 2 * PHASES]) };
    }

    pub(crate) struct Scope(Phase, Instant, Option<f64>);
    impl Scope {
        pub(crate) fn new(phase: Phase) -> Self { Self(phase, crate::time::now(), cpu_ms()) }
    }
    impl Drop for Scope {
        fn drop(&mut self) {
            if crate::tooling::perf_warn::budget::on() { crate::tooling::perf_warn::budget::add(self.0, self.1.elapsed().as_secs_f64() * 1000.0); }
            if crate::tooling::interaction::active() { crate::tooling::interaction::add(self.0, self.1.elapsed().as_secs_f64() * 1000.0); }
            TIMES.with(|times| {
                let mut row = times.get();
                row[self.0 as usize] += self.1.elapsed().as_secs_f64() * 1000.0;
                if let Some(start) = self.2 { row[self.0 as usize + PHASES] += cpu_ms().unwrap() - start; }
                times.set(row);
            });
        }
    }
    pub(crate) fn take() -> [f64; 2 * PHASES] { TIMES.replace([0.0; 2 * PHASES]) }
}

#[cfg(feature = "frame-profile")]
pub(crate) use enabled::{Scope, take};

/// One headless frame: phase times (wall then CPU), the update and encode split, and counts.
#[cfg(feature = "frame-profile")]
pub(crate) struct Row {
    pub phases: [f64; 2 * PHASES],
    /// CPU per stage after the update (`STAGES`).
    pub stages: [f64; STAGES],
    pub update_ms: f64,
    pub encode_ms: f64,
    /// Script events delivered before this frame.
    pub events: usize,
    pub allocations: u64,
    pub commits: u64,
    /// Why the frame committed, as `runtime::commit_why::CommitWhy` bits.
    pub why: u16,
    pub shaped: u32,
    pub animating: bool,
    pub cpu_ms: Option<f64>,
    /// Wall time from the frame's submit until the GPU had drawn it (0 with no compositor: nothing waits).
    pub gpu_wait_ms: f64,
}

/// A headless run's frames. With `RUNE_FRAME_LOG=FILE`, each frame is also written to FILE as a
/// JSON line as it ends, then a closing line with the process's memory footprint, so a run can
/// be read as a distribution (p99, worst, frames over budget) rather than the summary's means.
#[cfg(feature = "frame-profile")]
pub(crate) struct Log {
    rows: Vec<Row>,
    out: Option<std::io::BufWriter<std::fs::File>>,
    /// The CPU's counts at the last frame's end, for a log that is written (`counters`).
    #[cfg(target_os = "linux")]
    counted: Option<(counters::Counters, [u64; 4])>,
}

/// The CPU's own counts of this process's work in user space: cycles and instructions, of the
/// thread that draws and of every thread. A time says how long a frame took on a machine doing
/// other things at whatever clock it had; a cycle count ticks only while the thread runs and
/// counts the same at any clock, and an instruction count is the same every run. Kernel time is
/// not in them (a process may not count it here unprivileged), so they leave out what a driver
/// does in the kernel.
#[cfg(all(feature = "frame-profile", target_os = "linux"))]
#[allow(unsafe_code)]
mod counters {
    use std::os::fd::RawFd;

    /// `perf_event_attr`, its first 64 bytes (`PERF_ATTR_SIZE_VER0`).
    #[repr(C)]
    #[derive(Default)]
    struct Attr { kind: u32, size: u32, config: u64, period: u64, sample_type: u64, read_format: u64, flags: u64, wakeup: u32, bp_type: u32, config1: u64 }

    const INHERIT: u64 = 1 << 1;
    const EXCLUDE_KERNEL: u64 = 1 << 5;
    const EXCLUDE_HV: u64 = 1 << 6;

    /// Cycles and instructions of the drawing thread, and of the other threads (each with the
    /// threads it starts later), a pair for each kind of core the process may run on.
    pub(super) struct Counters { main: Vec<[RawFd; 2]>, others: Vec<[RawFd; 2]> }

    fn open(kind: u32, config: u64, tid: i32, inherit: bool) -> Option<RawFd> {
        let attr = Attr { kind, size: 64, config, flags: EXCLUDE_KERNEL | EXCLUDE_HV | if inherit { INHERIT } else { 0 }, ..Default::default() };
        // SAFETY: the kernel reads `size` bytes of `attr`, which lives through the call.
        let fd = unsafe { libc::syscall(libc::SYS_perf_event_open, &attr as *const Attr, tid, -1i32, -1i32, 0u64) };
        (fd >= 0).then_some(fd as RawFd)
    }

    fn read(fd: RawFd) -> u64 {
        let mut value = 0u64;
        // SAFETY: a counter's read is its 8-byte count, into 8 bytes this frame owns.
        let got = unsafe { libc::read(fd, (&mut value as *mut u64).cast(), 8) };
        if got == 8 { value } else { 0 }
    }

    /// Each CPU's counter unit that names both events: its type, and the events' numbers. Cores of
    /// two kinds (Apple's, Arm's big and little) have a unit each; where none names them, the
    /// kernel's generic cycles and instructions.
    fn units() -> Vec<(u32, u64, u64)> {
        let mut units = Vec::new();
        for entry in std::fs::read_dir("/sys/bus/event_source/devices").into_iter().flatten().flatten() {
            let path = entry.path();
            let event = |name: &str| std::fs::read_to_string(path.join("events").join(name)).ok()
                .and_then(|s| u64::from_str_radix(s.trim().strip_prefix("event=0x")?, 16).ok());
            let kind = std::fs::read_to_string(path.join("type")).ok().and_then(|s| s.trim().parse().ok());
            if let (Some(kind), Some(cycles), Some(instructions)) = (kind, event("cycles"), event("instructions")) { units.push((kind, cycles, instructions)); }
        }
        if units.is_empty() { units.push((0, 0, 1)); }
        units
    }

    impl Counters {
        /// Counters on the calling thread (the one that draws) and on every other thread there is.
        /// None where the kernel gives none (no counters, or not to this process).
        pub(super) fn new() -> Option<Self> {
            // SAFETY: gettid takes nothing and returns this thread's id.
            let me = unsafe { libc::syscall(libc::SYS_gettid) } as i32;
            let units = units();
            let pair = |tid: i32, inherit: bool| -> Vec<[RawFd; 2]> {
                units.iter().filter_map(|&(kind, cycles, instructions)| Some([open(kind, cycles, tid, inherit)?, open(kind, instructions, tid, inherit)?])).collect()
            };
            let main = pair(me, false);
            if main.is_empty() { return None; }
            let mut others = Vec::new();
            for entry in std::fs::read_dir("/proc/self/task").into_iter().flatten().flatten() {
                let Some(tid) = entry.file_name().to_str().and_then(|s| s.parse::<i32>().ok()) else { continue };
                if tid != me { others.extend(pair(tid, true)); }
            }
            Some(Self { main, others })
        }

        /// `[cycles, instructions]` of the drawing thread, then of every thread, since they were made.
        pub(super) fn read(&self) -> [u64; 4] {
            let sum = |fds: &[[RawFd; 2]], at: usize| fds.iter().map(|pair| read(pair[at])).sum::<u64>();
            let (cycles, instructions) = (sum(&self.main, 0), sum(&self.main, 1));
            [cycles, instructions, cycles + sum(&self.others, 0), instructions + sum(&self.others, 1)]
        }
    }

    impl Drop for Counters {
        fn drop(&mut self) {
            for fd in self.main.iter().chain(&self.others).flatten() {
                // SAFETY: each is a counter this value opened and nothing else holds.
                unsafe { libc::close(*fd) };
            }
        }
    }
}

#[cfg(feature = "frame-profile")]
impl Log {
    pub(crate) fn new(frames: usize) -> Self {
        install_text_hook();
        let out = std::env::var_os("RUNE_FRAME_LOG").map(|path| {
            let file = std::fs::File::create(&path).unwrap_or_else(|e| panic!("RUNE_FRAME_LOG {}: {e}", path.to_string_lossy()));
            std::io::BufWriter::new(file)
        });
        // Counted only for a log that is written: the counters are a few dozen files to read a frame.
        #[cfg(target_os = "linux")]
        let counted = out.as_ref().and_then(|_| counters::Counters::new()).map(|c| { let now = c.read(); (c, now) });
        Self { rows: Vec::with_capacity(frames), out, #[cfg(target_os = "linux")] counted }
    }

    pub(crate) fn push(&mut self, row: Row) {
        if let Some(out) = self.out.as_mut() {
            use std::io::Write;
            let p = &row.phases;
            let mut line = serde_json::json!({
                "k": self.rows.len(), "ms": row.update_ms + row.encode_ms, "cpu_ms": row.cpu_ms, "events": row.events, "update_ms": row.update_ms, "encode_ms": row.encode_ms,
                "view_cpu": p[PHASES], "commit_cpu": p[PHASES + 1], "layout_cpu": p[PHASES + 2], "text_cpu": p[PHASES + 3],
                "paint_cpu": p[PHASES + 4], "apply_cpu": p[PHASES + 5], "allocations": row.allocations,
                "commits": row.commits, "shaped": row.shaped, "animating": row.animating, "gpu_wait_ms": row.gpu_wait_ms,
            });
            if row.commits > 0 { line["why"] = crate::runtime::CommitWhy::names(row.why).into(); }
            for (name, ms) in STAGE_NAMES.iter().zip(row.stages) { line[*name] = ms.into(); }
            // Since the last frame ended, in user space: the drawing thread's cycles and instructions,
            // then every thread's (a software rasteriser's among them).
            #[cfg(target_os = "linux")]
            if let Some((counters, last)) = self.counted.as_mut() {
                let now = counters.read();
                for (at, name) in ["cyc", "ins", "cyc_all", "ins_all"].into_iter().enumerate() { line[name] = now[at].wrapping_sub(last[at]).into(); }
                *last = now;
            }
            writeln!(out, "{line}").expect("RUNE_FRAME_LOG write");
        }
        self.rows.push(row);
    }

    /// Prints the `rune-frame-profile` means (first frame left out) and closes the log.
    pub(crate) fn report(&mut self) {
        let rows = &self.rows[usize::from(self.rows.len() > 1)..];
        let n = rows.len() as f64;
        let avg = |f: &dyn Fn(&Row) -> f64| rows.iter().map(f).sum::<f64>() / n;
        let cpu = rows.first().and_then(|r| r.cpu_ms).is_some();
        let phase_cpu = |i: usize| cpu.then(|| avg(&|r| r.phases[PHASES + i]));
        eprintln!("rune-frame-profile {}", serde_json::json!({
            "view_ms": avg(&|r| r.phases[0]), "commit_inclusive_ms": avg(&|r| r.phases[1]),
            "layout_inclusive_ms": avg(&|r| r.phases[2]), "shape_ms": avg(&|r| r.phases[3]),
            "paint_ms": avg(&|r| r.phases[4]), "apply_ms": avg(&|r| r.phases[5]),
            "update_ms": avg(&|r| r.update_ms), "encode_submit_ms": avg(&|r| r.encode_ms),
            "allocations": avg(&|r| r.allocations as f64), "commits": avg(&|r| r.commits as f64),
            "shaped": avg(&|r| r.shaped as f64), "animating_fraction": avg(&|r| u8::from(r.animating) as f64),
            "thread_cpu_ms": cpu.then(|| avg(&|r| r.cpu_ms.unwrap_or(0.0))),
            "view_cpu_ms": phase_cpu(0), "commit_cpu_ms": phase_cpu(1), "layout_cpu_ms": phase_cpu(2),
            "shape_cpu_ms": phase_cpu(3), "paint_cpu_ms": phase_cpu(4), "apply_cpu_ms": phase_cpu(5),
        }));
        if let Some(mut out) = self.out.take() {
            use std::io::Write;
            let mem = procwatch::snapshot();
            let line = serde_json::json!({
                "end": true, "frames": self.rows.len(),
                "footprint": mem.as_ref().map(|m| m.footprint), "footprint_peak": mem.as_ref().map(|m| m.footprint_peak),
                "resident": mem.as_ref().map(|m| m.resident),
                "heap_live": mem.as_ref().and_then(|m| m.heap.as_ref()).map(|h| h.live),
            });
            writeln!(out, "{line}").and_then(|()| out.flush()).expect("RUNE_FRAME_LOG write");
        }
    }
}

/// Wall then CPU milliseconds per phase (view, commit, layout, text, paint, apply) on this thread
/// since the last call, for benches that drive a `Runtime` themselves.
#[cfg(feature = "frame-profile")]
pub fn take_frame_phases() -> [f64; 2 * PHASES] {
    install_text_hook();
    take()
}

/// This thread's CPU clock in milliseconds.
#[cfg(feature = "frame-profile")]
pub fn thread_cpu_ms() -> f64 { cpu_ms().unwrap_or(0.0) }

/// Without `frame-profile`, a phase is timed only for `RUNE_COMMIT_BUDGET_MS`
/// (`perf_warn::budget`) and in a frame interaction timing follows (`tooling::interaction`),
/// and otherwise costs a cached flag and a thread-local one.
#[cfg(not(feature = "frame-profile"))]
pub(crate) struct Scope(Option<(Phase, crate::time::Instant)>);
#[cfg(not(feature = "frame-profile"))]
impl Scope {
    #[inline(always)]
    pub(crate) fn new(phase: Phase) -> Self {
        Self((crate::tooling::perf_warn::budget::on() || crate::tooling::interaction::active()).then(|| (phase, crate::time::now())))
    }
}
#[cfg(not(feature = "frame-profile"))]
impl Drop for Scope {
    #[inline(always)]
    fn drop(&mut self) {
        let Some((phase, start)) = self.0 else { return };
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if crate::tooling::perf_warn::budget::on() { crate::tooling::perf_warn::budget::add(phase, ms); }
        if crate::tooling::interaction::active() { crate::tooling::interaction::add(phase, ms); }
    }
}

thread_local! {
    static TEXT_SCOPES: std::cell::RefCell<Vec<Scope>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn enter_text_phase() -> crate::text::profile::Token {
    #[cfg(not(feature = "frame-profile"))]
    if !crate::tooling::perf_warn::budget::on() && !crate::tooling::interaction::active() {
        return 0;
    }
    TEXT_SCOPES.with_borrow_mut(|scopes| scopes.push(Scope::new(Phase::Text)));
    1
}

fn exit_text_phase(_token: crate::text::profile::Token) {
    TEXT_SCOPES.with_borrow_mut(|scopes| {
        scopes.pop();
    });
}

/// Registers this crate's `Phase::Text` timing hooks with `rune-text`, once per process.
pub(crate) fn install_text_hook() {
    static INSTALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !INSTALLED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::text::profile::set_text_phase_hooks(enter_text_phase, exit_text_phase);
    }
}
