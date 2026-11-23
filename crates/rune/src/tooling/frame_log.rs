//! The frame log (`frame-profile`) for a runtime its app steps itself: an offscreen stage that
//! moves its own clock and delivers its own input. Each frame is drawn as an overlay draws its
//! drawable (whole, over transparent) into a target of the log's, waited on, and written as a
//! headless run writes its frames (`RUNE_FRAME_LOG`), with everything the thread did since the
//! last frame: the input it handled, the updates and the frames that settled them. Where the
//! device has timestamps inside encoders, the GPU's own time for the frame and its clock as it
//! drew come with it (`gpu_time_ms`, `gpu_mhz`).

use crate::render::{Gpu, Offscreen, Renderer};

/// A runtime's frames, drawn and logged.
pub struct FrameLog {
    log: crate::frame::profile::Log,
    target: Offscreen,
    timer: Option<Timer>,
    events: usize,
    cpu0: Option<f64>,
    commits0: u64,
    /// Frames logged so far.
    k: usize,
}

/// Two timestamps around a frame's commands, and where they're read back.
struct Timer {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    read: wgpu::Buffer,
    /// Nanoseconds a tick.
    period: f64,
}

impl Timer {
    fn new(gpu: &Gpu) -> Option<Self> {
        let wanted = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !gpu.device.features().contains(wanted) { return None; }
        let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("frame-log"), ty: wgpu::QueryType::Timestamp, count: 2 });
        let buffer = |usage| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("frame-log"), size: 16, usage, mapped_at_creation: false });
        let resolved = buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
        let read = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        Some(Self { queries, resolved, read, period: f64::from(gpu.queue.get_timestamp_period()) })
    }

    /// The milliseconds between the two timestamps, once the frame's work is done.
    fn read(&self, gpu: &Gpu) -> Option<f64> {
        let slice = self.read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        rune_gpu::health::poll(&gpu.device, wgpu::PollType::wait_indefinitely()).ok()?;
        let ticks = {
            let bytes = slice.get_mapped_range().ok()?;
            let at = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
            at(1).wrapping_sub(at(0))
        };
        self.read.unmap();
        Some(ticks as f64 * self.period / 1e6)
    }
}

/// The GPU's clock now (MHz), where the kernel says it: Intel's `gt_act_freq_mhz` on Linux. A file read
/// each frame, which only an offscreen judge's frames take: a window's loop never runs this.
fn gpu_mhz() -> Option<u32> {
    #[cfg(target_os = "linux")]
    for entry in std::fs::read_dir("/sys/class/drm").into_iter().flatten().flatten() {
        if let Some(mhz) = std::fs::read_to_string(entry.path().join("gt_act_freq_mhz")).ok().and_then(|s| s.trim().parse().ok()) {
            return Some(mhz);
        }
    }
    None
}

impl FrameLog {
    /// Frames `size` device pixels big, each a JSON line appended to `out`.
    pub fn new<S: 'static>(gpu: &Gpu, renderer: &Renderer, rt: &mut crate::Runtime<S>, size: (u32, u32), out: std::fs::File) -> Self {
        let target = Offscreen::with_format(&gpu.device, size.0.max(1), size.1.max(1), renderer.format());
        let mut log = Self { log: crate::frame::profile::Log::to(Some(out), 1024), target, timer: Timer::new(gpu), events: 0, cpu0: None, commits0: 0, k: 0 };
        log.skip(rt);
        log
    }

    /// An input event (or a change the app made for one) the next frame handles.
    pub fn input(&mut self) { self.events += 1; }

    /// Draws `rt`'s last frame (`rt.frame(..)` first; `animating` is what it returned) and logs it.
    pub fn frame<S: 'static>(&mut self, gpu: &Gpu, renderer: &mut Renderer, rt: &mut crate::Runtime<S>, animating: bool) {
        use crate::frame::profile::{STAGES, cpu_ms};
        let mut stages = [0.0; STAGES];
        let mut since = cpu_ms().unwrap_or(0.0);
        let mut stage = |at: usize, since: &mut f64| { let now = cpu_ms().unwrap_or(0.0); stages[at] += now - *since; *since = now; };
        let start = crate::time::now();
        let size = (self.target.width, self.target.height);
        let core = rt.core_mut();
        // `RUNE_DRAW_DUMP`: the draw list, for counting what the frame asks of a GPU. Drawn whole.
        #[cfg(not(target_os = "macos"))]
        super::headless::dump_draw_list(self.k, &crate::window::damage::Plan::All("overlay"), &core.instances, size, [0.0; 4], false, false);
        renderer.attach_text(&mut core.text.atlas);
        crate::frame::host_step::prepare_gpu_resources(renderer, gpu, core, size, false);
        stage(0, &mut since);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        if let Some(t) = &self.timer { encoder.write_timestamp(&t.queries, 0); }
        crate::frame::host_step::draw_ui(renderer, gpu, &mut encoder, &self.target.view, core, size, [0.0; 4]);
        if let Some(t) = &self.timer {
            encoder.write_timestamp(&t.queries, 1);
            encoder.resolve_query_set(&t.queries, 0..2, &t.resolved, 0);
            encoder.copy_buffer_to_buffer(&t.resolved, 0, &t.read, 0, 16);
        }
        stage(2, &mut since);
        renderer.submit(&gpu.queue, encoder);
        stage(3, &mut since);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        // The clock while the GPU draws, and the timestamps after: the log's own work, left out of the frame's.
        let before = cpu_ms().unwrap_or(0.0);
        let mhz = self.log.exclude(gpu_mhz);
        let mut excluded = cpu_ms().unwrap_or(before) - before;
        since += excluded;
        rune_gpu::health::poll(&gpu.device, wgpu::PollType::wait_indefinitely()).expect("poll");
        stage(4, &mut since);
        // On a software rasteriser with no threads of its own (lavapipe, `LP_NUM_THREADS=0`), what
        // drawing the frame's pixels cost.
        let gpu_wait_ms = start.elapsed().as_secs_f64() * 1000.0 - ms;
        let before = cpu_ms().unwrap_or(0.0);
        let gpu_time_ms = match &self.timer { Some(t) => self.log.exclude(|| t.read(gpu)), None => None };
        excluded += cpu_ms().unwrap_or(before) - before;
        self.log.push(crate::frame::profile::Row {
            phases: crate::frame::profile::take(),
            stages,
            update_ms: 0.0,
            encode_ms: ms,
            events: self.events,
            allocations: 0,
            commits: core.stats.commits - self.commits0,
            why: core.commit.take_commit_why(),
            shaped: core.stats.shaped,
            animating,
            cpu_ms: self.cpu0.map(|start| cpu_ms().unwrap_or(start) - start - excluded),
            gpu_wait_ms,
            gpu_time_ms,
            gpu_mhz: mhz,
        });
        self.events = 0;
        self.cpu0 = cpu_ms();
        self.commits0 = core.stats.commits;
        self.k += 1;
    }

    /// A turn no window on screen would draw: what the thread did in it is left out of the next frame.
    pub fn skip<S: 'static>(&mut self, rt: &mut crate::Runtime<S>) {
        let _ = crate::frame::profile::take();
        self.log.skip();
        self.events = 0;
        self.cpu0 = crate::frame::profile::cpu_ms();
        self.commits0 = rt.core_mut().stats.commits;
    }
}

impl Drop for FrameLog {
    /// Closes the log with its end line; the means go to stderr.
    fn drop(&mut self) { self.log.report(); }
}
