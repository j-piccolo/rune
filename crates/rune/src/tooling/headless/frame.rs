//! [`crate::frame::host_step::FrameHost`] adapter for the offscreen headless driver, plus the
//! shared `Tx::resize_window` frame-animation helpers used by both single-window headless runs
//! and the multi-window virtual desktop.

use std::cell::{Cell, RefCell};
use std::io::Write;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::app::SharedCore;
use crate::frame::clock::FrameClock;
use crate::frame::host_step::{
    Acquire, FrameHost, Gate, PostStepCheck, ReacquireReason, RenderPrep, StepInfo,
    draw_ui, prepare_gpu_resources,
};
use crate::render::{Gpu, OFFSCREEN_FORMAT, Offscreen, Renderer};
use crate::runtime::RuntimeCore;
use crate::time::Instant;
use crate::tooling::{
    Args, Chrome, FrameTiming, Shot, Trace, crop, die, loop_check, tree_json_core,
};

use super::app::{HeadlessApp, HeadlessRenderArgs};
use super::report_traffic_light_overlaps;

pub(super) struct PreparedHeadlessFrame {
    encoder: wgpu::CommandEncoder,
    shot: Shot,
}

/// Persistent offscreen rendering resources and frame-to-frame state owned by `run_headless`.
pub(super) struct HeadlessLoop<'a> {
    pub(super) launched: Instant,
    pub(super) state_ms: f64,
    pub(super) runtime_ms: f64,
    pub(super) init_ms: f64,
    pub(super) args: &'a Args,
    pub(super) chrome: Chrome,
    pub(super) dark: bool,
    pub(super) scale: f32,
    pub(super) size: (f32, f32),
    pub(super) wp: u32,
    pub(super) hp: u32,
    pub(super) limits: ((f32, f32), Option<(f32, f32)>),
    pub(super) has_comp: bool,
    pub(super) has_video: bool,
    pub(super) start_frame: usize,
    pub(super) frames: usize,
    pub(super) snap_frame: usize,
    pub(super) dt: f64,
    pub(super) shared_core: &'a Arc<SharedCore>,
    pub(super) desktop: &'a crate::desktop::displays::VirtualDesktop,
    pub(super) gpu: &'a Gpu,
    pub(super) target_format: wgpu::TextureFormat,
    pub(super) renderer: Renderer,
    pub(super) target: Offscreen,
    pub(super) readback: crate::tooling::record::Readback<Shot>,
    pub(super) alpha: Option<crate::render::surface_alpha::AlphaSurface>,
    pub(super) overlay: Option<(crate::input::keycaps::Overlay, Renderer)>,
    pub(super) reports: crate::frame::encoded::FrameReports,
    pub(super) trace: Option<Trace>,
    pub(super) gate: Option<crate::motion_gate::Sampler>,
    pub(super) frame_anim: Option<crate::window::frame::FrameAnim>,
    pub(super) idle: bool,
    pub(super) early_place: Option<crate::windows::PlaceRequest>,
    pub(super) placed: Option<Value>,
    pub(super) snap_crop: &'a Cell<Option<[u32; 4]>>,
    pub(super) cpu_boot: &'a RefCell<Option<crate::render::cpu_boot::tool::Tool>>,
    pub(super) loop_check: loop_check::Tally,
    /// What the last frames changed, so the next redraws only that, as a window's does (`damage`).
    #[cfg(not(target_os = "macos"))]
    pub(super) damage: crate::window::damage::Damage,
    /// Bumped when the target is made again: its first frame draws whole.
    #[cfg(not(target_os = "macos"))]
    pub(super) target_made: u64,
    /// `RUNE_TARGET_AGE`: the other targets this run draws into in turn (the one drawn longest ago
    /// first), and how many frames each target in turn has taken since they were made.
    #[cfg(not(target_os = "macos"))]
    pub(super) spares: std::collections::VecDeque<Offscreen>,
    #[cfg(not(target_os = "macos"))]
    pub(super) turns: u32,
    pub(super) cpu_ms: Vec<f64>,
    pub(super) anim_ms: Vec<f64>,
    pub(super) rest_ms: Vec<f64>,
    #[cfg(feature = "frame-profile")]
    pub(super) profile: crate::frame::profile::Log,
}

/// `RUNE_DRAW_DUMP=<dir>` (frame-profile): every frame's instance list (`instances.bin`, 128 bytes each)
/// and what the redraw planner made of it (`frames.jsonl`, with each list's offset), for tools that
/// count what a frame asks of a GPU from the draw list itself, on any machine.
#[cfg(all(feature = "frame-profile", not(target_os = "macos")))]
pub(crate) fn dump_draw_list(k: usize, plan: &crate::window::damage::Plan, instances: &[crate::render::Instance], size: (u32, u32), clear: [f32; 4], keeps: bool, into_kept: bool) {
    use std::sync::{Mutex, OnceLock};
    static DIR: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    static OUT: Mutex<Option<(std::fs::File, std::fs::File, u64)>> = Mutex::new(None);
    let Some(dir) = DIR.get_or_init(|| std::env::var_os("RUNE_DRAW_DUMP").map(Into::into)) else { return };
    let mut out = OUT.lock().unwrap();
    if out.is_none() {
        std::fs::create_dir_all(dir).ok();
        let (Ok(meta), Ok(bin)) = (std::fs::File::create(dir.join("frames.jsonl")), std::fs::File::create(dir.join("instances.bin"))) else { return };
        *out = Some((meta, bin, 0));
    }
    let Some((meta, bin, at)) = out.as_mut() else { return };
    let bytes: &[u8] = bytemuck::cast_slice(instances);
    bin.write_all(bytes).ok();
    let (kind, why, draw, shift, same) = match plan {
        crate::window::damage::Plan::All(why) => ("all", *why, Vec::new(), None, false),
        crate::window::damage::Plan::Part { draw, shift, same, .. } => ("part", "", draw.clone(), shift.map(|s| (s.from, s.to)), *same),
    };
    writeln!(meta, "{}", json!({ "k": k, "at": *at, "n": instances.len(), "size": [size.0, size.1], "clear": clear, "plan": kind, "why": why,
        "draw": draw, "shift": shift, "same": same, "keeps": keeps, "into_kept": into_kept, "age": crate::window::damage::target_age() })).ok();
    *at += bytes.len() as u64;
}

impl HeadlessLoop<'_> {
    #[inline]
    fn reblend_offscreen(&mut self, rt: &mut RuntimeCore) {
        let rt_srgb = rt.srgb_blend();
        self.target_format = if rt_srgb { OFFSCREEN_FORMAT.remove_srgb_suffix() } else { OFFSCREEN_FORMAT };
        crate::window::rebuild_renderer_core(&self.gpu.device, &mut self.renderer, self.target_format, rt);
        self.target = Offscreen::with_format(&self.gpu.device, self.wp, self.hp, self.target_format);
        #[cfg(not(target_os = "macos"))]
        { self.target_made += 1; self.spares.clear(); self.turns = 0; }
        if self.alpha.is_some() {
            self.alpha = Some(crate::render::surface_alpha::AlphaSurface::new(&self.gpu.device, self.target_format, (self.wp, self.hp)));
        }
        if let Some((ov, ui)) = self.overlay.as_mut() {
            ov.set_srgb_blend(rt_srgb);
            ui.set_format(&self.gpu.device, self.target_format);
        }
    }
}

/// Per-frame borrow view driving one headless iteration through [`crate::frame::host_step::run`].
pub(super) struct HeadlessFrame<'a, 'b, F> {
    pub(super) state: &'a mut HeadlessLoop<'b>,
    pub(super) app: &'a mut dyn HeadlessApp,
    pub(super) deliver: &'a mut F,
    pub(super) k: usize,
    pub(super) real_t: f64,
    pub(super) frame_start: Instant,
    pub(super) script_had_events: bool,
    pub(super) requested_place: Option<crate::windows::PlaceRequest>,
    pub(super) want_pixels: bool,
    pub(super) first_pixels: bool,
    pub(super) encoded_stamp: Option<Instant>,
    #[cfg(feature = "frame-profile")]
    pub(super) fp_alloc0: u64,
    #[cfg(feature = "frame-profile")]
    pub(super) fp_commits0: u64,
    #[cfg(feature = "frame-profile")]
    pub(super) fp_cpu0: Option<f64>,
    #[cfg(feature = "frame-profile")]
    pub(super) fp_events: usize,
    /// This thread's CPU in each stage after the update (`frame::profile::STAGES`).
    #[cfg(feature = "frame-profile")]
    pub(super) fp_stages: [f64; crate::frame::profile::STAGES],
}

/// Adds this thread's CPU since `since` to stage `stage` of the frame's profile, and returns now.
#[cfg(feature = "frame-profile")]
#[inline]
fn staged(stages: &mut [f64; crate::frame::profile::STAGES], stage: usize, since: f64) -> f64 {
    let now = crate::frame::profile::cpu_ms().unwrap_or(0.0);
    stages[stage] += now - since;
    now
}

impl<F: FnMut(Shot, Vec<u8>)> FrameHost for HeadlessFrame<'_, '_, F> {
    type Drawable = ();
    type Prepared = PreparedHeadlessFrame;

    #[inline]
    fn gate(&mut self) -> Gate { Gate::Draw }

    #[inline]
    fn before_acquire(&mut self) {
        let s = &mut *self.state;
        take_window_frame(self.app.rt_mut(), &mut s.frame_anim, self.real_t, s.limits);
        if !s.idle {
            step_window_frame(self.app.rt_mut(), &mut s.frame_anim, self.real_t, s.scale);
        }
        {
            let rt = self.app.rt_mut();
            rt.host.window_resting = s.frame_anim.is_none() && rt.host.window_frame.is_none();
        }
        if let Some(g) = s.gate.as_mut() { self.app.gate_before(g); }
    }

    #[inline]
    fn needs_reblend(&self) -> bool {
        self.state.renderer.srgb_blend() != self.app.rt().srgb_blend()
    }

    #[inline]
    fn reblend(&mut self) -> bool {
        self.state.reblend_offscreen(self.app.rt_mut());
        true
    }

    #[inline]
    fn acquire(&mut self) -> Acquire<()> { Acquire::Ready(()) }

    #[inline]
    fn step_runtime(&mut self) -> StepInfo {
        let s = &mut *self.state;
        let step = if self.k == 0 { 0.0 } else { s.dt };
        // `RUNE_LOOP_CHECK=1`: every frame asks for a loop-only step first, and the step is
        // compared with the ordinary frame that runs instead, or says why it wouldn't run. A
        // window never asks after a script's kind of event (`LoopFrame::touched`); here the
        // step's own refusal shows for those too, counted apart.
        let asked = (self.app.rt().loop_check_on() && self.k > 0 && s.frame_anim.is_none())
            .then(|| self.app.rt_mut().loop_step(step));
        let animating = self.app.frame(step);
        if let Some(asked) = asked {
            s.loop_check.add(self.real_t, self.script_had_events, asked, self.app.rt_mut().loop_checked());
        }
        StepInfo {
            runtime_animating: animating,
            animating,
            dt: step,
            update_ms: 0.0,
            presents_at: None,
        }
    }

    #[inline]
    fn post_step_check(&mut self, _info: &mut StepInfo) -> PostStepCheck {
        if self.state.renderer.srgb_blend() != self.app.rt().srgb_blend() {
            PostStepCheck::Reacquire(ReacquireReason::Reblend)
        } else {
            PostStepCheck::Keep
        }
    }

    #[inline]
    fn prepare_reacquire(&mut self, reason: ReacquireReason, _info: &mut StepInfo) -> bool {
        match reason {
            ReacquireReason::Reblend => {
                // `Tx::set_srgb_blend`: the same bytes (sRGB-encoded) reach the readback either way.
                self.state.reblend_offscreen(self.app.rt_mut());
                true
            }
            #[cfg(any(test, target_os = "macos"))]
            ReacquireReason::NativeMaterials => true,
        }
    }

    #[inline]
    fn sync_after_step(&mut self, info: &mut StepInfo) {
        let s = &mut *self.state;
        self.app.open_popup_headless();
        // As on macOS, a frame change this commit asked for starts with the springs it
        // started, which have shown a step unless they held for the window.
        if self.app.rt().host.window_frame.is_some() {
            let shown = if self.app.rt_mut().host.fit.take_held() { 0.0 } else { info.dt };
            take_window_frame(self.app.rt_mut(), &mut s.frame_anim, self.real_t - shown, s.limits);
        }
        s.idle = !info.runtime_animating && s.frame_anim.is_none();
        // Updates, poll and input handlers place too; the latest request this frame wins.
        if let Some(place) = s.shared_core.take_place().or(self.requested_place.take()).or(s.early_place.take()) {
            let (display, frame) = s.desktop.place(&place, s.size);
            self.app.rt_mut().set_window_origin((frame[0], frame[1]));
            let record = json!({ "t": self.real_t, "display": display, "frame": frame });
            println!("rune-placed {record}");
            s.placed = Some(record);
        }
        info.update_ms = self.frame_start.elapsed().as_secs_f64() * 1000.0;
        if let Some(tool) = s.cpu_boot.borrow_mut().as_mut() {
            tool.frame(self.app.rt_mut(), (s.wp, s.hp), self.k, (info.update_ms * 1e6) as u64, self.k == s.snap_frame);
        }
    }

    #[inline]
    fn prepare_render(&mut self, _drawable: (), _info: &mut StepInfo) -> RenderPrep<PreparedHeadlessFrame> {
        let s = &mut *self.state;
        // An image's pixels changed where its instances didn't: the whole frame draws (`damage`).
        #[cfg(not(target_os = "macos"))]
        let images_changed = self.app.rt().images.pages.iter().any(|p| p.dirty.is_some());
        #[cfg(feature = "frame-profile")]
        let cpu = crate::frame::profile::cpu_ms().unwrap_or(0.0);
        prepare_gpu_resources(&mut s.renderer, s.gpu, self.app.rt_mut(), (s.wp, s.hp), false);
        let encoder = s.gpu.device.create_command_encoder(&Default::default());
        // `RUNE_TARGET_AGE=n`: this frame's target is the one drawn `n` frames ago, as a swap chain's is.
        #[cfg(not(target_os = "macos"))]
        let buffers = crate::window::damage::target_age();
        #[cfg(not(target_os = "macos"))]
        if buffers > 1 {
            while s.spares.len() + 1 < buffers as usize { s.spares.push_back(Offscreen::with_format(&s.gpu.device, s.wp, s.hp, s.target_format)); }
            let next = s.spares.pop_front().unwrap();
            s.spares.push_back(std::mem::replace(&mut s.target, next));
        }
        #[cfg(feature = "frame-profile")]
        let cpu = staged(&mut self.fp_stages, 0, cpu);
        // Only what changed is redrawn, as a window redraws it (`window::damage`).
        #[cfg(not(target_os = "macos"))]
        {
            let rt = self.app.rt();
            let in_place = s.alpha.is_none() && !images_changed;
            let plan = s.damage.plan(&rt.instances, (s.wp, s.hp), rt.clear_color(), s.target_made, in_place, true);
            // The target is this run's own and only the renderer draws in it, unless an overlay
            // draws over the frame: then it keeps the last frame (`RUNE_TARGET_KEEPS=1`).
            let keeps = crate::window::damage::target_keeps() && s.overlay.is_none();
            // Every target has taken a frame of this renderer's, and nothing else draws on them.
            let age = if buffers > 0 && s.turns >= buffers && s.overlay.is_none() { buffers } else { 0 };
            s.turns = s.turns.saturating_add(1);
            #[cfg(feature = "frame-profile")]
            dump_draw_list(self.k, &plan, &rt.instances, (s.wp, s.hp), rt.clear_color(), keeps, s.alpha.is_none() && !crate::window::damage::whole_frames());
            match plan {
                crate::window::damage::Plan::Part { draw, shift, same, .. } => {
                    s.renderer.arm_damage(&s.gpu.device, &s.gpu.queue, &draw, rt.clear_color(), (s.wp, s.hp), s.target_format, shift.map(|s| (s.from, s.to)));
                    if same { s.renderer.arm_same(); } else { s.renderer.cull_armed(&rt.instances, (s.wp, s.hp)); }
                    if keeps { s.renderer.arm_kept_target(); } else { s.renderer.arm_aged_target(age); }
                }
                // Whole, but into the kept frame, so the next frame may redraw part of it.
                crate::window::damage::Plan::All(_) if s.alpha.is_none() && !crate::window::damage::whole_frames() => {
                    s.renderer.arm_whole(&s.gpu.device, &s.gpu.queue, rt.clear_color(), (s.wp, s.hp), s.target_format);
                    if keeps { s.renderer.arm_kept_target(); } else { s.renderer.arm_aged_target(age); }
                }
                crate::window::damage::Plan::All(_) => {}
            }
        }
        #[cfg(feature = "frame-profile")]
        staged(&mut self.fp_stages, 1, cpu);
        let in_range = self.k >= s.start_frame && self.k < s.start_frame + s.frames;
        let shot = Shot {
            k: self.k,
            dump: in_range && (s.args.out.is_some() || s.args.strip.is_some()) && (self.k - s.start_frame).is_multiple_of(s.args.every),
            video: in_range && s.has_video,
            snap: self.k == s.snap_frame && s.args.snap.is_some(),
        };
        let first_pixels = s.args.first_pixels && self.k == 0;
        let want_pixels = shot.dump || shot.video || shot.snap || first_pixels;
        self.want_pixels = want_pixels;
        self.first_pixels = first_pixels;
        RenderPrep::Ready(PreparedHeadlessFrame { encoder, shot })
    }

    #[inline]
    fn render(&mut self, prepared: &mut PreparedHeadlessFrame, info: &StepInfo) -> bool {
        let s = &mut *self.state;
        #[cfg(feature = "frame-profile")]
        let cpu = crate::frame::profile::cpu_ms().unwrap_or(0.0);
        let view = s.alpha.as_ref().map_or(&s.target.view, |a| &a.view);
        let comp_animating = if s.has_comp {
            self.app.render_compositor(HeadlessRenderArgs {
                device: &s.gpu.device,
                queue: &s.gpu.queue,
                encoder: &mut prepared.encoder,
                target: view,
                format: s.target_format,
                size: (s.wp, s.hp),
                dt: info.dt,
                update_ms: info.update_ms,
                ui: &mut s.renderer,
            })
        } else {
            let rt = self.app.rt_mut();
            let clear = rt.clear_color();
            draw_ui(&mut s.renderer, s.gpu, &mut prepared.encoder, view, rt, (s.wp, s.hp), clear);
            false
        };
        // Every frame, so a frame drawn whole leaves the renderer's kept frame behind.
        #[cfg(not(target_os = "macos"))]
        s.renderer.damage_drawn().ok();
        if let Some((ov, ui)) = s.overlay.as_mut() {
            let ort = ov.frame(self.real_t, info.dt, self.app.rt().backdrop_dark().unwrap_or(s.dark));
            if !ort.instances.is_empty() {
                ui.set_fade_stops(ort.fade_stops());
                ui.prepare_ui(&s.gpu.device, &s.gpu.queue, &mut ort.text.atlas, &ort.instances, (s.wp, s.hp));
                ui.draw_over(&mut prepared.encoder, view);
            }
        }
        if let Some(alpha) = &s.alpha {
            alpha.draw(&mut prepared.encoder, &s.target.view);
        }
        if prepared.shot.snap {
            s.snap_crop.set(crop::snap_rect(self.app.rt(), s.args, s.scale, (s.wp, s.hp)));
        }
        if self.want_pixels {
            for (shot, px) in s.readback.make_room(&s.gpu.device) { (self.deliver)(shot, px); }
            s.readback.copy(&mut prepared.encoder, &s.target, prepared.shot);
        }
        #[cfg(feature = "frame-profile")]
        staged(&mut self.fp_stages, 2, cpu);
        comp_animating
    }

    #[inline]
    fn submit_and_present(&mut self, prepared: PreparedHeadlessFrame, _info: &StepInfo) -> Instant {
        let s = &mut *self.state;
        #[cfg(feature = "frame-profile")]
        let cpu = crate::frame::profile::cpu_ms().unwrap_or(0.0);
        s.renderer.submit(&s.gpu.queue, prepared.encoder);
        #[cfg(feature = "frame-profile")]
        staged(&mut self.fp_stages, 3, cpu);
        self.encoded_stamp = s.reports.stamp();
        if self.want_pixels { s.readback.submitted(); }
        crate::time::now()
    }

    #[inline]
    fn present_targets(&mut self) -> (Option<&mut FrameClock>, &mut RuntimeCore) {
        (None, self.app.rt_mut())
    }

    #[inline]
    fn settle(&mut self, _submitted: Instant, info: &StepInfo) {
        let s = &mut *self.state;
        s.reports.drawn(self.encoded_stamp, None, self.frame_start, info.update_ms);
        let ms = self.frame_start.elapsed().as_secs_f64() * 1000.0;
        #[cfg(feature = "frame-profile")]
        let cpu = crate::frame::profile::cpu_ms().unwrap_or(0.0);
        if s.has_comp {
            // Pace like a display link so a compositor's async readbacks (GPU timers) land between frames.
            rune_gpu::health::poll(&s.gpu.device, wgpu::PollType::wait_indefinitely()).ok();
        }
        #[cfg(feature = "frame-profile")]
        staged(&mut self.fp_stages, 4, cpu);
        #[cfg(feature = "frame-profile")]
        {
            // The frame's wall time until the GPU had drawn it, less `ms`: on a software rasteriser with no
            // threads of its own (lavapipe, `LP_NUM_THREADS=0`), what drawing the frame's pixels cost.
            let gpu_wait_ms = self.frame_start.elapsed().as_secs_f64() * 1000.0 - ms;
            let rt = self.app.rt_mut();
            s.profile.push(crate::frame::profile::Row {
                phases: crate::frame::profile::take(),
                stages: self.fp_stages,
                update_ms: info.update_ms,
                encode_ms: ms - info.update_ms,
                events: self.fp_events,
                allocations: procwatch::alloc::thread_allocations() - self.fp_alloc0,
                commits: rt.stats.commits - self.fp_commits0,
                why: rt.commit.take_commit_why(),
                shaped: rt.stats.shaped,
                animating: info.animating,
                cpu_ms: self.fp_cpu0.map(|start| crate::frame::profile::cpu_ms().unwrap() - start),
                gpu_wait_ms,
                gpu_time_ms: None,
                gpu_layers_ms: None,
                gpu_mhz: None,
            });
        }
        s.cpu_ms.push(ms);
        if self.k > 0 {
            if info.animating { s.anim_ms.push(ms) } else { s.rest_ms.push(ms) }
        }
        if let Some(tr) = s.trace.as_mut() {
            tr.frame_timed_core(self.app.rt_mut(), self.real_t * 1000.0, ms, FrameTiming::default());
        }
        if let Some(g) = s.gate.as_mut() { self.app.gate_frame(g, info.animating); }
        if self.first_pixels {
            let wait = crate::time::now();
            let done = s.readback.finish(&s.gpu.device);
            let marker = json!({
                "headless_ms": s.launched.elapsed().as_secs_f64() * 1000.0,
                "state_ms": s.state_ms, "runtime_ms": s.runtime_ms, "gpu_init_ms": s.init_ms,
                "frame_cpu_ms": ms, "readback_ms": wait.elapsed().as_secs_f64() * 1000.0,
                "pixel_bytes": done.last().map_or(0, |(_, px)| px.len()),
            });
            println!("rune-first-pixels {marker}");
            std::io::stdout().flush().unwrap();
            for (shot, px) in done { (self.deliver)(shot, px); }
        } else if self.want_pixels {
            for (shot, px) in s.readback.ready(&s.gpu.device) { (self.deliver)(shot, px); }
        }
        if self.k == s.snap_frame
            && s.args.chrome_overlay
            && let Some(lights) = s.chrome.traffic_light_rects()
        {
            report_traffic_light_overlaps(self.app.rt(), &lights);
        }
        if self.k == s.snap_frame
            && let Some(p) = &s.args.tree
        {
            let mut tree = tree_json_core(self.app.rt(), self.real_t);
            if let Some(placed) = &s.placed { tree["placed"] = placed.clone(); }
            let text = serde_json::to_string_pretty(&tree).unwrap();
            std::fs::write(p, text).unwrap_or_else(|e| die(&format!("{}: {e}", p.display())));
        }
    }
}

/// Takes the size handlers asked for a headless window, inside the limits the app set; its
/// viewport moves there on the virtual clock from `start`. Centring has nothing to centre on.
pub(in crate::tooling) fn take_window_frame(
    rt: &mut RuntimeCore,
    anim: &mut Option<crate::window::frame::FrameAnim>,
    start: f64,
    (min, max): ((f32, f32), Option<(f32, f32)>),
) {
    use crate::window::frame::{FrameAnim, clamp_size};
    rt.host.fit.take_held();
    let Some(request) = rt.host.window_frame.take() else { return };
    let Some(size) = request.size else { return };
    let (w, h) = clamp_size(size, min, max);
    let from = [0.0, 0.0, rt.window_viewport().0, rt.window_viewport().1];
    *anim = Some(FrameAnim::retarget(anim.as_ref(), request.spring, start, from, [0.0, 0.0, w, h]));
}

/// Moves a headless window's viewport one frame along, before the frame lays out.
pub(in crate::tooling) fn step_window_frame(
    rt: &mut RuntimeCore,
    anim: &mut Option<crate::window::frame::FrameAnim>,
    now: f64,
    scale: f32,
) {
    let Some(moving) = *anim else { return };
    let [_, _, w, h] = crate::window::frame::snap(moving.sample(now).0, scale);
    rt.step_viewport((w, h), scale);
    if moving.done(now) { *anim = None; }
}
