//! Redrawing part of a frame: the renderer keeps the window's last frame in a texture of its own
//! (`Retained`), redraws only the rectangles that changed into it, and copies it to the
//! drawable. wgpu treats a drawable as holding nothing when it's acquired (a `LoadOp::Load` into
//! it loads zeros), so the frame can't be drawn over the drawable itself; the copy costs the GPU
//! a few hundred microseconds where the whole frame's raster costs milliseconds.
//!
//! The window arms the rectangles before the frame draws (`arm_damage`); `draw` (and so a
//! compositor's `Frame::draw_ui`) redraws only them when nothing but the instances draws there,
//! and the window reads back whether it did (`damage_drawn`), every frame: any frame drawn
//! another way leaves the kept texture behind, and the next armed one draws it whole. Every
//! instance still goes through the pass, each scissor kept inside the rectangle
//! (`ActivePass::borrowed_within`), so what lands there is exactly what a whole frame draws there.
//!
//! A scroll moves most of the window without changing it. The kept frame then holds the scrolled
//! viewport rolled (`Roll`): the scroll moves none of its pixels, only where the viewport starts
//! in it, and the copy to the drawable straightens it. What came in is redrawn on the drawable
//! and copied back to where the roll keeps it. A scroll frame so shades a strip and copies the
//! window once, as any partial frame does. A rectangle's clear is one colour written, not the
//! uber shader blending it (`SOLID_WGSL`), and each rectangle draws only the instances that
//! reach it (`cull_armed`).
//!
//! A target that keeps its pixels from one frame to the next (an offscreen one; a window's
//! buffer of a known age) needs no copy out: its host says so each frame (`arm_kept_target`),
//! and the rectangles are redrawn on the target itself, a frame that draws whole draws there, and
//! one that changes nothing draws nothing. The kept frame then serves only a scroll, which moves
//! pixels through it, and is brought up to date from the target where it fell behind (`lag`).

use super::*;

/// Rectangles a frame redraws at most.
pub(crate) const MAX_RECTS: usize = 4;
/// Rectangles `arm_damage` takes at most: a scroll's are more, and further apart.
pub(crate) const MAX_ARMED: usize = 8;

/// Rectangles the kept frame may be behind a target that keeps its pixels in, before they are one.
const LAG_RECTS: usize = 4;

/// Frames whose redrawn rectangles are remembered for a target that comes back older (`arm_aged_target`):
/// a swap chain of this many buffers and one more is served.
const HISTORY: usize = 4;
/// Rectangles such a target is brought up to date in, before they are their bounds.
const AGED_RECTS: usize = 8;

/// What the armed rectangles came to.
pub(super) type Drawn = Result<(), &'static str>;

/// The window's last frame, and whether it's current.
pub(super) struct Retained {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    current: bool,
}

/// A viewport (`band`, `[x, y, width, height]`) whose pixels the kept frame holds rolled: the
/// window's pixel (x, y) there is kept at `band.xy + ((x, y) - band.xy + off) mod band.wh`. A scroll
/// then moves nothing in the kept frame: it changes `off`, and what scrolls in is drawn where what
/// scrolled out was. The copy to the drawable, which a frame makes anyway, straightens it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Roll {
    band: [u32; 4],
    off: [u32; 2],
}

/// A rectangle of the window (`[x, y, width, height]`) and where the kept frame holds it.
type Piece = ([u32; 4], [u32; 2]);

impl Roll {
    /// The roll after the pixels at `from` (`[x, y, width, height]`) moved to `to`: `roll`, or a
    /// new one, of the viewport they moved in. `None`: `roll` is of another viewport.
    fn moved(roll: Option<Roll>, from: [u32; 4], to: [u32; 2]) -> Option<Roll> {
        let band = [from[0].min(to[0]), from[1].min(to[1]), from[2] + from[0].abs_diff(to[0]), from[3] + from[1].abs_diff(to[1])];
        let roll = roll.unwrap_or(Roll { band, off: [0, 0] });
        if roll.band != band { return None; }
        let step = |off: u32, from: u32, to: u32, size: u32| (i64::from(off) + i64::from(from) - i64::from(to)).rem_euclid(i64::from(size)) as u32;
        Some(Roll { band, off: [step(roll.off[0], from[0], to[0], band[2]), step(roll.off[1], from[1], to[1], band[3])] })
    }

    /// `rect` (`[x, y, width, height]`, the window's pixels) in pieces, each with where the kept
    /// frame holds it: itself outside the viewport, and inside it split where the roll wraps.
    fn pieces(roll: Option<Roll>, rect: [u32; 4], out: &mut Vec<Piece>) {
        let [x, y, w, h] = rect;
        if w == 0 || h == 0 { return; }
        let Some(Roll { band: [bx, by, bw, bh], off }) = roll else { return out.push((rect, [x, y])) };
        let (x1, y1) = (x + w, y + h);
        let (ix0, iy0, ix1, iy1) = (x.max(bx), y.max(by), x1.min(bx + bw), y1.min(by + bh));
        if ix0 >= ix1 || iy0 >= iy1 { return out.push((rect, [x, y])); }
        let mut plain = |r: [u32; 4]| if r[2] > 0 && r[3] > 0 { out.push((r, [r[0], r[1]])) };
        plain([x, y, w, iy0 - y]);
        plain([x, iy1, w, y1 - iy1]);
        plain([x, iy0, ix0 - x, iy1 - iy0]);
        plain([ix1, iy0, x1 - ix1, iy1 - iy0]);
        // Along one axis: (start, length, where it's kept), before the seam and from it.
        let spans = |lo: u32, hi: u32, b: u32, size: u32, off: u32| {
            let seam = b + size - off;
            [(lo, hi.min(seam).saturating_sub(lo), lo + off), (lo.max(seam), hi.saturating_sub(lo.max(seam)), (lo.max(seam) + off).wrapping_sub(size))]
        };
        for (sx, sw, kx) in spans(ix0, ix1, bx, bw, off[0]) {
            for (sy, sh, ky) in spans(iy0, iy1, by, bh, off[1]) {
                if sw > 0 && sh > 0 { out.push(([sx, sy, sw, sh], [kx, ky])); }
            }
        }
    }
}

/// One opaque colour over the viewport: the clear of a rectangle a partial frame redraws. Nothing
/// blends and nothing is computed for a pixel, where the uber shader's rect costs a rounded box,
/// a clip and a blend each.
const SOLID_WGSL: &str = "
struct In { @builtin(vertex_index) vi: u32, @location(2) fill: vec4<f32> };
struct Out { @builtin(position) pos: vec4<f32>, @location(0) @interpolate(flat) fill: vec4<f32> };
@vertex fn vs(i: In) -> Out {
    var o: Out;
    o.pos = vec4<f32>(f32(i.vi & 1u) * 2.0 - 1.0, 1.0 - f32(i.vi >> 1u) * 2.0, 0.0, 1.0);
    o.fill = i.fill;
    return o;
}
@fragment fn fs(i: Out) -> @location(0) vec4<f32> { return i.fill; }
";

fn solid_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("solid"), source: wgpu::ShaderSource::Wgsl(SOLID_WGSL.into()) });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("solid"),
        layout: None,
        vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[Some(metallib::uber_vertex_layout())] },
        primitive: metallib::uber_primitive(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn copy_at(t: &wgpu::Texture, x: u32, y: u32) -> wgpu::TexelCopyTextureInfo<'_> {
    let mut c = t.as_image_copy();
    c.origin = wgpu::Origin3d { x, y, z: 0 };
    c
}

fn extent(width: u32, height: u32) -> wgpu::Extent3d { wgpu::Extent3d { width, height, depth_or_array_layers: 1 } }

/// The renderer's side of a partial frame.
#[derive(Default)]
pub(super) struct DamageState {
    retained: Option<Retained>,
    armed: Vec<[u32; 4]>,
    /// For each armed rectangle, the instances whose quads reach it (`arm_cycle`), when only
    /// those are drawn there; empty: every instance is.
    runs: Vec<Vec<std::ops::Range<u32>>>,
    /// The paint and target size `arm_cycle` last found its rectangles and runs for, and them.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    cycle: Option<((u64, (u32, u32)), Result<Cull, &'static str>)>,
    /// The paint and target size `arm_loop` last found the units of, and them.
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    loop_units: Option<((u64, (u32, u32)), Result<std::sync::Arc<Units>, &'static str>)>,
    /// Whether a run of the armed rectangles holds a layer's scope.
    layers_reached: bool,
    clear: [f32; 4],
    /// Pixels the armed frame moves first, `([x, y, width, height], to)`: a scroll moving what it
    /// moved (`window::damage::Shift`).
    shift: Option<([u32; 4], [u32; 2])>,
    /// How the kept frame holds a scrolled viewport: rolled, so a scroll moves none of its pixels.
    roll: Option<Roll>,
    /// `runs` are of a frame whose list changed (`cull_armed`): they only say what to draw where,
    /// and its layers are planned and drawn as any frame's.
    listed: bool,
    /// The armed frame draws whole (`arm_whole`): into the kept frame, which is then current.
    behind: bool,
    /// The armed frame changes nothing (`arm_same`): no pass, only the copy out.
    same: bool,
    /// The armed frame's target still holds what this renderer last drew into it, and nothing
    /// drawn since (`arm_kept_target`).
    target_kept: bool,
    /// The last frame was drawn here, armed: its target holds that frame, all of it.
    target_drawn: bool,
    /// Where the kept frame, current otherwise, is behind such a target: what frames redrew on
    /// the target alone (`[x, y, width, height]` each).
    lag: Vec<[u32; 4]>,
    /// The armed frame's target holds, whole, the frame this renderer drew this many frames ago
    /// (`arm_aged_target`); 0: it holds nothing known.
    target_age: u32,
    /// What each of the last frames redrew in the kept frame, newest last (`[x, y, width, height]`
    /// each; none where nothing changed). `None`: a frame that changed the picture some other way
    /// (drawn whole, a scroll, drawn through the drawable, not armed).
    history: std::collections::VecDeque<Option<Vec<[u32; 4]>>>,
    /// This frame's entry for `history`, when it redrew rectangles of a current kept frame.
    redrew: Option<Vec<[u32; 4]>>,
    /// Fills a rectangle with one opaque colour, for the format it was made for (`solid_pipeline`).
    solid: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    drawn: Option<Drawn>,
    /// The armed rectangles are a loop-only frame's (`arm_loop`), for `RUNE_LOOP_CHECK=2`.
    pub(super) looped: bool,
    pub(super) pixel: Option<super::pixel_check::Check>,
}

impl Renderer {
    /// The next `draw` may redraw only `rects` (`[x, y, width, height]` each, target pixels, at
    /// most `MAX_RECTS`) of a `size` drawable of `format`, cleared to `clear`.
    /// With `shift`, the pixels at `shift.0` (`[x, y, width, height]`) move to `shift.1` first.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn arm_damage(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, rects: &[[u32; 4]], clear: [f32; 4], size: (u32, u32), format: wgpu::TextureFormat, shift: Option<([u32; 4], [u32; 2])>) {
        let mut d = self.damage.lock().unwrap();
        if d.retained.as_ref().is_none_or(|r| (r.texture.width(), r.texture.height()) != size || r.texture.format() != format) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("last frame"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[self.format],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor { format: Some(self.format), ..Default::default() });
            d.retained = Some(Retained { texture, view, current: false });
            d.roll = None;
        }
        // Where WGSL compiles as the renderer starts (Vulkan, GL); elsewhere the uber shader clears.
        if cfg!(not(any(target_os = "windows", target_os = "macos", target_os = "ios", target_arch = "wasm32"))) && d.solid.as_ref().is_none_or(|(f, _)| *f != self.format) {
            d.solid = Some((self.format, solid_pipeline(device, self.format)));
        }
        let fills: Vec<Instance> = rects.iter().take(MAX_ARMED).map(|&[x, y, w, h]| Instance {
            rect: [x as f32, y as f32, w as f32, h as f32],
            radii: [0.0; 4],
            fill: clear,
            border: [0.0; 4],
            params: [0.0; 4],
            uv: [0.0; 4],
            clip: [x as f32, y as f32, (x + w) as f32, (y + h) as f32],
            fade: [0; 4],
        }).collect();
        let quads = self.clear_quad.get_or_init(|| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("damage clear"),
            size: (std::mem::size_of::<Instance>() * MAX_ARMED) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        // Through the renderer's staging, as a layer's quad: `Queue::write_buffer` makes a buffer each call.
        if !fills.is_empty() { self.uploads.staging(device, queue).write_buffer(quads, 0, bytemuck::cast_slice(&fills)); }
        d.armed = rects.iter().take(MAX_ARMED).copied().collect();
        d.runs.clear();
        d.listed = false;
        d.behind = false;
        d.same = false;
        d.target_kept = false;
        d.target_age = 0;
        d.clear = clear;
        d.shift = shift;
        d.drawn = Some(Err("drawn by the compositor"));
    }

    /// A loop-only frame (`runtime/paint/cycle.rs`): only the instances at `slots` changed since
    /// the frame the kept texture holds, and only in colour. Arms the rectangles they draw in,
    /// each with only the instances whose quads reach it (what the rest drew there stays as
    /// drawn). `Err` says why this frame draws whole.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn arm_cycle(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, paint: u64, instances: &[Instance], slots: &[u32], clear: [f32; 4], size: (u32, u32), format: wgpu::TextureFormat) -> Result<(), &'static str> {
        // The clearing quad blends: only an opaque clear stands for `LoadOp::Clear`.
        if clear[3] != 1.0 { return Err("clear"); }
        if slots.is_empty() { return Err("no slots"); }
        self.redraws_in_place()?;
        let key = (paint, size);
        let cached = self.damage.lock().unwrap().cycle.as_ref().filter(|c| c.0 == key).map(|c| c.1.clone());
        let (rects, runs, layers) = match cached {
            Some(c) => c?,
            None => {
                let c = cycle_cull(instances, slots, size);
                self.damage.lock().unwrap().cycle = Some((key, c.clone()));
                c?
            }
        };
        self.arm_damage(device, queue, &rects, clear, size, format, None);
        let mut d = self.damage.lock().unwrap();
        d.runs = runs;
        d.layers_reached = layers;
        Ok(())
    }

    /// A window's loop-only frame (`Runtime::loop_step`): only the instances at `patched`
    /// (ascending) changed since the frame the kept texture and the instance buffer hold, both
    /// paint `paint`'s. Arms the pixels they drew on and draw on now, each with only the instances
    /// that reach it (what the rest drew there stays as drawn). Returns those rectangles (x0, y0,
    /// x1, y1) to present, or why this frame draws whole.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    pub(crate) fn arm_loop(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, paint: u64, instances: &[Instance], patched: &[u32], clear: [f32; 4], size: (u32, u32), format: wgpu::TextureFormat) -> Result<Vec<[i32; 4]>, &'static str> {
        // The clearing quad blends: only an opaque clear stands for `LoadOp::Clear`.
        if clear[3] != 1.0 { return Err("clear"); }
        self.redraws_in_place()?;
        let held = self.held_instances(paint, instances.len()).ok_or("buffer")?;
        let at = |i: &u32| *i as usize;
        // Only what differs from what the buffer holds, which is what the last frame drew (a
        // frame that missed its drawable uploaded nothing): a stepped spinner's dots between their
        // steps, while the shine drives frames, aren't redrawn; a dot a missed frame stepped is.
        let patched: Vec<u32> = patched.iter().copied().filter(|&i| bytemuck::bytes_of(&held[i as usize]) != bytemuck::bytes_of(&instances[i as usize])).collect();
        let patched = &patched[..];
        let mut rects = gather(patched.iter().map(at).flat_map(|i| [&held[i], &instances[i]]), size);
        // Nothing on screen changed: one pixel keeps the present, which DWM composes again as is.
        if rects.is_empty() { rects.push([0, 0, 1, 1]); }
        let key = (paint, size);
        let cached = self.damage.lock().unwrap().loop_units.as_ref().filter(|c| c.0 == key).map(|c| c.1.clone());
        let units = match cached {
            Some(units) => units?,
            None => {
                let units = units(instances, size).map(std::sync::Arc::new);
                self.damage.lock().unwrap().loop_units = Some((key, units.clone()));
                units?
            }
        };
        let (draw, runs, layers) = cull_runs(instances, &units, &rects, patched);
        self.arm_damage(device, queue, &draw, clear, size, format, None);
        if super::pixel_check::on() { self.pixel_check_arm(device, size, format); }
        let mut d = self.damage.lock().unwrap();
        d.runs = runs;
        d.layers_reached = layers;
        Ok(rects)
    }

    /// After `arm_damage`, for a frame drawing `instances` (the list uploaded for it): each armed
    /// rectangle draws only the instances that reach it. A frame's instances are a thousand quads
    /// to put through the vertex stage and bin for every rectangle, most of them nowhere near it.
    #[cfg_attr(any(target_os = "macos", target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn cull_armed(&self, instances: &[Instance], size: (u32, u32)) {
        let mut d = self.damage.lock().unwrap();
        if d.armed.is_empty() || instances.len() != self.count as usize { return; }
        let Ok(units) = units(instances, size) else { return };
        let rects: Vec<[i32; 4]> = d.armed.iter().map(|r| [r[0] as i32, r[1] as i32, (r[0] + r[2]) as i32, (r[1] + r[3]) as i32]).collect();
        let (_, runs, layers) = cull_runs(instances, &units, &rects, &[]);
        d.runs = runs;
        d.layers_reached = layers;
        d.listed = true;
    }

    /// After `arm_damage`, for a frame whose instances are the last frame's: the armed rectangle
    /// redraws nothing, since the kept frame holds it already.
    #[cfg_attr(any(target_os = "macos", target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn arm_same(&self) {
        let mut d = self.damage.lock().unwrap();
        d.runs = vec![Vec::new(); d.armed.len()];
        d.listed = true;
        d.same = true;
    }

    /// The next `draw` draws all of a `size` drawable of `format`, into the kept frame: a frame
    /// that changed too much to redraw in part. Drawn straight to the drawable it would leave the
    /// kept frame behind, and the frame after it would draw whole as well.
    #[cfg_attr(any(target_os = "macos", target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn arm_whole(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, clear: [f32; 4], size: (u32, u32), format: wgpu::TextureFormat) {
        self.arm_damage(device, queue, &[[0, 0, size.0, size.1]], clear, size, format, None);
        self.damage.lock().unwrap().behind = true;
    }

    /// After `arm_damage` or `arm_whole`: the texture the next `draw` draws into still holds what
    /// this renderer drew into it last frame, and nothing has drawn there since. The frame is then
    /// drawn on it as it stands, with no copy from the kept frame. For a target whose pixels last
    /// (an offscreen one), never a drawable a swap chain hands back with another frame's in it.
    #[cfg_attr(any(target_os = "macos", target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn arm_kept_target(&self) {
        self.damage.lock().unwrap().target_kept = true;
    }

    /// After `arm_damage` or `arm_whole`: the texture the next `draw` draws into holds, whole, the
    /// frame this renderer drew into a target `age` frames ago (1: the last frame), and nothing
    /// drawn since: a swap chain's buffer of a known age. The kept frame is then copied out only
    /// where the frames since redrew it, not all of it. For a host that knows its buffers' ages
    /// (`EGL_EXT_buffer_age`; a swap chain whose images keep their pixels); 0 says nothing is known.
    #[cfg_attr(any(target_os = "macos", target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn arm_aged_target(&self, age: u32) {
        self.damage.lock().unwrap().target_age = age;
    }

    /// The armed frame redraws its rectangles over a current kept frame, and no layer is among
    /// what it draws there: `prepare` leaves the layers as the last frame planned and drew them.
    pub(super) fn cycle_leaves_layers(&self) -> bool {
        let d = self.damage.lock().unwrap();
        !d.armed.is_empty() && !d.runs.is_empty() && !d.listed && !d.layers_reached && d.retained.as_ref().is_some_and(|r| r.current)
    }

    /// After every frame: whether it redrew only the armed rectangles, or why not. A frame that
    /// drew any other way leaves the kept frame behind.
    pub(crate) fn damage_drawn(&self) -> Drawn {
        let mut d = self.damage.lock().unwrap();
        d.armed.clear();
        d.shift = None;
        d.listed = false;
        d.behind = false;
        d.same = false;
        d.target_kept = false;
        d.target_age = 0;
        d.looped = false;
        let redrew = d.redrew.take();
        if d.history.len() == HISTORY { d.history.pop_front(); }
        d.history.push_back(redrew);
        if let Some(c) = d.pixel.as_mut() { c.frame_ended(); }
        let drawn = d.drawn.take().unwrap_or(Err("not armed"));
        if drawn.is_err() && let Some(r) = d.retained.as_mut() { r.current = false; }
        if drawn.is_err() { d.lag.clear(); }
        d.target_drawn = drawn.is_ok();
        drawn
    }

    /// Whether this frame can redraw only part of its target: nothing draws there but its own
    /// instances (no turned or blurred layer, effect slot, backdrop blur, native surface or GPU
    /// spring moves pixels the instances don't say).
    pub(super) fn redraws_in_place(&self) -> Drawn {
        if !self.effects.is_empty() { return Err("effect"); }
        if self.has_backdrop_blur() { return Err("backdrop blur"); }
        if self.springs.is_some() { return Err("springs"); }
        if !self.layers.in_place() { return Err("layer"); }
        if self.surfaces.as_ref().is_some_and(|s| !s.draws_nothing()) { return Err("surface"); }
        Ok(())
    }

    /// Rectangles are armed for the next `draw` (asked before it looks up its target's texture).
    pub(super) fn armed(&self) -> bool {
        !self.damage.lock().unwrap().armed.is_empty()
    }

    /// `draw`'s part, when armed: redraws the armed rectangles into the kept frame (all of it
    /// when it's behind) and copies it to `dst`. False when `draw` should draw whole.
    ///
    /// A scroll (`arm_damage`'s `shift`) goes the other way round, when `view` is `dst`'s and
    /// `dst` can be copied from: the kept frame is copied to `dst` first, moved (`Roll`), the
    /// rectangles are redrawn there, and only they are copied back. The frame then copies the
    /// window's pixels once, as any partial frame does, where moving them in the kept frame as
    /// well would copy them twice more.
    ///
    /// On a target that keeps its pixels (`arm_kept_target`) nothing is copied out: the frame is
    /// drawn on `dst` as it stands, all of it when it draws whole, and a scroll copies only the
    /// viewport from the kept frame, brought up to date from `dst` first.
    pub(crate) fn draw_armed(&self, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Texture, view: Option<&wgpu::TextureView>) -> bool {
        let mut guard = self.damage.lock().unwrap();
        let d = &mut *guard;
        let looped = std::mem::take(&mut d.looped);
        if d.armed.is_empty() { return false; }
        if let Err(why) = self.redraws_in_place() {
            d.drawn = Some(Err(why));
            return false;
        }
        let rects = std::mem::take(&mut d.armed);
        let runs = std::mem::take(&mut d.runs);
        let shift = d.shift.take();
        let layers_reached = d.layers_reached;
        let listed = std::mem::take(&mut d.listed);
        let behind = std::mem::take(&mut d.behind);
        let unchanged = std::mem::take(&mut d.same);
        let clear = d.clear;
        let (Some(quads), Some(kept)) = (self.clear_quad.get(), d.retained.as_mut()) else { return false };
        if behind { kept.current = false; }
        if (dst.width(), dst.height()) != (kept.texture.width(), kept.texture.height()) {
            d.drawn = Some(Err("size"));
            return false;
        }
        let size = [dst.width(), dst.height()];
        let solid = d.solid.as_ref().filter(|(format, _)| clear[3] == 1.0 && *format == self.format).map(|(_, p)| p);
        // frame-profile, `RUNE_LOOP_CHECK_SPOIL=1`: a loop-only frame skips its redraw, so the kept
        // picture goes stale and `RUNE_LOOP_CHECK=2` must find it (its failing row).
        #[cfg(feature = "frame-profile")]
        let spoil = looped && super::pixel_check::spoiled();
        #[cfg(not(feature = "frame-profile"))]
        let spoil = false;
        // No layer reaches the rectangles: their targets aren't drawn from, nor made again.
        // The uploads go into this encoder ahead of the pass either way (`encode_layers_timed`
        // flushes them too), not in a submission of their own: a submit costs the CPU about as
        // much as the rest of a loop-only frame.
        let uploads = |encoder: &mut wgpu::CommandEncoder| {
            if runs.is_empty() || layers_reached || listed { self.encode_layers_timed(encoder, None, &mut |_, _| {}); } else { self.flush_uploads(encoder); }
            self.submit_uploads();
        };
        let copy = |encoder: &mut wgpu::CommandEncoder, from: &wgpu::Texture, at: [u32; 2], to: &wgpu::Texture, to_at: [u32; 2], size: [u32; 2]| {
            encoder.copy_texture_to_texture(copy_at(from, at[0], at[1]), copy_at(to, to_at[0], to_at[1]), extent(size[0], size[1]));
        };
        let target_kept = std::mem::take(&mut d.target_kept);
        if let Some(view) = view.filter(|_| target_kept) {
            // The target holds the last frame: this one is drawn on it, and nothing is copied out.
            if behind || !d.target_drawn {
                // All of it, or a target whose last frame was drawn another way.
                self.draw_timed(encoder, view, clear, None);
                kept.current = false;
                d.lag.clear();
                d.roll = None;
                d.drawn = Some(Ok(()));
                return true;
            }
            if shift.is_some() && !dst.usage().contains(wgpu::TextureUsages::COPY_SRC) {
                kept.current = false;
                d.lag.clear();
                d.roll = None;
                d.drawn = Some(Err("no copies"));
                return false;
            }
            uploads(encoder);
            let mut pieces = Vec::new();
            if let Some((from, to)) = shift {
                // A scroll moves pixels through the kept frame, which first catches up with the
                // target: all of it, or where frames since redrew the target alone.
                if kept.current {
                    for rect in d.lag.drain(..) { Roll::pieces(d.roll, rect, &mut pieces); }
                    for (s, k) in pieces.drain(..) { copy(encoder, dst, [s[0], s[1]], &kept.texture, k, [s[2], s[3]]); }
                } else {
                    copy(encoder, dst, [0, 0], &kept.texture, [0, 0], size);
                    kept.current = true;
                    d.lag.clear();
                    d.roll = None;
                }
                let mut roll = Roll::moved(d.roll, from, to);
                if roll.is_none() {
                    // Another viewport's scroll: the kept frame is straightened from the target.
                    if let Some(Roll { band: [x, y, w, h], .. }) = d.roll { copy(encoder, dst, [x, y], &kept.texture, [x, y], [w, h]); }
                    roll = Roll::moved(None, from, to);
                }
                d.roll = roll;
                // The viewport, from where the roll now keeps each of its pixels: moved. The rest
                // of the target stands as it is.
                if let Some(Roll { band, .. }) = d.roll { Roll::pieces(d.roll, band, &mut pieces); }
                for (s, k) in pieces.drain(..) { copy(encoder, &kept.texture, k, dst, [s[0], s[1]], [s[2], s[3]]); }
                self.redraw(encoder, view, &rects, &runs, quads, solid, clear, size, spoil);
                // What was redrawn, back where the kept frame holds it.
                for &rect in &rects { Roll::pieces(d.roll, rect, &mut pieces); }
                for (s, k) in pieces.drain(..) { copy(encoder, dst, [s[0], s[1]], &kept.texture, k, [s[2], s[3]]); }
            } else if !unchanged {
                self.redraw(encoder, view, &rects, &runs, quads, solid, clear, size, spoil);
                // The kept frame is behind there now: one rectangle once they are many, since
                // each is a copy of its own when a scroll next needs the kept frame.
                if kept.current {
                    d.lag.extend_from_slice(&rects);
                    if d.lag.len() > LAG_RECTS {
                        let all = d.lag.drain(..).reduce(|a, b| {
                            let (x, y) = (a[0].min(b[0]), a[1].min(b[1]));
                            [x, y, (a[0] + a[2]).max(b[0] + b[2]) - x, (a[1] + a[3]).max(b[1] + b[3]) - y]
                        });
                        d.lag.extend(all);
                    }
                }
            }
            d.drawn = Some(Ok(()));
            return true;
        }
        // Behind a target in places, the kept frame isn't the last frame.
        if !d.lag.is_empty() {
            kept.current = false;
            d.lag.clear();
        }
        let partial = kept.current;
        let age = std::mem::take(&mut d.target_age) as usize;
        // Through the drawable: a scroll, and every frame after it while the kept frame is rolled.
        let through = kept.current && (shift.is_some() || d.roll.is_some());
        let through_view = view.filter(|_| dst.usage().contains(wgpu::TextureUsages::COPY_SRC));
        if through && through_view.is_none() {
            // The rectangles were planned for moved pixels, which this drawable can't give back.
            d.roll = None;
            d.drawn = Some(Err("no copies"));
            return false;
        }
        if kept.current {
            uploads(encoder);
            let mut pieces = Vec::new();
            if through {
                if let Some((from, to)) = shift {
                    let mut roll = Roll::moved(d.roll, from, to);
                    if roll.is_none() {
                        // Another viewport's scroll: the kept frame is straightened first, through
                        // the drawable, which this frame overwrites anyway.
                        Roll::pieces(d.roll, [0, 0, size[0], size[1]], &mut pieces);
                        for (s, k) in pieces.drain(..) { copy(encoder, &kept.texture, k, dst, [s[0], s[1]], [s[2], s[3]]); }
                        if let Some(Roll { band: [x, y, w, h], .. }) = d.roll { copy(encoder, dst, [x, y], &kept.texture, [x, y], [w, h]); }
                        roll = Roll::moved(None, from, to);
                    }
                    d.roll = roll;
                }
                // A target that holds an older frame is behind the kept frame only where the frames
                // since redrew it (this frame's rectangles are drawn on it below); a scroll moved
                // the viewport's every pixel, and its frame takes the whole kept frame.
                match aged(&d.history, age, &[]).filter(|_| shift.is_none()) {
                    Some(past) => for rect in past { Roll::pieces(d.roll, rect, &mut pieces); },
                    None => Roll::pieces(d.roll, [0, 0, size[0], size[1]], &mut pieces),
                }
                for (s, k) in pieces.drain(..) { copy(encoder, &kept.texture, k, dst, [s[0], s[1]], [s[2], s[3]]); }
            }
            // Nothing changed: no pass at all, where one would load and store a target to draw nothing.
            if !unchanged { self.redraw(encoder, through_view.filter(|_| through).unwrap_or(&kept.view), &rects, &runs, quads, solid, clear, size, spoil); }
            if through {
                // What was redrawn, back where the kept frame holds it.
                for &rect in rects.iter().filter(|_| !unchanged) { Roll::pieces(d.roll, rect, &mut pieces); }
                for (s, k) in pieces.drain(..) { copy(encoder, dst, [s[0], s[1]], &kept.texture, k, [s[2], s[3]]); }
                d.redrew = shift.is_none().then(|| if unchanged { Vec::new() } else { rects.clone() });
                d.drawn = Some(Ok(()));
                return true;
            }
        } else {
            // Behind: this frame draws whole, into the kept frame.
            self.draw_timed(encoder, &kept.view, clear, None);
            kept.current = true;
            d.roll = None;
        }
        // `RUNE_LOOP_CHECK=2`: this loop-only frame's picture, and the whole frame's beside it.
        if looped && partial && let Some(c) = d.pixel.as_mut() { self.pixel_check_encode(encoder, c, &kept.texture, clear, &rects); }
        // What this frame redrew in a kept frame that was current, for the frames after it.
        let redrew = partial.then(|| if unchanged { Vec::new() } else { rects.clone() });
        // A target that holds an older frame of this renderer's differs from the kept frame only
        // where the frames since redrew it, this one among them: those rectangles are copied out,
        // where a target that holds nothing known takes the whole kept frame.
        match redrew.as_ref().and_then(|now| aged(&d.history, age, now)) {
            Some(all) => for [x, y, w, h] in all { copy(encoder, &kept.texture, [x, y], dst, [x, y], [w, h]); },
            None => copy(encoder, &kept.texture, [0, 0], dst, [0, 0], size),
        }
        d.redrew = redrew;
        d.drawn = Some(Ok(()));
        true
    }

    /// Redraws `rects` of `target`, which holds the frame before: each cleared to `clear`, then
    /// drawn with what reaches it (`runs`), or with every instance where there are none.
    #[allow(clippy::too_many_arguments)]
    fn redraw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, rects: &[[u32; 4]], runs: &[Vec<std::ops::Range<u32>>], quads: &wgpu::Buffer,
        solid: Option<&wgpu::RenderPipeline>, clear: [f32; 4], size: [u32; 2], spoil: bool) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("frame (damage)"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        // An opaque clear is the pass's own clear over the rectangle: one colour, nothing blended, and
        // the rects at the head of the list that paint only that colour change nothing after it
        // (`cleared_start`, as in a frame drawn whole).
        let opaque = clear[3] == 1.0;
        let start = if opaque { self.cleared_start(wgpu::LoadOp::Clear(wgpu::Color { r: clear[0] as f64, g: clear[1] as f64, b: clear[2] as f64, a: clear[3] as f64 })) } else { 0 };
        for (n, &rect) in rects.iter().enumerate().filter(|_| !spoil) {
            let [x, y, w, h] = rect;
            pass.set_scissor_rect(x, y, w, h);
            pass.set_vertex_buffer(0, quads.slice(..));
            match solid {
                Some(solid) => {
                    pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
                    pass.set_pipeline(solid);
                    pass.draw(0..4, n as u32..n as u32 + 1);
                    pass.set_viewport(0.0, 0.0, size[0] as f32, size[1] as f32, 0.0, 1.0);
                }
                None => {
                    pass.set_pipeline(self.draw_pipeline(self.format, None));
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.draw(0..4, n as u32..n as u32 + 1);
                }
            }
            let mut active = ActivePass::borrowed_within(&mut pass, rect);
            active.set_scissor(None, size);
            match runs.get(n) {
                // Only what reaches the rectangle (`arm_cycle`): none of the rest draws there.
                Some(runs) => for run in runs {
                    let run = run.start.max(start)..run.end;
                    if !run.is_empty() { self.draw_layer_range(&mut active, self.format, run, None, &mut |_, _| {}); }
                },
                None => self.draw_layer_range(&mut active, self.format, start..self.count, None, &mut |_, _| {}),
            }
        }
    }
}

/// Where a target that holds the frame of `age` frames ago differs from the kept frame: what the
/// frames since redrew (`history`, newest last) and `now`, this frame's. `None` when one of them
/// changed the picture in a way its rectangles don't say, or the age isn't known (0) or remembered.
fn aged(history: &std::collections::VecDeque<Option<Vec<[u32; 4]>>>, age: usize, now: &[[u32; 4]]) -> Option<Vec<[u32; 4]>> {
    if age == 0 || age - 1 > history.len() { return None; }
    let mut all = now.to_vec();
    for past in history.iter().rev().take(age - 1) { all.extend_from_slice(past.as_ref()?); }
    if all.len() > AGED_RECTS {
        let bounds = all.drain(..).reduce(|a, b| {
            let (x, y) = (a[0].min(b[0]), a[1].min(b[1]));
            [x, y, (a[0] + a[2]).max(b[0] + b[2]) - x, (a[1] + a[3]).max(b[1] + b[3]) - y]
        });
        all.extend(bounds);
    }
    Some(all)
}

/// A loop-only frame's rectangles, the runs drawn in each, and whether a layer is among them.
type Cull = (Vec<[u32; 4]>, Vec<Vec<std::ops::Range<u32>>>, bool);

/// The rectangles a loop-only frame redraws (each slot's quad 2 px wider, merged with those it
/// comes within 8 px of, `MAX_RECTS` at most) and, for each, runs of what reaches it, a pixel
/// wider still: an instance, or a layer's whole scope (its own bounds and all it holds), which
/// `draw_layer_range` draws as one.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn cycle_cull(instances: &[Instance], slots: &[u32], size: (u32, u32)) -> Result<Cull, &'static str> {
    let quads = slots.iter().map(|&s| instances.get(s as usize).ok_or("slot")).collect::<Result<Vec<_>, _>>()?;
    let rects = gather(quads.into_iter(), size);
    if rects.is_empty() { return Err("nothing drawn"); }
    let units = units(instances, size)?;
    Ok(cull_runs(instances, &units, &rects, &[]))
}

/// The pixels `quads` draw on, each 2 px wider, merged with those it comes within 8 px of,
/// `MAX_RECTS` at most (x0, y0, x1, y1).
fn gather<'a>(quads: impl Iterator<Item = &'a Instance>, size: (u32, u32)) -> Vec<[i32; 4]> {
    let (w, h) = (size.0 as i32, size.1 as i32);
    let empty = |b: &[i32; 4]| b[2] <= b[0] || b[3] <= b[1];
    let near = |a: &[i32; 4], b: &[i32; 4]| a[0] <= b[2] + 8 && b[0] <= a[2] + 8 && a[1] <= b[3] + 8 && b[1] <= a[3] + 8;
    let union = |a: [i32; 4], b: [i32; 4]| if empty(&a) { b } else if empty(&b) { a } else { [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])] };
    let mut rects: Vec<[i32; 4]> = Vec::new();
    for q in quads {
        let b = q.pixel_bounds(w, h);
        if empty(&b) { continue; }
        let mut b = [(b[0] - 2).max(0), (b[1] - 2).max(0), (b[2] + 2).min(w), (b[3] + 2).min(h)];
        while let Some(k) = rects.iter().position(|r| near(r, &b)) { b = union(b, rects.swap_remove(k)); }
        rects.push(b);
    }
    while rects.len() > MAX_RECTS {
        let (Some(a), Some(b)) = (rects.pop(), rects.pop()) else { break };
        rects.push(union(a, b));
    }
    rects
}

/// What a frame draws, in units: an instance, or a layer's whole scope (its own bounds and all it
/// holds), which `draw_layer_range` draws as one, each with its pixel bounds.
type Units = Vec<(std::ops::Range<u32>, [i32; 4])>;

fn units(instances: &[Instance], size: (u32, u32)) -> Result<Units, &'static str> {
    let (w, h) = (size.0 as i32, size.1 as i32);
    let bounds = |i: &Instance| i.pixel_bounds(w, h);
    let empty = |b: &[i32; 4]| b[2] <= b[0] || b[3] <= b[1];
    let union = |a: [i32; 4], b: [i32; 4]| if empty(&a) { b } else if empty(&b) { a } else { [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])] };
    let mut units: Units = Vec::new();
    let mut n = 0;
    while n < instances.len() {
        let start = n;
        let mut b = bounds(&instances[n]);
        if instances[n].kind() == Some(Kind::Layer) {
            let mut depth = 1;
            while depth > 0 {
                n += 1;
                let i = instances.get(n).ok_or("open layer")?;
                b = union(b, bounds(i));
                match i.kind() {
                    Some(Kind::Layer) => depth += 1,
                    Some(Kind::LayerEnd) => depth -= 1,
                    _ => {}
                }
            }
        }
        n += 1;
        units.push((start as u32..n as u32, b));
    }
    Ok(units)
}

/// For each of `rects`, runs of the units that reach it, a pixel wider still, and of any unit
/// holding one of `always` (ascending): instances whose bounds were found before they moved.
fn cull_runs(instances: &[Instance], units: &Units, rects: &[[i32; 4]], always: &[u32]) -> Cull {
    let empty = |b: &[i32; 4]| b[2] <= b[0] || b[3] <= b[1];
    let holds = |u: &std::ops::Range<u32>| always.get(always.partition_point(|&a| a < u.start)).is_some_and(|&a| a < u.end);
    let runs: Vec<Vec<std::ops::Range<u32>>> = rects.iter().map(|r| {
        let mut runs: Vec<std::ops::Range<u32>> = Vec::new();
        for (unit, b) in units {
            let reaches = !empty(b) && b[0] - 1 < r[2] && r[0] < b[2] + 1 && b[1] - 1 < r[3] && r[1] < b[3] + 1;
            if !reaches && !holds(unit) { continue; }
            // A short gap is drawn through: what's in it doesn't reach the rectangle, and one
            // draw costs less than two.
            match runs.last_mut() {
                Some(run) if unit.start - run.end <= 16 => run.end = unit.end,
                _ => runs.push(unit.clone()),
            }
        }
        runs
    }).collect();
    let layers = units.iter().any(|(u, _)| instances[u.start as usize].kind() == Some(Kind::Layer) && runs.iter().flatten().any(|r| r.start <= u.start && u.end <= r.end));
    (rects.iter().map(|r| [r[0] as u32, r[1] as u32, (r[2] - r[0]) as u32, (r[3] - r[1]) as u32]).collect(), runs, layers)
}

#[cfg(test)]
#[path = "damaged_tests.rs"]
mod tests;
