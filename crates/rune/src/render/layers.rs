//! Bounded painter scopes. The ordinary path retains its instance buffer, batches and pass;
//! only scopes requiring isolation acquire a target. Scope records are also compositor fences.
use super::*;
#[cfg(test)]
use bytemuck::Zeroable;
use std::ops::Range;

mod plan;
pub(super) use plan::is_scissor_clip;
use plan::{Describe, Disposition, Held, Kept, Layer, Move, Planned, Planner, Pools};

// Linear premultiplied intermediates preserve HDR colours and avoid accumulating 8-bit
// quantization through nested layers. The destination format is applied only at composition.
pub(super) const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Default offscreen texture budget for [`El::cache`](crate::El::cache) subtrees (64 MiB).
pub const DEFAULT_CACHE_BUDGET: u64 = 64 * 1024 * 1024;
/// What spare layer targets may hold (Layers::spare): a full-window layer at 2880x1800 or two menus' worth.
const SPARE_BYTES: u64 = 32 * 1024 * 1024;
/// Maximum width or height of a cached subtree's offscreen texture in device pixels.
const MAX_CACHE_DIM: u32 = 4096;

#[cfg(test)]
#[path = "layers/plan_frame_tests.rs"]
mod plan_frame_tests;
#[cfg(test)]
#[path = "../runtime/layer_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "../runtime/layer_cache_tests.rs"]
mod cache_tests;
#[cfg(test)]
#[path = "layer_filter_tests.rs"]
mod filter_tests;
#[cfg(test)]
#[path = "../runtime/rounded_clip_tests.rs"]
mod rounded_clip_tests;
#[cfg(test)]
#[path = "continuous_corner_tests.rs"]
mod continuous_corner_tests;
#[cfg(test)]
#[path = "layer_show_tests.rs"]
mod show_tests;
#[cfg(test)]
#[path = "layer_rebind_tests.rs"]
mod rebind_tests;

pub(super) struct Layers {
    plan: Vec<Layer>,
    /// What each layer does this frame, and the moves the target pools take to get there.
    planner: Planner,
    pub targets: Vec<Option<Target>>,
    retained: Vec<Target>,
    /// Targets no layer drew into this frame (a menu closed, a layer's place in the plan moved),
    /// kept for the next layer of their size: opening a menu or a popover again makes no texture.
    spare: Vec<Target>,
    stack: Vec<usize>,
    pub(super) pipelines: Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    /// The isolation pipelines being made ahead of the first layer ([`warm_layers`]).
    warming: Warming,
    pub additive: Option<(wgpu::RenderPipeline, wgpu::RenderPipeline)>,
    pub blur: Option<layer_blur::Pipeline>,
    viewport: (u32, u32),
    pub(super) image_generation: u64,
    allocations: u64,
    /// The atlas and outline texture generation: every target's base bind group names the
    /// generation it was made against, and a parked target is bound again only when a layer takes
    /// it in a later one.
    bindings: u64,
    cache_budget: u64,
    frame_stamp: u64,
    /// Bumped whenever texels a layer's instances may sample are written (glyph atlas rows and
    /// fills, outline records, image pages): a target that holds a layer's picture holds it for
    /// one such generation (`Holds`).
    pub(super) texels: u64,
    /// Test-only: the most bytes of live targets and their blur scratches a frame has held at once.
    #[cfg(test)]
    peak: u64,
    /// Test-only: how many times a target's base bind group has been made again.
    #[cfg(test)]
    pub(super) binds: u64,
}

impl Layers {
    /// Whether every group of the frame lands where its instances say: drawn straight into its
    /// parent's pass, or into a target composited back at its own place, neither turned nor
    /// blurred, so a changed instance changes only the pixels its bounds cover
    /// (`Renderer::redraws_in_place`).
    #[cfg(not(target_os = "macos"))]
    pub(super) fn in_place(&self) -> bool {
        self.plan.iter().enumerate().all(|(i, l)| match self.planner.dispositions.get(i) {
            Some(Disposition::Inline | Disposition::Skip) | None => true,
            Some(Disposition::Render { .. } | Disposition::Cached { .. }) => l.turn.is_none() && !l.free && l.blur == 0.0,
        })
    }

    /// Kept textures get the budget left after this frame's visible caches.
    fn trim_retained(&mut self, active_bytes: u64) {
        let available = self.cache_budget.saturating_sub(active_bytes);
        let mut bytes: u64 = self.retained.iter().map(Renderer::target_bytes).sum();
        while bytes > available {
            let oldest = self.retained.iter().enumerate().min_by_key(|(_, t)| t.last_used).unwrap().0;
            let evicted = self.retained.swap_remove(oldest);
            bytes -= Renderer::target_bytes(&evicted);
        }
    }

    /// Keeps `t` for a later layer of its size, within `SPARE_BYTES`; the least recently used go.
    fn park(&mut self, mut t: Target) {
        t.cache = None;
        self.spare.push(t);
        self.trim_spare(SPARE_BYTES);
    }

    /// Drops the least recently used spare targets until they hold at most `limit` bytes.
    fn trim_spare(&mut self, limit: u64) {
        let mut bytes: u64 = self.spare.iter().map(Renderer::target_bytes).sum();
        while bytes > limit {
            let oldest = self.spare.iter().enumerate().min_by_key(|(_, t)| t.last_used).unwrap().0;
            bytes -= Renderer::target_bytes(&self.spare.swap_remove(oldest));
        }
    }

    /// Test-only: notes the peak of the bytes this frame holds live, targets and blur scratches
    /// alike, so a test can pin it against the peak dev held.
    #[cfg(test)]
    fn watch_peak(&mut self) {
        let bytes: u64 = self.targets.iter().flatten().chain(&self.retained).chain(&self.spare)
            .map(Renderer::target_bytes).sum();
        self.peak = self.peak.max(bytes);
    }

    /// Test-only: the most live target bytes the last frame held at once.
    #[cfg(test)]
    pub(super) fn peak_bytes(&self) -> u64 { self.peak }

    /// Makes every blurred layer write its blur uniforms again on its next prepare.
    pub(super) fn forget_blur_uniforms(&mut self) {
        let targets = self.targets.iter_mut().flatten().chain(&mut self.retained).chain(&mut self.spare);
        for blur in targets.filter_map(|t| t.blur.as_mut()) { blur.forget_uniforms(); }
    }
}

impl Default for Layers {
    fn default() -> Self {
        Self {
            plan: Vec::new(),
            planner: Planner::default(),
            targets: Vec::new(),
            retained: Vec::new(),
            spare: Vec::new(),
            stack: Vec::new(),
            pipelines: Vec::new(),
            warming: Warming::Unasked,
            additive: None,
            blur: None,
            viewport: (0, 0),
            image_generation: 0,
            allocations: 0,
            bindings: 0,
            cache_budget: DEFAULT_CACHE_BUDGET,
            frame_stamp: 0,
            texels: 0,
            #[cfg(test)]
            peak: 0,
            #[cfg(test)]
            binds: 0,
        }
    }
}

pub(super) struct Target {
    identity: u64,
    composite_parent: Option<u64>,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    projection: wgpu::Buffer,
    quad: wgpu::Buffer,
    composite: wgpu::BindGroup,
    pub base: wgpu::BindGroup,
    pub images: Vec<Option<wgpu::BindGroup>>,
    pub glyphs: Vec<wgpu::BindGroup>,
    image_generation: u64,
    blur: Option<layer_blur::Scratch>,
    /// The node and signature of the cached content, if this target is kept.
    cache: Option<(u64, u64)>,
    last_used: u64,
    content_size: [f32; 2],
    /// The `Layers::bindings` generation the base bind group was made against.
    binding: u64,
    /// What the texture holds, when the last pass into it drew a plain layer (`Holds`).
    holds: std::sync::Mutex<Option<Holds>>,
    /// What it will hold once this frame's pass has drawn into it.
    pending: std::sync::Mutex<Option<Holds>>,
    /// This frame's layer is what the texture holds already: no pass draws it again, and the
    /// frame only composites it (a fade changes a layer's opacity every frame, and nothing in it).
    unchanged: std::sync::atomic::AtomicBool,
    /// When the layer has an opaque base fill covering `rect`, its clear colour (lets full redraws
    /// use `LoadOp::Clear(bg)` and skip the base rect, and partial updates use `LoadOp::Load`).
    opaque_clear: Option<[f32; 4]>,
    /// Layer-local dirty scissor `[x, y, w, h]` when only a few instances inside an opaque-base
    /// layer changed since the last frame (e.g. a small spinner inside a full-screen video layer).
    dirty_rect: Option<[u32; 4]>,
    /// True when this top-level layer sits directly over the window's opaque clear colour with no
    /// overlapping instances drawn before it, allowing unblended `fast_layer_over_clear` compositing.
    over_clear: bool,
    /// Test-only: the outline texture's height the base bind group was made against.
    #[cfg(test)]
    outline_rows: u32,
}

/// The picture a layer's target holds: the instances drawn into it, where its origin was, and
/// the generations of everything else they draw from. A layer that asks for the same again is
/// composited from the texture as it stands.
struct Holds {
    rect: [f32; 4],
    viewport: (u32, u32),
    texels: u64,
    images: u64,
    binding: u64,
    instances: Vec<Instance>,
}

impl Holds {
    fn is(&self, rect: [f32; 4], viewport: (u32, u32), texels: u64, images: u64, binding: u64, instances: &[Instance]) -> bool {
        if self.rect[2] != rect[2] || self.rect[3] != rect[3]
            || self.viewport != viewport || self.texels != texels || self.images != images || self.binding != binding
            || self.instances.len() != instances.len()
        {
            return false;
        }
        if self.rect[0] == rect[0] && self.rect[1] == rect[1] {
            return bytemuck::cast_slice::<Instance, u8>(&self.instances) == bytemuck::cast_slice::<Instance, u8>(instances);
        }
        // A layer that only translated (e.g. an offset+opacity arrive transition) draws the exact
        // same texture pixels when every child instance shifted by the same delta, because the
        // layer projection subtracts `rect.xy`.
        let (dx, dy) = (rect[0] - self.rect[0], rect[1] - self.rect[1]);
        self.instances.iter().zip(instances).all(|(a, b)| {
            if !matches!(a.kind(), Some(Kind::Rect | Kind::Shadow | Kind::Glyph | Kind::Inset | Kind::Gradient | Kind::Image | Kind::Dashed | Kind::Radial)) {
                return false;
            }
            let clip_ok = (a.clip == b.clip && (a.clip[0] < -1e5
                || (a.clip[0] <= a.rect[0] && a.clip[1] <= a.rect[1]
                    && a.clip[2] >= a.rect[0] + a.rect[2] && a.clip[3] >= a.rect[1] + a.rect[3]
                    && b.clip[0] <= b.rect[0] && b.clip[1] <= b.rect[1]
                    && b.clip[2] >= b.rect[0] + b.rect[2] && b.clip[3] >= b.rect[1] + b.rect[3])))
                || ((b.clip[0] - a.clip[0] - dx).abs() <= 1e-3
                    && (b.clip[1] - a.clip[1] - dy).abs() <= 1e-3
                    && (b.clip[2] - a.clip[2] - dx).abs() <= 1e-3
                    && (b.clip[3] - a.clip[3] - dy).abs() <= 1e-3);
            a.rect[2] == b.rect[2] && a.rect[3] == b.rect[3]
                && (b.rect[0] - a.rect[0] - dx).abs() <= 1e-3
                && (b.rect[1] - a.rect[1] - dy).abs() <= 1e-3
                && clip_ok
                && a.radii == b.radii
                && a.fill == b.fill
                && a.border == b.border
                && a.params == b.params
                && a.uv == b.uv
                && a.fade == b.fade
        })
    }
}

fn opaque_base(first: Option<&Instance>, rect: [f32; 4]) -> Option<[f32; 4]> {
    let i = first?;
    if i.kind() != Some(Kind::Rect)
        || i.fill[3] != 1.0
        || i.params[0] != 0.0
        || i.params[2] > 0.5
        || i.radii != [0.0; 4]
        || i.fade != [0; 4]
    {
        return None;
    }
    let (rx1, ry1) = (rect[0] + rect[2], rect[1] + rect[3]);
    let covers = i.rect[0] <= rect[0]
        && i.rect[1] <= rect[1]
        && i.rect[0] + i.rect[2] >= rx1
        && i.rect[1] + i.rect[3] >= ry1
        && i.clip[0] <= rect[0]
        && i.clip[1] <= rect[1]
        && i.clip[2] >= rx1
        && i.clip[3] >= ry1;
    covers.then_some(i.fill)
}

fn opaque_interior_covers(first: Option<&Instance>, rect: [f32; 4], dirty: [u32; 4]) -> bool {
    let Some(i) = first else { return false };
    if i.kind() != Some(Kind::Rect) || i.fill[3] != 1.0 || i.params[2] > 0.5 || i.fade != [0; 4] {
        return false;
    }
    let inset = (i.radii.iter().copied().fold(0.0f32, f32::max).max(i.params[0]) + 1.0).ceil();
    let ix0 = (i.rect[0] + inset).max(i.clip[0]);
    let iy0 = (i.rect[1] + inset).max(i.clip[1]);
    let ix1 = (i.rect[0] + i.rect[2] - inset).min(i.clip[2]);
    let iy1 = (i.rect[1] + i.rect[3] - inset).min(i.clip[3]);
    let dx0 = rect[0] + dirty[0] as f32;
    let dy0 = rect[1] + dirty[1] as f32;
    let dx1 = dx0 + dirty[2] as f32;
    let dy1 = dy0 + dirty[3] as f32;
    ix0 <= dx0 && iy0 <= dy0 && ix1 >= dx1 && iy1 >= dy1
}

fn partial_layer_dirty(holds: &Holds, rect: [f32; 4], viewport: (u32, u32), texels: u64, images: u64, binding: u64, instances: &[Instance]) -> Option<[u32; 4]> {
    if holds.rect != rect || holds.viewport != viewport || holds.texels != texels || holds.images != images || holds.binding != binding || holds.instances.len() != instances.len() {
        return None;
    }
    let (rw, rh) = (rect[2] as u32, rect[3] as u32);
    if rw == 0 || rh == 0 { return None; }
    let mut changed = 0usize;
    let mut box_ = [u32::MAX, u32::MAX, 0u32, 0u32];
    let mut grow = |i: &Instance| {
        let Some((x0, y0, x1, y1)) = i.damage_quad_bounds() else { return };
        let lx0 = ((x0 - rect[0] - 1.0).floor().max(0.0) as u32).min(rw);
        let ly0 = ((y0 - rect[1] - 1.0).floor().max(0.0) as u32).min(rh);
        let lx1 = ((x1 - rect[0] + 1.0).ceil().max(0.0) as u32).min(rw);
        let ly1 = ((y1 - rect[1] + 1.0).ceil().max(0.0) as u32).min(rh);
        if lx1 > lx0 && ly1 > ly0 {
            box_[0] = box_[0].min(lx0);
            box_[1] = box_[1].min(ly0);
            box_[2] = box_[2].max(lx1);
            box_[3] = box_[3].max(ly1);
        }
    };
    for (a, b) in holds.instances.iter().zip(instances) {
        if bytemuck::bytes_of(a) != bytemuck::bytes_of(b) {
            changed += 1;
            if changed > 16 { return None; }
            grow(a);
            grow(b);
        }
    }
    if changed == 0 || box_[2] <= box_[0] || box_[3] <= box_[1] { return None; }
    let dw = box_[2] - box_[0];
    let dh = box_[3] - box_[1];
    if u64::from(dw) * u64::from(dh) * 4 >= u64::from(rw) * u64::from(rh) { return None; }
    Some([box_[0], box_[1], dw, dh])
}

fn disjoint_before(instances: &[Instance], from: usize, to: usize, rect: [f32; 4]) -> bool {
    let (rx0, ry0, rx1, ry1) = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]);
    for i in &instances[from..to] {
        if matches!(i.kind(), Some(Kind::Layer | Kind::LayerEnd)) {
            continue;
        }
        let Some((x0, y0, x1, y1)) = i.damage_quad_bounds() else { continue };
        let c = i.clip;
        let (x0, y0, x1, y1) = (x0.max(c[0]), y0.max(c[1]), x1.min(c[2]), y1.min(c[3]));
        if x1 > x0 && y1 > y0 && x0 < rx1 && rx0 < x1 && y0 < ry1 && ry0 < y1 {
            return false;
        }
    }
    true
}

/// Transforms a premultiplied colour `c` for affine in-place layer inlining at group `opacity`
/// over an opaque `bg` colour: `T(S, α) = (opacity * S + α * (1 - opacity) * bg, α)`.
#[inline]
fn affine_color(c: [f32; 4], opacity: f32, bg: [f32; 4]) -> [f32; 4] {
    let k = c[3] * (1.0 - opacity);
    [
        c[0] * opacity + bg[0] * k,
        c[1] * opacity + bg[1] * k,
        c[2] * opacity + bg[2] * k,
        c[3],
    ]
}

/// Inlines top-level plain `SourceOver` layers that have an opaque full-layer base and sit
/// directly over the window's opaque clear colour (no overlapping earlier instances, no nested
/// layers/images/effects) by applying the exact affine colour transform `T(S, α)` to each child.
/// Eliminates the offscreen texture, second render pass, and full-layer `textureLoad` composite.
pub(super) fn inline_affine_layers(instances: &[Instance], viewport: (u32, u32)) -> std::borrow::Cow<'_, [Instance]> {
    if !instances.iter().any(|i| i.kind() == Some(Kind::Layer)) {
        return std::borrow::Cow::Borrowed(instances);
    }
    let cleared = super::cleared::Cleared::of(instances, viewport);
    if cleared.len == 0 || cleared.color[3] != 1.0 {
        return std::borrow::Cow::Borrowed(instances);
    }
    let bg = cleared.color;
    let cleared_len = cleared.len as usize;
    let mut out: Option<Vec<Instance>> = None;
    let mut i = 0;
    let mut depth = 0usize;
    while i < instances.len() {
        match instances[i].kind() {
            Some(Kind::Layer) => {
                if depth == 0 {
                    let begin = i;
                    let mut end = begin + 1;
                    let mut d = 1usize;
                    while end < instances.len() && d > 0 {
                        match instances[end].kind() {
                            Some(Kind::Layer) => d += 1,
                            Some(Kind::LayerEnd) => d -= 1,
                            _ => {}
                        }
                        if d > 0 { end += 1; }
                    }
                    let l = &instances[begin];
                    let opacity = l.fill[3];
                    let clip_ok = l.clip[0] < -1e5 || l.clip.iter().all(|v| v.fract() == 0.0);
                    let vis_rect = if l.clip[0] < -1e5 {
                        l.rect
                    } else {
                        let x0 = l.rect[0].max(l.clip[0]);
                        let y0 = l.rect[1].max(l.clip[1]);
                        let x1 = (l.rect[0] + l.rect[2]).min(l.clip[2]);
                        let y1 = (l.rect[1] + l.rect[3]).min(l.clip[3]);
                        [x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0)]
                    };
                    let qualifies = end < instances.len()
                        && opacity > 0.0
                        && opacity < 1.0
                        && l.params[0] == 0.0
                        && l.params[2] == 0.0
                        && l.params[3] == 0.0
                        && l.border[1] == 0.0
                        && l.fade == [0; 4]
                        && l.radii[3] == 0.0
                        && clip_ok
                        && begin >= cleared_len
                        && disjoint_before(instances, cleared_len, begin, vis_rect)
                        && instances[begin + 1..end].iter().all(|c| match c.kind() {
                            Some(Kind::Rect | Kind::Shadow | Kind::Glyph | Kind::Inset | Kind::Gradient | Kind::Dashed | Kind::Radial | Kind::LayerEnd) => true,
                            Some(Kind::Layer) => {
                                c.fill[3] >= 1.0
                                    && c.params[0] == 0.0
                                    && c.params[2] == 0.0
                                    && c.params[3] == 0.0
                                    && c.border[1] == 0.0
                                    && c.fade == [0; 4]
                                    && c.radii[3] == 0.0
                                    && (c.clip[0] < -1e5 || c.clip.iter().all(|v| v.fract() == 0.0))
                            }
                            _ => false,
                        });
                    if qualifies {
                        let buf = out.get_or_insert_with(|| instances.to_vec());
                        buf[begin].fill[3] = 1.0;
                        for c in &mut buf[begin + 1..end] {
                            if matches!(c.kind(), Some(Kind::Layer | Kind::LayerEnd)) {
                                continue;
                            }
                            c.fill = affine_color(c.fill, opacity, bg);
                            if matches!(c.kind(), Some(Kind::Rect | Kind::Gradient | Kind::Dashed | Kind::Radial | Kind::Inset)) {
                                c.border = affine_color(c.border, opacity, bg);
                            }
                        }
                    }
                    i = end + 1;
                    continue;
                }
                depth += 1;
            }
            Some(Kind::LayerEnd) => {
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
        i += 1;
    }
    match out {
        Some(v) => std::borrow::Cow::Owned(v),
        None => std::borrow::Cow::Borrowed(instances),
    }
}

/// `RUNE_LAYER_KEEP=0`: every layer is drawn into its target again each frame, as before.
fn keeps_layers() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*OFF.get_or_init(|| std::env::var_os("RUNE_LAYER_KEEP").is_some_and(|v| v == "0"))
}

impl Describe for Target {
    /// What the plan reads about it: its size, what it keeps and what it costs, with no handle.
    fn held(&self) -> Held {
        Held {
            size: [self.texture.width(), self.texture.height()],
            bytes: Renderer::target_bytes(self),
            cache: self.cache,
            last_used: self.last_used,
            content: self.content_size,
        }
    }
}

/// Binds `t` to the frame's atlas and outline texture, and to `generation`: its base bind group
/// here, its composite and image and glyph pages at the next `prepare_layers`.
fn rebind_target(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, globals: &wgpu::Buffer,
    outlines: Option<&wgpu::Texture>, atlas: &wgpu::Texture, t: &mut Target, generation: u64) {
    t.base = Renderer::bind_group(device, layout, globals, outlines, atlas, atlas, Some(&t.projection));
    // No target's identity is 0, so the composite is made again.
    t.composite_parent = Some(0);
    t.image_generation = u64::MAX;
    t.binding = generation;
    #[cfg(test)]
    { t.outline_rows = outlines.map_or(0, wgpu::Texture::height); }
}

/// Resource/pass counters for diagnostics. The ordinary path creates no layer resources.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayerStats {
    pub targets: usize,
    pub bytes: u64,
    pub extra_passes: usize,
    pub allocations: u64,
}

fn union(a: &mut [f32; 4], b: [f32; 4]) {
    if b[2] <= b[0] || b[3] <= b[1] { return; }
    a[0] = a[0].min(b[0]); a[1] = a[1].min(b[1]);
    a[2] = a[2].max(b[2]); a[3] = a[3].max(b[3]);
}

impl Renderer {
    /// Target format for isolated layer textures: uses the window's 8-bit format when blending in
    /// gamma space (`srgb_blend`) to halve bandwidth and avoid software f16 conversions, and
    /// `Rgba16Float` (`FORMAT`) for linear/HDR rendering.
    pub(super) fn layer_format(&self) -> wgpu::TextureFormat {
        if self.srgb_blend { self.format } else { FORMAT }
    }

    fn target_bytes(t: &Target) -> u64 {
        let mut bytes = u64::from(t.texture.width()) * u64::from(t.texture.height())
            * u64::from(t.texture.format().block_copy_size(None).unwrap());
        if let Some(blur) = &t.blur {
            bytes += blur.bytes();
        }
        bytes
    }

    /// Maximum bytes of offscreen textures retained for [`El::cache`](crate::El::cache) subtrees
    /// before least-recently-used entries are evicted. Defaults to 64 MiB.
    pub fn set_cache_budget(&mut self, bytes: u64) {
        self.layers.cache_budget = bytes;
        self.layers.trim_retained(0);
    }

    /// Current [`El::cache`](crate::El::cache) texture memory budget in bytes.
    pub fn cache_budget(&self) -> u64 {
        self.layers.cache_budget
    }

    /// With `alloc-sites` (a measuring build), the heap dump's `rune-gpu` section lists what this
    /// renderer's pools hold after its latest frame or rest, as `<pool> <count> <bytes>`.
    #[cfg(all(feature = "alloc-sites", not(target_arch = "wasm32")))]
    pub(super) fn report_pools(&self) {
        static POOLS: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
        static ONCE: std::sync::Once = std::sync::Once::new();
        fn texture_bytes(t: &wgpu::Texture) -> u64 {
            let block = u64::from(t.format().block_copy_size(None).unwrap_or(4));
            (0..t.mip_level_count()).map(|m| u64::from((t.width() >> m).max(1)) * u64::from((t.height() >> m).max(1)) * block).sum()
        }
        let pool = |name: &str, ts: &mut dyn Iterator<Item = u64>| {
            let (n, bytes) = ts.fold((0, 0), |(n, b), x| (n + 1, b + x));
            format!("{name} {n} {bytes}\n")
        };
        let l = &self.layers;
        let out = [
            pool("layer-live", &mut l.targets.iter().flatten().filter(|t| t.cache.is_none()).map(Self::target_bytes)),
            pool("layer-cache-live", &mut l.targets.iter().flatten().filter(|t| t.cache.is_some()).map(Self::target_bytes)),
            pool("layer-retained", &mut l.retained.iter().map(Self::target_bytes)),
            pool("layer-spare", &mut l.spare.iter().map(Self::target_bytes)),
            pool("backdrops", &mut self.backdrops.iter().map(|b| texture_bytes(&b.texture))),
            pool("glyph-atlas", &mut std::iter::once(&self.atlas).chain(self.atlas_pages.iter().map(|p| &p.0)).map(texture_bytes)),
            pool("image-pages", &mut self.images.iter().flatten().map(|p| texture_bytes(&p.0))),
        ].concat();
        *POOLS.lock().unwrap_or_else(|e| e.into_inner()) = out;
        ONCE.call_once(|| procwatch::alloc::sites::contribute("rune-gpu", || POOLS.lock().unwrap_or_else(|e| e.into_inner()).clone()));
    }

    /// What a window at rest gives back (`window::rest`): every spare target (the first menu,
    /// hover or session after it makes its own again, within a frame's run-to-run spread on an
    /// Iris Xe), cache textures no layer drew last frame, and the backdrop copies and blur pyramid
    /// when the last frame read none. With `all`, also the scratch targets of this screen's
    /// layers and its backdrop copies, which its next frame makes again (a few ms on an integrated GPU). It keeps the
    /// textures of caches on screen, which the last frame's instances name instead of their
    /// contents, and takes the ones it drops out of `resident`, so their next frame paints them.
    /// What no submission still uses is freed here; the rest when the device next polls.
    pub(crate) fn give_back(&mut self, all: bool, resident: &mut std::collections::HashMap<u64, u64>) {
        for t in self.layers.retained.drain(..) {
            if let Some((node, _)) = t.cache { resident.remove(&node); }
        }
        self.layers.trim_spare(0);
        if all {
            for slot in &mut self.layers.targets {
                if slot.as_ref().is_some_and(|t| t.cache.is_none()) { *slot = None; }
            }
        }
        // The backdrop copies (one full-window texture per target format) and the blur pyramid:
        // out of sight, or when the last frame read none (a session's jump button had blurred
        // what was under it, and Home doesn't).
        if all || !self.backdrop_used {
            self.backdrops.clear();
            if let Some(blur) = &mut self.backdrop_blur { blur.give_back(); }
        }
        #[cfg(all(feature = "alloc-sites", not(target_arch = "wasm32")))]
        self.report_pools();
    }

    /// Frees the staging chunks kept for reuse (`window::rest`, out of sight only: the next
    /// frame's uploads would make new ones).
    pub(crate) fn trim_uploads(&self) {
        self.uploads.trim();
    }

    pub(super) fn sync_resident_caches(&self, out: &mut std::collections::HashMap<u64, u64>) {
        if !out.is_empty() {
            out.clear();
        }
        for t in self.layers.targets.iter().flatten().chain(self.layers.retained.iter()) {
            if let Some((node, sig)) = t.cache {
                out.insert(node, sig);
            }
        }
    }

    pub fn layer_stats(&self) -> LayerStats {
        let mut stats = LayerStats { allocations: self.layers.allocations, ..Default::default() };
        for (index, t) in self.layers.targets.iter().enumerate() {
            let Some(t) = t else { continue };
            stats.targets += 1;
            stats.bytes += Self::target_bytes(t);
            if matches!(self.layers.planner.dispositions.get(index), Some(Disposition::Render { .. })) {
                stats.extra_passes += 1 + t.blur.as_ref().map_or(0, layer_blur::Scratch::passes);
            }
        }
        for t in &self.layers.retained {
            stats.targets += 1;
            stats.bytes += Self::target_bytes(t);
        }
        stats
    }

    pub(super) fn prepare_layers(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, instances: &[Instance], viewport: (u32, u32)) {
        let staging = self.uploads.staging(device, queue);
        self.layers.viewport = viewport;
        #[cfg(test)]
        { self.layers.peak = 0; }
        let before = self.layers.plan.len();
        for slot in &mut self.layers.targets {
            if slot.as_ref().is_some_and(|t| t.cache.is_some()) {
                self.layers.retained.push(slot.take().unwrap());
            }
        }
        self.layers.plan.clear();
        self.layers.stack.clear();
        for (index, i) in instances.iter().enumerate() {
            if i.kind() == Some(Kind::Layer) {
                    let parent = self.layers.stack.last().copied();
                    let parent_cached_hit = parent.is_some_and(|p| {
                        self.layers.plan[p].cache.is_some_and(|key| {
                            self.layers.retained.iter().any(|t| t.cache == Some(key))
                        })
                    });
                    let opacity = i.fill[3].clamp(0.0, 1.0);
                    let planned = match parent.map(|p| self.layers.plan[p].planned) {
                        _ if opacity == 0.0 => Planned::Hidden,
                        Some(Planned::Hidden) => Planned::Hidden,
                        Some(Planned::Covered) => Planned::Covered,
                        _ if parent_cached_hit => Planned::Covered,
                        _ => Planned::Inline,
                    };
                    // Bound storage by the ancestor's required input, including filter support.
                    // Keep the actual clip separate so rounded corners never move.
                    let limit = parent.map_or(NO_CLIP, |p| {
                        let p = &self.layers.plan[p];
                        let grow = (3.0 * p.blur).ceil();
                        [p.clip[0].max(p.limit[0]) - grow, p.clip[1].max(p.limit[1]) - grow,
                            p.clip[2].min(p.limit[2]) + grow, p.clip[3].min(p.limit[3]) + grow]
                    });
                    let blur = layer_blur::sigma(i.params[2]);
                    let cache = (i.params[3] == 1.0).then(|| (unpack_u64_f32(i.radii), unpack_u64_f32(i.uv)));
                    let turn = (i.border[1] != 0.0).then_some((i.border[1], [i.border[2], i.border[3]]));
                    let parent_free = parent.is_some_and(|p| self.layers.plan[p].free);
                    let free = turn.is_some() || parent_free;
                    let frame = if free {
                        let outer = match parent {
                            Some(p) if parent_free => self.layers.plan[p].frame,
                            _ => {
                                let support = parent.map_or(0.0, |p| self.layers.plan[p].support);
                                [-support, -support, viewport.0 as f32 + support, viewport.1 as f32 + support]
                            }
                        };
                        let shown = [outer[0].max(i.clip[0]), outer[1].max(i.clip[1]), outer[2].min(i.clip[2]), outer[3].min(i.clip[3])];
                        match turn {
                            Some((angle, pivot)) if shown[2] > shown[0] && shown[3] > shown[1] => turned_bounds(shown, -angle, pivot),
                            _ => shown,
                        }
                    } else { [0.0; 4] };
                    let limit = if parent_free { NO_CLIP } else { limit };
                    self.layers.stack.push(self.layers.plan.len());
                    self.layers.plan.push(Layer {
                        begin: index as u32, end: 0, parent,
                        edges: [i.rect[0], i.rect[1], i.rect[0] + i.rect[2], i.rect[1] + i.rect[3]],
                        vis: [0.0; 4],
                        clip: i.clip,
                        limit,
                        fade: i.fade,
                        opacity,
                        blend: if i.params[0] == 1.0 { Blend::PlusLighter } else { Blend::SourceOver },
                        blur,
                        support: parent.map_or(0.0, |p| self.layers.plan[p].support) + (3.0 * blur).ceil(),
                        cache,
                        planned,
                        turn,
                        free,
                        frame,
                        // A cached scope's `radii` name its node instead.
                        hole: if cache.is_some() { [0.0; 4] } else { i.radii },
                    });
            } else if i.kind() == Some(Kind::LayerEnd) {
                    let at = self.layers.stack.pop().expect("unbalanced layer end");
                    let l = &mut self.layers.plan[at];
                    l.end = index as u32;
                    let grow = (3.0 * l.blur).ceil();
                    // Retain off-viewport sources that can contribute through this or an ancestor's
                    // filter. Clipping to the destination viewport before filtering loses those taps.
                    let e = l.edges;
                    let vis = if l.free {
                        [(e[0] - grow).max(l.frame[0] - grow).floor(), (e[1] - grow).max(l.frame[1] - grow).floor(),
                            (e[2] + grow).min(l.frame[2] + grow).ceil(), (e[3] + grow).min(l.frame[3] + grow).ceil()]
                    } else {
                        [(e[0] - grow).max(l.clip[0].max(l.limit[0]) - grow).max(-l.support).floor(),
                            (e[1] - grow).max(l.clip[1].max(l.limit[1]) - grow).max(-l.support).floor(),
                            (e[2] + grow).min(l.clip[2].min(l.limit[2]) + grow).min(viewport.0 as f32 + l.support).ceil(),
                            (e[3] + grow).min(l.clip[3].min(l.limit[3]) + grow).min(viewport.1 as f32 + l.support).ceil()]
                    };
                    l.vis = vis;
                    if l.cache.is_some() {
                        if vis[2] <= vis[0] || vis[3] <= vis[1] {
                            l.planned = Planned::Hidden;
                        }
                        l.edges = [e[0].floor(), e[1].floor(), e[2].ceil(), e[3].ceil()];
                    } else {
                        l.edges = vis;
                    }
                    let edges = match l.turn {
                        Some((angle, pivot)) if l.edges[2] > l.edges[0] && l.edges[3] > l.edges[1] => turned_bounds(l.edges, angle, pivot),
                        _ => l.edges,
                    };
                    if let Some(parent) = l.parent.filter(|&p| self.layers.plan[p].cache.is_none()) {
                        union(&mut self.layers.plan[parent].edges, edges);
                    }
            } else if let Some(&at) = self.layers.stack.last() {
                if self.layers.plan[at].cache.is_some() { continue; }
                let (x0, y0, x1, y1) = i.layer_bounds_non_scope();
                union(&mut self.layers.plan[at].edges, [x0, y0, x1, y1]);
            }
        }
        assert!(self.layers.stack.is_empty(), "unclosed layer scope");
        let isolate = self.check_layer_features(device);
        while self.layers.targets.len() > self.layers.plan.len() {
            if let Some(t) = self.layers.targets.pop().flatten() { self.layers.park(t); }
        }
        self.layers.targets.resize_with(self.layers.plan.len(), || None);
        if !self.layers.plan.is_empty() || !self.layers.retained.is_empty() {
            self.layers.frame_stamp += 1;
        }
        let max_dim = self.max_texture_dim.min(MAX_CACHE_DIM);
        if self.layers.plan.is_empty() {
            // No layers to plan: what closed at the last frame's end is all this frame keeps or
            // drops, exactly as dev trimmed the pool. Nothing here reads a target's size.
            self.layers.trim_retained(0);
        } else {
            // One disposition per layer, and every change to the target pools it needs, decided on
            // the pools as they are: `commit` then carries the moves out and `paint_layers` draws them.
            let layers = &mut self.layers;
            let pools = Pools { slots: &layers.targets, retained: &layers.retained, spare: &layers.spare };
            layers.planner.decide(&layers.plan, &pools, layers.cache_budget, isolate, max_dim);
            self.commit(device);
            self.paint_layers(device, &staging, instances);
        }
        // A frame whose layers fell gives back what a burst grew the plan's scratch to; a frame
        // that keeps its layers, or stays empty, leaves it alone.
        if self.layers.plan.len() < before {
            let layers = &mut self.layers;
            layers.planner.settle(layers.plan.len(), layers.retained.len(), layers.spare.len());
        }
        // Rebind child composites to their nearest materialized parent (opaque scopes flatten).
        for index in 0..self.layers.plan.len() {
            if self.layers.targets[index].is_none() { continue; }
            let parent = self.layer_target(self.layers.plan[index].parent);
            let t = self.layers.targets[index].as_ref().unwrap();
            let parent_identity = parent.map(|p| self.layers.targets[p].as_ref().unwrap().identity);
            if t.composite_parent == parent_identity { continue; }
            let composite = Self::bind_group(device, &self.bind_layout, &self.globals, self.uber.outlines(), &self.atlas, &t.texture,
                parent.map(|p| &self.layers.targets[p].as_ref().unwrap().projection));
            let t = self.layers.targets[index].as_mut().unwrap();
            t.composite = composite;
            t.composite_parent = parent_identity;
        }
        let lfmt = self.layer_format();
        if self.layers.targets.iter().any(Option::is_some) {
            self.add_format(device, lfmt);
            self.add_layer_format(device, self.format);
            for index in 0..self.extra.len() { self.add_layer_format(device, self.extra[index].0); }
        }
        if self.layers.plan.iter().any(|l| l.blend == Blend::PlusLighter) && let Some(s) = &mut self.surfaces {
            s.add_plus_lighter(device, lfmt);
        }
        debug_assert!(self.layers.planner.dispositions.iter().zip(&self.layers.targets)
            .all(|(d, t)| t.is_some() == matches!(d, Disposition::Render { .. } | Disposition::Cached { .. })),
            "only Render and Cached layers hold a target");
        self.refresh_layer_bindings(device);
        if before > 0 || !self.layers.plan.is_empty() { self.position_layer_surfaces(queue); }
    }

    /// Carries out the moves the plan decided, in its order: the one place a target changes pool.
    fn commit(&mut self, device: &wgpu::Device) {
        for i in 0..self.layers.planner.moves.len() {
            match self.layers.planner.moves[i] {
                Move::Park(slot) => {
                    // The plan's `Drop`s trim the spares; nothing trims here.
                    if let Some(mut t) = self.layers.targets[slot].take() { t.cache = None; self.layers.spare.push(t); }
                }
                Move::Set { slot, from } => {
                    let mut t = match from {
                        Kept::Retained(at) => self.layers.retained.swap_remove(at),
                        Kept::Spare(at) => self.layers.spare.swap_remove(at),
                    };
                    // A target parked before the atlas or outline texture changed binds it now,
                    // once; a spare no layer takes again never makes a bind group at all.
                    self.rebind_if_stale(device, &mut t);
                    self.layers.targets[slot] = Some(t);
                }
                Move::Create { slot, size } => self.layers.targets[slot] = Some(self.make_target(device, size)),
                Move::Unblur(slot) => {
                    if let Some(t) = self.layers.targets[slot].as_mut() { t.blur = None; }
                }
                Move::Drop(Kept::Retained(at)) => { self.layers.retained.swap_remove(at); }
                Move::Drop(Kept::Spare(at)) => { self.layers.spare.swap_remove(at); }
            }
            #[cfg(test)]
            self.layers.watch_peak();
        }
    }

    /// A target of `size`, with the buffers and bind groups a layer draws into it through.
    fn make_target(&mut self, device: &wgpu::Device, size: [u32; 2]) -> Target {
        let (width, height) = (size[0], size[1]);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("isolated group"), size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: self.layer_format(), usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let projection = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer projection"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let base = Self::bind_group(device, &self.bind_layout, &self.globals, self.uber.outlines(), &self.atlas, &self.atlas, Some(&projection));
        let composite = Self::bind_group(device, &self.bind_layout, &self.globals, self.uber.outlines(), &self.atlas, &texture, None);
        self.layers.allocations += 1;
        Target { texture, view, projection,
            identity: self.layers.allocations, composite_parent: None,
            quad: Self::instance_buffer(device, 1), base, composite, images: Vec::new(), glyphs: Vec::new(), image_generation: u64::MAX, blur: None,
            cache: None, last_used: 0, content_size: [0.0; 2], binding: self.layers.bindings,
            holds: Default::default(), pending: Default::default(), unchanged: Default::default(),
            opaque_clear: None, dirty_rect: None, over_clear: false,
            #[cfg(test)]
            outline_rows: self.uber.outlines().map_or(0, wgpu::Texture::height) }
    }

    /// Draws what each layer's disposition says: its target's descriptor, its blur, and the quad
    /// that composites it.
    fn paint_layers(&mut self, device: &wgpu::Device, staging: &rune_staging::Staging, instances: &[Instance]) {
        let stamp = self.layers.frame_stamp;
        // Only instances draw into a layer this frame: no effect's pass, GPU spring, backdrop or
        // native surface changes what lands there without changing them.
        let quiet = keeps_layers() && self.effects.is_empty() && self.springs.is_none() && !self.has_backdrop_blur()
            && self.surfaces.as_ref().is_none_or(|s| s.draws_nothing());
        let (viewport, texels, images) = (self.layers.viewport, self.layers.texels, self.layers.image_generation);
        for index in 0..self.layers.plan.len() {
            let l = self.layers.plan[index];
            match self.layers.planner.dispositions[index] {
                Disposition::Skip | Disposition::Inline => {}
                Disposition::Cached { rect, cache } => {
                    let t = self.layers.targets[index].as_mut().expect("a claimed cache has its target");
                    t.cache = cache;
                    if cache.is_some() { t.last_used = stamp; }
                    t.over_clear = false;
                    let mut quad = Instance::begin_layer(rect, l.clip, LayerSpec {
                        opacity: l.opacity, blend: l.blend, ..Default::default() });
                    // A hit composites the same turned quad a miss drew: the turn belongs to the
                    // scope, not to the pass that filled its texture.
                    if let Some((angle, pivot)) = l.turn { quad = quad.turned(angle, pivot); }
                    quad.fade = l.fade;
                    if (t.content_size[0] - rect[2]).abs() > 0.5 || (t.content_size[1] - rect[3]).abs() > 0.5 {
                        quad.uv = [0.0, 0.0, t.content_size[0], t.content_size[1]];
                    }
                    staging.write_buffer(&t.quad, 0, bytemuck::bytes_of(&quad));
                }
                Disposition::Render { rect, cache } => {
                    let t = self.layers.targets[index].as_mut().expect("a rendering layer has its target");
                    t.cache = cache;
                    t.last_used = stamp;
                    t.content_size = [rect[2], rect[3]];
                    // A plain layer (no blur, no turn, no layer or effect inside it) that draws what
                    // its target holds already isn't drawn again: the frame composites the texture.
                    let content = instances.get(l.begin as usize + 1..l.end as usize).unwrap_or(&[]);
                    let plain = quiet && cache.is_none() && l.blur == 0.0 && !l.free && l.turn.is_none() && !content.is_empty()
                        && content.iter().all(|i| match i.kind() {
                            Some(Kind::Effect | Kind::BackdropBlur) | None => false,
                            Some(Kind::Layer) => {
                                i.fill[3] >= 1.0
                                    && i.params[0] == 0.0
                                    && i.params[2] == 0.0
                                    && i.params[3] == 0.0
                                    && i.border[1] == 0.0
                                    && i.fade == [0; 4]
                                    && i.radii[3] == 0.0
                            }
                            _ => true,
                        });
                    let holds = t.holds.get_mut().unwrap_or_else(|e| e.into_inner());
                    let same = plain && holds.as_ref().is_some_and(|h| h.is(rect, viewport, texels, images, t.binding, content));
                    let ob = if l.blur == 0.0 && !l.free && l.turn.is_none() { opaque_base(content.first(), rect) } else { None };
                    t.opaque_clear = ob;
                    t.dirty_rect = if !same && plain {
                        holds.as_ref()
                            .and_then(|h| partial_layer_dirty(h, rect, viewport, texels, images, t.binding, content))
                            .filter(|&dirty| ob.is_some() || opaque_interior_covers(content.first(), rect, dirty))
                    } else {
                        None
                    };
                    *t.unchanged.get_mut() = same;
                    if !same {
                        *holds = None;
                        *t.pending.get_mut().unwrap_or_else(|e| e.into_inner()) = plain.then(|| Holds { rect, viewport, texels, images, binding: t.binding, instances: content.to_vec() });
                    }
                    // A scratch the layer no longer wants was dropped by `Move::Unblur`, before any
                    // later layer's texture existed.
                    if l.blur > 0.0 {
                        let pipeline = self.layers.blur.as_ref().expect("prepared layer blur");
                        if t.blur.is_none() {
                            t.blur = Some(pipeline.scratch(device, &t.texture));
                            self.layers.allocations += 1;
                        }
                        if t.blur.as_mut().unwrap().prepare(device, pipeline, staging,
                            [rect[2], rect[3]], l.blur, l.blend == Blend::PlusLighter) {
                            self.layers.allocations += 1;
                        }
                    }
                    staging.write_buffer(&t.projection, 0, bytemuck::cast_slice(&rect));
                    let over_clear = l.parent.is_none()
                        && l.turn.is_none()
                        && l.fade == [0; 4]
                        && l.hole[3] <= 0.0
                        && l.blend == Blend::SourceOver
                        && self.cleared.len > 0
                        && self.cleared.color[3] == 1.0
                        && (self.cleared.len as usize) <= (l.begin as usize)
                        && disjoint_before(instances, self.cleared.len as usize, l.begin as usize, rect);
                    t.over_clear = over_clear;
                    let mut quad = Instance::begin_layer(rect, l.clip, LayerSpec {
                        opacity: l.opacity, blend: l.blend, ..Default::default() });
                    if let Some((angle, pivot)) = l.turn { quad = quad.turned(angle, pivot); }
                    else if over_clear { quad.border = self.cleared.color; }
                    quad.fade = l.fade;
                    quad.radii = l.hole;
                    staging.write_buffer(&t.quad, 0, bytemuck::bytes_of(&quad));
                }
            }
            #[cfg(test)]
            self.layers.watch_peak();
        }
    }

    /// Builds and checks the pipelines this frame's layers need (`breaker`), and plans the frame
    /// without the ones the device refused or hasn't taken yet: a dissolve as source-over, a blur
    /// unblurred. Whether layers can isolate; when they can't, each group draws in its parent.
    fn check_layer_features(&mut self, device: &wgpu::Device) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        self.warm_isolation(device);
        // A layer a cache covers counts: the cache can still miss and draw it this frame.
        let shown = || self.layers.plan.iter().enumerate().filter(|(_, l)| l.planned != Planned::Hidden);
        let isolation = shown().any(|(index, _)| self.isolates(index));
        let plus_lighter = shown().any(|(_, l)| l.blend == Blend::PlusLighter);
        let blur = shown().any(|(_, l)| l.blur > 0.0);
        if !isolation && !plus_lighter && !blur && !self.breakers.checking() { return false; }
        self.check_features(device, |feature| match feature {
            breaker::Feature::Isolation => isolation,
            breaker::Feature::PlusLighter => plus_lighter,
            breaker::Feature::LayerBlur => blur,
        });
        let working = |feature| self.breakers.health(feature).working();
        let (isolate, add, blur) = (working(breaker::Feature::Isolation), working(breaker::Feature::PlusLighter), working(breaker::Feature::LayerBlur));
        for l in &mut self.layers.plan {
            if !add { l.blend = Blend::SourceOver; }
            if !blur { l.blur = 0.0; }
        }
        isolate
    }

    /// Whether layer `index` needs a target of its own to draw as asked, while planning.
    fn isolates(&self, index: usize) -> bool {
        let l = &self.layers.plan[index];
        let additive_parent = l.parent.is_some_and(|p| self.layers.plan[p].blend == Blend::PlusLighter);
        l.cache.is_some() || l.opacity != 1.0 || l.blend != Blend::SourceOver || additive_parent
            || l.blur != 0.0 || l.fade[3] != 0 || l.clip != NO_CLIP || l.turn.is_some() || l.hole[3] != 0.0
    }

    /// Drops the composite pipelines (`breaker`), and with `targets` every target, which no
    /// frame will draw into again.
    pub(super) fn drop_layer_pipelines(&mut self, targets: bool) {
        self.layers.pipelines.clear();
        // Pipelines still being made ahead are the old shader's.
        if matches!(self.layers.warming, Warming::Making(_)) { self.layers.warming = Warming::Done; }
        if targets {
            self.layers.targets.clear();
            self.layers.retained.clear();
            self.layers.spare.clear();
        }
    }

    /// Drops the layer blur's pipelines and every target's blur scratch (`breaker`).
    pub(super) fn drop_layer_blur(&mut self) {
        self.layers.blur = None;
        for t in self.layers.targets.iter_mut().flatten().chain(&mut self.layers.retained).chain(&mut self.layers.spare) { t.blur = None; }
    }

    pub(super) fn add_layer_format(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        if !self.layers.pipelines.iter().any(|(f, _)| *f == format) {
            let p = if cfg!(not(any(target_os = "windows", target_os = "macos", target_os = "ios", target_arch = "wasm32"))) {
                super::fast::layer_composite_pipeline(device, &self.layout, format, wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING)
            } else {
                Self::pipeline(device, &self.layout, &self.shader, format, "vs", Blend::SourceOver)
            };
            self.layers.pipelines.push((format, p));
        }
    }

    fn layer_target(&self, mut layer: Option<usize>) -> Option<usize> {
        while let Some(index) = layer {
            if self.layers.targets[index].is_some() { return Some(index); }
            layer = self.layers.plan[index].parent;
        }
        None
    }

    pub(super) fn layer_blend(&self, target: Option<usize>) -> Blend {
        target.map_or(Blend::SourceOver, |p| self.layers.plan[p].blend)
    }

    pub(super) fn images_changed(&mut self) { self.layers.image_generation += 1; self.layers.texels += 1; }

    /// Makes every live target's bind groups again (a new atlas or outline texture), and starts a
    /// generation: the slots and retained caches bind it here, a parked target binds it when a
    /// layer takes it back, and a spare no layer takes again makes nothing.
    pub(super) fn rebind_layers(&mut self, device: &wgpu::Device) {
        self.layers.bindings += 1;
        let generation = self.layers.bindings;
        let (layout, globals, outlines, atlas) = (&self.bind_layout, &self.globals, self.uber.outlines(), &self.atlas);
        for t in self.layers.targets.iter_mut().flatten().chain(&mut self.layers.retained) {
            rebind_target(device, layout, globals, outlines, atlas, t, generation);
            #[cfg(test)]
            { self.layers.binds += 1; }
        }
    }

    /// Binds `t` to the atlas and outline texture this frame draws with, if it was made or parked
    /// against an earlier generation.
    fn rebind_if_stale(&mut self, device: &wgpu::Device, t: &mut Target) {
        let generation = self.layers.bindings;
        if t.binding == generation { return; }
        rebind_target(device, &self.bind_layout, &self.globals, self.uber.outlines(), &self.atlas, t, generation);
        #[cfg(test)]
        { self.layers.binds += 1; }
    }

    pub(super) fn refresh_layer_bindings(&mut self, device: &wgpu::Device) {
        for t in self.layers.targets.iter_mut().flatten() {
            if t.image_generation == self.layers.image_generation { continue; }
            t.images.clear();
            t.images.extend(self.images.iter().map(|page| page.as_ref().map(|(texture, _)|
                Self::bind_group_images(device, &self.bind_layout, &self.globals, self.uber.outlines(), &self.atlas, texture, Some(&t.projection), self.srgb_blend))));
            t.glyphs.clear();
            t.glyphs.extend(self.atlas_pages.iter().map(|(texture, _)|
                Self::bind_group(device, &self.bind_layout, &self.globals, self.uber.outlines(), texture, texture, Some(&t.projection))));
            t.image_generation = self.layers.image_generation;
        }
    }

    pub(super) fn position_layer_surfaces(&self, queue: &wgpu::Queue) {
        let Some(s) = &self.surfaces else { return };
        for index in 0..s.len() {
            let before = s.before_instance(index);
            let layer = self.layers.plan.iter().rposition(|l| before > l.begin && before <= l.end);
            let bounds = self.layer_target(layer).map_or(
                [0.0, 0.0, self.layers.viewport.0 as f32, self.layers.viewport.1 as f32],
                |p| self.layers.planner.dispositions.get(p).map_or([0.0; 4], |d| d.rect()));
            // prepare_surfaces may precede prepare on the first frame.
            if bounds[2] > 0.0 && bounds[3] > 0.0 { s.target(queue, index, bounds); }
        }
    }

    /// Encode isolated scopes before opening a compositor render pass that calls `draw_range`.
    /// `draw`/`draw_timed` do this automatically. Contents are redrawn each presentation so GPU
    /// springs and newly latched native surfaces stay live; textures, not content, are pooled.
    pub fn encode_layers(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_layers_timed(encoder, None, &mut |_, _| {});
    }

    /// `encode_layers` that calls `hook` at each effect slot inside an isolated layer, in painter
    /// order within that layer's pass. The layer's opacity, blur, dissolve and clip then apply to
    /// whatever the hook drew when the layer composites. Slots outside any layer are reached by
    /// `draw_range_with` or `draw_with`.
    pub fn encode_layers_with(&self, encoder: &mut wgpu::CommandEncoder, hook: &mut EffectHook<'_>) {
        self.encode_layers_timed(encoder, None, &mut |phase, site| {
            if let EffectPhase::Draw(pass) = phase { hook(pass, site); }
        });
    }

    /// `encode_layers_with` that also calls `hook(EffectPhase::Backdrop(encoder), site)` outside
    /// the render pass at each backdrop slot (`El::effect_backdrop(true)`) inside an isolated
    /// layer, right after copying the layer's target up to that slot into `EffectSite::backdrop`
    /// and before re-opening the layer's pass for `EffectPhase::Draw(pass)`.
    pub fn encode_layers_with_backdrop(&self, encoder: &mut wgpu::CommandEncoder, hook: &mut EffectBackdropHook<'_>) {
        self.encode_layers_timed(encoder, None, hook);
    }

    /// Origin and size of a pass's target: a layer's bounds, or the frame viewport.
    pub(super) fn target_frame(&self, target: Option<usize>) -> ([f32; 2], [u32; 2]) {
        match target {
            Some(index) => {
                let b = self.layers.planner.dispositions.get(index).map_or([0.0; 4], |d| d.rect());
                ([b[0], b[1]], [b[2] as u32, b[3] as u32])
            }
            None => ([0.0, 0.0], [self.layers.viewport.0, self.layers.viewport.1]),
        }
    }

    pub(super) fn encode_layers_timed(&self, encoder: &mut wgpu::CommandEncoder, timestamps: Option<&wgpu::RenderPassTimestampWrites<'_>>, hook: &mut EffectBackdropHook<'_>) {
        self.flush_uploads(encoder);
        let mut first = true;
        for (index, l) in self.layers.plan.iter().enumerate().rev() {
            let disposition = self.layers.planner.dispositions.get(index).copied().unwrap_or(Disposition::Skip);
            if !matches!(disposition, Disposition::Render { .. }) { continue; }
            let Some(target) = &self.layers.targets[index] else { continue };
            if target.unchanged.load(std::sync::atomic::Ordering::Relaxed) { continue; }
            let rect = disposition.rect();
            let (load, start_at) = if target.dirty_rect.is_some() {
                (wgpu::LoadOp::Load, l.begin + 1)
            } else if let Some(bg) = target.opaque_clear {
                (wgpu::LoadOp::Clear(wgpu::Color { r: bg[0] as f64, g: bg[1] as f64, b: bg[2] as f64, a: bg[3] as f64 }), l.begin + 2)
            } else {
                (wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), l.begin + 1)
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("isolated group"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view, depth_slice: None, resolve_target: None,
                    ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                })], depth_stencil_attachment: None,
                timestamp_writes: if first { timestamps.map(|t| wgpu::RenderPassTimestampWrites {
                    query_set: t.query_set, beginning_of_pass_write_index: t.beginning_of_pass_write_index,
                    end_of_pass_write_index: None,
                }) } else { None },
                occlusion_query_set: None, multiview_mask: None,
            }).forget_lifetime();
            first = false;
            pass.set_viewport(0.0, 0.0, rect[2], rect[3], 0.0, 1.0);
            let mut active = ActivePass::owned(pass, encoder, &target.view);
            if let Some(dirty) = target.dirty_rect {
                active.set_scissor(Some(dirty), [rect[2] as u32, rect[3] as u32]);
            }
            self.draw_layer_range(&mut active, target.texture.format(), start_at..l.end, Some(index), hook);
            drop(active);
            if let Some(blur) = &target.blur {
                self.layers.blur.as_ref().unwrap().encode(encoder, blur, &target.view);
            }
            // The texture now holds what `paint_layers` said this pass draws.
            if let Some(drawn) = target.pending.lock().unwrap_or_else(|e| e.into_inner()).take() {
                *target.holds.lock().unwrap_or_else(|e| e.into_inner()) = Some(drawn);
            }
        }
    }

    pub(super) fn draw_layer_range(&self, pass: &mut ActivePass<'_, '_>, format: wgpu::TextureFormat, range: Range<u32>, target: Option<usize>, hook: &mut EffectBackdropHook<'_>) {
        let mut cursor = range.start;
        for (index, l) in self.layers.plan.iter().enumerate() {
            if l.begin < cursor || l.begin >= range.end { continue; }
            assert!(l.end < range.end, "draw_range must not split a layer scope");
            self.draw_plain(pass, format, cursor..l.begin, target, true, hook);
            let disposition = self.layers.planner.dispositions.get(index).copied().unwrap_or(Disposition::Skip);
            if let (Disposition::Render { rect, .. } | Disposition::Cached { rect, .. }, Some(t)) = (disposition, &self.layers.targets[index]) {
                let plain_fast = self.springs.is_none()
                    && self.layer_blend(target) == Blend::SourceOver
                    && l.blend == Blend::SourceOver
                    && l.turn.is_none()
                    && l.fade == [0; 4]
                    && l.hole[3] <= 0.0
                    && (t.content_size[0] - rect[2]).abs() <= 0.5
                    && (t.content_size[1] - rect[3]).abs() <= 0.5
                    && rect.iter().all(|v| v.fract() == 0.0)
                    && (l.clip[0] < -1e5 || l.clip.iter().all(|v| v.fract() == 0.0));
                // Otherwise an upright, unfaded, not additive composite still skips the general composite's work.
                let plain_layer = self.layer_blend(target) != Blend::PlusLighter
                    && l.turn.is_none() && !l.free && l.hole[3] <= 0.0 && l.blend != Blend::PlusLighter
                    && l.cache.is_none()
                    && l.fade[0] == 0 && l.fade[1] == 0 && l.fade[2] == 0
                    && (l.fade[3] & 0x8000_0000) == 0;
                let plain_pipe = plain_layer.then(|| {
                    self.solid.as_ref().filter(|(f, p)| *f == self.format && p.len() == 2 * super::variant::SOLID_COUNT).and_then(|(_, p)| {
                        let (window, layer) = p.split_at(super::variant::SOLID_COUNT);
                        let slice = if format == self.format { Some(window) } else if target.is_some() && format == FORMAT { Some(layer) } else { None };
                        slice?.get(super::variant::PLAIN_LAYER_SLOT)?.as_ref()
                    })
                }).flatten();
                let pipeline = if self.layer_blend(target) == Blend::PlusLighter {
                    &self.layers.additive.as_ref().expect("prepared dissolve").1
                } else if plain_fast && let Some((_, fp)) = self.fast.iter().find(|(f, _)| *f == format) {
                    if target.is_none() && t.over_clear { &fp.fast_layer_over_clear } else { &fp.fast_layer }
                } else if let Some(p) = plain_pipe {
                    p
                } else {
                    &self.layers.pipelines.iter().find(|(f, _)| *f == format).expect("Renderer::add_format first").1
                };
                pass.set_pipeline(pipeline);
                if let Some(properties) = &self.springs { pass.set_bind_group(1, &properties.group, &[]); }
                pass.set_bind_group(0, &t.composite, &[]);
                pass.set_vertex_buffer(0, t.quad.slice(..));
                // A layer's kind is its composite: `RUNE_SKIP_KIND=8` leaves it undrawn, the layer's contents drawn.
                if !self.kind_skipped(l.begin) { pass.draw(0..4, 0..1); }
            } else if disposition == Disposition::Inline {
                // An opaque group that needs no target, or any group while layers can't isolate
                // (`breaker`).
                if l.clip != NO_CLIP {
                    let (origin, size) = self.target_frame(target);
                    let edge = |v: f32, o: f32, max: u32| (v - o).clamp(0.0, max as f32) as u32;
                    let (mut x0, mut y0) = (edge(l.clip[0].floor(), origin[0], size[0]), edge(l.clip[1].floor(), origin[1], size[1]));
                    let (mut x1, mut y1) = (edge(l.clip[2].ceil(), origin[0], size[0]), edge(l.clip[3].ceil(), origin[1], size[1]));
                    // Inside an enclosing inline clip: both apply.
                    let outer = pass.scissor();
                    if let Some([ox, oy, ow, oh]) = outer {
                        (x0, y0, x1, y1) = (x0.max(ox), y0.max(oy), x1.min(ox + ow), y1.min(oy + oh));
                    }
                    if x1 > x0 && y1 > y0 {
                        pass.set_scissor(Some([x0, y0, x1 - x0, y1 - y0]), size);
                        self.draw_layer_range(pass, format, l.begin + 1..l.end, target, hook);
                        pass.set_scissor(outer, size);
                    }
                } else {
                    self.draw_layer_range(pass, format, l.begin + 1..l.end, target, hook);
                }
            }
            cursor = l.end + 1;
        }
        let include_end = range.end == self.count || self.layers.plan.iter().any(|l| l.end == range.end);
        self.draw_plain(pass, format, cursor..range.end, target, include_end, hook);
    }

    pub(super) fn validate_layer_range(&self, range: &Range<u32>) {
        for l in &self.layers.plan {
            assert!(!(l.begin < range.start && range.start <= l.end)
                && !(l.begin < range.end && range.end <= l.end), "draw_range must not split a layer scope");
        }
    }

    fn draw_plain(&self, pass: &mut ActivePass<'_, '_>, format: wgpu::TextureFormat, range: Range<u32>, target: Option<usize>, include_end: bool, hook: &mut EffectBackdropHook<'_>) {
        let mut cursor = range.start;
        if let Some(s) = &self.surfaces {
            for index in 0..s.len() {
                let before = s.before_instance(index);
                if before < range.start || before > range.end || (before == range.end && !include_end) { continue; }
                self.draw_with_effects(pass, format, cursor..before, target, hook);
                // Only the window's own pass can punch a video's place out for its layer.
                match target {
                    None => s.draw_main(pass, index, format, self.layer_blend(target)),
                    Some(_) => s.draw_isolated(pass, index, format, self.layer_blend(target)),
                }
                cursor = before;
            }
        }
        self.draw_with_effects(pass, format, cursor..range.end, target, hook);
    }
}

static WARM_LAYERS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Has every renderer make the pipelines an isolated layer draws with (a group's opacity, a
/// fade, a rounded clip cut by its viewport) on a thread of their own, from its first frame,
/// rather than in the first frame that shows a layer. Making them compiles shaders, some 20 ms
/// on a slow machine: in the frame of the first hover or fade, that is a stall the user sees.
/// A layer shown before the thread is done makes them in its frame, as it would have.
/// Call it before the window opens.
pub fn warm_layers() {
    WARM_LAYERS.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// The isolation pipelines a thread makes, with what the device said of them.
type Warmed = std::sync::Arc<std::sync::Mutex<Option<(Vec<(wgpu::TextureFormat, &'static str, wgpu::RenderPipeline)>, bool)>>>;

enum Warming {
    Unasked,
    Making(Warmed),
    Done,
}

#[cfg(not(target_arch = "wasm32"))]
impl Renderer {
    /// Starts the isolation pipelines on their thread, once, and takes them when they are made:
    /// the breaker then finds them built and has only the device's answer to hear.
    fn warm_isolation(&mut self, device: &wgpu::Device) {
        if !matches!(self.breakers.health(breaker::Feature::Isolation), breaker::Health::Untried) {
            // A layer got there first, and its frame made them.
            if matches!(self.layers.warming, Warming::Making(_)) { self.layers.warming = Warming::Done; }
            return;
        }
        match &self.layers.warming {
            Warming::Done => {}
            Warming::Unasked => {
                if !WARM_LAYERS.load(std::sync::atomic::Ordering::Relaxed) { return; }
                let made = Warmed::default();
                let (to, made_on, layout, shader, format, entry) = (made.clone(), device.clone(), self.layout.clone(), self.shader.clone(), self.format, self.vertex_entry);
                let composite_wgsl = self.composite_wgsl();
                let thread = std::thread::Builder::new().name("rune-warm-layers".into()).spawn(move || {
                    // What `build_feature` makes for isolation: the uber pipeline for layer
                    // targets, and the composite for the window's format and theirs.
                    let mut pipelines = Vec::new();
                    let composite = composite_wgsl.and_then(|wgsl| {
                        super::variant::compile(&made_on, &wgsl, |pinned| Renderer::pipeline(&made_on, &layout, pinned, format, "vs", Blend::SourceOver)).ok()
                    });
                    let verdict = breaker::checked(&made_on, || {
                        pipelines.push((FORMAT, entry, Renderer::pipeline(&made_on, &layout, &shader, FORMAT, entry, Blend::SourceOver)));
                        for format in [format, FORMAT] {
                            let pipeline = match &composite {
                                Some((pinned, _)) => Renderer::pipeline(&made_on, &layout, pinned, format, "vs", Blend::SourceOver),
                                None => Renderer::pipeline(&made_on, &layout, &shader, format, "vs", Blend::SourceOver),
                            };
                            pipelines.push((format, "vs", pipeline));
                        }
                    });
                    let taken = matches!(verdict, breaker::Verdict::Now(None));
                    *to.lock().unwrap_or_else(|e| e.into_inner()) = Some((pipelines, taken));
                });
                self.layers.warming = match thread {
                    // Frame-profile builds wait for them, as for the pinned pipelines, and take them now.
                    #[cfg(feature = "frame-profile")]
                    Ok(thread) => {
                        thread.join().ok();
                        self.layers.warming = Warming::Making(made);
                        return self.warm_isolation(device);
                    }
                    #[cfg(not(feature = "frame-profile"))]
                    Ok(_) => Warming::Making(made),
                    Err(_) => Warming::Done,
                };
            }
            Warming::Making(made) => {
                let Some((pipelines, taken)) = made.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
                self.layers.warming = Warming::Done;
                // Refused on the thread: the first layer builds them in the breaker, which reports it.
                if !taken { return; }
                let mut pipelines = pipelines.into_iter();
                if let Some((format, _, uber)) = pipelines.next() && !self.extra.iter().any(|(f, _)| *f == format) {
                    self.extra.push((format, uber));
                }
                for (format, _, composite) in pipelines {
                    if !self.layers.pipelines.iter().any(|(f, _)| *f == format) { self.layers.pipelines.push((format, composite)); }
                }
                self.check_features(device, |feature| feature == breaker::Feature::Isolation);
            }
        }
    }
}
