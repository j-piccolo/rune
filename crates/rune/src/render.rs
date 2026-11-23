//! The instanced wgpu renderer, with lazy bounded group layers and offscreen readback.

use crate::pixels::ImageAtlas;
use crate::text::{ATLAS_SIZE, Atlas};

mod backdrop_blur;
mod breaker;
mod cleared;
#[cfg(not(target_os = "macos"))]
mod damaged;
mod device;
mod dxbc;
mod effect_warm;
mod fast;
mod fill;
mod first_use;
mod gpu_loss;
mod instance;
mod layers;
mod metallib;
mod offscreen;
mod outlines;
mod pass;
mod pipelines;
// The kept frame it checks exists only where partial frames do (`damaged`).
#[cfg(not(target_os = "macos"))]
mod pixel_check;
#[cfg(target_os = "macos")]
mod prewarm;
#[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
#[path = "render/warm.rs"]
mod prewarm;
#[cfg(target_arch = "wasm32")]
mod prewarm {
    pub(super) fn start() {}
    pub(super) fn take() -> Option<(super::Gpu, super::Renderer)> { None }
    pub(super) fn discard() {}
}
mod shader_check;
mod surface_setup;
mod uber_wgsl;
mod upload;
mod slim;
mod variant;
#[cfg(any(target_os = "linux", all(test, not(target_arch = "wasm32"))))]
mod vulkan_icd;
#[cfg(not(target_arch = "wasm32"))]
mod warm_machine;
mod warmup;
#[cfg(target_os = "macos")]
mod windfoil_metallib;

pub(crate) mod csd_corners;
pub(crate) mod gpu_events;
pub(crate) mod layer_blur;
pub(crate) mod surface_alpha;
pub(crate) mod translucency;

pub mod cpu_boot;

#[cfg(test)]
mod abi_tests;
#[cfg(test)]
mod backdrop_tests;
#[cfg(test)]
mod image_tests;
#[cfg(test)]
#[path = "render/native_surface_tests.rs"]
mod native_surface;
#[cfg(test)]
mod prewarm_tests;
#[cfg(test)]
#[path = "render/warm_tests.rs"]
mod warm_tests;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod webgl_shader_tests;

use pass::{ActivePass, Backdrop};

#[cfg(not(target_os = "macos"))]
pub(crate) use damaged::{MAX_ARMED, MAX_RECTS};
pub(crate) use shader_check::ShaderError;
pub(crate) use surface_setup::{SurfaceSetupError, format_window_setup_error};
#[cfg(target_os = "windows")]
pub(crate) use surface_setup::with_wine_hwnd_hint;
pub(crate) use device::choose_gpu_backend;
pub(crate) use device::discard_prewarm;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use device::reopen_prewarm;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use device::shutdown_prewarm;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use warm_machine::ShutdownOutcome;
pub(crate) use instance::Globals;
pub(crate) use instance::INSTANCE_WORDS;
pub(crate) use instance::effect_mark;
pub(crate) use instance::end_layer;
pub(crate) use instance::pack_u64_f32;
pub(crate) use instance::turned_bounds;
pub(crate) use instance::unpack_u64_f32;
pub(crate) use instance::word;
pub(crate) use warmup::park_gpu;
pub(crate) use warmup::surface_gpu;
#[cfg(target_os = "linux")]
pub(crate) use warmup::{warm_gpu_made, warm_gpu_then};

pub use backdrop_blur::MAX_BACKDROP_BLURS;
pub use device::Gpu;
pub use device::prewarm_gpu;
pub use device::prewarm_gpu_on;
pub use device::set_gpu_memory_policy;
pub use effect_warm::PrewarmEffects;
#[doc(hidden)]
pub use fill::FillDrift;
pub use first_use::make_layer_blur;
pub use first_use::make_straight_alpha;
pub use first_use::precompiled_misses;
pub use instance::Instance;
pub use instance::KIND_BACKDROP_BLUR;
pub use instance::KIND_DASHED;
pub use instance::KIND_EFFECT;
pub use instance::KIND_GLYPH;
pub use instance::KIND_GRADIENT;
pub use instance::KIND_IMAGE;
pub use instance::KIND_INSET;
pub use instance::KIND_LAYER;
pub use instance::KIND_LAYER_END;
pub use instance::KIND_RADIAL;
pub use instance::KIND_RECT;
pub use instance::KIND_SEGMENT;
pub use instance::KIND_SHADOW;
pub use instance::KIND_TRIANGLE;
pub use instance::Kind;
pub use instance::NO_CLIP;
pub(crate) use instance::STOP_FADE_BIAS;
pub use instance::pack_clip_radii;
pub use instance::pack_fade;
pub use instance::unpack_clip_radii;
pub use instance::unpack_fade;
pub use layers::LayerStats;
pub use offscreen::OFFSCREEN_FORMAT;
pub use offscreen::WINDOW_FORMAT;
pub use offscreen::Offscreen;
pub use offscreen::write_png;
#[cfg(not(target_arch = "wasm32"))]
pub use offscreen::snapshot;
pub use rune_core::Blend;
pub use warmup::warm_gpu;
pub use warmup::warm_gpu_cache;
pub use warmup::warm_gpu_with;

/// Rows the first glyph atlas page's texture starts with: a window starts with 2 MiB instead of
/// 4 MiB and avoids mid-session atlas grows when browsing text-heavy views, doubling up to
/// `ATLAS_SIZE` (`prepare`).
pub(crate) const ATLAS_START_ROWS: u32 = 1024;

/// `RUNE_LOOP_CHECK=2` found a picture that differed from the whole frame's (`pixel_check`), so the
/// process's exit code is 3. Never on macOS and Linux, which have no loop-only frames.
pub(crate) fn loop_pixel_check_failed() -> bool {
    #[cfg(not(target_os = "macos"))]
    return pixel_check::failed();
    #[cfg(target_os = "macos")]
    false
}

/// The kinds the frame pass has a pipeline of its own for (`regen_dxbc` compiles each).
#[doc(hidden)]
pub const PINNED_KINDS: [u32; 3] = variant::PINNED;

/// BLAKE3 of the WGSL the frame pass's pipeline pinned to `kind` compiles from (`variant::pinned_wgsl`, `\r\n` read as
/// `\n`): what `regen_dxbc` records beside each pinned fragment and `dxbc::uber_pinned` checks before using one.
#[doc(hidden)]
pub fn pinned_wgsl_blake3(kind: u32) -> Option<String> {
    variant::pinned_wgsl(kind).map(|wgsl| blake3::hash(wgsl.replace("\r\n", "\n").as_bytes()).to_hex().to_string())
}

pub(crate) fn blend_state(blend: Blend) -> wgpu::BlendState {
    match blend {
        Blend::SourceOver => wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
        Blend::PlusLighter => {
            let add = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
            wgpu::BlendState { color: add, alpha: add }
        }
    }
}

/// Composition properties of an isolated subtree. Filters can extend this contract without
/// changing painter-stream scopes or the target pool.
#[derive(Clone, Copy, Debug)]
pub struct LayerSpec {
    pub opacity: f32,
    pub blend: Blend,
    /// Gaussian standard deviation in device pixels, clamped to 64. Zero bypasses filtering.
    pub blur: f32,
    /// Radius of the rounded `clip` rectangle, in device pixels; zero is rectangular and a
    /// negative radius rounds with continuous corners (`El::continuous_corners`). Applied to
    /// the completed layer after filtering. The caller supplies finite clip edges.
    pub clip_radius: f32,
}

impl Default for LayerSpec {
    fn default() -> Self { Self { opacity: 1.0, blend: Blend::SourceOver, blur: 0.0, clip_radius: 0.0 } }
}

/// WGSL for clipping like Rune does: `sd_rounded`, `ramp` and `clip_alpha(p, clip, fade)`. `p` is
/// a window device-pixel position, `clip` an instance's `clip` and `fade` (`vec4<u32>`) its packed
/// soft edges and rounded-clip radius, so a compositor shader cuts its effect with the same
/// rectangle, feathers and circular or continuous corners as the primitives around it. Prepend it
/// to the shader source.
pub const CLIP_WGSL: &str = include_str!("shaders/clip.wgsl");

/// The WGSL the precompiled DX12 bytecode is compiled from (Windfoil's, CRLF read as LF), its
/// BLAKE3 and the hash of its pipeline layout, as the bytecode records them;
/// `examples/regen_dxbc.rs` regenerates that bytecode from them.
#[cfg(target_os = "windows")]
#[doc(hidden)]
pub fn uber_shader_identity() -> (&'static str, &'static str, u64) {
    (variant::windfoil_wgsl(), uber_wgsl::WINDFOIL_BLAKE3, variant::windfoil_abi_hash())
}

/// Where a compositor is asked to draw an effect slot (`Renderer::draw_range_with`).
pub struct EffectSite<'a> {
    /// The slot's position in the painter stream.
    pub index: u32,
    /// The slot's record (`KIND_EFFECT`): rect, radii, clip and fade.
    pub instance: &'a Instance,
    /// The slot's `El::effect` tag, so a compositor can find the resources it keeps for it.
    pub mark: u32,
    /// The slot's `El::effect_data` payload, as the element was last built. A slot playing its
    /// exit transition keeps the payload it had when it was removed.
    pub data: [f32; 4],
    /// Device pixels per point at this slot (`instance.params[3]`: `Env::scale` times any
    /// element or ancestor `scale` / `live_scale`).
    pub scale: f32,
    /// Format of the pass being recorded.
    pub format: wgpu::TextureFormat,
    /// Window device-pixel position of this pass's top-left pixel: zero in the frame target,
    /// the layer's bounds origin inside an isolated layer. Subtract it from window coordinates.
    pub origin: [f32; 2],
    /// Size of this pass's target in device pixels.
    pub size: [u32; 2],
    /// True inside an isolated layer: the pass is that layer's offscreen target.
    pub in_layer: bool,
    /// Whether colours are sRGB-encoded for blending (`Renderer::srgb_blend`). Choose a
    /// pipeline's encoding from this: encode the shader's output when it's true, write linear
    /// values when it's false. Don't use `format.is_srgb()`: an isolated layer's `Rgba16Float`
    /// target is neither, and holds whichever encoding the frame uses.
    pub srgb_blend: bool,
    /// Viewport-sized copy of the pass's target up to this slot when the slot asked for a
    /// backdrop (`El::effect_backdrop(true)`, `instance.params[0] != 0.0`) and the pass was run
    /// with a command encoder (`Renderer::draw_with`, `Renderer::draw_with_backdrop`,
    /// `Renderer::draw_target_range_with`, or `Renderer::encode_layers_with`). Has the same
    /// `format` as the pass (`*Srgb` / `Rgba16Float` in linear frames, non-sRGB / `Rgba16Float`
    /// holding sRGB-encoded premultiplied values when `srgb_blend` is true; pass it directly to
    /// `FxRenderer::blur_backdrop` or `FxRenderer::set_layer_backdrop`). Inside an isolated layer
    /// (`in_layer: true`), holds that layer's offscreen target copied at `origin` in window
    /// coordinates.
    pub backdrop: Option<&'a wgpu::TextureView>,
    /// Packed explicit scroll-fade stops (`Runtime::fade_stops`) for this frame, matching
    /// `clip_alpha_stops` in `CLIP_WGSL` (`[0.0, 0.0]` when no scroller uses `ScrollFade::stops`).
    pub fade_stops: [f32; 2],
}

/// Called at each effect slot, with the pass open and everything painted before the slot already
/// recorded, and the scissor set to the slot's clip rectangle (a slot clipped away entirely is
/// skipped). Soft edges and rounded corners on a slot outside a layer are the hook's to apply
/// (`CLIP_WGSL`). It may record draws; the viewport and scissor are reset to the full target after.
pub type EffectHook<'h> = dyn FnMut(&mut wgpu::RenderPass<'_>, &EffectSite<'_>) + 'h;

/// Phase of an effect slot callback in `Renderer::draw_with_backdrop`,
/// `Renderer::encode_layers_with_backdrop`, and `Renderer::draw_target_range_with`.
pub enum EffectPhase<'a, 'p> {
    /// Outside the render pass, right after the target up to a backdrop slot
    /// (`El::effect_backdrop(true)`) was copied into `site.backdrop` and before the pass re-opens
    /// with `LoadOp::Load` (e.g. for `FxRenderer::blur_backdrop`).
    Backdrop(&'a mut wgpu::CommandEncoder),
    /// Inside the open render pass at the slot's painter position, with the scissor set to the
    /// slot's clip rectangle.
    Draw(&'a mut wgpu::RenderPass<'p>),
}

/// Called at each effect slot: first with `EffectPhase::Backdrop(encoder)` while the pass is
/// closed if the slot requested a backdrop (`El::effect_backdrop(true)`), then with
/// `EffectPhase::Draw(pass)` inside the open pass.
pub type EffectBackdropHook<'h> = dyn FnMut(EffectPhase<'_, '_>, &EffectSite<'_>) + 'h;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum QuadMode {
    /// Ordinary 4-vertex quad on its kind or frame pipeline.
    Whole,
    /// Borderless, circular-corner, unfaded `Kind::Rect` drawn whole on `simple_rect_wgsl`.
    SimpleRect,
    /// Large shape split into middle (`SPLIT_MIDDLE`) + bands (`SPLIT_BANDS`).
    Split,
    /// Large borderless, circular-corner, unfaded `Kind::Rect`: `solid_rect_wgsl` middle + `simple_rect_wgsl` bands.
    SplitSimpleRect,
    /// Large linear `Kind::Radial` whose middle lies in `[t0, t1]`: `linear_grad_middle_wgsl` middle + bands.
    SplitLinear,
    /// Plain whole-pixel borderless unrounded rect `>= BIG` inside an unfaded clip: 4-vertex middle only, no bands.
    MiddleOnly,
    /// Plain 1:1 grayscale atlas glyph: slim 4-vertex glyph pipeline.
    PlainGlyph,
    /// One band of a border ring with no fill, circular corners and no fade: `simple_ring_wgsl`.
    Ring,
}

impl QuadMode {
    #[inline]
    pub(super) fn is_split(self) -> bool {
        matches!(self, QuadMode::Split | QuadMode::SplitSimpleRect | QuadMode::SplitLinear)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BatchBind {
    /// Glyph atlas page 0 with image atlas page `usize`.
    Image(usize),
    /// Secondary glyph atlas page `usize` (index in `atlas_pages`, i.e. glyph page - 1).
    Glyph(usize),
}

pub struct Renderer {
    /// Post-first-frame background effect-pipeline warmer: listed first so its Drop drains
    /// any active worker before GPU resources on this renderer drop.
    effect_warm: effect_warm::EffectWarm,
    first_frame_submitted: std::sync::atomic::AtomicBool,
    surfaces: Option<crate::native_surface::SurfaceRenderer>,
    /// The native surface renderer while its pipeline compiles (`build_surfaces`).
    surfaces_building: Option<crate::native_surface::Building>,
    pipeline: wgpu::RenderPipeline,
    format: wgpu::TextureFormat,
    shader: dxbc::Shaders,
    layout: wgpu::PipelineLayout,
    vertex_entry: &'static str,
    /// Pipelines for other target formats a compositor draws into (`add_format`).
    extra: Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    bind_group: wgpu::BindGroup,
    bind_layout: wgpu::BindGroupLayout,
    globals: wgpu::Buffer,
    atlas: wgpu::Texture,
    /// Which uber shader draws (`variant::Uber`), with the outline store's texture under Windfoil.
    uber: variant::Uber,
    /// Windfoil's subpixel pipeline for the window (`variant::Lcd`), and whether this frame's
    /// text draws subpixel masks.
    lcd: variant::Lcd,
    /// The frame pass's pipelines pinned to one kind each (`variant::PINNED`), once their thread has made them
    /// (`variant::start_pinned`); empty where there's no bytecode for them, None until they arrive.
    pinned: Option<Vec<Option<wgpu::RenderPipeline>>>,
    /// Those of the formats `extra` keeps a frame pipeline for, so a switch back makes none.
    pinned_extra: Vec<(wgpu::TextureFormat, Vec<Option<wgpu::RenderPipeline>>)>,
    /// The pinned pipelines being made on their thread (`variant::start_pinned`), for this format.
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    pinned_building: Option<(wgpu::TextureFormat, std::sync::mpsc::Receiver<Vec<Option<wgpu::RenderPipeline>>>)>,
    /// Scratch buffer for paint-order-preserving wave batching (`upload::wave_batch`).
    reordered: Vec<Instance>,
    /// Each instance's kind (`params[1]`), as batched: where a run of one kind starts and ends.
    kinds: Vec<Option<Kind>>,
    /// Each instance's fast-path classification, as batched (`fast.rs`).
    fast_kinds: Vec<fast::FastKind>,
    /// Fast-path pipelines per target format (`fast.rs`).
    fast: Vec<(wgpu::TextureFormat, fast::FastPipelines)>,
    /// Uniform buffer & bind group for fused multi-stop radial + vignette stacks (`fast.rs`).
    fused_res: Option<fast::FusedGradResources>,
    /// Detected fused gradient span + payload for the current frame (`fast.rs`).
    fused_grad: Option<(std::ops::Range<u32>, fast::FusedGradUniform)>,
    /// Instance skipped during a dominant-rect clear-swap pass (`cleared.rs`).
    skip_instance: std::cell::Cell<Option<u32>>,
    /// Each instance, as batched: which quad path draws it (`classify_quad`).
    quad_mode: Vec<QuadMode>,
    /// Each instance, as batched: whether it draws apart with the one before it in one draw, middles then bands
    /// (`together`): the same kind, both split, and clear of every other in their group.
    together: Vec<bool>,
    /// Each instance, as batched: whether it's big enough to draw alone with its kind's pipeline (`BIG`).
    big: Vec<bool>,
    /// Frame-profile builds: a timestamp written between a frame's layer passes and its own (`tooling::FrameLog`).
    #[cfg(feature = "frame-profile")]
    pub(crate) pass_mark: Option<(wgpu::QuerySet, u32)>,
    /// The pipelines a shape's middle draws with where the frame's own pipeline would draw it (`variant::middle_wgsl`,
    /// `variant::MIDDLE_KINDS` order), for the format they were made for (`None`: that one can't be made).
    solid: Option<(wgpu::TextureFormat, Vec<Option<wgpu::RenderPipeline>>)>,
    /// The middle pipelines being made on their thread (`variant::middles_for_format`), for this format.
    solid_building: Option<(wgpu::TextureFormat, std::sync::mpsc::Receiver<Vec<Option<wgpu::RenderPipeline>>>)>,
    lcd_frame: bool,
    /// The window's wake, for work a frame starts on another thread (set_frame_wake).
    frame_wake: Option<std::sync::Arc<crate::live::Wake>>,
    /// The 1×1 image the base bind group holds until the first image page.
    stand_in: wgpu::Texture,
    /// Secondary glyph atlas pages (`1..`).
    atlas_pages: Vec<(wgpu::Texture, wgpu::BindGroup)>,
    /// Stable image pages; the initial bind group uses a 1×1 stand-in.
    images: Vec<Option<(wgpu::Texture, wgpu::BindGroup)>>,
    image_batches: Vec<(std::ops::Range<u32>, BatchBind)>,
    instances: rune_staging::GrowBuffer,
    count: u32,
    /// The rects at the head of the list that a pass clearing to their colour leaves out (`cleared`).
    cleared: cleared::Cleared,
    /// The instances ahead of this one lie under a picture the compositor shows (`set_underlay_start`).
    under: u32,
    springs: Option<crate::gpu::springs::Bindings>,
    spring_instances: usize,
    layers: layers::Layers,
    srgb_blend: bool,
    fade_stops: [f32; 2],
    /// Effect slots of the uploaded instances, by index.
    effects: Vec<(u32, Instance)>,
    /// Pooled viewport-sized backdrop copies, allocated only when a slot sets `effect_backdrop`
    /// or an element sets `backdrop_blur`.
    backdrops: Vec<Backdrop>,
    /// The last prepared frame had an effect reading the backdrop: a rest keeps the copies.
    backdrop_used: bool,
    /// Pipelines and pyramid for backdrop blur records, built on the first one.
    backdrop_blur: Option<backdrop_blur::BackdropBlur>,
    /// Which lazily built pipelines the device took (`breaker`).
    breakers: breaker::Breakers,
    /// Health of the primary uber shader and pipeline (`shader_check`), tracking async WebGPU
    /// error scopes on the web.
    uber_check: shader_check::UberCheck,
    uber_check_seq: crate::machine::Seq,
    #[cfg(target_arch = "wasm32")]
    uber_check_wake: shader_check::WakeSlot,
    uploads: upload::Uploads,
    /// Atlas fills `prepare` staged, recorded into the frame's encoder with the uploads.
    fills: std::sync::Mutex<Option<fill::PendingFills>>,
    /// Unique to this renderer, so to the device it was made on (`Frame::device_key`).
    device_key: crate::compose::DeviceKey,
    /// The device's largest 2D texture, read once: in a browser each `Device::limits` reads every
    /// limit from JavaScript (about 35 µs), which a frame's layers used to pay.
    max_texture_dim: u32,
    /// The one instance per rectangle that clears a damaged one (`arm_damage`), made on first use.
    #[cfg(not(target_os = "macos"))]
    clear_quad: std::sync::OnceLock<wgpu::Buffer>,
    /// The kept frame and the rectangles the next `draw` may redraw alone (`damaged`).
    #[cfg(not(target_os = "macos"))]
    damage: std::sync::Mutex<damaged::DamageState>,
}

impl Renderer {
    /// `Frame::device_key`: a renderer belongs to one device, and a new device gets a new one.
    pub fn device_key(&self) -> crate::compose::DeviceKey {
        self.device_key
    }

    /// Creates a renderer for `device` drawing into `format`, panicking if the GPU rejects
    /// Rune's last-resort uber shader or pipeline. Windows and headless runs use [`Self::try_new`].
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Renderer {
        Self::try_new(device, format).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Creates a renderer for `device` drawing into `format`, returning an error instead of
    /// leaving an invalid pipeline if the GPU rejects Rune's last-resort uber shader or pipeline.
    ///
    /// On native devices and WebGL2 the shader and pipeline are checked synchronously inside
    /// validation, out-of-memory and internal error scopes. On browser WebGPU, error scopes
    /// resolve asynchronously and are checked by the web host without blocking startup.
    pub fn try_new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Result<Renderer, String> {
        Self::try_new_classified(device, format).map_err(Into::into)
    }

    pub(crate) fn try_new_classified(device: &wgpu::Device, format: wgpu::TextureFormat) -> Result<Renderer, ShaderError> {
        Self::try_create(device, format, None, cfg!(target_os = "macos"))
    }

    pub fn set_srgb_blend(&mut self, enabled: bool) {
        self.srgb_blend = enabled;
        if let Some(s) = &mut self.surfaces {
            s.set_srgb_blend(enabled);
        }
    }

    pub fn set_fade_stops(&mut self, stops: [f32; 2]) {
        self.fade_stops = stops;
    }

    pub fn fade_stops(&self) -> [f32; 2] {
        self.fade_stops
    }

    /// Opt-in retained geometry. IDs map instances to shared translation/opacity nodes.
    /// Upload geometry once with `prepare`, then use `present_at` without rebuilding instances.
    /// Panics if the GPU rejects the spring uber shader or pipeline; use [`Self::try_with_springs`]
    /// to handle that failure.
    pub fn with_springs(device: &wgpu::Device, format: wgpu::TextureFormat, nodes: &[crate::gpu::springs::Node], ids: &[u32]) -> Renderer {
        Self::try_with_springs(device, format, nodes, ids).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Like [`Self::with_springs`], returning an error if the GPU rejects the shader or pipeline.
    pub fn try_with_springs(device: &wgpu::Device, format: wgpu::TextureFormat, nodes: &[crate::gpu::springs::Node], ids: &[u32]) -> Result<Renderer, String> {
        Self::try_create(device, format, Some((nodes, ids)), true).map_err(Into::into)
    }

    /// A renderer that starts with the Classic uber shader, as on macOS, and builds Windfoil when
    /// text first draws from outlines (render/variant.rs); for checking that path elsewhere.
    #[doc(hidden)]
    pub fn new_classic(device: &wgpu::Device, format: wgpu::TextureFormat) -> Renderer {
        Self::try_new_classic(device, format).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Like [`Self::new_classic`], returning an error if the GPU rejects the Classic shader or
    /// pipeline.
    #[doc(hidden)]
    pub fn try_new_classic(device: &wgpu::Device, format: wgpu::TextureFormat) -> Result<Renderer, String> {
        Self::try_create(device, format, None, true).map_err(Into::into)
    }

    /// Draws uploaded instances `range` into a pass whose target has `format` (the renderer's
    /// own, or one added with `add_format`). Call `encode_layers` before opening the pass.
    /// Scopes are atomic: list/range boundaries must not bisect a layer.
    pub fn draw_range(&self, pass: &mut wgpu::RenderPass<'_>, format: wgpu::TextureFormat, range: std::ops::Range<u32>) {
        if range.is_empty() && range.end != self.count { return; }
        self.submit_uploads();
        self.validate_layer_range(&range);
        let mut active = ActivePass::borrowed(pass);
        self.draw_layer_range(&mut active, format, range, None, &mut |_, _| {});
    }

    /// `draw_range` that stops at each effect slot (`El::effect`) in `range` and calls `hook`
    /// there, in painter order. Effects inside an isolated layer are reached through
    /// `encode_layers_with`, which draws that layer's pass; scopes nest as in `draw_range`.
    /// Because `pass` is already open, `EffectSite::backdrop` is `None`; use `draw_with`,
    /// `draw_with_backdrop`, or `draw_target_range_with` when slots request a backdrop copy.
    pub fn draw_range_with(&self, pass: &mut wgpu::RenderPass<'_>, format: wgpu::TextureFormat, range: std::ops::Range<u32>, hook: &mut EffectHook<'_>) {
        if range.is_empty() && range.end != self.count { return; }
        self.submit_uploads();
        self.validate_layer_range(&range);
        let mut active = ActivePass::borrowed(pass);
        self.draw_layer_range(&mut active, format, range, None, &mut |phase, site| {
            if let EffectPhase::Draw(pass) = phase { hook(pass, site); }
        });
    }

    /// Encodes isolated layers and draws the frame into `view` (cleared to `clear`), calling
    /// `hook` at each effect slot (`El::effect`) in painter order. At slots with
    /// `El::effect_backdrop(true)`, ends the current pass, copies the target up to that slot into
    /// a pooled backdrop texture, re-opens the pass with `LoadOp::Load`, and hands the backdrop
    /// view in `EffectSite::backdrop`.
    pub fn draw_with(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, clear: [f32; 4], hook: &mut EffectHook<'_>) {
        self.draw_with_backdrop(encoder, view, clear, &mut |phase, site| {
            if let EffectPhase::Draw(pass) = phase { hook(pass, site); }
        });
    }

    /// Like `draw_with`, also calling `hook(EffectPhase::Backdrop(encoder), site)` while the
    /// render pass is closed at each backdrop slot (`El::effect_backdrop(true)`) so the compositor
    /// can blur or pre-process `site.backdrop` (e.g. `FxRenderer::blur_backdrop`) before the pass
    /// re-opens for `EffectPhase::Draw(pass)`.
    pub fn draw_with_backdrop(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        clear: [f32; 4],
        hook: &mut EffectBackdropHook<'_>,
    ) {
        // A partial frame (`damaged`), as `draw` takes one: only when nothing but the instances
        // draws there, so there's no effect slot for `hook` to fill. (The web arms and draws its
        // own, `platform/web/gpu.rs`.)
        #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
        if self.armed() && self.draw_armed(encoder, &view.texture(), Some(view)) { return; }
        self.encode_layers_with_backdrop(encoder, hook);
        let clear = wgpu::Color { r: clear[0] as f64, g: clear[1] as f64, b: clear[2] as f64, a: clear[3] as f64 };
        self.draw_target_range_with(encoder, view, self.format, wgpu::LoadOp::Clear(clear), 0..self.count, hook);
    }

    /// Draws `range` into `view` (opening the pass with `load`), splitting the pass on `encoder`
    /// only at backdrop slots (`El::effect_backdrop(true)`) in `range` to copy `view` into
    /// `EffectSite::backdrop` and run `EffectPhase::Backdrop(encoder)` before re-opening with
    /// `LoadOp::Load`. Call `encode_layers_with` / `encode_layers_with_backdrop` before drawing
    /// the first range.
    /// `view` must be a single-sample, full-size, mip-zero 2D target; backdrop slots require
    /// its texture to have `COPY_SRC` usage. These owned passes have no depth/stencil or resolve
    /// attachment. Use `draw_range_with` for a caller-owned pass instead (no backdrop copies).
    pub fn draw_target_range_with(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        load: wgpu::LoadOp<wgpu::Color>,
        range: std::ops::Range<u32>,
        hook: &mut EffectBackdropHook<'_>,
    ) {
        if range.is_empty() && range.end != self.count { return; }
        self.validate_layer_range(&range);
        self.flush_uploads(encoder);
        let (load, range) = if range.start == 0 {
            let (load, start) = self.cleared_pass(load, 0);
            (load, start.min(range.end)..range.end)
        } else {
            (load, range)
        };
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("frame"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }).forget_lifetime();
        let mut active = ActivePass::owned(pass, encoder, view);
        self.draw_layer_range(&mut active, format, range, None, hook);
    }

    /// `draw_instances`, split after each effect slot in `range` so `hook` runs at its place.
    fn draw_with_effects(&self, pass: &mut ActivePass<'_, '_>, format: wgpu::TextureFormat, range: std::ops::Range<u32>, target: Option<usize>, hook: &mut EffectBackdropHook<'_>) {
        let first = self.effects.partition_point(|(at, _)| *at < range.start);
        let mut cursor = range.start;
        for (at, instance) in self.effects[first..].iter().take_while(|(at, _)| *at < range.end) {
            self.draw_instances(pass, format, cursor..*at + 1, target);
            cursor = *at + 1;
            if instance.kind() == Some(Kind::BackdropBlur) {
                self.draw_backdrop_blur(pass, format, *at, instance, target);
                continue;
            }
            let (origin, size) = self.target_frame(target);
            // Scissor to the slot's clip rectangle, so a compositor that ignores `clip` still
            // stays inside its scroll view or clipped ancestor.
            let c = instance.clip;
            let edge = |v: f32, o: f32, max: u32| (v - o).clamp(0.0, max as f32) as u32;
            let (x0, y0) = (edge(c[0].floor(), origin[0], size[0]), edge(c[1].floor(), origin[1], size[1]));
            let (x1, y1) = (edge(c[2].ceil(), origin[0], size[0]), edge(c[3].ceil(), origin[1], size[1]));
            if x1 <= x0 || y1 <= y0 { continue; }
            let backdrop = if instance.params[0] != 0.0 {
                pass.split_for_backdrop(self, *at, instance, format, origin, size, target.is_some(), hook)
            } else {
                None
            };
            pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
            hook(EffectPhase::Draw(pass), &EffectSite { index: *at, instance, mark: effect_mark(instance), data: instance.uv, scale: instance.params[3],
                format, origin, size, in_layer: target.is_some(), srgb_blend: self.srgb_blend, backdrop, fade_stops: self.fade_stops });
            pass.set_viewport(0.0, 0.0, size[0] as f32, size[1] as f32, 0.0, 1.0);
            pass.set_scissor(pass.scissor(), size);
        }
        self.draw_instances(pass, format, cursor..range.end, target);
    }

    fn draw_instances(&self, pass: &mut wgpu::RenderPass<'_>, format: wgpu::TextureFormat, range: std::ops::Range<u32>, target: Option<usize>) {
        if range.is_empty() {
            return;
        }
        let frame_pipeline = self.draw_pipeline(format, target);
        pass.set_pipeline(frame_pipeline);
        // The pinned pipelines draw runs of one kind (`variant::PINNED`), in paint order, where the frame's own pipeline
        // would draw: the window's, or a layer's drawn over as the window is (not an additive layer or subpixel text).
        // Made from WGSL they come for the window's format and then the layers' (`layers::FORMAT`).
        let n = variant::PINNED_FROM_WGSL.len();
        let pinned = self.pinned.as_deref().filter(|p| p.iter().any(Option::is_some)).and_then(|p| {
            // A layer in the window's own format (8-bit, where the frame blends sRGB-encoded) draws as the window does.
            if format == self.format && std::ptr::eq(frame_pipeline, &self.pipeline) {
                Some(&p[..p.len().min(n)])
            } else if p.len() == 2 * n && target.is_some() && format == layers::FORMAT && self.layer_blend(target) != Blend::PlusLighter
                && self.extra.iter().any(|(f, e)| *f == format && std::ptr::eq(frame_pipeline, e)) {
                Some(&p[n..])
            } else {
                None
            }
        });
        // Frame-profile builds: `RUNE_PINNED=off` draws every span with the general shader, `all` splits every span
        // whatever its runs; the screen comparison runs both against the general build.
        #[cfg(feature = "frame-profile")]
        let pinned = pinned.filter(|_| pinned_forced() != Some(false));
        let mut bound = frame_pipeline;
        // A shape's middle (and plain atlas glyphs) draws with its slim pipeline where the frame's own pipeline
        // would draw it: the window's, or a layer's (`layers::FORMAT`) drawn over as the window is (not additively).
        let middles = self.solid.as_ref().filter(|(f, p)| *f == self.format && p.len() == 2 * variant::SOLID_COUNT).and_then(|(_, p)| {
            let (window, layer) = p.split_at(variant::SOLID_COUNT);
            // A layer in the window's own format (8-bit, where the frame blends sRGB-encoded) draws as the window does.
            if format == self.format && std::ptr::eq(frame_pipeline, &self.pipeline) {
                Some(window)
            } else if target.is_some() && format == layers::FORMAT && self.layer_blend(target) != Blend::PlusLighter
                && self.extra.iter().any(|(f, p)| *f == format && std::ptr::eq(frame_pipeline, p)) {
                Some(layer)
            } else {
                None
            }
        });
        if let Some(properties) = &self.springs { pass.set_bind_group(1, &properties.group, &[]); }
        pass.set_vertex_buffer(0, self.instances.slice(..));
        let start = self.image_batches.partition_point(|(batch, _)| batch.end <= range.start);
        for (batch, bind_kind) in self.image_batches[start..].iter().take_while(|(batch, _)| batch.start < range.end) {
            let bind = match (*bind_kind, target) {
                (BatchBind::Image(page), Some(index)) => {
                    let t = self.layers.targets[index].as_ref().unwrap();
                    t.images.get(page).and_then(Option::as_ref).unwrap_or(&t.base)
                }
                (BatchBind::Image(page), None) => {
                    self.images.get(page).and_then(Option::as_ref).map_or(&self.bind_group, |(_, bind)| bind)
                }
                (BatchBind::Glyph(page), Some(index)) => {
                    let t = self.layers.targets[index].as_ref().unwrap();
                    t.glyphs.get(page).unwrap_or(&t.base)
                }
                (BatchBind::Glyph(page), None) => {
                    self.atlas_pages.get(page).map_or(&self.bind_group, |(_, bind)| bind)
                }
            };
            pass.set_bind_group(0, bind, &[]);
            let span = batch.start.max(range.start)..batch.end.min(range.end);
            // Fast-path pipelines (fast.rs): tiny specialized shaders (and unblended opaque draws)
            // sharing `@group(0)`. Used whenever the frame pass is SourceOver without GPU springs
            // or dual-source LCD blending.
            let fast_pipes = if self.uploads.has_fast
                && self.springs.is_none()
                && self.layer_blend(target) == Blend::SourceOver
                && (target.is_some() || !self.lcd_frame)
            {
                self.fast.iter().find(|(f, _)| *f == format).map(|(_, p)| p)
            } else {
                None
            };
            let skip_idx = if target.is_none() { self.skip_instance.get() } else { None };
            if let Some(fp) = fast_pipes {
                let mut at = span.start as usize;
                let end = span.end as usize;
                let fk = &self.fast_kinds[..end.min(self.fast_kinds.len())];
                let mut general_start: Option<usize> = None;
                while at < end {
                    if skip_idx == Some(at as u32) {
                        if let Some(s) = general_start.take() {
                            self.draw_span(pass, s as u32..at as u32, frame_pipeline, pinned, middles, &mut bound);
                        }
                        at += 1;
                        continue;
                    }
                    if let (Some((fg_span, _)), Some(res)) = (&self.fused_grad, &self.fused_res)
                        && fg_span.start as usize == at
                        && fg_span.end as usize <= end
                    {
                        if let Some(s) = general_start.take() {
                            self.draw_span(pass, s as u32..at as u32, frame_pipeline, pinned, middles, &mut bound);
                        }
                        let p = &fp.fused_grad;
                        if !std::ptr::eq(p, bound) { pass.set_pipeline(p); bound = p; }
                        pass.set_bind_group(1, &res.bind_group, &[]);
                        pass.draw(0..4, 0..1);
                        at = fg_span.end as usize;
                        continue;
                    }
                    let kind = fk.get(at).copied().unwrap_or(fast::FastKind::General);
                    let mut cur = at + 1;
                    while fk.get(cur).copied() == Some(kind) && cur < end && skip_idx != Some(cur as u32) { cur += 1; }
                    if matches!(kind, fast::FastKind::General) {
                        general_start.get_or_insert(at);
                        at = cur;
                        continue;
                    }
                    if let Some(s) = general_start.take() {
                        self.draw_span(pass, s as u32..at as u32, frame_pipeline, pinned, middles, &mut bound);
                    }
                    let run = at as u32..cur as u32;
                    match kind {
                        fast::FastKind::OpaqueRoundedRect => {
                            let p_in = &fp.opaque_rect_interior;
                            if !std::ptr::eq(p_in, bound) { pass.set_pipeline(p_in); }
                            pass.draw(0..4, run.clone());
                            let p_ring = &fp.rounded_rect_ring;
                            pass.set_pipeline(p_ring);
                            bound = p_ring;
                            pass.draw(0..24, run);
                        }
                        fast::FastKind::OpaqueRoundedLinear => {
                            let p_in = &fp.opaque_linear_interior;
                            if !std::ptr::eq(p_in, bound) { pass.set_pipeline(p_in); }
                            pass.draw(0..4, run.clone());
                            let p_ring = &fp.rounded_linear_ring;
                            pass.set_pipeline(p_ring);
                            bound = p_ring;
                            pass.draw(0..24, run);
                        }
                        fast::FastKind::OpaqueRect => {
                            let p = &fp.opaque_rect;
                            if !std::ptr::eq(p, bound) { pass.set_pipeline(p); bound = p; }
                            pass.draw(0..4, run);
                        }
                        fast::FastKind::OpaqueLinear => {
                            let p = &fp.opaque_linear;
                            if !std::ptr::eq(p, bound) { pass.set_pipeline(p); bound = p; }
                            pass.draw(0..4, run);
                        }
                        fast::FastKind::RoundedLinear => {
                            let p = &fp.rounded_linear;
                            if !std::ptr::eq(p, bound) { pass.set_pipeline(p); bound = p; }
                            pass.draw(0..4, run);
                        }
                        fast::FastKind::OpaqueRadial => {
                            let p = &fp.opaque_radial;
                            if !std::ptr::eq(p, bound) { pass.set_pipeline(p); bound = p; }
                            pass.draw(0..4, run);
                        }
                        fast::FastKind::LinearBand => {
                            let p = &fp.linear_band;
                            if !std::ptr::eq(p, bound) { pass.set_pipeline(p); bound = p; }
                            pass.draw(0..4, run);
                        }
                        fast::FastKind::Shadow => {
                            let p = &fp.shadow;
                            if !std::ptr::eq(p, bound) { pass.set_pipeline(p); bound = p; }
                            pass.draw(0..4, run);
                        }
                        fast::FastKind::General => unreachable!(),
                    }
                    at = cur;
                }
                if let Some(s) = general_start.take() {
                    self.draw_span(pass, s as u32..end as u32, frame_pipeline, pinned, middles, &mut bound);
                }
                continue;
            }
            self.draw_span(pass, span, frame_pipeline, pinned, middles, &mut bound);
        }
    }

    /// Draws `span` (one batch's, or a run the fast pipelines leave to the general path) with the pinned
    /// pipelines, each shape's slim pipeline (`draw_quads`) and the frame's pipeline, in paint order.
    #[allow(clippy::too_many_arguments)]
    fn draw_span<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'_>,
        span: std::ops::Range<u32>,
        frame_pipeline: &'a wgpu::RenderPipeline,
        pinned: Option<&'a [Option<wgpu::RenderPipeline>]>,
        middles: Option<&'a [Option<wgpu::RenderPipeline>]>,
        bound: &mut &'a wgpu::RenderPipeline,
    ) {
        // A span whose kinds alternate (a chip behind each token, a terminal's cell backgrounds between its
        // glyphs) draws in one call with the general shader: a draw and a pipeline switch per run cost more
        // than the smaller shader saves.
        #[cfg(not(feature = "frame-profile"))]
        let split = pinned.filter(|_| few_runs(&self.kinds, span.clone()));
        #[cfg(feature = "frame-profile")]
        let split = pinned.filter(|_| pinned_forced() == Some(true) || few_runs(&self.kinds, span.clone()));
        #[cfg(feature = "frame-profile")]
        PINNED_DRAWN[usize::from(split.is_some())].fetch_add(u64::from(span.end - span.start), std::sync::atomic::Ordering::Relaxed);
        let Some(pinned) = split else {
            // A span whose kinds alternate still draws with each kind's pipeline where that pays: a run of
            // `LONG_RUN` or more of one kind (a line of text, a list's rows), and an instance big enough
            // (`BIG`) that the smaller shader shading its many pixels at a GPU's full rate outweighs a draw.
            // The rest draws with the general shader, in as few draws as paint order allows.
            let choose = |run: &std::ops::Range<u32>, i: u32| -> Option<&wgpu::RenderPipeline> {
                let p = pinned?;
                let pipeline = pinned_pipeline(p, self.kinds.get(i as usize).copied().flatten())?;
                (run.len() >= LONG_RUN || self.big.get(i as usize).copied().unwrap_or(false)).then_some(pipeline)
            };
            let mut cur_start: Option<u32> = None;
            let mut cur_pick: Option<&wgpu::RenderPipeline> = None;
            let mut cur_end = 0u32;
            let same_pick = |a: Option<&wgpu::RenderPipeline>, b: Option<&wgpu::RenderPipeline>| match (a, b) {
                (Some(x), Some(y)) => std::ptr::eq(x, y),
                (None, None) => true,
                _ => false,
            };
            for run in kind_runs(&self.kinds, span.clone()) {
                for i in run.clone() {
                    let pick = choose(&run, i);
                    match cur_start {
                        Some(_) if same_pick(cur_pick, pick) && cur_end == i => {
                            cur_end = i + 1;
                        }
                        Some(s) => {
                            let fallback = cur_pick.unwrap_or(frame_pipeline);
                            // Unsplit alternating runs stay in one frame_pipeline draw without slim-pipeline switches.
                            let run_middles = cur_pick.and(middles);
                            self.draw_quads(pass, s..cur_end, fallback, bound, run_middles);
                            cur_start = Some(i);
                            cur_pick = pick;
                            cur_end = i + 1;
                        }
                        None => {
                            cur_start = Some(i);
                            cur_pick = pick;
                            cur_end = i + 1;
                        }
                    }
                }
            }
            if let Some(s) = cur_start {
                let fallback = cur_pick.unwrap_or(frame_pipeline);
                let run_middles = cur_pick.and(middles);
                self.draw_quads(pass, s..cur_end, fallback, bound, run_middles);
            }
            return;
        };
        for run in kind_runs(&self.kinds, span) {
            let kind = self.kinds[run.start as usize];
            // A kind with no pinned pipeline draws with the pass's own: the window's, or the layer format's.
            let fallback = pinned_pipeline(pinned, kind).unwrap_or(frame_pipeline);
            self.draw_quads(pass, run, fallback, bound, middles);
        }
    }

    /// Draws `range`'s quads with `fallback`, switching to slim pipelines (`middles`) for split shapes,
    /// middle-only whole-pixel rects, and plain atlas glyphs; updates `bound` to avoid redundant pipeline binds.
    fn draw_quads<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'_>,
        range: std::ops::Range<u32>,
        fallback: &'a wgpu::RenderPipeline,
        bound: &mut &'a wgpu::RenderPipeline,
        middles: Option<&'a [Option<wgpu::RenderPipeline>]>,
    ) {
        #[cfg(feature = "frame-profile")]
        let skipping = !skipped_kinds().is_empty() || !skipped_modes().is_empty() || !skipped_rects().is_empty();
        #[cfg(feature = "frame-profile")]
        let skip = |i: u32| skipping && (self.kinds.get(i as usize).copied().flatten().is_some_and(|k| skipped_kinds().contains(&k.as_u32()))
            || skipped_modes().contains(&(self.quad_mode.get(i as usize).copied().unwrap_or(QuadMode::Whole) as u32 + if middles.is_some() { 0 } else { 10 }))
            || self.uploads.instance(i).is_some_and(|inst| skipped_rects().iter().any(|r| r.iter().zip(inst.rect).all(|(a, b)| (a - b).abs() < 0.75))));
        #[cfg(not(feature = "frame-profile"))]
        let skip = |_: u32| false;
        #[cfg(feature = "frame-profile")]
        if draw_trace() {
            let modes: Vec<String> = (range.start..range.end).map(|i| format!("{}", self.quad_mode.get(i as usize).copied().unwrap_or(QuadMode::Whole) as u32)).collect();
            eprintln!("rune-draw: {}..{} middles={} kinds={:?} modes={}", range.start, range.end, middles.is_some(),
                self.kinds.get(range.start as usize).copied().flatten().map(Kind::as_u32), modes.join(""));
        }
        let bind = |pass: &mut wgpu::RenderPass<'_>, bound: &mut &'a wgpu::RenderPipeline, p: &'a wgpu::RenderPipeline| {
            if !std::ptr::eq(*bound, p) {
                pass.set_pipeline(p);
                *bound = p;
                #[cfg(feature = "frame-profile")]
                DRAW_TALLY[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        };
        let mut at = range.start;
        while at < range.end {
            if skip(at) {
                at += 1;
                continue;
            }
            let mode = self.quad_mode.get(at as usize).copied().unwrap_or(QuadMode::Whole);
            let mut end = at + 1;
            if middles.is_none() {
                let apart = mode.is_split();
                while end < range.end && !skip(end) && self.quad_mode.get(end as usize).copied().unwrap_or(QuadMode::Whole).is_split() == apart { end += 1; }
                bind(pass, bound, fallback);
                if apart {
                    tally_draw();
                    pass.draw(SPLIT_VERTICES, at..end);
                } else {
                    tally_draw();
                    pass.draw(0..4, at..end);
                }
                at = end;
                continue;
            }
            while end < range.end && !skip(end) && self.quad_mode.get(end as usize).copied().unwrap_or(QuadMode::Whole) == mode { end += 1; }
            let middles = middles.unwrap();
            match mode {
                QuadMode::Whole => {
                    bind(pass, bound, fallback);
                    tally_draw();
                    pass.draw(0..4, at..end);
                }
                QuadMode::SimpleRect => {
                    let p = middles.get(variant::SIMPLE_RECT_SLOT).and_then(Option::as_ref).unwrap_or(fallback);
                    bind(pass, bound, p);
                    tally_draw();
                    pass.draw(0..4, at..end);
                }
                QuadMode::MiddleOnly => {
                    let p = middles.first().and_then(Option::as_ref).unwrap_or(fallback);
                    bind(pass, bound, p);
                    tally_draw();
                    pass.draw(0..4, at..end);
                }
                QuadMode::Ring => {
                    let p = middles.get(variant::RING_SLOT).and_then(Option::as_ref).unwrap_or(fallback);
                    bind(pass, bound, p);
                    tally_draw();
                    pass.draw(0..4, at..end);
                }
                QuadMode::PlainGlyph => {
                    let p = middles.get(variant::PLAIN_GLYPH_SLOT).and_then(Option::as_ref).unwrap_or(fallback);
                    bind(pass, bound, p);
                    tally_draw();
                    pass.draw(0..4, at..end);
                }
                QuadMode::Split | QuadMode::SplitSimpleRect | QuadMode::SplitLinear => {
                    // Shapes of one kind clear of each other draw their middles in one draw and their bands in
                    // another: with none over another, the order between them changes no pixel.
                    let bands_pipe = if mode == QuadMode::SplitSimpleRect {
                        middles.get(variant::SIMPLE_RECT_SLOT).and_then(Option::as_ref).unwrap_or(fallback)
                    } else {
                        fallback
                    };
                    let mut i = at;
                    while i < end {
                        let mut j = i + 1;
                        while j < end && !skip(j) && self.together.get(j as usize).copied().unwrap_or(false) { j += 1; }
                        let middle = if mode == QuadMode::SplitLinear {
                            middles.get(variant::LINEAR_MIDDLE_SLOT).and_then(Option::as_ref)
                        } else {
                            let kind = self.kinds.get(i as usize).copied().flatten().map(Kind::as_u32);
                            variant::MIDDLE_KINDS.iter().position(|&k| Some(k) == kind).and_then(|at| middles.get(at)?.as_ref())
                        };
                        match middle {
                            Some(middle) => {
                                bind(pass, bound, middle);
                                tally_draw();
                                pass.draw(SPLIT_MIDDLE, i..j);
                                bind(pass, bound, bands_pipe);
                                tally_draw();
                                pass.draw(SPLIT_BANDS, i..j);
                            }
                            None => {
                                bind(pass, bound, bands_pipe);
                                tally_draw();
                                pass.draw(SPLIT_VERTICES, i..j);
                            }
                        }
                        i = j;
                    }
                }
            }
            at = end;
        }
    }

    /// The format of the target the renderer was built for (`draw_range` with this format).
    pub fn format(&self) -> wgpu::TextureFormat { self.format }

    /// True when colours are sRGB-encoded for blending: the target is a non-sRGB view and every
    /// pass, offscreen layers included, holds encoded premultiplied values. False means linear
    /// premultiplied values, encoded by the `*Srgb` target on write.
    pub fn srgb_blend(&self) -> bool { self.srgb_blend }

    pub fn submit(&self, queue: &wgpu::Queue, mut encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.before_submit(&mut encoder);
        let index = if let Some(s) = &self.surfaces { s.submit(queue, encoder) } else { queue.submit([encoder.finish()]) };
        self.first_frame_submitted.store(true, std::sync::atomic::Ordering::Release);
        index
    }

    /// Whether `submit` is `queue.submit` after `before_submit` and nothing more: no native video
    /// surface waits on the frame's submission, so another thread may make it.
    #[cfg(target_os = "macos")]
    pub(crate) fn submits_plainly(&self) -> bool {
        self.surfaces.as_ref().is_none_or(|s| s.draws_nothing())
    }

    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, clear: [f32; 4]) {
        #[cfg(not(target_os = "macos"))]
        if self.armed() && self.draw_armed(encoder, &view.texture(), Some(view)) { return; }
        self.draw_timed(encoder, view, clear, None);
    }

    /// Constant-time scheduler check. Present the terminal frame, then stop the display link.
    /// Deadlines are recomputed only on commits, so no per-node CPU sampling occurs on ticks.
    pub fn springs_animating(&self, time: f64) -> bool {
        self.springs.as_ref().is_some_and(|p| time < p.settle_at)
    }

    /// The next frames leave out the instances ahead of `from`: a picture the compositor shows
    /// under the window covers them (`RuntimeCore::underlay`). 0 draws them all.
    pub fn set_underlay_start(&mut self, from: u32) { self.under = from; }

    /// Draws over what `view` already holds instead of clearing it (headless overlays).
    pub(crate) fn draw_over(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        self.draw_load(encoder, view, wgpu::LoadOp::Load, None);
    }

    pub fn draw_timed(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, clear: [f32; 4], timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>) {
        let clear = wgpu::Color { r: clear[0] as f64, g: clear[1] as f64, b: clear[2] as f64, a: clear[3] as f64 };
        self.draw_load(encoder, view, wgpu::LoadOp::Clear(clear), timestamps);
    }

    fn draw_load(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, load: wgpu::LoadOp<wgpu::Color>, timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>) {
        let timestamps = timestamps.or_else(|| self.surfaces.as_ref().and_then(|s| s.timestamp_writes()));
        self.encode_layers_timed(encoder, timestamps.as_ref(), &mut |_, _| {});
        #[cfg(feature = "frame-profile")]
        if let Some((queries, at)) = &self.pass_mark { encoder.write_timestamp(queries, *at); }
        let timestamps = timestamps.map(|mut t| {
            if self.layer_stats().extra_passes > 0 { t.beginning_of_pass_write_index = None; }
            t
        });
        let blurs = self.has_backdrop_blur();
        let (load, start, skip) = match (load, self.cleared.swap) {
            (wgpu::LoadOp::Clear(c), Some((idx, c2)))
                if [c.r, c.g, c.b, c.a] == self.cleared.color.map(f64::from)
                    && self.under == 0
                    && self.surfaces.as_ref().is_none_or(|s| (0..s.len()).all(|i| s.before_instance(i) > idx)) =>
            {
                (wgpu::LoadOp::Clear(wgpu::Color { r: c2[0] as f64, g: c2[1] as f64, b: c2[2] as f64, a: c2[3] as f64 }), 1, Some(idx))
            }
            _ => {
                let (load, start) = self.cleared_pass(load, self.under.min(self.count));
                (load, start, None)
            }
        };
        self.skip_instance.set(skip);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("frame"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: timestamps.clone(),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if !blurs {
            self.draw_range(&mut pass, self.format, start..self.count);
            self.skip_instance.set(None);
            return;
        }
        // Backdrop blurs split the frame pass, which needs it owned along with the encoder.
        let pass = pass.forget_lifetime();
        let mut active = ActivePass::owned_timed(pass, encoder, view, timestamps);
        self.draw_layer_range(&mut active, self.format, start..self.count, None, &mut |_, _| {});
        self.skip_instance.set(None);
    }
}

/// How far a corner of radius `radius` reaches along each edge. Radii are signed: a negative
/// radius draws Apple's continuous corner, whose curve starts `CONTINUOUS_K` times the radius from
/// the corner (`CONTINUOUS_K` in clip.wgsl).
pub(crate) fn corner_extent(radius: f32) -> f32 {
    if radius < 0.0 { -radius * CONTINUOUS_K } else { radius }
}

pub(crate) const CONTINUOUS_K: f32 = instance::shader_consts::CONTINUOUS_K;

/// Frame-profile builds: `RUNE_SKIP_MODE=0,6` leaves the instances drawn on those `QuadMode`s undrawn (10 more
/// for a span drawn without the slim pipelines), as `RUNE_SKIP_KIND` does kinds.
#[cfg(feature = "frame-profile")]
fn skipped_modes() -> &'static [u32] {
    static SKIPPED: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    SKIPPED.get_or_init(|| std::env::var("RUNE_SKIP_MODE").map(|v| v.split(',').filter_map(|k| k.trim().parse().ok()).collect()).unwrap_or_default())
}

/// Frame-profile builds: `RUNE_DRAW_TRACE` prints each `draw_quads` span, its kinds and quad modes.
#[cfg(feature = "frame-profile")]
fn draw_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RUNE_DRAW_TRACE").is_some())
}

/// Frame-profile builds: `RUNE_SKIP_RECT=x,y,w,h;...` leaves the instances with those rects undrawn.
#[cfg(feature = "frame-profile")]
fn skipped_rects() -> &'static [[f32; 4]] {
    static SKIPPED: std::sync::OnceLock<Vec<[f32; 4]>> = std::sync::OnceLock::new();
    SKIPPED.get_or_init(|| std::env::var("RUNE_SKIP_RECT").map(|v| v.split(';').filter_map(|r| {
        let n: Vec<f32> = r.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (n.len() == 4).then(|| [n[0], n[1], n[2], n[3]])
    }).collect()).unwrap_or_default())
}

/// Frame-profile builds: pipeline binds and draws in `draw_quads`, reported with `RUNE_PINNED` (`report_pinned`).
#[cfg(feature = "frame-profile")]
pub(super) static DRAW_TALLY: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Frame-profile builds: `RUNE_SKIP_KIND=0,7` leaves the instances of those kinds (`Kind::as_u32`) undrawn in the
/// frame's passes, so a frame budget can say what each kind costs a GPU: the time with them less the time without.
#[cfg(feature = "frame-profile")]
fn skipped_kinds() -> &'static [u32] {
    static SKIPPED: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    SKIPPED.get_or_init(|| std::env::var("RUNE_SKIP_KIND").map(|v| v.split(',').filter_map(|k| k.trim().parse().ok()).collect()).unwrap_or_default())
}

/// Frame-profile builds: `RUNE_PINNED` forcing the split on (`all`) or off (`off`), and the instances drawn
/// unsplit and split since the last report (`report_pinned`).
#[cfg(feature = "frame-profile")]
pub(super) fn pinned_forced() -> Option<bool> {
    static FORCED: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| match rune_core::env::var("RUNE_PINNED").as_deref() { Some("all") => Some(true), Some("off") => Some(false), _ => None })
}

#[cfg(feature = "frame-profile")]
static PINNED_DRAWN: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Frame-profile builds with `RUNE_PINNED` set: every 60th frame, how many instances of the window's pass the
/// pinned pipelines drew and how many went unsplit, so a comparison shows it exercised the pinned path.
#[cfg(feature = "frame-profile")]
pub(crate) fn report_pinned() {
    static FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let frame = FRAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if std::env::var_os("RUNE_PINNED").is_none() || frame % 60 != 0 { return; }
    let [whole, split] = [0, 1].map(|i| PINNED_DRAWN[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    let [binds, draws] = [0, 1].map(|i| DRAW_TALLY[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    eprintln!("rune-pinned: {split} instances split into runs, {whole} drawn unsplit, {binds} binds and {draws} draws in 60 frames (frame {frame})");
}

/// The vertices a box drawn apart takes (shader.wgsl `base_vs`): its middle, where it and its clip cover every
/// pixel whole and the fragment shader returns its fill, and four bands around it. A big panel's middle is most of
/// its pixels, and working out its rounded corners, soft edge and clip there costs a small or software GPU several
/// times what the fill does.
const SPLIT_VERTICES: std::ops::Range<u32> = 4..34;
/// Its middle's quad, and the four bands around it.
const SPLIT_MIDDLE: std::ops::Range<u32> = 4..10;
const SPLIT_BANDS: std::ops::Range<u32> = 10..34;

/// Below this many device pixels either way a shape draws whole: its middle (`split_hole`, 8 pixels at least past
/// its corners and edges) would be small or empty, and the five quads cost more than they save.
const SPLIT_MIN: f32 = 32.0;

/// The pipeline pinned to `kind`, among the renderer's (`variant::PINNED_FROM_WGSL` order), if there is one.
fn pinned_pipeline(pinned: &[Option<wgpu::RenderPipeline>], kind: Option<Kind>) -> Option<&wgpu::RenderPipeline> {
    let at = variant::PINNED_FROM_WGSL.iter().position(|&p| Some(p) == kind.map(Kind::as_u32))?;
    pinned.get(at)?.as_ref()
}

/// Device pixels an instance covers from which it draws alone with its kind's pipeline in a span whose kinds
/// alternate: a panel, a card, a backdrop's picture. A draw and two pipeline switches cost the CPU a few
/// microseconds; shading that many pixels with the general shader costs a small GPU far more.
const BIG: f32 = 32.0 * 32.0;

/// Instances of one kind in a row from which a run draws with that kind's pipeline in a span whose kinds alternate.
const LONG_RUN: usize = 8;

/// Which quad path draws `instance`: plain atlas glyphs on their slim pipeline, plain whole-pixel
/// borderless unrounded rects on the rect middle pipeline with no bands draw, borderless circular
/// rects on `simple_rect_wgsl`, or large shapes split into middle + bands (`split_hole`).
pub(super) fn classify_quad(instance: &Instance) -> QuadMode {
    match instance.kind() {
        Some(Kind::Glyph)
            if instance.radii[3] == 0.0
                && instance.radii[0] <= 0.5
                && instance.params[0] == 0.0
                && instance.uv[2] == 0.0
                && instance.params[3] >= 0.0 =>
        {
            QuadMode::PlainGlyph
        }
        Some(Kind::Rect) => {
            let [x, y, w, h] = instance.rect;
            let [x1, y1] = [x + w, y + h];
            let [cx0, cy0, cx1, cy1] = instance.clip;
            let simple = instance.params[0] == 0.0
                && instance.uv == [0.0; 4]
                && instance.radii.iter().all(|&r| r >= 0.0)
                && instance.fade[0] == 0
                && instance.fade[1] == 0
                && instance.fade[2] == 0
                && (instance.fade[3] & 0x8000_0000) == 0;
            if simple
                && w * h >= BIG
                && instance.params[2] <= 0.5
                && instance.radii == [0.0; 4]
                && instance.fade[3] == 0
                && [x, y, x1, y1].iter().all(|v| v.fract() == 0.0)
                && cx0 <= x && cy0 <= y && cx1 >= x1 && cy1 >= y1
            {
                QuadMode::MiddleOnly
            } else if instance.uv[2] >= 0.0 && instance.fade[2] == 0 && w >= SPLIT_MIN && h >= SPLIT_MIN {
                if simple { QuadMode::SplitSimpleRect } else { QuadMode::Split }
            } else if simple {
                QuadMode::SimpleRect
            } else if instance.uv[2] < 0.0 && instance.params[0] > 0.0 && instance.fill == [0.0; 4]
                && instance.radii.iter().all(|&r| r >= 0.0)
                && instance.fade[0] == 0 && instance.fade[1] == 0 && instance.fade[2] == 0 && (instance.fade[3] & 0x8000_0000) == 0
            {
                QuadMode::Ring
            } else {
                QuadMode::Whole
            }
        }
        Some(Kind::Radial) if instance.fade[2] == 0 && instance.rect[2] >= SPLIT_MIN && instance.rect[3] >= SPLIT_MIN => {
            if instance.uv[2] == 0.0 && (instance.uv[0] != 0.0 || instance.uv[1] != 0.0) {
                let lower_open = instance.params[0] >= 1.5;
                let upper_open = instance.params[3] >= 1.5;
                let t0 = if lower_open { instance.params[0] - 2.0 } else { instance.params[0] };
                let t1 = if upper_open { instance.params[3] - 2.0 } else { instance.params[3] };
                if t1 > t0 + 1e-5 {
                    // The middle (`split_hole`) lies on whole pixels inside the box less `e`: its colour can be
                    // interpolated from its corners when `t` there stays between the stops.
                    let reach = instance.radii.into_iter().map(corner_extent).fold(0.0f32, f32::max);
                    let e = reach + instance.params[2].max(0.5) + 1.0;
                    let [x, y, w, h] = instance.rect;
                    let (x0, x1) = ((x + e).ceil(), (x + w - e).floor());
                    let (y0, y1) = ((y + e).ceil(), (y + h - e).floor());
                    let [cx, cy] = [x + w * 0.5, y + h * 0.5];
                    let t = |px: f32, py: f32| 0.5 + (px - cx) * instance.uv[0] + (py - cy) * instance.uv[1];
                    let ts = [t(x0, y0), t(x1, y0), t(x0, y1), t(x1, y1)];
                    if x1 > x0 && y1 > y0 && ts.iter().all(|&v| v >= t0 + 1e-5 && v <= t1 - 1e-5) {
                        return QuadMode::SplitLinear;
                    }
                }
            }
            QuadMode::Split
        }
        Some(Kind::Image)
            if instance.fade[2] == 0 && instance.rect[2] >= SPLIT_MIN && instance.rect[3] >= SPLIT_MIN =>
        {
            QuadMode::Split
        }
        _ => QuadMode::Whole,
    }
}

/// Most runs a span is split into (`few_runs`), and fewest instances a run has on average.
const MAX_RUNS: usize = 64;
const MIN_RUN: usize = 8;

/// Whether `span` splits into few enough runs of one kind to draw them with the pinned pipelines: one run,
/// or at most `MAX_RUNS` of `MIN_RUN` instances or more on average.
fn few_runs(kinds: &[Option<Kind>], span: std::ops::Range<u32>) -> bool {
    let len = span.len();
    let runs = kind_runs(kinds, span).take(MAX_RUNS + 1).count();
    runs <= 1 || (runs <= MAX_RUNS && runs * MIN_RUN <= len)
}

/// `span` split into runs of instances of one kind (`kinds` by instance).
fn kind_runs(kinds: &[Option<Kind>], span: std::ops::Range<u32>) -> impl Iterator<Item = std::ops::Range<u32>> + '_ {
    let mut at = span.start as usize;
    let end = span.end as usize;
    let slice = &kinds[..end.min(kinds.len())];
    std::iter::from_fn(move || {
        if at >= end { return None; }
        let start = at;
        let Some(&first) = slice.get(at) else {
            at = end;
            return Some(start as u32..end as u32);
        };
        let mut cur = at + 1;
        while let Some(&k) = slice.get(cur) && k == first { cur += 1; }
        at = cur;
        Some(start as u32..cur as u32)
    })
}

/// Counts a `draw_quads` draw in frame-profile builds (`DRAW_TALLY`).
#[inline]
fn tally_draw() {
    #[cfg(feature = "frame-profile")]
    DRAW_TALLY[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
