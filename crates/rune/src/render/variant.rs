//! Which uber shader a renderer draws with, as a state machine ([`Uber`]).
//!
//! `Classic` is shader.wgsl alone, the shader every renderer drew before glyphs could draw from
//! their outlines: its WGSL, layout and precompiled libraries are unchanged, so a Mac that keeps
//! the default text path draws exactly what it drew before. `Windfoil` is shader.wgsl with
//! outline.wgsl, whose glyphs can also draw from the outline store (binding 4). Renderers start
//! with Windfoil except on macOS (and with GPU springs, which keep Classic); a Classic renderer
//! builds Windfoil the first time text asks for outlines. A Windfoil that can't be built leaves
//! the renderer Classic for good (`Unavailable`), and text falls back to its other rasterisers.

use super::instance::shader_consts;
use super::*;
use super::slim::*;
use crate::machine::Effects;

const _: () = {
    assert!(shader_consts::PINNED_KIND < 0.0);
    assert!(!shader_consts::OUTLINE_SUBPIXEL);
};

/// Windfoil's WGSL: shader.wgsl with outline.wgsl's glyph branch, joined when Rune builds
/// (`build.rs`, render/uber_wgsl.rs).
pub(crate) fn windfoil_wgsl() -> &'static str {
    super::uber_wgsl::WINDFOIL
}

/// Windfoil's fragment entry, which the subpixel variant renames for its own entry to call.
const FS_ENTRY: &str = shader_consts::FS_ENTRY_DECL;
/// Mean subpixel coverage floor below which dual-source text emits zero (`DUAL_ENTRY`).
pub(super) const LCD_MEAN_EPSILON: f32 = 1.0e-5;
pub(super) const DUAL_ENTRY: &str = r#"
struct DualOut {
    @location(0) @blend_src(0) color: vec4<f32>,
    @location(0) @blend_src(1) blend: vec4<f32>,
}

// Windfoil's fragment, with each colour channel blended by its own coverage where the glyph
// drawn set one (`lcd_cov`): subpixel text. Everything else blends as premultiplied alpha.
@fragment
fn fs(i: Out) -> DualOut {
    let c = fs_premul(i);
    if (lcd_cov.w == 0.0) {
        return DualOut(c, vec4<f32>(c.a));
    }
    let e = i.params.w;
    let mean = pow((lcd_cov.x + lcd_cov.y + lcd_cov.z) / 3.0, e);
    if (mean < 1.0e-5) {
        return DualOut(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    let k = pow(lcd_cov.xyz, vec3<f32>(e)) / mean;
    return DualOut(vec4<f32>(c.rgb * k, c.a), vec4<f32>(c.a * k, c.a));
}
"#;

/// The kinds the frame pass has a pipeline of its own for (`KIND_RECT`, `KIND_SHADOW`, `KIND_GLYPH`):
/// what most of a frame's fragments draw.
pub(crate) const PINNED: [u32; 3] = [Kind::Rect.as_u32(), Kind::Shadow.as_u32(), Kind::Glyph.as_u32()];
/// The kinds pinned where the pipelines compile from WGSL: `PINNED`'s, in its order, then the gradient kinds and images,
/// which have no shipped bytecode. `Renderer::pinned` is in this order, as long as it has pipelines for.
pub(crate) const PINNED_FROM_WGSL: [u32; 6] = [Kind::Rect.as_u32(), Kind::Shadow.as_u32(), Kind::Glyph.as_u32(), Kind::Radial.as_u32(), Kind::Gradient.as_u32(), Kind::Image.as_u32()];

/// shader.wgsl's marker for the kind a fragment draws: -1, every kind.
const PINNED_OFF: &str = shader_consts::PINNED_KIND_OFF_DECL;

/// `base_vs`'s line ahead of a box's split (and of the line that picks a quad's corner): the banded variant
/// narrows the quad there, so a segment's middle is taken within the band it draws in.
const CORNER: &str = "    // The banded variant narrows a gradient segment's quad here (render/variant.rs `BAND`), before a split.";

/// What the banded variant adds to `base_vs`. A linear gradient's stop segment (kind 11) covers its
/// whole element, and its fragments are zero wherever another segment draws: past an open stop
/// edge, and past a closed end whose colour is clear. A gradient of five stops over a hero so
/// shades the hero four times to colour it once. The part that
/// isn't zero is a band across the gradient, and here the quad rasterizes only the band's extent
/// along the axis the gradient mostly runs along, with a pixel of slack for the rounding between
/// this edge and the fragment's own comparison. The fragment stage is the
/// same code on the same flat inputs at the same pixel centres, so every pixel still shaded gets
/// the value the whole quad gave it, and a pixel no longer shaded got zero, which blends to
/// nothing.
const BAND: &str = r#"    if (kind > 10.5 && kind < 11.5 && i.uv.z == 0.0 && any(i.uv.xy != vec2<f32>(0.0))) {
        let lower_open = i.params.x >= 1.5;
        let upper_open = i.params.w >= 1.5;
        let t0 = select(i.params.x, i.params.x - 2.0, lower_open);
        let t1 = select(i.params.w, i.params.w - 2.0, upper_open);
        let below = lower_open || all(i.fill == vec4<f32>(0.0));
        let above = upper_open || all(i.border == vec4<f32>(0.0));
        // The axis the gradient mostly runs along, and how far a stop's edge moves along it from
        // one side of the quad to the other (nothing for a gradient straight along the axis; an
        // angle of 90 degrees is a few parts in a hundred million off it in f32).
        let down = abs(i.uv.y) >= abs(i.uv.x);
        let slope = select(i.uv.x, i.uv.y, down);
        let across = select(i.uv.y, i.uv.x, down);
        let reach = abs(across / slope) * select(hi.y - lo.y, hi.x - lo.x, down) * 0.5 + 1.0;
        let middle = select(i.rect.x + i.rect.z * 0.5, i.rect.y + i.rect.w * 0.5, down);
        let at0 = middle + (t0 - 0.5) / slope;
        let at1 = middle + (t1 - 0.5) / slope;
        var near = select(lo.x, lo.y, down);
        var far = select(hi.x, hi.y, down);
        if (slope > 0.0) {
            if (below) { near = max(near, at0 - reach); }
            if (above) { far = min(far, at1 + reach); }
        } else {
            if (below) { far = min(far, at0 + reach); }
            if (above) { near = max(near, at1 - reach); }
        }
        far = max(far, near);
        if (down) { lo.y = near; hi.y = far; } else { lo.x = near; hi.x = far; }
    }
"#;

/// Windfoil's WGSL for a pipeline compiled from source, with gradient stop segments banded in the
/// vertex stage (`BAND`). The shipped Metal and DX12 bytecode is compiled from shader.wgsl as it
/// is, and draws the same pixels with whole quads. `RUNE_GRADIENT_BANDS=0` compiles it unbanded.
pub(super) fn banded_wgsl() -> std::borrow::Cow<'static, str> {
    let windfoil = windfoil_wgsl();
    let off = std::env::var_os("RUNE_GRADIENT_BANDS").is_some_and(|v| v == "0");
    if off || windfoil.matches(CORNER).count() != 1 { return windfoil.into(); }
    windfoil.replacen(CORNER, &format!("{BAND}{CORNER}"), 1).into()
}

/// Windfoil's WGSL with its fragment entry pinned to `kind` (shader.wgsl `fragment_kind`): the
/// compiler drops every other kind's code. The vertex stage is unchanged.
pub(crate) fn pinned_wgsl(kind: u32) -> Option<String> {
    let windfoil = windfoil_wgsl();
    (windfoil.matches(PINNED_OFF).count() == 1).then(|| windfoil.replacen(PINNED_OFF, &format!("const PINNED_KIND: f32 = {kind}.0;"), 1))
}

/// `pinned_wgsl` with the banded variant's vertex stage (`banded_wgsl`), for the pinned pipelines made from WGSL here:
/// a gradient stop segment they draw rasterizes only its band, as it does with the frame's own pipeline.
pub(crate) fn pinned_banded_wgsl(kind: u32) -> Option<String> {
    let banded = banded_wgsl();
    (banded.matches(PINNED_OFF).count() == 1).then(|| banded.replacen(PINNED_OFF, &format!("const PINNED_KIND: f32 = {kind}.0;"), 1))
}

/// The kinds whose middle (shader.wgsl `split_hole`) draws with a pipeline of its own (`middle_wgsl`): boxes,
/// pictures and gradient segments, in the order `Renderer`'s middle pipelines are in.
pub(super) const MIDDLE_KINDS: [u32; 3] = [Kind::Rect.as_u32(), Kind::Image.as_u32(), Kind::Radial.as_u32()];
/// Slot of the slim plain-atlas-glyph pipeline in `Renderer::solid`, right after `MIDDLE_KINDS`.
pub(super) const PLAIN_GLYPH_SLOT: usize = MIDDLE_KINDS.len();
/// Slot of the vertex-interpolated linear-gradient middle pipeline in `Renderer::solid`.
pub(super) const LINEAR_MIDDLE_SLOT: usize = MIDDLE_KINDS.len() + 1;
/// Slot of the slim borderless circular-corner rect pipeline in `Renderer::solid`.
pub(super) const SIMPLE_RECT_SLOT: usize = MIDDLE_KINDS.len() + 2;
/// Slot of the ultra-slim upright 1:1 non-additive, non-cutout, unfaded layer composite pipeline in `Renderer::solid`.
pub(super) const PLAIN_LAYER_SLOT: usize = MIDDLE_KINDS.len() + 3;
/// Slot of the slim border-ring band pipeline in `Renderer::solid`.
pub(super) const RING_SLOT: usize = MIDDLE_KINDS.len() + 4;
/// Total per-format pipelines in `Renderer::solid`.
pub(super) const SOLID_COUNT: usize = MIDDLE_KINDS.len() + 5;

/// shader.wgsl's switch for a pipeline that draws middles only.
const MIDDLE_OFF: &str = "const MIDDLE: bool = false;";

/// The pipeline a `kind` shape's middle draws with: Windfoil pinned to that kind (`pinned_wgsl`), and every
/// fragment in a middle, so the compiler keeps only what a middle works out (a box's fill, a picture's texel, a
/// gradient's colour) and none of the coverage and clip around it. A small or software GPU runs that at its full
/// rate, where the uber shader's size holds it down, and it's the same code on the same inputs: the same pixels.
pub(super) fn middle_wgsl(kind: u32) -> Option<String> {
    let pinned = pinned_banded_wgsl(kind)?;
    (pinned.matches(MIDDLE_OFF).count() == 1).then(|| pinned.replacen(MIDDLE_OFF, "const MIDDLE: bool = true;", 1))
}

/// outline.wgsl's switch for subpixel outline glyphs, off but in the subpixel variant.
const SUBPIXEL_OFF: &str = shader_consts::OUTLINE_SUBPIXEL_OFF_DECL;

/// The subpixel variant's WGSL: Windfoil's, with subpixel outline coverage and its fragment
/// entry blending by a second output. Only this variant pays for the three coverages.
pub(super) fn dual_wgsl() -> Option<String> {
    let windfoil = windfoil_wgsl();
    let marked = windfoil.matches(FS_ENTRY).count() == 1 && windfoil.matches(SUBPIXEL_OFF).count() == 1;
    marked.then(|| {
        let body = windfoil.replacen(FS_ENTRY, "fn fs_premul(i: Out) -> vec4<f32> {", 1)
            .replacen(SUBPIXEL_OFF, "const OUTLINE_SUBPIXEL: bool = true;", 1);
        format!("enable dual_source_blending;\n{body}\n{DUAL_ENTRY}")
    })
}

/// Windfoil's subpixel pipeline for the window, built the first frame text draws subpixel masks
/// (`FontSmoothing::WindfoilRgb`, `ClearType`) on a Windfoil renderer whose device blends by a
/// second output. Layers keep the plain pipeline, so text in them draws grayscale.
pub(super) enum Lcd {
    Unbuilt,
    /// No shipped bytecode takes it here, so its WGSL compiles on a thread of its own (FXC takes
    /// about a minute for it on a Windows laptop) and subpixel text draws grayscale meanwhile.
    Building(std::sync::mpsc::Receiver<Result<wgpu::RenderPipeline, String>>),
    Built(wgpu::RenderPipeline),
    /// It can't be built here: subpixel text draws grayscale from now on.
    Refused,
}

/// Classic's bindings and the outline store's.
pub(crate) const fn windfoil_bind_entries() -> [wgpu::BindGroupLayoutEntry; 5] {
    let [a, b, c, d] = metallib::uber_bind_entries();
    let outlines = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Uint,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    };
    [a, b, c, d, wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: outlines, count: None }]
}

/// Windfoil's pipeline ABI, as `metallib::uber_abi_hash` hashes Classic's.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(crate) const fn windfoil_abi_hash() -> u64 {
    metallib::abi_hash_of(&windfoil_bind_entries())
}

/// Where a Windfoil shader came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    /// The bytecode Rune ships for it (DXBC on Windows, a metallib on macOS).
    #[cfg_attr(not(any(target_os = "macos", target_os = "windows")), allow(dead_code))]
    Precompiled,
    /// Its WGSL, compiled at run time.
    Compiled,
}

pub(super) enum Uber {
    /// shader.wgsl alone, and no outline store.
    Classic,
    /// shader.wgsl and outline.wgsl, with the outline store's texture and the atlas fill pass.
    Windfoil { source: Source, outlines: wgpu::Texture, fill: fill::FillPass },
    /// Windfoil was asked for and couldn't be built: Classic from now on.
    Unavailable,
}

pub(super) enum Event {
    /// Text drew glyphs from outlines.
    OutlinesWanted,
    /// Windfoil's pipeline was built.
    Built { source: Source, outlines: wgpu::Texture, fill: fill::FillPass },
    /// Windfoil's pipeline couldn't be built, and why.
    Refused(String),
}

#[derive(Debug, PartialEq)]
pub(super) enum Effect {
    /// Build Windfoil's pipeline, and send `Built` or `Refused`.
    Build,
    /// Say why text falls back from outlines.
    Fallback(String),
}

impl Uber {
    pub(super) fn step(&mut self, event: Event) -> Effects<Effect> {
        match (&*self, event) {
            (Uber::Classic, Event::OutlinesWanted) => Effects::one(Effect::Build),
            (Uber::Classic, Event::Built { source, outlines, fill }) => {
                *self = Uber::Windfoil { source, outlines, fill };
                Effects::none()
            }
            (Uber::Classic, Event::Refused(why)) => {
                *self = Uber::Unavailable;
                Effects::one(Effect::Fallback(why))
            }
            _ => Effects::none(),
        }
    }

    /// The outline store's texture, which Windfoil's bind groups hold.
    pub(super) fn outlines(&self) -> Option<&wgpu::Texture> {
        match self {
            Uber::Windfoil { outlines, .. } => Some(outlines),
            Uber::Classic | Uber::Unavailable => None,
        }
    }

    /// What text can expect of outlines from this renderer.
    pub(super) fn support(&self) -> crate::text::outline::Support {
        match self {
            Uber::Classic => crate::text::outline::Support::Untried,
            Uber::Windfoil { .. } => crate::text::outline::Support::Drawn,
            Uber::Unavailable => crate::text::outline::Support::Unavailable,
        }
    }
}

/// What building Windfoil needs from its renderer.
pub(super) struct BuildContext<'a> {
    pub format: wgpu::TextureFormat,
    pub vertex_entry: &'static str,
    pub globals: &'a wgpu::Buffer,
    pub stand_in: &'a wgpu::Texture,
}

/// A built Windfoil: its layouts, shader and frame pipeline.
pub(super) struct Built {
    pub bind_layout: wgpu::BindGroupLayout,
    pub layout: wgpu::PipelineLayout,
    pub shader: dxbc::Shaders,
    pub pipeline: wgpu::RenderPipeline,
    pub source: Source,
}

impl Renderer {
    /// Which uber shader draws, for diagnostics: `classic`, `windfoil (precompiled)`,
    /// `windfoil (compiled)` or `unavailable` (Windfoil couldn't be built).
    #[doc(hidden)]
    pub fn uber_shader(&self) -> &'static str {
        match self.uber {
            Uber::Classic => "classic",
            Uber::Windfoil { source: Source::Precompiled, .. } => "windfoil (precompiled)",
            Uber::Windfoil { source: Source::Compiled, .. } => "windfoil (compiled)",
            Uber::Unavailable => "unavailable",
        }
    }

    /// Tells `atlas` what this renderer does with outlines (`text::outline::Support`) before it
    /// first draws, so the first frame's glyph masks are queued for the GPU fill rather than
    /// filled on the CPU. `prepare` tells it again every frame.
    pub fn attach_text(&self, atlas: &mut Atlas) {
        atlas.outlines.set_support(self.uber.support());
    }

    /// Steps the shader machine `state` on `event`, carrying out each effect (building Windfoil
    /// feeds `Built` or `Refused` back in) until none is left; with the Windfoil pipeline it
    /// built on the way, if it built one.
    fn run(device: &wgpu::Device, cx: &BuildContext<'_>, state: &mut Uber, event: Event) -> Option<Built> {
        let mut built = None;
        let mut queue: Vec<Effect> = state.step(event).into_iter().collect();
        while let Some(effect) = queue.pop() {
            match effect {
                Effect::Build => {
                    let next = match Self::build_windfoil(device, cx.format, cx.vertex_entry) {
                        Ok(b) => {
                            let outlines = Self::outline_texture(device, outlines::MIN_ROWS);
                            let fill = fill::FillPass::new(device, &b.layout, &b.shader, fill::FillBindings {
                                layout: &b.bind_layout, globals: cx.globals, outlines: &outlines, stand_in: cx.stand_in });
                            let event = Event::Built { source: b.source, outlines, fill };
                            built = Some(b);
                            event
                        }
                        Err(why) => Event::Refused(why),
                    };
                    queue.extend(state.step(next));
                }
                Effect::Fallback(why) => eprintln!("rune: glyphs can't draw from outlines here ({why}); text draws from the atlas alone"),
            }
        }
        built
    }
    /// A new renderer's shader machine, asked for Windfoil at once.
    pub(super) fn start_windfoil(device: &wgpu::Device, cx: &BuildContext<'_>) -> (Uber, Option<Built>) {
        let mut state = Uber::Classic;
        let built = Self::run(device, cx, &mut state, Event::OutlinesWanted);
        (state, built)
    }

    /// Steps this renderer's shader machine on `event`, taking any Windfoil pipeline it builds;
    /// whether it took one (the caller then makes its atlas pages ones the GPU can fill).
    pub(super) fn select(&mut self, device: &wgpu::Device, event: Event) -> bool {
        let cx = BuildContext { format: self.format, vertex_entry: self.vertex_entry, globals: &self.globals, stand_in: &self.stand_in };
        if let Some(built) = Self::run(device, &cx, &mut self.uber, event) {
            self.adopt(device, built);
            self.rebind_all(device);
            return true;
        }
        false
    }

    /// Windfoil's layouts, shader and pipeline for `format`, from the shipped bytecode where it
    /// matches, else from WGSL; an error where neither builds.
    pub(super) fn build_windfoil(device: &wgpu::Device, format: wgpu::TextureFormat, vertex_entry: &'static str) -> Result<Built, String> {
        if vertex_entry != "vs" {
            return Err("GPU springs draw the classic shader".into());
        }
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &windfoil_bind_entries() });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bind_layout)], immediate_size: 0 });
        let make_pipeline = |shader: &dxbc::Shaders| Self::pipeline(device, &layout, shader, format, vertex_entry, Blend::SourceOver);
        #[cfg(target_os = "windows")]
        if let Some((shader, pipeline)) = dxbc::uber(device, make_pipeline) {
            return Ok(Built { bind_layout, layout, shader, pipeline, source: Source::Precompiled });
        }
        #[cfg(target_os = "macos")]
        if let Some((shader, pipeline)) = windfoil_metallib::load(device, make_pipeline) {
            return Ok(Built { bind_layout, layout, shader: dxbc::Shaders::one(shader), pipeline, source: Source::Precompiled });
        }
        let (shader, pipeline) = compile(device, &banded_wgsl(), make_pipeline)?;
        Ok(Built { bind_layout, layout, shader, pipeline, source: Source::Compiled })
    }

    /// Whether this frame's subpixel text can blend by channel, starting the pipeline the first
    /// time it's asked: from the shipped bytecode at once where it matches, else compiled on a
    /// thread, which `frame_wake` tells when it's done. Until then subpixel text draws grayscale.
    /// `want`: this frame's text draws subpixel masks. A compile already running is collected
    /// either way, since text draws grayscale while it runs.
    pub(super) fn lcd_ready(&mut self, device: &wgpu::Device, want: bool) -> bool {
        if want && matches!(self.lcd, Lcd::Unbuilt) && self.uber.outlines().is_some() {
            self.lcd = match self.start_lcd(device) {
                Ok(lcd) => lcd,
                Err(why) => Self::refuse_lcd(&why),
            };
            if matches!(self.lcd, Lcd::Built(_)) { self.lcd_built(); }
        }
        if let Lcd::Building(done) = &self.lcd {
            match done.try_recv() {
                Ok(Ok(pipeline)) => {
                    self.lcd = Lcd::Built(pipeline);
                    self.lcd_built();
                }
                Ok(Err(why)) => self.lcd = Self::refuse_lcd(&why),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.lcd = Self::refuse_lcd("its compile ended without a pipeline"),
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        want && matches!(self.lcd, Lcd::Built(_))
    }

    /// Subpixel text can blend by channel from now on: it switches at its next frame, on the CPU
    /// and the GPU together, which the window is asked for.
    fn lcd_built(&self) {
        crate::text::note_dual_source(true);
        if let Some(wake) = &self.frame_wake { wake.notify(); }
    }

    fn refuse_lcd(why: &str) -> Lcd {
        eprintln!("rune: subpixel text can't blend by channel here ({why}); it draws grayscale");
        crate::text::note_dual_source(false);
        Lcd::Refused
    }

    fn start_lcd(&self, device: &wgpu::Device) -> Result<Lcd, String> {
        if !device.features().contains(wgpu::Features::DUAL_SOURCE_BLENDING) {
            return Err("the device has no dual-source blending".into());
        }
        let wgsl = dual_wgsl().ok_or("shader.wgsl's fragment entry moved")?;
        let (layout, format, entry) = (self.layout.clone(), self.format, self.vertex_entry);
        #[cfg(target_os = "windows")]
        if let Some((_, pipeline)) = dxbc::uber_dual(device, |shader| lcd_pipeline(device, &layout, format, entry, shader)) {
            return Ok(Lcd::Built(pipeline));
        }
        #[cfg(target_arch = "wasm32")]
        return compile(device, &wgsl, |shader| lcd_pipeline(device, &layout, format, entry, shader)).map(|(_, pipeline)| Lcd::Built(pipeline));
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (sent, done) = std::sync::mpsc::channel();
            let (device, wake) = (device.clone(), self.frame_wake.clone());
            std::thread::Builder::new()
                .name("rune-subpixel-shader".into())
                .spawn(move || {
                    // A renderer dropped meanwhile has no receiver: nothing to tell.
                    sent.send(compile(&device, &wgsl, |shader| lcd_pipeline(&device, &layout, format, entry, shader)).map(|(_, pipeline)| pipeline)).ok();
                    if let Some(wake) = wake { wake.notify(); }
                })
                .map_err(|e| format!("no thread to compile it on: {e}"))?;
            Ok(Lcd::Building(done))
        }
    }

    /// The window's wake, which a subpixel pipeline compiled off this thread tells when it's
    /// ready, so a window that rests meanwhile still draws its next frame with it.
    pub(crate) fn set_frame_wake(&mut self, wake: &std::sync::Arc<crate::live::Wake>) {
        if !self.frame_wake.as_ref().is_some_and(|w| std::sync::Arc::ptr_eq(w, wake)) {
            self.frame_wake = Some(wake.clone());
        }
    }

    #[cfg(test)]
    pub(crate) fn has_frame_wake(&self) -> bool {
        self.frame_wake.is_some()
    }

    /// The subpixel pipeline compiled from WGSL now, on this thread, for the tools that capture
    /// its HLSL (`regen_dxbc`): it bypasses the shipped bytecode.
    #[doc(hidden)]
    pub fn compile_subpixel_now(&mut self, device: &wgpu::Device) -> Result<(), String> {
        let wgsl = dual_wgsl().ok_or("shader.wgsl's fragment entry moved")?;
        let (layout, format, entry) = (self.layout.clone(), self.format, self.vertex_entry);
        let (_, pipeline) = compile(device, &wgsl, |shader| lcd_pipeline(device, &layout, format, entry, shader))?;
        self.lcd = Lcd::Built(pipeline);
        Ok(())
    }

    /// Compiles the pipeline pinned to `KIND_*` `kind` (`pinned_wgsl`) from WGSL now, so wgpu logs
    /// its HLSL (`regen_dxbc`): it bypasses the shipped bytecode.
    #[doc(hidden)]
    pub fn compile_pinned_now(&mut self, device: &wgpu::Device, kind: u32) -> Result<(), String> {
        let wgsl = pinned_wgsl(kind).ok_or("shader.wgsl's PINNED_KIND moved")?;
        let (layout, format, entry) = (self.layout.clone(), self.format, self.vertex_entry);
        compile(device, &wgsl, |shader| Self::pipeline(device, &layout, shader, format, entry, Blend::SourceOver))?;
        Ok(())
    }

    /// Starts making the frame pass's pipelines pinned to one kind each (`PINNED`) from the shipped bytecode, on a
    /// thread of their own, for Windows renderers drawing Windfoil from its precompiled shader: on a cold driver cache
    /// (the first launch after an install or a driver update) the driver takes about 150 ms for the three, which no
    /// frame should wait for. The general shader draws every kind until they arrive (`take_pinned`), the same picture.
    /// Anywhere else, or when the bytecode doesn't match, none: every kind draws with the one shader, as before.
    /// Never from WGSL, which takes FXC tens of seconds each.
    #[allow(unsafe_code)]
    pub(super) fn start_pinned(&mut self, device: &wgpu::Device) {
        self.pinned = None;
        self.pinned_building = None;
        #[cfg(target_os = "windows")]
        if matches!(self.uber, Uber::Windfoil { source: Source::Precompiled, .. }) && self.vertex_entry == "vs" {
            let (sent, done) = std::sync::mpsc::channel();
            let (device, layout, format) = (device.clone(), self.layout.clone(), self.format);
            let spawned = std::thread::Builder::new()
                .name("rune-pinned-pipelines".into())
                // A renderer dropped meanwhile has no receiver: nothing to tell.
                .spawn(move || {
                    // Below normal, as the CPU frame's helpers are: on a two-core machine a cold driver cache makes this a
                    // 150 ms compile, which mustn't compete with the first input.
                    // SAFETY: this thread's pseudo-handle.
                    unsafe { windows_sys::Win32::System::Threading::SetThreadPriority(windows_sys::Win32::System::Threading::GetCurrentThread(), windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL) };
                    sent.send(make_pinned(&device, &layout, format)).ok();
                });
            if spawned.is_ok() {
                self.pinned_building = Some((format, done));
                return;
            }
        }
        // Where the uber shader is compiled from WGSL as the renderer starts (Vulkan, GL), its pinned
        // pipelines can be too, on a thread: naga and the driver take milliseconds for each, where FXC
        // takes tens of seconds. The smaller shaders shade a pixel for less (a tenth off the GPU's time
        // on scrolling pages on an M1), and the draw per run of a kind costs the CPU about as much in
        // that machine's driver. On a small GPU with a small CPU (Intel's 24-EU UHD on an N100, a bar's
        // sheets and launcher) they took a fifth off the GPU's time for a few percent more CPU, and the
        // frames inside a 240 fps budget went from 31% to 42%. `RUNE_PINNED_WGSL=0` turns them off.
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "ios", target_arch = "wasm32")))]
        if matches!(self.uber, Uber::Windfoil { source: Source::Compiled, .. }) && self.vertex_entry == "vs"
            && std::env::var_os("RUNE_PINNED_WGSL").is_none_or(|v| v != "0") {
            let (sent, done) = std::sync::mpsc::channel();
            let (device, layout, format) = (device.clone(), self.layout.clone(), self.format);
            let spawned = std::thread::Builder::new().name("rune-pinned-pipelines".into()).spawn(move || {
                // For the window's format, then for its layers' (`layers::FORMAT`), where a fading page draws.
                let made: Vec<_> = PINNED_FROM_WGSL.iter().map(|&kind| {
                    let wgsl = pinned_banded_wgsl(kind)?;
                    compile(&device, &wgsl, |shader| Renderer::pipeline(&device, &layout, shader, format, "vs", Blend::SourceOver)).ok()
                }).collect();
                let layer = made.iter().map(|m| m.as_ref().map(|(shader, _)| Renderer::pipeline(&device, &layout, shader, super::layers::FORMAT, "vs", Blend::SourceOver))).collect::<Vec<_>>();
                let pinned: Vec<_> = made.into_iter().map(|m| m.map(|(_, p)| p)).chain(layer).collect();
                sent.send(if pinned.iter().any(Option::is_some) { pinned } else { Vec::new() }).ok();
            });
            if spawned.is_ok() {
                self.pinned_building = Some((format, done));
                return;
            }
        }
        let _ = device;
        self.pinned = Some(Vec::new());
    }

    /// The WGSL a layer's composite draws with, pinned to it (`Kind::Layer`), where the uber shader compiles from
    /// WGSL here: drawn with the uber shader, every fragment of a composite pays for registers its other kinds hold,
    /// and a small GPU hides a float texel's fetch behind fewer fragments at a time.
    pub(super) fn composite_wgsl(&self) -> Option<String> {
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "ios", target_arch = "wasm32")))]
        if matches!(self.uber, Uber::Windfoil { source: Source::Compiled, .. }) && self.vertex_entry == "vs"
            && std::env::var_os("RUNE_PINNED_WGSL").is_none_or(|v| v != "0") {
            return Some(layer_composite_wgsl());
        }
        None
    }

    /// The pipelines shapes' middles draw with (`middle_wgsl`) for the renderer's format, made on a thread the first
    /// frame that has none (where the uber shader compiles from WGSL here) and taken once they're made. Meanwhile,
    /// and where one can't be made, a middle draws with the uber shader, which works out the same there (`i.whole`).
    pub(super) fn solid_for_format(&mut self, device: &wgpu::Device) {
        if self.solid.as_ref().is_some_and(|(f, _)| *f == self.format) { return; }
        if let Some((format, done)) = &self.solid_building {
            // Frame-profile builds wait for it, as for the pinned pipelines (`take_pinned`).
            #[cfg(feature = "frame-profile")]
            let got = done.recv().map_err(|_| std::sync::mpsc::TryRecvError::Disconnected);
            #[cfg(not(feature = "frame-profile"))]
            let got = done.try_recv();
            match got {
                Ok(made) => {
                    if *format == self.format { self.solid = Some((*format, made)); }
                    self.solid_building = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.solid = Some((*format, Vec::new()));
                    self.solid_building = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            if self.solid_building.is_some() || self.solid.as_ref().is_some_and(|(f, _)| *f == self.format) { return; }
        }
        // Asked for once per format: where it can't be made, or isn't made from WGSL, the uber shader draws middles.
        if !matches!(self.uber, Uber::Windfoil { source: Source::Compiled, .. }) || self.vertex_entry != "vs" {
            self.solid = Some((self.format, Vec::new()));
            return;
        }
        let (sent, done) = std::sync::mpsc::channel();
        let (made_on, layout, format) = (device.clone(), self.layout.clone(), self.format);
        let spawned = std::thread::Builder::new().name("rune-middle-pipelines".into()).spawn(move || {
            // For the window's format, then for its layers' (`layers::FORMAT`), where a fading page draws.
            let mut made: Vec<_> = MIDDLE_KINDS.iter().map(|&kind| {
                let wgsl = if kind == Kind::Rect.as_u32() {
                    solid_rect_wgsl()
                } else if kind == Kind::Radial.as_u32() {
                    radial_middle_wgsl()
                } else {
                    middle_wgsl(kind)?
                };
                compile(&made_on, &wgsl, |shader| Renderer::pipeline(&made_on, &layout, shader, format, "vs", Blend::SourceOver)).ok()
            }).collect();
            let glyph_wgsl = plain_glyph_wgsl();
            made.push(compile(&made_on, &glyph_wgsl, |shader| Renderer::pipeline(&made_on, &layout, shader, format, "vs", Blend::SourceOver)).ok());
            let linear_wgsl = linear_grad_middle_wgsl();
            made.push(compile(&made_on, &linear_wgsl, |shader| Renderer::pipeline(&made_on, &layout, shader, format, "vs", Blend::SourceOver)).ok());
            let simple_wgsl = simple_rect_wgsl();
            made.push(compile(&made_on, &simple_wgsl, |shader| Renderer::pipeline(&made_on, &layout, shader, format, "vs", Blend::SourceOver)).ok());
            let plain_layer_wgsl = plain_layer_composite_wgsl();
            made.push(compile(&made_on, &plain_layer_wgsl, |shader| Renderer::pipeline(&made_on, &layout, shader, format, "vs", Blend::SourceOver)).ok());
            let ring_wgsl = simple_ring_wgsl();
            made.push(compile(&made_on, &ring_wgsl, |shader| Renderer::pipeline(&made_on, &layout, shader, format, "vs", Blend::SourceOver)).ok());
            let layer = made.iter().map(|m| m.as_ref().map(|(shader, _)| Renderer::pipeline(&made_on, &layout, shader, super::layers::FORMAT, "vs", Blend::SourceOver))).collect::<Vec<_>>();
            let made = made.into_iter().map(|m| m.map(|(_, p)| p)).chain(layer).collect::<Vec<_>>();
            sent.send(made).ok();
        });
        if spawned.is_ok() {
            self.solid_building = Some((format, done));
            // Frame-profile builds take it now, as they take the pinned pipelines.
            #[cfg(feature = "frame-profile")]
            self.solid_for_format(device);
        }
    }

    /// Takes the pinned pipelines once their thread has made them for this format (one made for a format the
    /// renderer has since left is dropped, and that format's next use starts again).
    pub(super) fn take_pinned(&mut self) {
        let Some((format, done)) = &self.pinned_building else { return };
        // Frame-profile builds wait for them, so every frame a capture or a frame budget times draws as the last does
        // and no compile on their thread lands in a frame's counts; every other build never waits.
        #[cfg(feature = "frame-profile")]
        let got = done.recv().map_err(|_| std::sync::mpsc::TryRecvError::Disconnected);
        #[cfg(not(feature = "frame-profile"))]
        let got = done.try_recv();
        match got {
            Ok(pinned) => {
                if *format == self.format { self.pinned = Some(pinned); }
                self.pinned_building = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pinned = Some(Vec::new());
                self.pinned_building = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
    }
    /// Takes `built` as the renderer's shader: every pipeline again from it. Bind groups follow
    /// once the outline texture exists (`rebind_all`).
    fn adopt(&mut self, device: &wgpu::Device, built: Built) {
        self.bind_layout = built.bind_layout;
        self.layout = built.layout;
        self.shader = built.shader;
        self.pipeline = built.pipeline;
        for index in 0..self.extra.len() {
            let format = self.extra[index].0;
            self.extra[index].1 = Self::pipeline(device, &self.layout, &self.shader, format, self.vertex_entry, Blend::SourceOver);
        }
        self.shader_changed(device);
        // Built on the old bind layout; made again at the next backdrop blur and subpixel frame.
        self.backdrop_blur = None;
        self.lcd = Lcd::Unbuilt;
        // The pinned pipelines again from the new shader, on their thread: the general shader draws until they're in.
        self.pinned_extra.clear();
        self.start_pinned(device);
    }
}

/// The subpixel pipeline from `shader`'s stages: Windfoil's, blending each channel by the
/// fragment's second output.
fn lcd_pipeline(device: &wgpu::Device, layout: &wgpu::PipelineLayout, format: wgpu::TextureFormat, vertex_entry: &'static str, shader: &dxbc::Shaders) -> wgpu::RenderPipeline {
    let blend = wgpu::BlendState {
        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrc1, operation: wgpu::BlendOperation::Add },
        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("uber subpixel"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader.vertex,
            entry_point: Some(vertex_entry),
            compilation_options: Default::default(),
            buffers: &[Some(metallib::uber_vertex_layout())],
        },
        primitive: metallib::uber_primitive(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader.fragment,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend: Some(blend), write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// `wgsl`'s module and pipeline, compiled now, or the error the device reported.
pub(super) fn compile(device: &wgpu::Device, wgsl: &str, make_pipeline: impl FnOnce(&dxbc::Shaders) -> wgpu::RenderPipeline) -> Result<(dxbc::Shaders, wgpu::RenderPipeline), String> {
    #[cfg(not(target_arch = "wasm32"))]
    let scopes = (device.push_error_scope(wgpu::ErrorFilter::Validation), device.push_error_scope(wgpu::ErrorFilter::Internal));
    let shader = dxbc::Shaders::one(device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("uber"), source: wgpu::ShaderSource::Wgsl(wgsl.into()) }));
    let pipeline = make_pipeline(&shader);
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(error) = pollster::block_on(scopes.1.pop()).or(pollster::block_on(scopes.0.pop())) {
        return Err(error.to_string());
    }
    Ok((shader, pipeline))
}

/// The pinned pipelines for `format` (`Renderer::start_pinned`'s thread): none taken (refused, or every fragment
/// stale) is none at all, so nothing is split.
#[cfg(target_os = "windows")]
fn make_pinned(device: &wgpu::Device, layout: &wgpu::PipelineLayout, format: wgpu::TextureFormat) -> Vec<Option<wgpu::RenderPipeline>> {
    #[cfg(feature = "frame-profile")]
    let started = crate::time::now();
    let pinned: Vec<_> = PINNED.iter().map(|&kind| dxbc::uber_pinned(device, kind, |shader| Renderer::pipeline(device, layout, shader, format, "vs", Blend::SourceOver)).map(|(_, p)| p)).collect();
    // Frame-profile builds: what making them costs, and what the run-time freshness check before `PINNED_FRESH`
    // cost (three `pinned_wgsl` and their hashes).
    #[cfg(feature = "frame-profile")]
    {
        let made = started.elapsed();
        let checked = crate::time::now();
        let old: usize = PINNED.iter().filter_map(|&kind| super::pinned_wgsl_blake3(kind)).map(|h| h.len()).sum();
        eprintln!("rune-warm pinned pipelines: {:.0} us to make {} of 3 on their thread (the run-time check they replace: {:.0} us, {old} hex)", made.as_secs_f64() * 1e6, pinned.iter().filter(|p| p.is_some()).count(), checked.elapsed().as_secs_f64() * 1e6);
    }
    if pinned.iter().any(Option::is_some) { pinned } else { Vec::new() }
}