//! Specialized fast-path pipelines for common UI primitives.
//!
//! All pipelines share the uber shader's bind group layout (`@group(0)`) and vertex buffer layout
//! (`uber_vertex_layout`), so switching between them costs only `set_pipeline` with zero bind
//! group or vertex buffer rebinds. Each shader is tiny and — for opaque primitives and opaque
//! interiors of large rounded boxes — disables alpha blending (`blend: None`), avoiding
//! destination reads and blend work on both software rasterizers (lavapipe) and tile-based mobile
//! GPUs (PowerVR / Mali).

use super::{CLIP_WGSL, Instance, Kind, metallib};

/// Minimum pixel area for an instance to take a dedicated fast-path pipeline (avoids pipeline
/// switches for tiny chips, dots, and icons that batch cheaply in the general shader).
const MIN_AREA: f32 = 4096.0;
/// Minimum pixel area for the 2-pass interior+ring split (avoids doubling draw calls for medium
/// cards while eliminating alpha blending on large panels).
const SPLIT_AREA: f32 = 120_000.0;

/// Which fast-path pipeline an instance qualifies for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FastKind {
    General,
    OpaqueRect,
    OpaqueRoundedRect,
    OpaqueLinear,
    RoundedLinear,
    OpaqueRoundedLinear,
    OpaqueRadial,
    LinearBand,
    Shadow,
}

#[inline]
fn is_int(v: f32) -> bool {
    v.is_finite() && (v - v.round()).abs() <= 1e-4
}

#[inline]
fn int_rect(r: [f32; 4]) -> bool {
    is_int(r[0]) && is_int(r[1]) && is_int(r[2]) && is_int(r[3])
}

#[inline]
fn int_or_no_clip(c: [f32; 4]) -> bool {
    c[0] < -1e5 || (is_int(c[0]) && is_int(c[1]) && is_int(c[2]) && is_int(c[3]))
}

#[inline]
fn max_radius(r: [f32; 4]) -> f32 {
    r[0].max(r[1]).max(r[2]).max(r[3])
}

/// Classifies an instance for fast-path dispatch.
#[inline]
pub(super) fn classify(i: &Instance) -> FastKind {
    let Some(kind) = i.kind() else { return FastKind::General };
    if i.fade != [0; 4] {
        return FastKind::General;
    }
    let area = i.rect[2] * i.rect[3];
    match kind {
        Kind::Rect => {
            if area < MIN_AREA || i.fill[3] < 1.0 || i.params[2] > 0.5 || !int_or_no_clip(i.clip) {
                return FastKind::General;
            }
            if i.params[0] == 0.0 && i.radii == [0.0; 4] && int_rect(i.rect) {
                FastKind::OpaqueRect
            } else if area >= SPLIT_AREA
                && i.radii.iter().all(|&r| r >= 0.0)
                && (i.params[0] == 0.0 || (i.uv[2] >= 0.0 && i.uv == [0.0; 4]))
            {
                let inset = (max_radius(i.radii).max(i.params[0]) + 1.0).ceil();
                if i.rect[2] > 2.0 * inset + 16.0 && i.rect[3] > 2.0 * inset + 16.0 {
                    FastKind::OpaqueRoundedRect
                } else {
                    FastKind::General
                }
            } else {
                FastKind::General
            }
        }
        Kind::Shadow => {
            // One corner radius for every corner (`radii.x`), as the fast shader reads it; a hole
            // (`uv.z > 0`) keeps its own in `radii.w`. Mixed corners take the uber shader.
            let one_corner = i.uv[2] > 0.0 || i.radii[1..].iter().all(|r| r.abs() == i.radii[0].abs());
            if one_corner && int_or_no_clip(i.clip) {
                FastKind::Shadow
            } else {
                FastKind::General
            }
        }
        Kind::Radial => {
            if area < MIN_AREA || i.params[2] > 0.5 || !int_or_no_clip(i.clip) {
                return FastKind::General;
            }
            if i.uv[2] > 0.0 {
                if i.fill[3] >= 1.0
                    && i.border[3] >= 1.0
                    && i.radii == [0.0; 4]
                    && int_rect(i.rect)
                {
                    FastKind::OpaqueRadial
                } else {
                    FastKind::General
                }
            } else if i.params[0] < 1.5 && i.params[3] < 1.5 && i.params[3] > i.params[0] + 1e-6 {
                if i.fill[3] >= 1.0 && i.border[3] >= 1.0 && i.radii == [0.0; 4] && int_rect(i.rect) {
                    FastKind::OpaqueLinear
                } else if area >= SPLIT_AREA && i.fill[3] >= 1.0 && i.border[3] >= 1.0 && i.radii.iter().all(|&r| r >= 0.0) {
                    let inset = (max_radius(i.radii) + 1.0).ceil();
                    if i.rect[2] > 2.0 * inset + 16.0 && i.rect[3] > 2.0 * inset + 16.0 {
                        FastKind::OpaqueRoundedLinear
                    } else {
                        FastKind::RoundedLinear
                    }
                } else if i.radii.iter().all(|&r| r >= 0.0) {
                    FastKind::RoundedLinear
                } else {
                    FastKind::General
                }
            } else if i.radii == [0.0; 4] && int_rect(i.rect) {
                FastKind::LinearBand
            } else {
                FastKind::General
            }
        }
        Kind::Glyph | Kind::Image => FastKind::General,
        _ => FastKind::General,
    }
}

/// One linear-gradient band fused into [`FusedGradUniform`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FusedBand {
    pub(super) fill: [f32; 4],
    pub(super) border: [f32; 4],
    pub(super) dir_stops: [f32; 4],
    pub(super) flags: [f32; 4],
}

/// Uniform payload for a single-pass 3-stop opaque radial gradient + up to 4 overlaid linear
/// gradient bands (e.g. a hero backdrop with horizontal and vertical vignettes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FusedGradUniform {
    pub(super) rect: [f32; 4],
    pub(super) clip: [f32; 4],
    pub(super) rad_center_inv: [f32; 4],
    pub(super) rad_stops: [f32; 4],
    pub(super) rad_c0: [f32; 4],
    pub(super) rad_c1: [f32; 4],
    pub(super) rad_c2: [f32; 4],
    pub(super) bands: [FusedBand; 4],
}

/// GPU buffer and `@group(1)` bind group for [`FusedGradUniform`].
pub(super) struct FusedGradResources {
    pub(super) layout: wgpu::BindGroupLayout,
    pub(super) buffer: wgpu::Buffer,
    pub(super) bind_group: wgpu::BindGroup,
}

impl FusedGradResources {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fused_grad_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<FusedGradUniform>() as u64),
                },
                count: None,
            }],
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_grad_ub"),
            size: std::mem::size_of::<FusedGradUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_grad_bg"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self { layout, buffer, bind_group }
    }
}

/// Detects a 2-segment (3-stop) opaque radial gradient optionally followed by up to 4 overlaid
/// `LinearBand`s sharing the same box and clip, so they can be shaded in one unblended pass.
pub(super) fn detect_fused_grad(instances: &[Instance], fast_kinds: &[FastKind]) -> Option<(std::ops::Range<u32>, FusedGradUniform)> {
    let n = instances.len().min(fast_kinds.len());
    let mut i = 0;
    while i + 1 < n {
        if fast_kinds[i] == FastKind::OpaqueRadial && fast_kinds[i + 1] == FastKind::OpaqueRadial {
            let a = &instances[i];
            let b = &instances[i + 1];
            let a_lower_open = a.params[0] >= 1.5;
            let a_upper_open = a.params[3] >= 1.5;
            let b_lower_open = b.params[0] >= 1.5;
            let b_upper_open = b.params[3] >= 1.5;
            if !a_lower_open && a_upper_open && b_lower_open && !b_upper_open && a.clip == b.clip && a.uv[2] == b.uv[2] && a.uv[3] == b.uv[3] && a.border == b.fill {
                let cx_a = a.rect[0] + a.uv[0];
                let cy_a = a.rect[1] + a.uv[1];
                let cx_b = b.rect[0] + b.uv[0];
                let cy_b = b.rect[1] + b.uv[1];
                let t0 = a.params[0];
                let t1_a = a.params[3] - 2.0;
                let t1_b = b.params[0] - 2.0;
                let t2 = b.params[3];
                let encloses = b.rect[0] <= a.rect[0] + 0.5
                    && b.rect[1] <= a.rect[1] + 0.5
                    && b.rect[0] + b.rect[2] >= a.rect[0] + a.rect[2] - 0.5
                    && b.rect[1] + b.rect[3] >= a.rect[1] + a.rect[3] - 0.5;
                if encloses && (cx_a - cx_b).abs() <= 1e-2 && (cy_a - cy_b).abs() <= 1e-2 && (t1_a - t1_b).abs() <= 1e-4 && t1_a > t0 + 1e-6 && t2 > t1_a + 1e-6 {
                    let mut bands = [FusedBand { fill: [0.0; 4], border: [0.0; 4], dir_stops: [0.0; 4], flags: [0.0; 4] }; 4];
                    let mut nb = 0usize;
                    let mut end = i + 2;
                    while end < n && nb < 4 && fast_kinds[end] == FastKind::LinearBand {
                        let ln = &instances[end];
                        if ln.rect != b.rect || ln.clip != b.clip {
                            break;
                        }
                        let lo = ln.params[0] >= 1.5;
                        let hi = ln.params[3] >= 1.5;
                        let s0 = if lo { ln.params[0] - 2.0 } else { ln.params[0] };
                        let s1 = if hi { ln.params[3] - 2.0 } else { ln.params[3] };
                        let inv_span = if s1 > s0 + 1e-6 { 1.0 / (s1 - s0) } else { 0.0 };
                        bands[nb] = FusedBand {
                            fill: ln.fill,
                            border: ln.border,
                            dir_stops: [ln.uv[0], ln.uv[1], s0, s1],
                            flags: [if lo { 1.0 } else { 0.0 }, if hi { 1.0 } else { 0.0 }, inv_span, 1.0],
                        };
                        nb += 1;
                        end += 1;
                    }
                    // Check if all bands are pure vignettes fading toward `b.border` (same RGB):
                    // lets `fs` combine them with one `mix(rad_c2, col, keep)` and skip `t >= s2`.
                    let matches_c2 = |c: [f32; 4]| {
                        c[3] <= 1e-5
                            || ((c[0] - b.border[0] * c[3]).abs() <= 2e-3
                                && (c[1] - b.border[1] * c[3]).abs() <= 2e-3
                                && (c[2] - b.border[2] * c[3]).abs() <= 2e-3)
                    };
                    if nb > 0 && bands[..nb].iter().all(|bn| matches_c2(bn.fill) && matches_c2(bn.border)) {
                        bands[0].flags[3] = 2.0;
                    }
                    let uniform = FusedGradUniform {
                        rect: b.rect,
                        clip: b.clip,
                        rad_center_inv: [cx_b, cy_b, 1.0 / b.uv[2], 1.0 / b.uv[3]],
                        rad_stops: [t0, t1_a, t2, nb as f32],
                        rad_c0: a.fill,
                        rad_c1: a.border,
                        rad_c2: b.border,
                        bands,
                    };
                    return Some((i as u32..end as u32, uniform));
                }
            }
        }
        i += 1;
    }
    None
}

/// Fast-path pipelines built for one target `wgpu::TextureFormat`.
pub(super) struct FastPipelines {
    pub(super) opaque_rect: wgpu::RenderPipeline,
    pub(super) opaque_rect_interior: wgpu::RenderPipeline,
    pub(super) rounded_rect_ring: wgpu::RenderPipeline,
    pub(super) opaque_linear: wgpu::RenderPipeline,
    pub(super) rounded_linear: wgpu::RenderPipeline,
    pub(super) opaque_linear_interior: wgpu::RenderPipeline,
    pub(super) rounded_linear_ring: wgpu::RenderPipeline,
    pub(super) opaque_radial: wgpu::RenderPipeline,
    pub(super) linear_band: wgpu::RenderPipeline,
    pub(super) shadow: wgpu::RenderPipeline,
    pub(super) fused_grad: wgpu::RenderPipeline,
    pub(super) fast_layer: wgpu::RenderPipeline,
    pub(super) fast_layer_over_clear: wgpu::RenderPipeline,
}

impl FastPipelines {
    pub(super) fn new(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        bind_layout: &wgpu::BindGroupLayout,
        fused_bgl: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
    ) -> Self {
        let make = |label: &str, src: &str, topology: wgpu::PrimitiveTopology, blend: Option<wgpu::BlendState>| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(metallib::uber_vertex_layout())],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let strip = wgpu::PrimitiveTopology::TriangleStrip;
        let list = wgpu::PrimitiveTopology::TriangleList;
        let premul = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let fused_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fused_grad_layout"),
            bind_group_layouts: &[Some(bind_layout), Some(fused_bgl)],
            immediate_size: 0,
        });
        let fused_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fast_fused_grad"),
            source: wgpu::ShaderSource::Wgsl(FUSED_GRAD_WGSL.into()),
        });
        let fused_grad = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fast_fused_grad"),
            layout: Some(&fused_layout),
            vertex: wgpu::VertexState {
                module: &fused_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState { topology: strip, ..Default::default() },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fused_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            opaque_rect: make("fast_opaque_rect", OPAQUE_RECT_WGSL, strip, None),
            opaque_rect_interior: make("fast_opaque_rect_interior", OPAQUE_RECT_INTERIOR_WGSL, strip, None),
            rounded_rect_ring: make("fast_rounded_rect_ring", ROUNDED_RECT_RING_WGSL, list, premul),
            opaque_linear: make("fast_opaque_linear", OPAQUE_LINEAR_WGSL, strip, None),
            rounded_linear: make("fast_rounded_linear", ROUNDED_LINEAR_WGSL, strip, premul),
            opaque_linear_interior: make("fast_opaque_linear_interior", OPAQUE_LINEAR_INTERIOR_WGSL, strip, None),
            rounded_linear_ring: make("fast_rounded_linear_ring", ROUNDED_LINEAR_RING_WGSL, list, premul),
            opaque_radial: make("fast_opaque_radial", OPAQUE_RADIAL_WGSL, strip, None),
            linear_band: make("fast_linear_band", LINEAR_BAND_WGSL, strip, premul),
            shadow: make("fast_shadow", SHADOW_WGSL, strip, premul),
            fused_grad,
            fast_layer: fast_layer_composite_pipeline(device, layout, format),
            fast_layer_over_clear: make("fast_layer_over_clear", FAST_LAYER_OVER_CLEAR_WGSL, strip, None),
        }
    }
}

/// Dedicated pipeline for compositing isolated layers (`Kind::Layer`), replacing the full uber shader.
pub(super) fn layer_composite_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    let src = format!("{CLIP_WGSL}\n{LAYER_COMPOSITE_WGSL}");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("layer_composite"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("layer_composite"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(metallib::uber_vertex_layout())],
        },
        primitive: metallib::uber_primitive(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Branchless 1:1 layer composite pipeline for plain unrotated, unscaled, hole-free, fade-free
/// SourceOver layers with integer bounds/clip (the 99% common case).
pub(super) fn fast_layer_composite_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fast_layer_composite"),
        source: wgpu::ShaderSource::Wgsl(FAST_LAYER_COMPOSITE_WGSL.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("fast_layer_composite"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(metallib::uber_vertex_layout())],
        },
        primitive: metallib::uber_primitive(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

const OPAQUE_RECT_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) fill: vec4<f32>,
};

@vertex fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32(i.vi >> 1u));
    let lo = max(i.rect.xy, i.clip.xy);
    let hi = max(min(i.rect.xy + i.rect.zw, i.clip.zw), lo);
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.fill = i.fill;
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    return i.fill;
}
"#;

const OPAQUE_RECT_INTERIOR_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) radii: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) fill: vec4<f32>,
};

@vertex fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32(i.vi >> 1u));
    let rmax = max(max(i.radii.x, i.radii.y), max(i.radii.z, i.radii.w));
    let inset = ceil(max(rmax, i.params.x) + 1.0);
    let lo = max(floor(i.rect.xy) + vec2<f32>(inset), i.clip.xy);
    let hi = max(min(ceil(i.rect.xy + i.rect.zw) - vec2<f32>(inset), i.clip.zw), lo);
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.fill = i.fill;
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    return i.fill;
}
"#;

const ROUNDED_RECT_RING_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) radii: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(3) border: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) center: vec2<f32>,
    @location(1) @interpolate(flat) half: vec2<f32>,
    @location(2) @interpolate(flat) radii: vec4<f32>,
    @location(3) @interpolate(flat) fill: vec4<f32>,
    @location(4) @interpolate(flat) border: vec4<f32>,
    @location(5) @interpolate(flat) params: vec4<f32>,
};

@vertex fn vs(i: In) -> Out {
    let strip = i.vi / 6u;
    let lv = i.vi % 6u;
    // Triangle list for a quad: (0,0), (1,0), (0,1), (1,0), (1,1), (0,1)
    let cx = select(0.0, 1.0, lv == 1u || lv == 3u || lv == 4u);
    let cy = select(0.0, 1.0, lv == 2u || lv == 4u || lv == 5u);
    let rmax = max(max(i.radii.x, i.radii.y), max(i.radii.z, i.radii.w));
    let inset = ceil(max(rmax, i.params.x) + 1.0);
    let ox0 = floor(i.rect.x) - 1.0;
    let oy0 = floor(i.rect.y) - 1.0;
    let ox1 = ceil(i.rect.x + i.rect.z) + 1.0;
    let oy1 = ceil(i.rect.y + i.rect.w) + 1.0;
    let ix0 = floor(i.rect.x) + inset;
    let iy0 = floor(i.rect.y) + inset;
    let ix1 = ceil(i.rect.x + i.rect.z) - inset;
    let iy1 = ceil(i.rect.y + i.rect.w) - inset;
    var slo: vec2<f32>;
    var shi: vec2<f32>;
    if (strip == 0u) { slo = vec2<f32>(ox0, oy0); shi = vec2<f32>(ox1, iy0); }
    else if (strip == 1u) { slo = vec2<f32>(ox0, iy1); shi = vec2<f32>(ox1, oy1); }
    else if (strip == 2u) { slo = vec2<f32>(ox0, iy0); shi = vec2<f32>(ix0, iy1); }
    else { slo = vec2<f32>(ix1, iy0); shi = vec2<f32>(ox1, iy1); }
    let lo = max(slo, i.clip.xy);
    let hi = max(min(shi, i.clip.zw), lo);
    let p = vec2<f32>(select(lo.x, hi.x, cx > 0.5), select(lo.y, hi.y, cy > 0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.half = i.rect.zw * 0.5;
    o.center = i.rect.xy + o.half;
    o.radii = i.radii;
    o.fill = i.fill;
    o.border = i.border;
    o.params = i.params;
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    let p = i.pos.xy + projection.xy - i.center;
    var r = i.radii.x;
    if (p.x > 0.0) { r = select(i.radii.y, i.radii.z, p.y > 0.0); }
    else { r = select(i.radii.x, i.radii.w, p.y > 0.0); }
    r = min(r, min(i.half.x, i.half.y));
    let q = abs(p) - i.half + vec2<f32>(r);
    let d = min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
    let bw = i.params.x;
    if (bw > 0.0) {
        let cov = clamp(0.5 - d, 0.0, 1.0);
        let ring = max(cov - clamp(0.5 - (d + bw), 0.0, 1.0), 0.0);
        let hide = clamp(i.border.a / clamp(i.params.w, 1e-6, 1.0), 0.0, 1.0);
        let shrink = any(i.radii != vec4<f32>(0.0)) && hide > 0.999;
        let under = clamp(0.5 - (d + select(0.0, bw * 0.5, shrink)), 0.0, 1.0);
        return i.border * ring + i.fill * (under * (1.0 - hide * ring));
    }
    return i.fill * clamp(0.5 - d, 0.0, 1.0);
}
"#;

const OPAQUE_LINEAR_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(3) border: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) uv: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) u: f32,
    @location(1) @interpolate(flat) fill: vec4<f32>,
    @location(2) @interpolate(flat) border: vec4<f32>,
};

@vertex fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32(i.vi >> 1u));
    let lo = max(i.rect.xy, i.clip.xy);
    let hi = max(min(i.rect.xy + i.rect.zw, i.clip.zw), lo);
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let center = i.rect.xy + i.rect.zw * 0.5;
    let t = 0.5 + dot(p - center, i.uv.xy);
    let u = (t - i.params.x) / (i.params.w - i.params.x);
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.u = u;
    o.fill = i.fill;
    o.border = i.border;
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    return mix(i.fill, i.border, clamp(i.u, 0.0, 1.0));
}
"#;

const OPAQUE_LINEAR_INTERIOR_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) radii: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(3) border: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) uv: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) u: f32,
    @location(1) @interpolate(flat) fill: vec4<f32>,
    @location(2) @interpolate(flat) border: vec4<f32>,
};

@vertex fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32(i.vi >> 1u));
    let rmax = max(max(i.radii.x, i.radii.y), max(i.radii.z, i.radii.w));
    let inset = ceil(rmax + 1.0);
    let lo = max(floor(i.rect.xy) + vec2<f32>(inset), i.clip.xy);
    let hi = max(min(ceil(i.rect.xy + i.rect.zw) - vec2<f32>(inset), i.clip.zw), lo);
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let center = i.rect.xy + i.rect.zw * 0.5;
    let t = 0.5 + dot(p - center, i.uv.xy);
    let u = (t - i.params.x) / (i.params.w - i.params.x);
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.u = u;
    o.fill = i.fill;
    o.border = i.border;
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    return mix(i.fill, i.border, clamp(i.u, 0.0, 1.0));
}
"#;

const ROUNDED_LINEAR_RING_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) radii: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(3) border: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) uv: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) u: f32,
    @location(1) @interpolate(flat) center: vec2<f32>,
    @location(2) @interpolate(flat) half: vec2<f32>,
    @location(3) @interpolate(flat) radii: vec4<f32>,
    @location(4) @interpolate(flat) fill: vec4<f32>,
    @location(5) @interpolate(flat) border: vec4<f32>,
};

@vertex fn vs(i: In) -> Out {
    let strip = i.vi / 6u;
    let lv = i.vi % 6u;
    let cx = select(0.0, 1.0, lv == 1u || lv == 3u || lv == 4u);
    let cy = select(0.0, 1.0, lv == 2u || lv == 4u || lv == 5u);
    let rmax = max(max(i.radii.x, i.radii.y), max(i.radii.z, i.radii.w));
    let inset = ceil(rmax + 1.0);
    let ox0 = floor(i.rect.x) - 1.0;
    let oy0 = floor(i.rect.y) - 1.0;
    let ox1 = ceil(i.rect.x + i.rect.z) + 1.0;
    let oy1 = ceil(i.rect.y + i.rect.w) + 1.0;
    let ix0 = floor(i.rect.x) + inset;
    let iy0 = floor(i.rect.y) + inset;
    let ix1 = ceil(i.rect.x + i.rect.z) - inset;
    let iy1 = ceil(i.rect.y + i.rect.w) - inset;
    var slo: vec2<f32>;
    var shi: vec2<f32>;
    if (strip == 0u) { slo = vec2<f32>(ox0, oy0); shi = vec2<f32>(ox1, iy0); }
    else if (strip == 1u) { slo = vec2<f32>(ox0, iy1); shi = vec2<f32>(ox1, oy1); }
    else if (strip == 2u) { slo = vec2<f32>(ox0, iy0); shi = vec2<f32>(ix0, iy1); }
    else { slo = vec2<f32>(ix1, iy0); shi = vec2<f32>(ox1, iy1); }
    let lo = max(slo, i.clip.xy);
    let hi = max(min(shi, i.clip.zw), lo);
    let p = vec2<f32>(select(lo.x, hi.x, cx > 0.5), select(lo.y, hi.y, cy > 0.5));
    let half = i.rect.zw * 0.5;
    let center = i.rect.xy + half;
    let t = 0.5 + dot(p - center, i.uv.xy);
    let u = (t - i.params.x) / (i.params.w - i.params.x);
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.u = u;
    o.center = center;
    o.half = half;
    o.radii = i.radii;
    o.fill = i.fill;
    o.border = i.border;
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    let p = i.pos.xy + projection.xy - i.center;
    var r = i.radii.x;
    if (p.x > 0.0) { r = select(i.radii.y, i.radii.z, p.y > 0.0); }
    else { r = select(i.radii.x, i.radii.w, p.y > 0.0); }
    r = min(r, min(i.half.x, i.half.y));
    let q = abs(p) - i.half + vec2<f32>(r);
    let d = min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
    return mix(i.fill, i.border, clamp(i.u, 0.0, 1.0)) * clamp(0.5 - d, 0.0, 1.0);
}
"#;

const OPAQUE_RADIAL_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(3) border: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) uv: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) d: vec2<f32>,
    @location(1) @interpolate(flat) fill: vec4<f32>,
    @location(2) @interpolate(flat) border: vec4<f32>,
    @location(3) @interpolate(flat) stops: vec4<f32>,
    @location(4) @interpolate(flat) flags: vec2<f32>,
};

@vertex fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32(i.vi >> 1u));
    let lower_open = i.params.x >= 1.5;
    let upper_open = i.params.w >= 1.5;
    let t0 = select(i.params.x, i.params.x - 2.0, lower_open);
    let t1 = select(i.params.w, i.params.w - 2.0, upper_open);
    var lo = max(i.rect.xy, i.clip.xy);
    var hi = max(min(i.rect.xy + i.rect.zw, i.clip.zw), lo);
    let c = i.rect.xy + i.uv.xy;
    if (upper_open) {
        let rb = abs(i.uv.zw) * max(t1, 0.0) + vec2<f32>(1.0);
        lo = max(lo, c - rb);
        hi = max(min(hi, c + rb), lo);
    }
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.d = (p - c) / i.uv.zw;
    o.fill = i.fill;
    o.border = i.border;
    let inv_span = select(1.0 / (t1 - t0), 0.0, t1 <= t0 + 1e-6);
    o.stops = vec4<f32>(t0, t1, inv_span, t0 * t0);
    o.flags = vec2<f32>(select(0.0, 1.0, lower_open), select(-1.0, t1 * t1, upper_open));
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    let t2 = dot(i.d, i.d);
    if ((i.flags.x > 0.5 && t2 < i.stops.w) || (i.flags.y >= 0.0 && t2 >= i.flags.y)) {
        discard;
    }
    if (i.flags.y < 0.0 && t2 >= i.stops.y * i.stops.y) {
        return i.border;
    }
    let t = sqrt(t2);
    let u = select(clamp((t - i.stops.x) * i.stops.z, 0.0, 1.0), select(0.0, 1.0, t >= i.stops.x), i.stops.z == 0.0);
    return mix(i.fill, i.border, u);
}
"#;

const ROUNDED_LINEAR_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) radii: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(3) border: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) uv: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) u: f32,
    @location(1) @interpolate(flat) center: vec2<f32>,
    @location(2) @interpolate(flat) half: vec2<f32>,
    @location(3) @interpolate(flat) radii: vec4<f32>,
    @location(4) @interpolate(flat) fill: vec4<f32>,
    @location(5) @interpolate(flat) border: vec4<f32>,
    @location(6) @interpolate(flat) rmax: f32,
};

@vertex fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32(i.vi >> 1u));
    let lo = max(i.rect.xy - vec2<f32>(1.0), i.clip.xy);
    let hi = max(min(i.rect.xy + i.rect.zw + vec2<f32>(1.0), i.clip.zw), lo);
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let half = i.rect.zw * 0.5;
    let center = i.rect.xy + half;
    let t = 0.5 + dot(p - center, i.uv.xy);
    let u = (t - i.params.x) / (i.params.w - i.params.x);
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.u = u;
    o.center = center;
    o.half = half;
    o.radii = i.radii;
    o.fill = i.fill;
    o.border = i.border;
    o.rmax = max(max(i.radii.x, i.radii.y), max(i.radii.z, i.radii.w)) + 0.5;
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    let p = i.pos.xy + projection.xy - i.center;
    let ap = abs(p);
    var cov = 1.0;
    if (ap.x > i.half.x - i.rmax || ap.y > i.half.y - i.rmax) {
        var r = i.radii.x;
        if (p.x > 0.0) { r = select(i.radii.y, i.radii.z, p.y > 0.0); }
        else { r = select(i.radii.x, i.radii.w, p.y > 0.0); }
        r = min(r, min(i.half.x, i.half.y));
        let q = ap - i.half + vec2<f32>(r);
        let d = min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
        cov = clamp(0.5 - d, 0.0, 1.0);
    }
    return mix(i.fill, i.border, clamp(i.u, 0.0, 1.0)) * cov;
}
"#;

const SHADOW_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) radii: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) uv: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) center: vec2<f32>,
    @location(1) @interpolate(flat) half: vec2<f32>,
    @location(2) @interpolate(flat) fill: vec4<f32>,
    @location(3) @interpolate(flat) sigma_k: vec4<f32>,
    @location(4) @interpolate(flat) uv: vec4<f32>,
    @location(5) @interpolate(flat) rect_xy: vec2<f32>,
    @location(6) @interpolate(flat) hole_r: f32,
};

fn erf2(x: vec2<f32>) -> vec2<f32> {
    let s = sign(x);
    let a = abs(x);
    var r = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
    r = r * r;
    return s - s / (r * r);
}

@vertex fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32(i.vi >> 1u));
    let grow = i.params.z * 3.0 + 1.0;
    var lo = i.rect.xy - vec2<f32>(grow);
    var hi = i.rect.xy + i.rect.zw + vec2<f32>(grow);
    if (i.uv.z < 0.0) {
        lo = max(lo, i.rect.xy + i.uv.xy);
        hi = max(min(hi, i.rect.xy + vec2<f32>(-i.uv.z, i.uv.w)), lo);
    }
    lo = max(lo, i.clip.xy);
    hi = max(min(hi, i.clip.zw), lo);
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    let sigma = max(i.params.z, 0.01);
    let half = i.rect.zw * 0.5;
    let cr = min(abs(i.radii.x), min(half.x, half.y));
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.center = i.rect.xy + half;
    o.half = half;
    o.fill = i.fill;
    o.sigma_k = vec4<f32>(sigma, 0.70710678 / sigma, 0.5 / (sigma * sigma), cr);
    o.uv = i.uv;
    o.rect_xy = i.rect.xy;
    o.hole_r = i.radii.w;
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    let screen = i.pos.xy + projection.xy;
    var hole_cov = 1.0;
    if (i.uv.z > 0.0) {
        let hole = i.uv.zw * 0.5;
        let hp = abs(screen - (i.rect_xy + i.uv.xy + hole)) - hole + vec2<f32>(i.hole_r);
        let hd = min(max(hp.x, hp.y), 0.0) + length(max(hp, vec2<f32>(0.0))) - i.hole_r;
        hole_cov = clamp(0.5 + hd, 0.0, 1.0);
        if (hole_cov <= 0.0) { discard; }
    }
    let p = screen - i.center;
    let sigma = i.sigma_k.x;
    let inv_s = i.sigma_k.y;
    let corner = i.sigma_k.w;
    var cov: f32;
    if (abs(p.x) <= i.half.x - 4.0 * sigma && abs(p.y) <= i.half.y - corner - 4.0 * sigma) {
        cov = 0.99863125;
    } else if (corner <= 0.45 * sigma || abs(p.y) + 2.0 * sigma <= i.half.y - corner) {
        // Exact separable 2D Gaussian where the vertical window stays on the straight sides
        // (or where the blur radius dominates the corner radius).
        let ix = erf2((p.x + vec2<f32>(-i.half.x, i.half.x)) * inv_s);
        let iy = erf2((p.y + vec2<f32>(-i.half.y, i.half.y)) * inv_s);
        cov = 0.25 * (ix.y - ix.x) * (iy.y - iy.x);
    } else {
        let lo = p.y - i.half.y;
        let hi = p.y + i.half.y;
        let start = clamp(-3.0 * sigma, lo, hi);
        let end = clamp(3.0 * sigma, lo, hi);
        let step = (end - start) / 4.0;
        if (step <= 0.0) { discard; }
        let inv2s2 = i.sigma_k.z;
        let norm = step / (2.50662827 * sigma);
        var y = start + step * 0.5;
        var v = 0.0;
        for (var k = 0; k < 4; k = k + 1) {
            let delta = min(i.half.y - corner - abs(p.y - y), 0.0);
            let curved = i.half.x - corner + sqrt(max(0.0, corner * corner - delta * delta));
            let ix = erf2((p.x + vec2<f32>(-curved, curved)) * inv_s);
            v = v + 0.5 * (ix.y - ix.x) * exp(-(y * y) * inv2s2);
            y = y + step;
        }
        cov = v * norm;
    }
    return i.fill * (cov * hole_cov);
}
"#;

const LINEAR_BAND_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(3) border: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) uv: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) t: f32,
    @location(1) @interpolate(flat) fill: vec4<f32>,
    @location(2) @interpolate(flat) border: vec4<f32>,
    @location(3) @interpolate(flat) stops: vec4<f32>,
    @location(4) @interpolate(flat) flags: vec2<f32>,
};

@vertex fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32(i.vi >> 1u));
    let lower_open = i.params.x >= 1.5;
    let upper_open = i.params.w >= 1.5;
    let t0 = select(i.params.x, i.params.x - 2.0, lower_open);
    let t1 = select(i.params.w, i.params.w - 2.0, upper_open);
    let below = lower_open || all(i.fill == vec4<f32>(0.0));
    let above = upper_open || all(i.border == vec4<f32>(0.0));
    var lo = max(i.rect.xy, i.clip.xy);
    var hi = max(min(i.rect.xy + i.rect.zw, i.clip.zw), lo);
    if (any(i.uv.xy != vec2<f32>(0.0))) {
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
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let center = i.rect.xy + i.rect.zw * 0.5;
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.t = 0.5 + dot(p - center, i.uv.xy);
    o.fill = i.fill;
    o.border = i.border;
    let inv_span = select(1.0 / (t1 - t0), 0.0, t1 <= t0 + 1e-6);
    o.stops = vec4<f32>(t0, t1, inv_span, 0.0);
    o.flags = vec2<f32>(select(0.0, 1.0, lower_open), select(0.0, 1.0, upper_open));
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    if ((i.flags.x > 0.5 && i.t < i.stops.x) || (i.flags.y > 0.5 && i.t >= i.stops.y)) {
        discard;
    }
    let u = select(clamp((i.t - i.stops.x) * i.stops.z, 0.0, 1.0), select(0.0, 1.0, i.t >= i.stops.x), i.stops.z == 0.0);
    return mix(i.fill, i.border, u);
}
"#;

const LAYER_COMPOSITE_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(2) var images: texture_2d<f32>;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) radii: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(3) border: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) uv: vec4<f32>,
    @location(6) clip: vec4<f32>,
    @location(7) fade: vec4<u32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) rect: vec4<f32>,
    @location(1) @interpolate(flat) radii: vec4<f32>,
    @location(2) @interpolate(flat) fill: vec4<f32>,
    @location(3) @interpolate(flat) border: vec4<f32>,
    @location(4) @interpolate(flat) params: vec4<f32>,
    @location(5) @interpolate(flat) uv: vec4<f32>,
    @location(6) @interpolate(flat) clip: vec4<f32>,
    @location(7) @interpolate(flat) fade: vec4<u32>,
};

fn layer_load(t: vec2<i32>, clamp_unit: bool) -> vec4<f32> {
    let c = textureLoad(images, t, 0);
    return select(c, clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)), clamp_unit);
}

fn layer_texel(t: vec2<i32>, size: vec2<f32>, clamp_unit: bool) -> vec4<f32> {
    if (any(t < vec2<i32>(0)) || any(t >= vec2<i32>(size))) { return vec4<f32>(0.0); }
    return layer_load(t, clamp_unit);
}

fn hole_alpha(p: vec2<f32>, h: vec4<f32>) -> f32 {
    if (h.w <= 0.0) { return 1.0; }
    return clamp((length(p - h.xy) - (h.z + h.w) * 0.5) / max(h.w - h.z, 1.0) + 0.5, 0.0, 1.0);
}

@vertex fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32(i.vi >> 1u));
    var lo = i.rect.xy - vec2<f32>(1.0);
    var hi = i.rect.xy + i.rect.zw + vec2<f32>(1.0);
    if (i.border.y == 0.0) {
        lo = max(lo, i.clip.xy - vec2<f32>(1.0));
        hi = max(min(hi, i.clip.zw + vec2<f32>(1.0)), lo);
    }
    var p = select(lo, hi, corner > vec2<f32>(0.5));
    if (i.border.y != 0.0) {
        let q = p - i.border.zw;
        let c = cos(i.border.y);
        let s = sin(i.border.y);
        p = i.border.zw + vec2<f32>(c * q.x - s * q.y, s * q.x + c * q.y);
    }
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.rect = i.rect;
    o.radii = i.radii;
    o.fill = i.fill;
    o.border = i.border;
    o.params = i.params;
    o.uv = i.uv;
    o.clip = i.clip;
    o.fade = i.fade;
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    let screen = i.pos.xy + projection.xy;
    var p = screen - i.rect.xy;
    let additive = i.params.x == 1.0;
    var color: vec4<f32>;
    if (i.border.y != 0.0) {
        let q = screen - i.border.zw;
        let c = cos(i.border.y);
        let s = sin(i.border.y);
        p = i.border.zw + vec2<f32>(c * q.x + s * q.y, c * q.y - s * q.x) - i.rect.xy;
        let local = p - vec2<f32>(0.5);
        let base = floor(local);
        let f = local - base;
        let t0 = vec2<i32>(base);
        color = mix(mix(layer_texel(t0, i.rect.zw, additive), layer_texel(t0 + vec2<i32>(1, 0), i.rect.zw, additive), f.x),
                    mix(layer_texel(t0 + vec2<i32>(0, 1), i.rect.zw, additive), layer_texel(t0 + vec2<i32>(1, 1), i.rect.zw, additive), f.x), f.y);
    } else if (any(p < vec2<f32>(0.0)) || any(p >= i.rect.zw)) {
        discard;
    } else if (i.uv.z > 0.0) {
        let local = p * i.uv.zw / i.rect.zw - vec2<f32>(0.5);
        let hi = vec2<i32>(i.uv.zw) - vec2<i32>(1);
        let base = floor(local);
        let f = local - base;
        let t0 = clamp(vec2<i32>(base), vec2<i32>(0), hi);
        let t1 = clamp(vec2<i32>(base) + vec2<i32>(1, 1), vec2<i32>(0), hi);
        color = mix(mix(layer_load(t0, additive), layer_load(vec2<i32>(t1.x, t0.y), additive), f.x),
                    mix(layer_load(vec2<i32>(t0.x, t1.y), additive), layer_load(t1, additive), f.x), f.y);
    } else {
        color = layer_load(vec2<i32>(floor(p)), additive);
    }
    if (color.a <= 0.0) { discard; }
    if (i.fade.w == 0u && i.radii.w <= 0.0 && i.border.y == 0.0) {
        let unclipped = i.clip.x < -1e5
            || (screen.x >= i.clip.x + 0.5 && screen.x <= i.clip.z - 0.5
                && screen.y >= i.clip.y + 0.5 && screen.y <= i.clip.w - 0.5);
        if (unclipped) {
            return color * i.fill.a;
        }
    }
    return color * (i.fill.a * clip_alpha_stops(screen, i.clip, i.fade, g.flags.zw) * hole_alpha(screen, i.radii));
}
"#;

const FAST_LAYER_COMPOSITE_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(2) var images: texture_2d<f32>;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) origin: vec2<f32>,
    @location(1) @interpolate(flat) alpha: f32,
};

@vertex fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32(i.vi >> 1u));
    let lo = max(i.rect.xy, i.clip.xy);
    let hi = max(min(i.rect.xy + i.rect.zw, i.clip.zw), lo);
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.origin = i.rect.xy;
    o.alpha = i.fill.a;
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(i.pos.xy + projection.xy - i.origin));
    return textureLoad(images, p, 0) * i.alpha;
}
"#;

const FAST_LAYER_OVER_CLEAR_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(2) var images: texture_2d<f32>;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(3) border: vec4<f32>,
    @location(6) clip: vec4<f32>,
};
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) origin: vec2<f32>,
    @location(1) @interpolate(flat) alpha: f32,
    @location(2) @interpolate(flat) bg: vec3<f32>,
};

@vertex fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32(i.vi >> 1u));
    let lo = max(i.rect.xy, i.clip.xy);
    let hi = max(min(i.rect.xy + i.rect.zw, i.clip.zw), lo);
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.origin = i.rect.xy;
    o.alpha = i.fill.a;
    o.bg = i.border.rgb;
    return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(i.pos.xy + projection.xy - i.origin));
    let src = textureLoad(images, p, 0) * i.alpha;
    return vec4<f32>(src.rgb + i.bg * (1.0 - src.a), 1.0);
}
"#;

const FUSED_GRAD_WGSL: &str = r#"
struct Globals { viewport: vec2<f32>, pad: vec2<f32>, flags: vec4<f32> };
@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct Band {
    fill: vec4<f32>,
    border: vec4<f32>,
    dir_stops: vec4<f32>,
    flags: vec4<f32>,
};
struct Fused {
    rect: vec4<f32>,
    clip: vec4<f32>,
    rad_center_inv: vec4<f32>,
    rad_stops: vec4<f32>,
    rad_c0: vec4<f32>,
    rad_c1: vec4<f32>,
    rad_c2: vec4<f32>,
    b0: Band,
    b1: Band,
    b2: Band,
    b3: Band,
};
@group(1) @binding(0) var<uniform> u: Fused;

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) d: vec2<f32>,
    @location(1) lt: vec4<f32>,
};

@vertex fn vs(@builtin(vertex_index) vi: u32) -> Out {
    let corner = vec2<f32>(f32(vi & 1u), f32(vi >> 1u));
    let lo = max(u.rect.xy, u.clip.xy);
    let hi = max(min(u.rect.xy + u.rect.zw, u.clip.zw), lo);
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    let rel = p - (u.rect.xy + u.rect.zw * 0.5);
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.d = (p - u.rad_center_inv.xy) * u.rad_center_inv.zw;
    o.lt = vec4<f32>(
        0.5 + dot(rel, u.b0.dir_stops.xy),
        0.5 + dot(rel, u.b1.dir_stops.xy),
        0.5 + dot(rel, u.b2.dir_stops.xy),
        0.5 + dot(rel, u.b3.dir_stops.xy),
    );
    return o;
}

fn apply_band(col: vec3<f32>, lt: f32, b: Band) -> vec3<f32> {
    let in_lo = b.flags.x < 0.5 || lt >= b.dir_stops.z;
    let in_hi = b.flags.y < 0.5 || lt < b.dir_stops.w;
    if (in_lo && in_hi) {
        let f = clamp((lt - b.dir_stops.z) * b.flags.z, 0.0, 1.0);
        let bc = mix(b.fill, b.border, f);
        return bc.rgb + col * (1.0 - bc.a);
    }
    return col;
}

fn band_alpha(lt: f32, b: Band) -> f32 {
    let in_lo = b.flags.x < 0.5 || lt >= b.dir_stops.z;
    let in_hi = b.flags.y < 0.5 || lt < b.dir_stops.w;
    if (in_lo && in_hi) {
        let f = clamp((lt - b.dir_stops.z) * b.flags.z, 0.0, 1.0);
        return mix(b.fill.a, b.border.a, f);
    }
    return 0.0;
}

@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    let t2 = dot(i.d, i.d);
    let s0 = u.rad_stops.x;
    let s1 = u.rad_stops.y;
    let s2 = u.rad_stops.z;
    let uniform_vignette = u.b0.flags.w > 1.5;
    if (uniform_vignette && t2 >= s2 * s2) {
        return u.rad_c2;
    }
    var col: vec3<f32>;
    if (t2 >= s2 * s2) {
        col = u.rad_c2.rgb;
    } else {
        let t = sqrt(t2);
        if (t < s1) {
            col = mix(u.rad_c0.rgb, u.rad_c1.rgb, clamp((t - s0) / (s1 - s0), 0.0, 1.0));
        } else {
            col = mix(u.rad_c1.rgb, u.rad_c2.rgb, clamp((t - s1) / (s2 - s1), 0.0, 1.0));
        }
    }
    let nb = u.rad_stops.w;
    if (uniform_vignette) {
        var keep = 1.0;
        if (nb > 0.5) { keep = keep * (1.0 - band_alpha(i.lt.x, u.b0)); }
        if (nb > 1.5) { keep = keep * (1.0 - band_alpha(i.lt.y, u.b1)); }
        if (nb > 2.5) { keep = keep * (1.0 - band_alpha(i.lt.z, u.b2)); }
        if (nb > 3.5) { keep = keep * (1.0 - band_alpha(i.lt.w, u.b3)); }
        return vec4<f32>(mix(u.rad_c2.rgb, col, keep), 1.0);
    }
    if (nb > 0.5) { col = apply_band(col, i.lt.x, u.b0); }
    if (nb > 1.5) { col = apply_band(col, i.lt.y, u.b1); }
    if (nb > 2.5) { col = apply_band(col, i.lt.z, u.b2); }
    if (nb > 3.5) { col = apply_band(col, i.lt.w, u.b3); }
    return vec4<f32>(col, 1.0);
}
"#;

/// Issues a tiny 2×2 warm-up draw for every fast pipeline of `format` so drivers with lazy
/// fragment shader compilation (such as lavapipe's LLVM JIT) compile them during initialization
/// rather than stalling a later interactive frame.
pub(super) fn warm_draw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bind_group: &wgpu::BindGroup,
    fused_buf: &wgpu::Buffer,
    fused_bg: &wgpu::BindGroup,
    format: wgpu::TextureFormat,
    fp: &FastPipelines,
    layer_pipe: Option<&wgpu::RenderPipeline>,
    extra_pipe: Option<&wgpu::RenderPipeline>,
) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fast_warm"),
        size: wgpu::Extent3d { width: 2, height: 2, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let mut inst: Instance = bytemuck::Zeroable::zeroed();
    inst.rect = [0.0, 0.0, 4096.0, 4096.0];
    inst.fill = [1.0, 1.0, 1.0, 1.0];
    inst.border = [1.0, 1.0, 1.0, 1.0];
    inst.params = [0.0, 0.0, 1.0, 1.0];
    inst.uv = [0.0, 0.0, 1.0, 1.0];
    inst.clip = super::NO_CLIP;
    let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fast_warm_vb"),
        size: std::mem::size_of::<Instance>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    if let Ok(mut view) = vbuf.slice(..).get_mapped_range_mut() {
        view.copy_from_slice(bytemuck::bytes_of(&inst));
    }
    vbuf.unmap();
    let mut dummy_fg: FusedGradUniform = bytemuck::Zeroable::zeroed();
    dummy_fg.rect = [0.0, 0.0, 4096.0, 4096.0];
    dummy_fg.clip = super::NO_CLIP;
    dummy_fg.rad_center_inv = [2048.0, 2048.0, 1.0 / 2048.0, 1.0 / 2048.0];
    dummy_fg.rad_stops = [0.0, 0.5, 1.0, 1.0];
    dummy_fg.rad_c0 = [1.0; 4];
    dummy_fg.rad_c1 = [0.5; 4];
    dummy_fg.rad_c2 = [0.0, 0.0, 0.0, 1.0];
    queue.write_buffer(fused_buf, 0, bytemuck::bytes_of(&dummy_fg));
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fast_warm") });
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fast_warm"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, vbuf.slice(..));
        for p in [
            &fp.opaque_rect,
            &fp.opaque_rect_interior,
            &fp.opaque_linear,
            &fp.rounded_linear,
            &fp.opaque_linear_interior,
            &fp.opaque_radial,
            &fp.linear_band,
            &fp.shadow,
            &fp.fast_layer,
            &fp.fast_layer_over_clear,
        ] {
            pass.set_pipeline(p);
            pass.draw(0..4, 0..1);
        }
        for p in [&fp.rounded_rect_ring, &fp.rounded_linear_ring] {
            pass.set_pipeline(p);
            pass.draw(0..24, 0..1);
        }
        if let Some(p) = layer_pipe {
            pass.set_pipeline(p);
            pass.draw(0..4, 0..1);
        }
        if let Some(p) = extra_pipe {
            pass.set_pipeline(p);
            pass.draw(0..4, 0..1);
        }
        pass.set_pipeline(&fp.fused_grad);
        pass.set_bind_group(1, fused_bg, &[]);
        pass.draw(0..4, 0..1);
    }
    queue.submit([enc.finish()]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{Gpu, Renderer};
    use crate::text::Atlas;

    fn solid_rect(rect: [f32; 4], fill: [f32; 4]) -> Instance {
        let mut i: Instance = bytemuck::Zeroable::zeroed();
        i.rect = rect;
        i.fill = fill;
        i.params = [0.0, Kind::Rect.as_f32(), 0.0, 1.0];
        i.clip = super::super::NO_CLIP;
        i
    }

    fn radial_seg(rect: [f32; 4], center_r: [f32; 4], fill: [f32; 4], border: [f32; 4], s0: f32, s1: f32) -> Instance {
        let mut i: Instance = bytemuck::Zeroable::zeroed();
        i.rect = rect;
        i.fill = fill;
        i.border = border;
        i.params = [s0, Kind::Radial.as_f32(), 0.0, s1];
        i.uv = center_r;
        i.clip = super::super::NO_CLIP;
        i
    }

    #[test]
    fn transparent_image_uses_general_pipeline() {
        let mut img: Instance = bytemuck::Zeroable::zeroed();
        img.rect = [0.0, 0.0, 352.0, 160.0];
        img.fill = [1.0, 1.0, 1.0, 1.0];
        img.params = [0.0, Kind::Image.as_f32(), 0.0, 1.0];
        img.uv = [0.0, 0.0, 352.0, 160.0];
        img.clip = super::super::NO_CLIP;
        assert_eq!(classify(&img), FastKind::General);
    }

    #[test]
    fn opaque_rect_fading_updates_fast_kind_without_params_change() {
        let gpu = Gpu::headless_wgpu();
        let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Bgra8Unorm);
        let mut atlas = Atlas::new();
        let mut list = vec![solid_rect([0.0, 0.0, 200.0, 200.0], [0.2, 0.4, 0.8, 1.0])];
        renderer.prepare(&gpu.device, &gpu.queue, &mut atlas, &list, (400, 400));
        assert_eq!(renderer.fast_kinds, vec![FastKind::OpaqueRect]);

        // Fade fill alpha below 1 without changing count or params.
        list[0].fill = [0.1, 0.2, 0.4, 0.5];
        renderer.prepare(&gpu.device, &gpu.queue, &mut atlas, &list, (400, 400));
        assert_eq!(renderer.fast_kinds, vec![FastKind::General]);

        // Also test the loop-only `write_patched` path.
        list[0].fill = [0.2, 0.4, 0.8, 1.0];
        renderer.next_instances(1, None);
        renderer.prepare(&gpu.device, &gpu.queue, &mut atlas, &list, (400, 400));
        assert_eq!(renderer.fast_kinds, vec![FastKind::OpaqueRect]);

        list[0].fill = [0.1, 0.2, 0.4, 0.5];
        renderer.next_instances(1, Some(vec![0]));
        renderer.prepare(&gpu.device, &gpu.queue, &mut atlas, &list, (400, 400));
        assert_eq!(renderer.fast_kinds, vec![FastKind::General]);
    }

    #[test]
    fn fused_grad_color_change_updates_uniform_without_params_change() {
        let gpu = Gpu::headless_wgpu();
        let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Bgra8Unorm);
        let mut atlas = Atlas::new();
        let rect = [0.0, 0.0, 400.0, 200.0];
        let mut list = vec![
            radial_seg(rect, [200.0, 100.0, 200.0, 100.0], [1.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0], 0.0, 2.5),
            radial_seg(rect, [200.0, 100.0, 200.0, 100.0], [0.5, 0.0, 0.0, 1.0], [0.1, 0.1, 0.1, 1.0], 2.5, 1.0),
        ];
        renderer.prepare(&gpu.device, &gpu.queue, &mut atlas, &list, (400, 200));
        let (span, u0) = renderer.fused_grad.as_ref().expect("fused grad detected");
        assert_eq!(*span, 0..2);
        assert_eq!(u0.rad_c0, [1.0, 0.0, 0.0, 1.0]);

        // Change only slide colours (same count, same params).
        list[0].fill = [0.0, 0.0, 1.0, 1.0];
        list[0].border = [0.0, 0.0, 0.5, 1.0];
        list[1].fill = [0.0, 0.0, 0.5, 1.0];
        renderer.prepare(&gpu.device, &gpu.queue, &mut atlas, &list, (400, 200));
        let (span2, u1) = renderer.fused_grad.as_ref().expect("fused grad still detected");
        assert_eq!(*span2, 0..2);
        assert_eq!(u1.rad_c0, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(u1.rad_c1, [0.0, 0.0, 0.5, 1.0]);
    }
}
