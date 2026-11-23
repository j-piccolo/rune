//! Slim shaders for the commonest quads: plain box middles, gradient middles, borderless boxes, plain
//! atlas glyphs and layer composites. Each does what shader.wgsl does for the instances
//! `classify_quad` (render.rs) gives it, with only the work those instances need.

/// Ultra-slim shader for `Kind::Rect` middles (`QuadMode::MiddleOnly` on `0..4`, and `QuadMode::Split`
/// middle on `4..10`): 1 flat varying (`fill`, byte-rounded once per vertex) and a 1-instruction fragment shader.
pub(super) fn solid_rect_wgsl() -> String {
    format!("{}\n{}", super::CLIP_WGSL, r#"
struct Globals {
    viewport: vec2<f32>,
    pad: vec2<f32>,
    flags: vec4<f32>,
};

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
    @location(7) fade: vec4<u32>,
};

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) fill: vec4<f32>,
};

@vertex
fn vs(i: In) -> Out {
    var corner: vec2<f32>;
    var lo: vec2<f32>;
    var hi: vec2<f32>;
    if (i.vi < 4u) {
        corner = vec2<f32>(f32(i.vi & 1u), f32((i.vi >> 1u) & 1u));
        lo = i.rect.xy;
        hi = i.rect.xy + i.rect.zw;
    } else {
        let j = i.vi - 4u;
        let k = u32(clamp(i32(j % 6u) - 1, 0, 3));
        corner = vec2<f32>(f32(k & 1u), f32(k >> 1u));
        let r = i.radii;
        let reach = max(max(max(r.x, r.y), max(r.z, r.w)), -CONTINUOUS_K * min(min(r.x, r.y), min(r.z, r.w)));
        let border = max(i.params.x, max(max(i.uv.x, i.uv.y), max(i.uv.z, i.uv.w)));
        let soft = max(i.params.z, 0.5);
        let e = reach + soft + border + 1.0;
        let f = vec4<f32>(unpack2x16float(i.fade.x), unpack2x16float(i.fade.y));
        var edge = max(abs(f), vec4<f32>(0.5)) + vec4<f32>(1.0);
        if (i.fade.w != 0u) {
            let c = unpack_clip_radii(i.fade.w);
            let cr = max(max(max(c.x, c.y), max(c.z, c.w)), -CONTINUOUS_K * min(min(c.x, c.y), min(c.z, c.w)));
            edge = max(edge, vec4<f32>(cr + 1.0));
        }
        lo = ceil(max(i.rect.xy + vec2<f32>(e), i.clip.xy + edge.xy));
        hi = floor(min(i.rect.xy + i.rect.zw - vec2<f32>(e), i.clip.zw - edge.zw));
        if (any(hi - lo < vec2<f32>(8.0)) || i.fade.z != 0u || i.uv.z < 0.0) {
            hi = lo;
        }
    }
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var fill = i.fill;
    if (projection.z == 0.0 && g.flags.y > 0.5 && i.params.x == 0.0 && fill.a > 0.0 && fill.a < 1.0) {
        fill = round(clamp(fill, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0) / 255.0;
    }
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.fill = fill;
    return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4<f32> {
    return i.fill;
}
"#)
}

/// Slim shader for `Kind::Radial` (radial and angled linear gradient) middles (`4..10`):
/// 5 flat varyings instead of 10, no `sd_rounded` or `clip_alpha_stops` in the fragment stage.
pub(super) fn radial_middle_wgsl() -> String {
    format!("{}\n{}", super::CLIP_WGSL, r#"
struct Globals {
    viewport: vec2<f32>,
    pad: vec2<f32>,
    flags: vec4<f32>,
};

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
    @location(7) fade: vec4<u32>,
};

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) rect: vec4<f32>,
    @location(1) @interpolate(flat) fill: vec4<f32>,
    @location(2) @interpolate(flat) border: vec4<f32>,
    @location(3) @interpolate(flat) params: vec4<f32>,
    @location(4) @interpolate(flat) uv: vec4<f32>,
};

@vertex
fn vs(i: In) -> Out {
    let j = i.vi - 4u;
    let k = u32(clamp(i32(j % 6u) - 1, 0, 3));
    let corner = vec2<f32>(f32(k & 1u), f32(k >> 1u));
    let grow = 1.0 + i.params.z * 2.0;
    var lo = i.rect.xy - vec2<f32>(grow);
    var hi = lo + i.rect.zw + vec2<f32>(2.0 * grow);
    if (i.uv.z == 0.0 && any(i.uv.xy != vec2<f32>(0.0))) {
        let lower_open = i.params.x >= 1.5;
        let upper_open = i.params.w >= 1.5;
        let t0 = select(i.params.x, i.params.x - 2.0, lower_open);
        let t1 = select(i.params.w, i.params.w - 2.0, upper_open);
        let below = lower_open || all(i.fill == vec4<f32>(0.0));
        let above = upper_open || all(i.border == vec4<f32>(0.0));
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
    let r = i.radii;
    let reach = max(max(max(r.x, r.y), max(r.z, r.w)), -CONTINUOUS_K * min(min(r.x, r.y), min(r.z, r.w)));
    let soft = max(i.params.z, 0.5);
    let e = reach + soft + 1.0;
    let f = vec4<f32>(unpack2x16float(i.fade.x), unpack2x16float(i.fade.y));
    var edge = max(abs(f), vec4<f32>(0.5)) + vec4<f32>(1.0);
    if (i.fade.w != 0u) {
        let c = unpack_clip_radii(i.fade.w);
        let cr = max(max(max(c.x, c.y), max(c.z, c.w)), -CONTINUOUS_K * min(min(c.x, c.y), min(c.z, c.w)));
        edge = max(edge, vec4<f32>(cr + 1.0));
    }
    var hlo = ceil(max(i.rect.xy + vec2<f32>(e), i.clip.xy + edge.xy));
    var hhi = floor(min(i.rect.xy + i.rect.zw - vec2<f32>(e), i.clip.zw - edge.zw));
    if (any(hhi - hlo < vec2<f32>(8.0)) || i.fade.z != 0u) {
        hlo = vec2<f32>(0.0);
        hhi = vec2<f32>(0.0);
    }
    hlo = max(hlo, lo);
    hhi = min(hhi, hi);
    if (hhi.x <= hlo.x || hhi.y <= hlo.y) {
        hhi = hlo;
    }
    let p = select(hlo, hhi, corner > vec2<f32>(0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.rect = i.rect;
    o.fill = i.fill;
    o.border = i.border;
    o.params = i.params;
    o.uv = i.uv;
    return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4<f32> {
    let screen = i.pos.xy + projection.xy;
    let center = i.rect.xy + i.rect.zw * 0.5;
    let lower_open = i.params.x >= 1.5;
    let upper_open = i.params.w >= 1.5;
    let t0 = select(i.params.x, i.params.x - 2.0, lower_open);
    let t1 = select(i.params.w, i.params.w - 2.0, upper_open);
    var t: f32;
    if (i.uv.z > 0.0) {
        t = length((screen - i.rect.xy - i.uv.xy) / i.uv.zw);
    } else {
        t = 0.5 + dot(screen - center, i.uv.xy);
    }
    if ((lower_open && t < t0) || (upper_open && t >= t1)) {
        return vec4<f32>(0.0);
    }
    let u = select(clamp((t - t0) / (t1 - t0), 0.0, 1.0), select(0.0, 1.0, t >= t0), t1 <= t0 + 1e-6);
    return mix(i.fill, i.border, u);
}
"#)
}

/// Vertex-interpolated shader for linear `Kind::Radial` middles (`4..10`) whose `split_hole` lies
/// strictly within `[t0, t1]`: evaluates `mix(fill, border, u)` at the 4 vertices and lets the
/// hardware rasterizer interpolate `col`, making the fragment shader a single `return i.col;`.
pub(super) fn linear_grad_middle_wgsl() -> String {
    format!("{}\n{}", super::CLIP_WGSL, r#"
struct Globals {
    viewport: vec2<f32>,
    pad: vec2<f32>,
    flags: vec4<f32>,
};

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
    @location(7) fade: vec4<u32>,
};

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) col: vec4<f32>,
};

@vertex
fn vs(i: In) -> Out {
    let j = i.vi - 4u;
    let k = u32(clamp(i32(j % 6u) - 1, 0, 3));
    let corner = vec2<f32>(f32(k & 1u), f32(k >> 1u));
    let grow = 1.0 + i.params.z * 2.0;
    var lo = i.rect.xy - vec2<f32>(grow);
    var hi = lo + i.rect.zw + vec2<f32>(2.0 * grow);
    let lower_open = i.params.x >= 1.5;
    let upper_open = i.params.w >= 1.5;
    let t0 = select(i.params.x, i.params.x - 2.0, lower_open);
    let t1 = select(i.params.w, i.params.w - 2.0, upper_open);
    if (any(i.uv.xy != vec2<f32>(0.0))) {
        let below = lower_open || all(i.fill == vec4<f32>(0.0));
        let above = upper_open || all(i.border == vec4<f32>(0.0));
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
    let r = i.radii;
    let reach = max(max(max(r.x, r.y), max(r.z, r.w)), -CONTINUOUS_K * min(min(r.x, r.y), min(r.z, r.w)));
    let soft = max(i.params.z, 0.5);
    let e = reach + soft + 1.0;
    let f = vec4<f32>(unpack2x16float(i.fade.x), unpack2x16float(i.fade.y));
    var edge = max(abs(f), vec4<f32>(0.5)) + vec4<f32>(1.0);
    if (i.fade.w != 0u) {
        let c = unpack_clip_radii(i.fade.w);
        let cr = max(max(max(c.x, c.y), max(c.z, c.w)), -CONTINUOUS_K * min(min(c.x, c.y), min(c.z, c.w)));
        edge = max(edge, vec4<f32>(cr + 1.0));
    }
    var hlo = ceil(max(i.rect.xy + vec2<f32>(e), i.clip.xy + edge.xy));
    var hhi = floor(min(i.rect.xy + i.rect.zw - vec2<f32>(e), i.clip.zw - edge.zw));
    if (any(hhi - hlo < vec2<f32>(8.0)) || i.fade.z != 0u) {
        hlo = vec2<f32>(0.0);
        hhi = vec2<f32>(0.0);
    }
    hlo = max(hlo, lo);
    hhi = min(hhi, hi);
    if (hhi.x <= hlo.x || hhi.y <= hlo.y) {
        hhi = hlo;
    }
    let p = select(hlo, hhi, corner > vec2<f32>(0.5));
    let center = i.rect.xy + i.rect.zw * 0.5;
    let t = 0.5 + dot(p - center, i.uv.xy);
    let u = clamp((t - t0) / max(t1 - t0, 1e-6), 0.0, 1.0);
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.col = mix(i.fill, i.border, u);
    return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4<f32> {
    return i.col;
}
"#)
}

/// Slim shader for borderless, circular-corner, unfaded `Kind::Rect`s (whole `0..4` or `SPLIT_BANDS`
/// `10..34`): no `border_over`, no per-side borders, no `sd_continuous` (`atan2`), no `ramp_stops`,
/// and discards zero-coverage apron pixels before blending.
pub(super) fn simple_rect_wgsl() -> String {
    format!("{}\n{}", super::CLIP_WGSL, r#"
struct Globals {
    viewport: vec2<f32>,
    pad: vec2<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) radii: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(6) clip: vec4<f32>,
    @location(7) fade: vec4<u32>,
};

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) rect: vec4<f32>,
    @location(1) @interpolate(flat) radii: vec4<f32>,
    @location(2) @interpolate(flat) fill: vec4<f32>,
    @location(3) @interpolate(flat) soft: f32,
    @location(4) @interpolate(flat) clip: vec4<f32>,
    @location(5) @interpolate(flat) clip_rad: u32,
};

fn sd_circ(p: vec2<f32>, half: vec2<f32>, radii: vec4<f32>) -> f32 {
    var r = radii.x;
    if (p.x > 0.0) {
        r = select(radii.y, radii.z, p.y > 0.0);
    } else {
        r = select(radii.x, radii.w, p.y > 0.0);
    }
    r = min(r, min(half.x, half.y));
    let q = abs(p) - half + vec2<f32>(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

@vertex
fn vs(i: In) -> Out {
    var corner = vec2<f32>(f32(i.vi & 1u), f32((i.vi >> 1u) & 1u));
    var sub = 0u;
    if (i.vi >= 4u) {
        let j = i.vi - 4u;
        sub = j / 6u + 1u;
        let k = u32(clamp(i32(j % 6u) - 1, 0, 3));
        corner = vec2<f32>(f32(k & 1u), f32(k >> 1u));
    }
    let grow = 1.0 + i.params.z * 2.0;
    var lo = i.rect.xy - vec2<f32>(grow);
    var hi = lo + i.rect.zw + vec2<f32>(2.0 * grow);
    if (sub > 0u) {
        let r = i.radii;
        let reach = max(max(r.x, r.y), max(r.z, r.w));
        let soft = max(i.params.z, 0.5);
        let e = reach + soft + 1.0;
        var edge = 1.5;
        if (i.fade.w != 0u) {
            let c = unpack_clip_radii(i.fade.w);
            edge = max(edge, max(max(c.x, c.y), max(c.z, c.w)) + 1.0);
        }
        var hlo = ceil(max(i.rect.xy + vec2<f32>(e), i.clip.xy + vec2<f32>(edge)));
        var hhi = floor(min(i.rect.xy + i.rect.zw - vec2<f32>(e), i.clip.zw - vec2<f32>(edge)));
        if (any(hhi - hlo < vec2<f32>(8.0))) {
            if (sub != 2u) { hi = lo; }
        } else {
            hlo = max(hlo, lo);
            hhi = min(hhi, hi);
            if (hhi.x > hlo.x && hhi.y > hlo.y) {
                switch sub {
                    case 1u: { lo = hlo; hi = hhi; }
                    case 2u: { hi = vec2<f32>(hi.x, hlo.y); }
                    case 3u: { lo = vec2<f32>(lo.x, hlo.y); hi = vec2<f32>(hlo.x, hhi.y); }
                    case 4u: { lo = vec2<f32>(hhi.x, hlo.y); hi = vec2<f32>(hi.x, hhi.y); }
                    default: { lo = vec2<f32>(lo.x, hhi.y); }
                }
            } else if (sub != 2u) {
                hi = lo;
            }
        }
    }
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.rect = i.rect;
    o.radii = i.radii;
    o.fill = i.fill;
    o.soft = max(i.params.z, 0.5);
    o.clip = i.clip;
    o.clip_rad = i.fade.w;
    return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4<f32> {
    let screen = i.pos.xy + projection.xy;
    let half = i.rect.zw * 0.5;
    let center = i.rect.xy + half;
    let d = sd_circ(screen - center, half, i.radii);
    let cov = clamp(0.5 - d / (i.soft * 2.0), 0.0, 1.0);
    if (cov == 0.0) { discard; }
    let hard = vec4<f32>(screen - i.clip.xy, i.clip.zw - screen);
    var ca = 1.0;
    if (!all(hard >= vec4<f32>(0.5625))) {
        let r = clamp(hard + vec4<f32>(0.5), vec4<f32>(0.0), vec4<f32>(1.0));
        ca = r.x * r.y * r.z * r.w;
    }
    if (i.clip_rad != 0u) {
        let ch = (i.clip.zw - i.clip.xy) * 0.5;
        ca *= clamp(0.5 - sd_circ(screen - i.clip.xy - ch, ch, unpack_clip_radii(i.clip_rad)), 0.0, 1.0);
    }
    if (ca == 0.0) { discard; }
    let source = i.fill * (cov * ca);
    if (g.flags.y > 0.5 && projection.z == 0.0 && i.fill.a > 0.0 && i.fill.a < 1.0) {
        return round(clamp(source, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0) / 255.0;
    }
    return source;
}
"#)
}

/// Slim shader for `Kind::Layer` composites: minimal vertex stage and early `discard` on transparent
/// layer texels (skipping `clip_alpha_stops`, `hole_alpha`, and framebuffer blend writes on empty space).
pub(super) fn layer_composite_wgsl() -> String {
    format!("{}\n{}", super::CLIP_WGSL, r#"
struct Globals {
    viewport: vec2<f32>,
    pad: vec2<f32>,
    flags: vec4<f32>,
};

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
    @location(2) @interpolate(flat) border: vec4<f32>,
    @location(3) @interpolate(flat) uv: vec4<f32>,
    @location(4) @interpolate(flat) clip: vec4<f32>,
    @location(5) @interpolate(flat) fade: vec4<u32>,
    @location(6) @interpolate(flat) alpha_add: vec2<f32>,
};

@vertex
fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32((i.vi >> 1u) & 1u));
    let lo = i.rect.xy - vec2<f32>(1.0);
    let hi = i.rect.xy + i.rect.zw + vec2<f32>(1.0);
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    var screen = p;
    if (i.border.y != 0.0) {
        let q = p - i.border.zw;
        let c = cos(i.border.y);
        let s = sin(i.border.y);
        screen = i.border.zw + vec2<f32>(c * q.x - s * q.y, s * q.x + c * q.y);
    }
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = screen - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.rect = i.rect;
    o.radii = i.radii;
    o.border = i.border;
    o.uv = i.uv;
    o.clip = i.clip;
    o.fade = i.fade;
    o.alpha_add = vec2<f32>(i.fill.a, i.params.x);
    return o;
}

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

@fragment
fn fs(i: Out) -> @location(0) vec4<f32> {
    let screen = i.pos.xy + projection.xy;
    var p = screen - i.rect.xy;
    let additive = i.alpha_add.y == 1.0;
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
    if (all(color == vec4<f32>(0.0))) { discard; }
    return color * i.alpha_add.x * clip_alpha_stops(screen, i.clip, i.fade, g.flags.zw) * hole_alpha(screen, i.radii);
}
"#)
}

/// Ultra-slim shader for upright 1:1 non-additive, non-cutout, unfaded, circular-or-rectangular-clipped
/// `Kind::Layer` composites: 4 flat varyings, no `sd_continuous`/`atan2`, no `ramp_stops`, no `hole_alpha`,
/// no bilinear 4-tap path, and early `discard` on transparent texels for maximum SIMD occupancy hiding
/// `Rgba16Float` texture fetch latency.
pub(super) fn plain_layer_composite_wgsl() -> String {
    format!("{}\n{}", super::CLIP_WGSL, r#"
struct Globals {
    viewport: vec2<f32>,
    pad: vec2<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(2) var images: texture_2d<f32>;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(6) clip: vec4<f32>,
    @location(7) fade: vec4<u32>,
};

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) rect: vec4<f32>,
    @location(1) @interpolate(flat) alpha: f32,
    @location(2) @interpolate(flat) clip: vec4<f32>,
    @location(3) @interpolate(flat) clip_rad: u32,
};

fn sd_circ(p: vec2<f32>, half: vec2<f32>, radii: vec4<f32>) -> f32 {
    var r = radii.x;
    if (p.x > 0.0) {
        r = select(radii.y, radii.z, p.y > 0.0);
    } else {
        r = select(radii.x, radii.w, p.y > 0.0);
    }
    r = min(r, min(half.x, half.y));
    let q = abs(p) - half + vec2<f32>(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

@vertex
fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32((i.vi >> 1u) & 1u));
    let p = i.rect.xy + corner * i.rect.zw;
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.rect = i.rect;
    o.alpha = i.fill.a;
    o.clip = i.clip;
    o.clip_rad = i.fade.w;
    return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4<f32> {
    let screen = i.pos.xy + projection.xy;
    let p = screen - i.rect.xy;
    if (any(p < vec2<f32>(0.0)) || any(p >= i.rect.zw)) { discard; }
    let color = textureLoad(images, vec2<i32>(floor(p)), 0);
    if (all(color == vec4<f32>(0.0))) { discard; }
    let hard = vec4<f32>(screen - i.clip.xy, i.clip.zw - screen);
    var ca = 1.0;
    if (!all(hard >= vec4<f32>(0.5625))) {
        let r = clamp(hard + vec4<f32>(0.5), vec4<f32>(0.0), vec4<f32>(1.0));
        ca = r.x * r.y * r.z * r.w;
    }
    if (i.clip_rad != 0u) {
        let ch = (i.clip.zw - i.clip.xy) * 0.5;
        ca *= clamp(0.5 - sd_circ(screen - i.clip.xy - ch, ch, unpack_clip_radii(i.clip_rad)), 0.0, 1.0);
    }
    return color * (i.alpha * ca);
}
"#)
}

/// Slim shader for plain 1:1 grayscale atlas glyphs (no outline, LCD, rotation, zoom or atlas-fill):
/// skips split/band/grow vertex math, carries 5 flat varyings instead of 10, and discards zero-coverage
/// texels before `pow` and clipping.
pub(super) fn plain_glyph_wgsl() -> String {
    format!("{}\n{}", super::CLIP_WGSL, r#"
struct Globals {
    viewport: vec2<f32>,
    pad: vec2<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(2) fill: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) uv: vec4<f32>,
    @location(6) clip: vec4<f32>,
    @location(7) fade: vec4<u32>,
};

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) rect: vec4<f32>,
    @location(1) @interpolate(flat) fill: vec4<f32>,
    @location(2) @interpolate(flat) uv_exp: vec3<f32>,
    @location(3) @interpolate(flat) clip: vec4<f32>,
    @location(4) @interpolate(flat) fade: vec4<u32>,
};

@vertex
fn vs(i: In) -> Out {
    let corner = vec2<f32>(f32(i.vi & 1u), f32((i.vi >> 1u) & 1u));
    let p = i.rect.xy + corner * i.rect.zw;
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.rect = i.rect;
    o.fill = i.fill;
    o.uv_exp = vec3<f32>(i.uv.xy, i.params.w);
    o.clip = i.clip;
    o.fade = i.fade;
    return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4<f32> {
    let screen = i.pos.xy + projection.xy;
    let texel = vec2<i32>(i.uv_exp.xy + floor(screen - i.rect.xy));
    let cov = textureLoad(atlas, texel, 0).r;
    if (cov == 0.0) { discard; }
    let a = select(pow(cov, i.uv_exp.z), cov, i.uv_exp.z == 1.0);
    let source = i.fill * (a * clip_alpha_stops(screen, i.clip, i.fade, g.flags.zw));
    if (g.flags.y > 0.5 && projection.z == 0.0) {
        return round(clamp(source, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0) / 255.0;
    }
    return source;
}
"#)
}

/// Slim shader for one band of a border ring (`runtime/bands.rs`): a box with a border, no fill,
/// circular corners and no fade. The uber shader's box path there is `border * ring` (`border_over`
/// with a clear fill), so this works out only the ring and its clip, and discards where it's empty.
pub(super) fn simple_ring_wgsl() -> String {
    format!("{}\n{}", super::CLIP_WGSL, r#"
struct Globals {
    viewport: vec2<f32>,
    pad: vec2<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(3) var<uniform> projection: vec4<f32>;

struct In {
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) radii: vec4<f32>,
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
    @location(2) @interpolate(flat) border: vec4<f32>,
    @location(3) @interpolate(flat) soft_bw: vec2<f32>,
    @location(4) @interpolate(flat) clip: vec4<f32>,
    @location(5) @interpolate(flat) clip_rad: u32,
};

fn sd_circ(p: vec2<f32>, half: vec2<f32>, radii: vec4<f32>) -> f32 {
    var r = radii.x;
    if (p.x > 0.0) {
        r = select(radii.y, radii.z, p.y > 0.0);
    } else {
        r = select(radii.x, radii.w, p.y > 0.0);
    }
    r = min(r, min(half.x, half.y));
    let q = abs(p) - half + vec2<f32>(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

@vertex
fn vs(i: In) -> Out {
    // shader.wgsl `base_vs` for a box's quad, cut to its band's window.
    let corner = vec2<f32>(f32(i.vi & 1u), f32((i.vi >> 1u) & 1u));
    let grow = 1.0 + i.params.z * 2.0;
    var lo = i.rect.xy - vec2<f32>(grow);
    var hi = lo + (i.rect.zw + vec2<f32>(2.0 * grow));
    if (i.uv.z < 0.0) {
        lo = max(lo, i.rect.xy + i.uv.xy);
        hi = max(min(hi, i.rect.xy + vec2<f32>(-i.uv.z, i.uv.w)), lo);
    }
    let p = select(lo, hi, corner > vec2<f32>(0.5));
    let size_px = select(g.viewport, projection.zw, projection.z > 0.0);
    let local = p - projection.xy;
    var o: Out;
    o.pos = vec4<f32>(local.x / size_px.x * 2.0 - 1.0, 1.0 - local.y / size_px.y * 2.0, 0.0, 1.0);
    o.rect = i.rect;
    o.radii = i.radii;
    o.border = i.border;
    o.soft_bw = vec2<f32>(max(i.params.z, 0.5), i.params.x);
    o.clip = i.clip;
    o.clip_rad = i.fade.w;
    return o;
}

@fragment
fn fs(i: Out) -> @location(0) vec4<f32> {
    let screen = i.pos.xy + projection.xy;
    let half = i.rect.zw * 0.5;
    let d = sd_circ(screen - (i.rect.xy + half), half, i.radii);
    let soft2 = i.soft_bw.x * 2.0;
    let ring = max(clamp(0.5 - d / soft2, 0.0, 1.0) - clamp(0.5 - (d + i.soft_bw.y) / soft2, 0.0, 1.0), 0.0);
    if (ring == 0.0) { discard; }
    let hard = vec4<f32>(screen - i.clip.xy, i.clip.zw - screen);
    var ca = 1.0;
    if (!all(hard >= vec4<f32>(0.5625))) {
        let r = clamp(hard + vec4<f32>(0.5), vec4<f32>(0.0), vec4<f32>(1.0));
        ca = r.x * r.y * r.z * r.w;
    }
    if (i.clip_rad != 0u) {
        let ch = (i.clip.zw - i.clip.xy) * 0.5;
        ca *= clamp(0.5 - sd_circ(screen - i.clip.xy - ch, ch, unpack_clip_radii(i.clip_rad)), 0.0, 1.0);
    }
    return i.border * ring * ca;
}
"#)
}
