//! The frame pass clears its target to the app root's fill (`RuntimeCore::clear_color`), and
//! the root's fill is the first thing painted: an opaque rect over the whole window in that same
//! colour. A frame's GPU time is mostly the pixels it shades, and the root's fill is a whole
//! window of them that change nothing. So the rects at the head of the list that paint only the
//! clear colour, each on whole pixels, are left out of a pass that clears to it: the target
//! holds exactly the bytes it would have held after them.

use super::{Instance, Kind, Renderer};

impl Renderer {
    /// Where the frame pass opening with `load` starts in the uploaded list (`Cleared::start`).
    pub(super) fn cleared_start(&self, load: wgpu::LoadOp<wgpu::Color>) -> u32 {
        let before = self.surfaces.as_ref().and_then(|s| (0..s.len()).map(|i| s.before_instance(i)).min());
        self.cleared.start(load, before.unwrap_or(u32::MAX))
    }
}

/// The instances at the head of the list that, over a target holding `color` everywhere, leave
/// every pixel as it was.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Cleared {
    pub(super) color: [f32; 4],
    pub(super) len: u32,
    /// When the viewport is partitioned into `[0, 0, W, y0]` (`instances[1]`, `color`) and a
    /// dominant opaque rect `[0, y0, W, H - y0]` (`instances[skip]`, `swap_color`) disjoint from
    /// everything in `2..skip`, clearing to `swap_color` and starting at `1` skips shading the
    /// dominant rect and draws only the smaller top strip.
    pub(super) swap: Option<(u32, [f32; 4])>,
}

impl Cleared {
    pub(super) fn of(instances: &[Instance]) -> Self {
        let Some(color) = instances.first().map(|i| i.fill).filter(|c| c[3] == 1.0) else { return Self::default() };
        let len = instances.iter().take_while(|i| paints_only(i, color)).count();
        let mut swap = None;
        if len == 2 {
            let r0 = instances[0].rect;
            let r1 = instances[1].rect;
            if r0[0] == 0.0 && r0[1] == 0.0 && r1[0] == 0.0 && r1[1] == 0.0 && r1[2] == r0[2] && r1[3] > 0.0 && r1[3] < r0[3] * 0.25 {
                let want = [0.0, r1[3], r0[2], r0[3] - r1[3]];
                let mut depth = 0usize;
                for (k, inst) in instances.iter().enumerate().skip(2) {
                    match inst.kind() {
                        Some(Kind::Layer) => {
                            if inst.fill[3] < 1.0 || inst.params[0] != 0.0 || inst.params[2] != 0.0 || inst.params[3] != 0.0 || inst.border[1] != 0.0 || inst.fade != [0; 4] || inst.radii[3] != 0.0 {
                                depth += 1;
                            }
                            continue;
                        }
                        Some(Kind::LayerEnd) => {
                            depth = depth.saturating_sub(1);
                            continue;
                        }
                        _ => {}
                    }
                    if depth == 0 && inst.rect == want && inst.fill[3] == 1.0 && paints_only(inst, inst.fill) {
                        let disjoint = instances[2..k].iter().all(|prev| {
                            if matches!(prev.kind(), Some(Kind::Layer | Kind::LayerEnd)) { return true; }
                            let Some((x0, y0, x1, y1)) = prev.damage_quad_bounds() else { return true };
                            let c = prev.clip;
                            let (x0, y0, x1, y1) = (x0.max(c[0]), y0.max(c[1]), x1.min(c[2]), y1.min(c[3]));
                            !(x1 > x0 && y1 > y0 && y1 > want[1])
                        });
                        if disjoint {
                            swap = Some((k as u32, inst.fill));
                        }
                        break;
                    }
                }
            }
        }
        Self { color, len: len as u32, swap }
    }

    /// Where a pass opening with `load` starts drawing the list: past these instances when it
    /// clears to their colour. `before` is the first instance something else draws ahead of (a
    /// native surface), which they must not reach past: a rect after a video draws over it.
    pub(super) fn start(&self, load: wgpu::LoadOp<wgpu::Color>, before: u32) -> u32 {
        match load {
            wgpu::LoadOp::Clear(c) if [c.r, c.g, c.b, c.a] == self.color.map(f64::from) => self.len.min(before),
            _ => 0,
        }
    }
}

/// `i` writes exactly `color`, which is opaque, to the pixels inside its rect and nothing to
/// the rest: a plain rect (no border, softened edge or rounded corner) whose edges lie on pixel
/// boundaries, inside a clip with no fade. The shader's coverage and clip ramps are then exactly
/// 1 at every pixel centre inside and 0 outside, so over pixels already `color` it changes none.
fn paints_only(i: &Instance, color: [f32; 4]) -> bool {
    let [x, y, w, h] = i.rect;
    let [x1, y1] = [x + w, y + h];
    let [cx0, cy0, cx1, cy1] = i.clip;
    i.kind() == Some(Kind::Rect)
        && i.fill == color
        && i.params[0] == 0.0
        && i.params[2] <= 0.5
        && i.radii == [0.0; 4]
        && [x, y, x1, y1].iter().all(|v| v.fract() == 0.0)
        && cx0 <= x && cy0 <= y && cx1 >= x1 && cy1 >= y1
        && i.fade == [0; 4]
}

#[cfg(test)]
#[path = "cleared_tests.rs"]
mod tests;
