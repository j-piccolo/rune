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
    color: [f32; 4],
    len: u32,
}

impl Cleared {
    pub(super) fn of(instances: &[Instance]) -> Self {
        let Some(color) = instances.first().map(|i| i.fill).filter(|c| c[3] == 1.0) else { return Self::default() };
        let len = instances.iter().take_while(|i| paints_only(i, color)).count();
        Self { color, len: len as u32 }
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
