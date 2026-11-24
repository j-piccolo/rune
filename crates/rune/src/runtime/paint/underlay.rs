//! A picture the compositor shows under the window (`El::underlay`), found in a painted frame.

use super::*;
use crate::render::Kind;

impl RuntimeCore {
    /// For the frame just painted, `size` device pixels: the picture an `El::underlay` element shows,
    /// if it lies whole over the window as the frame's first picture (opaque, unclipped, one texel a
    /// pixel) with only plain shapes under it, and the first instance the window still draws over it.
    pub(crate) fn underlay(&self, size: (u32, u32)) -> Option<(u32, crate::pixels::Pixels)> {
        let pixels = self.tree.nodes.iter().find(|n| n.vis.underlay).and_then(|n| n.vis.pixels.clone())?;
        if pixels.size() != size || size.0 == 0 || size.1 == 0 { return None; }
        let (w, h) = (size.0 as f32, size.1 as f32);
        let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
        for (at, i) in self.instances.iter().enumerate() {
            match i.kind() {
                Some(Kind::Image) => {
                    let whole = near(i.rect[0], 0.0) && near(i.rect[1], 0.0) && near(i.rect[2], w) && near(i.rect[3], h)
                        && near(i.uv[2], w) && near(i.uv[3], h) && i.fill[3] >= 1.0 && i.radii == [0.0; 4]
                        && i.fade == [0; 4] && i.clip[0] <= 0.0 && i.clip[1] <= 0.0 && i.clip[2] >= w && i.clip[3] >= h;
                    return whole.then_some((at as u32 + 1, pixels));
                }
                // Something under it that isn't a plain shape: a layer, an effect, a blur.
                Some(Kind::Layer | Kind::LayerEnd | Kind::Effect | Kind::BackdropBlur) | None => return None,
                _ => {}
            }
        }
        None
    }
}
