//! Paint keep for a subtree that only moved: what it painted last time, copied where it is now.
//!
//! A scroll moves everything in its viewport by the same whole number of device pixels (boxes
//! snap, and the scroll's offset is rounded apart from them) and changes nothing else about it, yet each node inherits a different offset from the one it
//! painted under, so [`keep_or_paint`](RuntimeCore::keep_or_paint) would paint it all again: a
//! page of cards, every frame of the scroll. Here the subtree's record is copied with that move
//! applied to it, when the move is all that changed.
//!
//! What a move changes, and so what the copy rewrites:
//! - an instance's rect, by the move; its clip, which is the clip the subtree inherits (given
//!   again this paint, whether or not it moved) or a clip from inside the subtree (a truncated
//!   text's box), which moved with it;
//! - the hit, wheel, overlay, help and drop rects, each a node's box cut by the inherited clip;
//! - each node's drawn rect, and the offset, clips, hover group box and rigid translation in
//!   what it inherits and hands down.
//!
//! What the copy can't know, it doesn't guess. It gives the subtree back to paint when:
//! - anything inherited differs but the offset, the clips and what follows from them;
//! - a box in it doesn't snap to whole pixels, and so moved by the offset's fraction too;
//! - the subtree is scaled, turned, blurred, faded by an ancestor's soft edge, under a rounded
//!   clip, in a cache, or holds a node that clips, scrolls or offsets its own children;
//! - a text lies left of or above the window's origin, where its glyphs round the other way;
//! - a node would cross the edge of what is drawn (it is culled on one side of the move), or a
//!   text had a glyph outside the clip, which paint left out and the record doesn't hold;
//! - an edge that was cut by a clip that stayed put may be cut elsewhere now: a hit rect's, or
//!   a truncated text's clip, whose uncut edge the record doesn't hold;
//! - it holds an instance whose geometry isn't its rect alone (a segment, a triangle, a layer).

use super::*;

/// The move between what a subtree inherited last paint (`a`) and inherits now (`b`).
#[derive(Clone, Copy)]
pub(super) struct Shift {
    /// The display scale.
    s: f32,
    /// The move in points, and in device pixels (whole).
    d: [f32; 2],
    px: [f32; 2],
    /// Whether what draws at its exact place moved as far as what snaps, on each axis.
    exact: [bool; 2],
    /// The inherited offset (x, y), then and now.
    at: [[f32; 2]; 2],
    /// The inherited paint clip and logical clip, and the logical clip inside the paint clip,
    /// which every node below the root inherits: then and now.
    paint: [[f32; 4]; 2],
    logical: [[f32; 4]; 2],
    full: [[f32; 4]; 2],
    group: [[f32; 4]; 2],
    rigid: [[f32; 2]; 2],
    anchor: [f32; 2],
    cover: [f32; 2],
}

/// Whether `x` is the plain case a move can be applied to: unscaled, unturned, unblurred, with
/// no soft edges or rounded clip above it, outside any cache or scaled snap layer.
fn plain(x: &Inh) -> bool {
    x.scale == 1.0 && x.anchor == x.anchor_to && x.blur == 0.0 && x.turn == 0 && !x.in_cache && !x.skip_paint
        && x.rounded.is_none() && x.outer.is_none() && x.fade == [0.0; 4] && x.paint_fade == [0.0; 4] && x.snap.layer == 0
}

/// `RUNE_PAINT_SHIFT=0`: paint what moved again, as `RUNE_PAINT_KEEP=0` paints everything.
fn shift_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("RUNE_PAINT_SHIFT").is_some_and(|v| v == "0"))
}

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3
}

/// Whether two rects (or clips) agree to within what float arithmetic on them may differ by.
pub(super) fn near4(a: [f32; 4], b: [f32; 4]) -> bool {
    (0..4).all(|k| near(a[k], b[k]) || a[k] == b[k])
}

/// One edge of a box that moved by `by`, cut by a clip whose edge was `from` and is `to` (all in
/// one unit): `v` was that edge last paint. `rigid`: the clip's edge moved with the box, or is
/// no edge at all. `cut`: `v` at the clip's edge may stand for an edge beyond it.
#[inline]
fn edge(v: f32, from: f32, to: f32, by: f32, low: bool, rigid: bool, cut: bool) -> Option<f32> {
    if v != from {
        // The box's own edge: it moved, and the clip cuts it where the clip is now.
        let moved = v + by;
        return Some(if low { moved.max(to) } else { moved.min(to) });
    }
    // The clip's edge, which the box's lay on or beyond. A clip that moved with the box still
    // cuts it there; one that stayed may not, unless nothing was beyond it.
    if rigid || !cut { Some(to) } else { None }
}

impl Shift {
    /// The move from `a` to `b` at display scale `s`, when a move is all that sets them apart.
    pub(super) fn between(a: &Inh, b: &Inh, s: f32) -> Option<Shift> {
        if !plain(a) || !plain(b) || a.snap.off || b.snap.off { return None; }
        let same = a.opacity == b.opacity && a.paint_opacity == b.paint_opacity && a.first_fade == b.first_fade
            && a.anchor == b.anchor && a.cover == b.cover && a.mark == b.mark && a.group == b.group
            && a.pressed == b.pressed && a.press_to == b.press_to && a.hit_opacity == b.hit_opacity
            && a.scale_motion == b.scale_motion && a.still == b.still
            && a.fade_origin == b.fade_origin && a.paint_fade_origin == b.paint_fade_origin
            && a.snap.steps == b.snap.steps && a.snap.shifted == b.snap.shifted
            && a.snap.steady == b.snap.steady && a.snap.contain == b.snap.contain && a.snap.parent == b.snap.parent;
        if !same { return None; }
        let d = [b.dx - a.dx, b.dy - a.dy];
        // What draws at its exact place lands on the nearest 64th of a pixel (`snap_px`), which
        // a move that isn't a whole number of them doesn't carry over: a scroll's is, a row
        // gliding open's needn't be.
        if !(0..2).all(|k| near(d[k] * s * 64.0, (d[k] * s * 64.0).round())) { return None; }
        // A box draws at its place less the rigid translation, rounded, plus that translation
        // rounded on its own (`pixel_snap::snap`): however far the scroll went, what it drew
        // moved by the difference of the two roundings.
        let whole = |v: f32| (v * s + 0.5).floor();
        let px = [whole(b.snap.rigid[0]) - whole(a.snap.rigid[0]), whole(b.snap.rigid[1]) - whole(a.snap.rigid[1])];
        // The rigid translation moved by the same amount, and the hover group's box with it or
        // not at all (a group above the scroll).
        if !(0..2).all(|k| near(b.snap.rigid[k] - a.snap.rigid[k], d[k])) { return None; }
        let moved = [a.group_rect[0] + d[0], a.group_rect[1] + d[1], a.group_rect[2] + d[0], a.group_rect[3] + d[1]];
        if a.group_rect != b.group_rect && !near4(moved, b.group_rect) { return None; }
        Some(Shift {
            s, d, px,
            exact: [near(px[0], d[0] * s), near(px[1], d[1] * s)],
            at: [[a.dx, a.dy], [b.dx, b.dy]],
            paint: [a.paint_clip, b.paint_clip],
            logical: [a.clip, b.clip],
            full: [intersect(a.clip, a.paint_clip), intersect(b.clip, b.paint_clip)],
            group: [a.group_rect, b.group_rect],
            rigid: [a.snap.rigid, b.snap.rigid],
            anchor: a.anchor,
            cover: a.cover,
        })
    }

    /// Whether edge `e` of a clip that was `from` and is `to` (device px) moved with the
    /// subtree, or is the edge of no clip.
    #[inline]
    fn rigid(&self, from: [f32; 4], to: [f32; 4], e: usize) -> bool {
        to[e] == from[e] + self.px[e % 2] || (from[e].abs() >= NO_CLIP[2] && to[e] == from[e])
    }

    /// An instance's clip `c` (device px) under the inherited paint clip.
    #[inline]
    fn clip(&self, c: [f32; 4], cut: bool) -> Option<[f32; 4]> {
        let [from, to] = self.paint;
        if c == from && !cut { return Some(to); }
        let mut out = [0.0; 4];
        for e in 0..4 {
            out[e] = edge(c[e], from[e], to[e], self.px[e % 2], e < 2, self.rigid(from, to, e), cut)?;
        }
        Some(out)
    }

    /// An instance where the move puts it. `text`: its node's text, if it has one, is clipped
    /// to its own box (`Some(true)`), so a glyph's clip may be the inherited one standing in
    /// for a box beyond it. A glyph of a node with no text is an icon.
    #[inline]
    pub(super) fn instance(&self, q: &Instance, text: Option<bool>) -> Option<Instance> {
        let kind = q.params[1];
        let glyph = kind == KIND_GLYPH;
        // Kinds whose geometry is the rect alone; an outline glyph also holds its pen.
        if !(kind == crate::render::KIND_RECT || kind == crate::render::KIND_SHADOW || kind == crate::render::KIND_GRADIENT || kind == crate::render::KIND_RADIAL || kind == crate::render::KIND_DASHED || (glyph && q.radii[3] <= 0.0)) {
            return None;
        }
        // A box too thin to snap draws at its exact edges, and an icon at its exact place: both
        // moved by the offset itself. What snapped lies on whole pixels; a text's glyph sits
        // where its line's box put it.
        for axis in 0..2 {
            let exact = if glyph { text.is_none() } else { q.rect[axis].fract() != 0.0 || q.rect[axis + 2].fract() != 0.0 };
            if exact && !self.exact[axis] { return None; }
        }
        let clip = self.clip(q.clip, text == Some(true) && glyph)?;
        // A soft or rounded edge is packed against the clip: it holds only where the clip moved
        // as one with the instance.
        let whole = [q.clip[0] + self.px[0], q.clip[1] + self.px[1], q.clip[2] + self.px[0], q.clip[3] + self.px[1]];
        if q.fade != [0; 4] && clip != whole { return None; }
        let rect = [q.rect[0] + self.px[0], q.rect[1] + self.px[1], q.rect[2], q.rect[3]];
        // A glyph outside the clip is left out by paint: one that moved there, or to its very
        // edge, is no copy.
        let c = self.paint[1];
        if glyph && (rect[1] >= c[3] || rect[1] + rect[3] <= c[1] || rect[0] >= c[2] || rect[0] + rect[2] <= c[0]) { return None; }
        // A text's baseline and pens round half away from zero, so a glyph left of or above the
        // window's origin rounds the other way from one inside it: no whole-pixel move there.
        for axis in 0..2 {
            if glyph && (self.px[axis] != 0.0 || self.d[axis] != 0.0) && (q.rect[axis] < 0.0 || rect[axis] < 0.0) { return None; }
        }
        Some(Instance { rect, clip, ..*q })
    }

    /// A rect in points (x0, y0, x1, y1) that a node's box cut by its logical clip gave: the
    /// subtree's root's (`root`), or one below it.
    #[inline]
    pub(super) fn cut_rect(&self, r: [f32; 4], root: bool) -> Option<[f32; 4]> {
        let [from, to] = if root { self.logical } else { self.full };
        let mut out = [0.0; 4];
        for e in 0..4 {
            // The box moved by the offset itself, the clip by whole pixels: they went together
            // only where those are one move.
            let rigid = from[e].abs() >= NO_CLIP[2] && to[e] == from[e] || (self.exact[e % 2] && self.rigid(from, to, e));
            out[e] = edge(r[e], from[e] / self.s, to[e] / self.s, self.d[e % 2], e < 2, rigid, true)?;
        }
        Some(out)
    }

    /// Whether a node drawn at `r` (points; x, y, w, h), moved or not yet, lies wholly outside
    /// the logical clip `c` (device px): it takes no wheel, and a wheel hit it would have
    /// given is in no record.
    #[inline]
    fn outside(&self, r: [f32; 4], c: [f32; 4]) -> bool {
        r[0].max(c[0] / self.s) >= (r[0] + r[2]).min(c[2] / self.s) || r[1].max(c[1] / self.s) >= (r[1] + r[3]).min(c[3] / self.s)
    }

    /// A node's drawn rect in points (x, y, w, h).
    #[inline]
    pub(super) fn drawn(&self, r: [f32; 4]) -> [f32; 4] {
        [r[0] + self.d[0], r[1] + self.d[1], r[2], r[3]]
    }

    /// Whether a node drawn at `r` last paint is on screen now, as paint's own test has it for
    /// a node with nothing that reaches past its box.
    #[inline]
    fn visible(&self, r: [f32; 4], vp: [f32; 4]) -> bool {
        let r = self.drawn(r);
        let sr = [r[0] * self.s, r[1] * self.s, (r[0] + r[2]) * self.s, (r[1] + r[3]) * self.s];
        let (c, m) = (self.paint[1], 2.0);
        c[2] > c[0] && c[3] > c[1] && sr[2] >= c[0].max(vp[0]) - m && sr[0] <= c[2].min(vp[2]) + m
            && sr[3] >= c[1].max(vp[1]) - m && sr[1] <= c[3].min(vp[3]) + m
    }

    /// Whether `x`, which a node below the root inherits or any node of the subtree hands
    /// down, is the subtree's own inherited state passed along: no offset, clip or scale of
    /// its own for its children.
    #[inline]
    fn passes(&self, x: &Inh) -> bool {
        [x.dx, x.dy] == self.at[0] && x.paint_clip == self.paint[0] && x.clip == self.full[0] && x.snap.rigid == self.rigid[0]
            && plain(x) && x.anchor == self.anchor && x.cover == self.cover
    }

    /// `x` (which [`passes`](Self::passes)) after the move, in place.
    #[inline]
    fn carry(&self, x: &mut Inh) {
        [x.dx, x.dy] = self.at[1];
        x.paint_clip = self.paint[1];
        x.clip = self.full[1];
        let g = x.group_rect;
        x.group_rect = if g == self.group[0] { self.group[1] } else { [g[0] + self.d[0], g[1] + self.d[1], g[2] + self.d[0], g[3] + self.d[1]] };
        x.snap.rigid = self.rigid[1];
    }

    /// What [`carry`](Self::carry) makes of `x`, or None where it doesn't pass.
    pub(super) fn inh(&self, x: &Inh) -> Option<Inh> {
        let mut y = *x;
        self.passes(x).then(|| { self.carry(&mut y); y })
    }
}

/// How long each of a paint's lists was before a subtree's copy was added to it.
#[derive(Clone, Copy)]
pub(super) struct Starts {
    pub instances: usize,
    pub hits: usize,
    pub wheel: usize,
    pub overlay: usize,
    pub help: usize,
    pub drop: usize,
}

impl RuntimeCore {
    pub(super) fn keep_starts(&self) -> Starts {
        Starts { instances: self.instances.len(), hits: self.hit.hits.len(), wheel: self.hit.wheel_hits.len(), overlay: self.hit.overlay_hits.len(), help: self.hit.help_hits.len(), drop: self.hit.drop_hits.len() }
    }

    /// Takes back what was added since `to`.
    pub(super) fn keep_unwind(&mut self, to: Starts) {
        self.instances.truncate(to.instances);
        if self.track_marks { self.marks.truncate(to.instances); }
        self.hit.hits.truncate(to.hits);
        self.hit.wheel_hits.truncate(to.wheel);
        self.hit.overlay_hits.truncate(to.overlay);
        self.hit.help_hits.truncate(to.help);
        self.hit.drop_hits.truncate(to.drop);
    }

    /// Copies what the subtree of `len` nodes at last paint's index `o` (this paint's `i`)
    /// painted to where it is under `pinh`, when a move is all that changed for it: its
    /// instances, hits, drawn rects and inherited state, as `keep_copy` copies them for a
    /// subtree that stayed. The move, or None with nothing added.
    pub(super) fn keep_move(&mut self, i: usize, o: usize, len: usize, pinh: &Inh) -> Option<Shift> {
        if self.tree.has_sticky || shift_off() { return None; }
        let sh = Shift::between(&self.paint.keep.prev_pinh[o], pinh, self.env.scale)?;
        let starts = self.keep_starts();
        if self.keep_move_into(i, o, len, &sh).is_none() {
            self.keep_unwind(starts);
            return None;
        }
        Some(sh)
    }

    fn keep_move_into(&mut self, i: usize, o: usize, len: usize, sh: &Shift) -> Option<()> {
        let k = &mut self.paint.keep;
        let r = k.prev[o];
        let range = |a: [u32; 2]| a[0] as usize..a[1] as usize;
        if self.track_marks {
            self.marks.resize(self.instances.len(), 0);
            self.marks.extend_from_slice(&k.marks[range(r.inst)]);
        }
        let first = self.instances.len();
        for j in 0..len {
            let p = k.prev[o + j];
            // A text that left a glyph out for lying outside the clip isn't all here, and a
            // box that doesn't snap moved by the offset's fraction too.
            if p.cut || (self.tree.nodes[i + j].vis_flags.no_pixel_snap() && sh.exact != [true; 2]) { return None; }
            if sh.visible(k.drawn[o + j], k.vp) != k.seen[o + j] { return None; }
            // Clipped away whole, a node gives no wheel hit: none to copy once it shows.
            let [from, to] = if j == 0 { sh.logical } else { sh.full };
            if sh.outside(k.drawn[o + j], from) && !sh.outside(sh.drawn(k.drawn[o + j]), to) { return None; }
            let (mut down, mut up) = (k.inh[o + j], k.prev_pinh[o + j]);
            if !sh.passes(&down) || (j > 0 && !sh.passes(&up)) { return None; }
            sh.carry(&mut down);
            self.paint.inh_scratch[i + j] = down;
            if j > 0 {
                sh.carry(&mut up);
                k.pinh[i + j] = up;
            }
            self.paint.drawn[i + j] = sh.drawn(k.drawn[o + j]);
            // Its own instances: up to its first child's.
            let own = p.inst[0] as usize..if j + 1 < len && ((o + j + 1) as u32) < p.end { k.prev[o + j + 1].inst[0] } else { p.inst[1] } as usize;
            let text = self.tree.nodes[i + j].text.as_ref().map(|t| t.truncate_fade || t.cold.live.is_some());
            for q in &k.instances[own] { self.instances.push(sh.instance(q, text)?); }
        }
        // An instance after a node's children (a layer's end) belongs to none of them.
        if self.instances.len() - first != range(r.inst).len() { return None; }
        // The root's rects come first, cut by the clip it inherits; the rest by the clip inside it.
        let next = if len > 1 { k.prev[o + 1] } else { r };
        let below = |all: [u32; 2], first: u32| if len > 1 { first as usize } else { all[1] as usize };
        let (hits, wheel, overlay, help, drop) = (below(r.hits, next.hits[0]), below(r.wheel, next.wheel[0]), below(r.overlay, next.overlay[0]), below(r.help, next.help[0]), below(r.drop, next.drop[0]));
        let seq = self.hit.hits.len().wrapping_sub(r.hits[0] as usize);
        for at in range(r.help) {
            let (id, rect, s, l) = k.help[at];
            self.hit.help_hits.push((id, sh.cut_rect(rect, at < help)?, s.wrapping_add(seq), l));
        }
        for at in range(r.drop) {
            let (id, rect, s) = k.drop[at];
            self.hit.drop_hits.push((id, sh.cut_rect(rect, at < drop)?, s.wrapping_add(seq)));
        }
        for at in range(r.hits) { self.hit.hits.push((k.hits[at].0, sh.cut_rect(k.hits[at].1, at < hits)?)); }
        for at in range(r.wheel) {
            // Clipped away whole it takes no wheel (`push_wheel_hit`), and its record goes.
            let rect = sh.cut_rect(k.wheel[at].1, at < wheel)?;
            if rect[2] <= rect[0] || rect[3] <= rect[1] { return None; }
            self.hit.wheel_hits.push((k.wheel[at].0, rect));
        }
        for at in range(r.overlay) { self.hit.overlay_hits.push(sh.cut_rect(k.overlay[at], at < overlay)?); }
        self.paint.drawn_scale[i..i + len].copy_from_slice(&k.drawn_scale[o..o + len]);
        self.paint.seen[i..i + len].copy_from_slice(&k.seen[o..o + len]);
        k.shifted += len;
        Some(())
    }

    /// Debug builds: what the subtree at `r` painted, against what `sh` would have copied for
    /// it from last paint's index `o`. The names of what differs.
    pub(super) fn keep_check_shifted(&self, r: usize, o: usize, sh: &Shift, differ: &mut Vec<&'static str>) {
        let k = &self.paint.keep;
        let (now, was) = (k.recs[r], k.prev[o]);
        let range = |a: [u32; 2]| a[0] as usize..a[1] as usize;
        let len = k.end[r] as usize - r;
        // Instances, node by node, as the copy takes them.
        let mut at = now.inst[0] as usize;
        let mut same = range(now.inst).len() == range(was.inst).len();
        for j in 0..len {
            let p = k.prev[o + j];
            let own = p.inst[0] as usize..if j + 1 < len && ((o + j + 1) as u32) < p.end { k.prev[o + j + 1].inst[0] } else { p.inst[1] } as usize;
            let text = self.tree.nodes[r + j].text.as_ref().map(|t| t.truncate_fade || t.cold.live.is_some());
            for q in &k.instances[own] {
                let (Some(want), Some(got)) = (sh.instance(q, text), self.instances.get(at)) else { same = false; break };
                at += 1;
                let rest = |q: &Instance| Instance { rect: [0.0; 4], clip: [0.0; 4], ..*q };
                same &= near4(want.rect, got.rect) && near4(want.clip, got.clip) && bytemuck::bytes_of(&rest(&want)) == bytemuck::bytes_of(&rest(got));
            }
        }
        if !same { differ.push("moved instances"); }
        let rects = |name: &'static str, got: &mut dyn Iterator<Item = [f32; 4]>, want: &mut dyn Iterator<Item = Option<[f32; 4]>>, differ: &mut Vec<&'static str>| {
            loop {
                match (got.next(), want.next()) {
                    (None, None) => break,
                    (Some(g), Some(Some(w))) if near4(g, w) => {}
                    _ => { differ.push(name); break }
                }
            }
        };
        let below = |all: [u32; 2], first: u32| if len > 1 { first as usize } else { all[1] as usize };
        let next = if len > 1 { k.prev[o + 1] } else { was };
        let first = range(was.hits).start;
        let root = below(was.hits, next.hits[0]);
        rects("moved hits", &mut self.hit.hits[range(now.hits)].iter().map(|h| h.1), &mut k.hits[range(was.hits)].iter().enumerate().map(|(n, h)| sh.cut_rect(h.1, first + n < root)), differ);
        let first = range(was.wheel).start;
        let root = below(was.wheel, next.wheel[0]);
        rects("moved wheel hits", &mut self.hit.wheel_hits[range(now.wheel)].iter().map(|h| h.1), &mut k.wheel[range(was.wheel)].iter().enumerate().map(|(n, h)| sh.cut_rect(h.1, first + n < root)), differ);
        let first = range(was.overlay).start;
        let root = below(was.overlay, next.overlay[0]);
        rects("moved overlay hits", &mut self.hit.overlay_hits[range(now.overlay)].iter().copied(), &mut k.overlay[range(was.overlay)].iter().enumerate().map(|(n, h)| sh.cut_rect(*h, first + n < root)), differ);
        let first = range(was.help).start;
        let root = below(was.help, next.help[0]);
        rects("moved help hits", &mut self.hit.help_hits[range(now.help)].iter().map(|h| h.1), &mut k.help[range(was.help)].iter().enumerate().map(|(n, h)| sh.cut_rect(h.1, first + n < root)), differ);
        let first = range(was.drop).start;
        let root = below(was.drop, next.drop[0]);
        rects("moved drop hits", &mut self.hit.drop_hits[range(now.drop)].iter().map(|h| h.1), &mut k.drop[range(was.drop)].iter().enumerate().map(|(n, h)| sh.cut_rect(h.1, first + n < root)), differ);
        if (0..len).any(|j| !near4(self.paint.drawn[r + j], sh.drawn(k.drawn[o + j])) || self.paint.drawn_scale[r + j] != k.drawn_scale[o + j] || self.paint.seen[r + j] != k.seen[o + j]) {
            differ.push("moved drawn rects");
        }
        // A hover group's box is moved by the offset here and placed by it there: the two may
        // differ in the last bit, and by nothing else.
        let alike = |want: Option<Inh>, got: &Inh| want.is_some_and(|mut x| {
            if near4(x.group_rect, got.group_rect) { x.group_rect = got.group_rect; }
            inh_key(&x) == inh_key(got)
        });
        if (0..len).any(|j| !alike(sh.inh(&k.inh[o + j]), &self.paint.inh_scratch[r + j])) {
            differ.push("moved inherited state");
        }
        if (1..len).any(|j| k.recs[r + j].keep && !alike(sh.inh(&k.prev_pinh[o + j]), &k.pinh[r + j])) {
            differ.push("moved inherited state of a child");
        }
        if self.track_marks && self.marks.get(range(now.inst)) != k.marks.get(range(was.inst)) { differ.push("marks"); }
        if now.glyphs[1] - now.glyphs[0] != was.glyphs[1] - was.glyphs[0] { differ.push("glyph count"); }
        if k.pages[range(now.pages)] != k.prev_pages[range(was.pages)] { differ.push("atlas pages"); }
    }
}
