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
//! - the hit, wheel, overlay, help and drop rects, each a node's box cut by the inherited clip:
//!   the record holds the box uncut (`PaintKeep::hit_raw`), so it is cut again where it is now;
//! - a text clipped to its own box (`PaintKeep::text_cut` holds the box and its soft edge), cut
//!   and faded again where it is now;
//! - each node's drawn rect, and the offset, clips, hover group box and rigid translation in
//!   what it inherits and hands down; where sticky elements may stick inside it.
//!
//! What the copy can't know, it doesn't guess. It gives the subtree back to paint when:
//! - anything inherited differs but the offset, the clips and what follows from them;
//! - a box in it doesn't snap to whole pixels, and so moved by the offset's fraction too;
//! - the subtree is scaled, turned, blurred, faded by an ancestor's soft edge, under a rounded
//!   clip, in a cache, or holds a node that clips, scrolls or offsets its own children;
//! - a text lies left of or above the window's origin, where its glyphs round the other way;
//! - a node would cross the edge of what is drawn (it is culled on one side of the move), or a
//!   text had a glyph outside the clip, which paint left out and the record doesn't hold;
//! - a node's hit rect is cut away whole on one side of the move and not the other;
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

/// A text clipped to its own box: the box (device px), and the soft edge its glyphs fade
/// through, as `clip_text_instances` packs them against the clip.
#[derive(Clone, Copy)]
pub(in crate::runtime) struct TextCut {
    pub bounds: [f32; 4],
    pub widths: [f32; 4],
    pub origins: [f32; 4],
}

/// `r` (x0, y0, x1, y1) inside `c`.
#[inline]
fn cut(r: [f32; 4], c: [f32; 4]) -> [f32; 4] {
    [r[0].max(c[0]), r[1].max(c[1]), r[2].min(c[2]), r[3].min(c[3])]
}

#[inline]
fn empty(r: [f32; 4]) -> bool {
    r[2] <= r[0] || r[3] <= r[1]
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
        // Under a hover group that stayed put, a node shown only on hover takes hits inside the
        // group's box, which the uncut rect on record already holds.
        if a.group_rect == b.group_rect && d != [0.0; 2] && a.group_rect.iter().all(|v| v.is_finite()) { return None; }
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

    /// A text's own box where the move puts it.
    #[inline]
    pub(super) fn text_cut(&self, t: &TextCut) -> TextCut {
        let by = |v: [f32; 4]| [0, 1, 2, 3].map(|e| if v[e].abs() >= NO_CLIP[2] { v[e] } else { v[e] + self.px[e % 2] });
        TextCut { bounds: by(t.bounds), widths: t.widths, origins: by(t.origins) }
    }

    /// An instance where the move puts it. `text`: its node's text, if it has one, with the
    /// box it is clipped to if it is; a glyph of a node with no text is an icon.
    #[inline]
    pub(super) fn instance(&self, q: &Instance, text: Option<Option<&TextCut>>) -> Option<Instance> {
        let kind = q.params[1];
        let glyph = kind == KIND_GLYPH;
        // Kinds whose geometry is the rect alone; an outline glyph also holds its pen.
        // A picture's image is here only where its place in the atlas held (`keep_pictures_hold`).
        let image = kind == KIND_IMAGE;
        if !(kind == crate::render::KIND_RECT || kind == crate::render::KIND_SHADOW || kind == crate::render::KIND_GRADIENT || kind == crate::render::KIND_RADIAL || kind == crate::render::KIND_DASHED || image || (glyph && q.radii[3] <= 0.0)) {
            return None;
        }
        // A box too thin to snap draws at its exact edges, and an icon at its exact place: both
        // moved by the offset itself. What snapped lies on whole pixels; a text's glyph sits
        // where its line's box put it.
        for axis in 0..2 {
            // A picture fitted inside its box sits a fraction off the box's snapped edge, and moves with it.
            let exact = if glyph { text.is_none() } else { !image && (q.rect[axis].fract() != 0.0 || q.rect[axis + 2].fract() != 0.0) };
            if exact && !self.exact[axis] { return None; }
        }
        let rect = [q.rect[0] + self.px[0], q.rect[1] + self.px[1], q.rect[2], q.rect[3]];
        let [from, to] = self.paint;
        let (clip, fade) = match text.flatten().filter(|_| glyph) {
            // Its text's box, cut by the clip where both are now, and its soft edge against that.
            Some(t) => {
                if q.clip != intersect(from, t.bounds) { return None; }
                let t = self.text_cut(t);
                let clip = intersect(to, t.bounds);
                (clip, crate::render::pack_fade(t.widths, t.origins, clip))
            }
            // The clip the subtree inherits, with no soft or rounded edge packed against it.
            None => {
                if q.clip != from || q.fade != [0; 4] { return None; }
                (to, q.fade)
            }
        };
        if glyph {
            // A glyph outside the clip is left out by paint: one that moved there, or to its
            // very edge, is no copy.
            if rect[1] >= to[3] || rect[1] + rect[3] <= to[1] || rect[0] >= to[2] || rect[0] + rect[2] <= to[0] { return None; }
            // A text's baseline and pens round half away from zero, so a glyph left of or above
            // the window's origin rounds the other way from one inside it: no whole-pixel move.
            for axis in 0..2 {
                if (self.px[axis] != 0.0 || self.d[axis] != 0.0) && (q.rect[axis] < 0.0 || rect[axis] < 0.0) { return None; }
            }
        }
        Some(Instance { rect, clip, fade, ..*q })
    }

    /// A node's uncut hit rect in points (x0, y0, x1, y1) where the move puts it.
    #[inline]
    pub(super) fn raw(&self, r: [f32; 4]) -> [f32; 4] {
        [r[0] + self.d[0], r[1] + self.d[1], r[2] + self.d[0], r[3] + self.d[1]]
    }

    /// The logical clip in points a node's hit rects are cut by, then and now: the subtree's
    /// root's (`root`), or a node's below it.
    #[inline]
    fn hit_clips(&self, root: bool) -> [[f32; 4]; 2] {
        let [from, to] = if root { self.logical } else { self.full };
        [from.map(|v| v / self.s), to.map(|v| v / self.s)]
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

/// Debug builds: what a moved copy added, taken back out so the subtree paints, and kept to
/// check that paint against.
pub(super) struct Moved {
    instances: Vec<Instance>,
    hits: Vec<(NodeId, [f32; 4])>,
    wheel: Vec<(NodeId, [f32; 4])>,
    overlay: Vec<[f32; 4]>,
    help: Vec<(NodeId, [f32; 4], usize, Option<usize>)>,
    drop: Vec<(NodeId, [f32; 4], usize)>,
    drawn: Vec<[f32; 4]>,
    inh: Vec<Inh>,
    sticky: Vec<[f32; 8]>,
    /// The hit list's length where the copy began, which help and drop hits count from.
    hits_at: usize,
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

    /// Debug builds: takes a moved copy of this paint's `i..end` back out, from `to` on, for
    /// `keep_check_moved` to hold the subtree's paint against.
    pub(super) fn keep_take_moved(&mut self, i: usize, end: usize, to: Starts) -> Moved {
        let moved = Moved {
            instances: self.instances[to.instances..].to_vec(),
            hits: self.hit.hits[to.hits..].to_vec(),
            wheel: self.hit.wheel_hits[to.wheel..].to_vec(),
            overlay: self.hit.overlay_hits[to.overlay..].to_vec(),
            help: self.hit.help_hits[to.help..].to_vec(),
            drop: self.hit.drop_hits[to.drop..].to_vec(),
            drawn: self.paint.drawn[i..end].to_vec(),
            inh: self.paint.inh_scratch[i..end].to_vec(),
            sticky: if self.tree.has_sticky { self.scroll.sticky_frame[i..end].to_vec() } else { Vec::new() },
            hits_at: to.hits,
        };
        self.keep_unwind(to);
        self.paint.keep.shifted -= end - i;
        moved
    }

    /// Copies what the subtree of `len` nodes at last paint's index `o` (this paint's `i`)
    /// painted to where it is under `pinh`, when a move is all that changed for it: its
    /// instances, hits, drawn rects and inherited state, as `keep_copy` copies them for a
    /// subtree that stayed. The move, or None with nothing added.
    pub(super) fn keep_move(&mut self, i: usize, o: usize, len: usize, pinh: &Inh) -> Option<Shift> {
        if shift_off() { return None; }
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
        // Help and drop hits count from the hit list.
        let seq = self.hit.hits.len().wrapping_sub(r.hits[0] as usize);
        // Where sticky elements may stick: the viewport's bounds, handed down from the root's
        // parent as it painted them now, and each node's own content box, which moved.
        let sticky = self.tree.has_sticky.then(|| self.tree.nodes[i].parent.map(|p| self.scroll.sticky_frame[p])).flatten();
        if self.tree.has_sticky && sticky.is_none() { return None; }
        for j in 0..len {
            let p = k.prev[o + j];
            let n = &self.tree.nodes[i + j];
            // A text that left a glyph out for lying outside the clip isn't all here, and a
            // box that doesn't snap moved by the offset's fraction too. The window's root has
            // no clip above it to cut its hits by.
            if p.cut || (n.vis_flags.no_pixel_snap() && sh.exact != [true; 2]) || n.parent.is_none() { return None; }
            if sh.visible(k.drawn[o + j], k.vp) != k.seen[o + j] { return None; }
            let (mut down, mut up) = (k.inh[o + j], k.prev_pinh[o + j]);
            if !sh.passes(&down) || (j > 0 && !sh.passes(&up)) { return None; }
            sh.carry(&mut down);
            self.paint.inh_scratch[i + j] = down;
            if j > 0 {
                sh.carry(&mut up);
                k.pinh[i + j] = up;
            }
            self.paint.drawn[i + j] = sh.drawn(k.drawn[o + j]);
            if let Some(up) = sticky {
                let f = k.sticky_frame[o + j];
                self.scroll.sticky_frame[i + j] = [up[0], up[1], f[2] + sh.d[1], f[3] + sh.d[1], up[4], up[5], f[6] + sh.d[0], f[7] + sh.d[0]];
            }
            // What is its own: up to its first child's.
            let child = if j + 1 < len && ((o + j + 1) as u32) < p.end { Some(k.prev[o + j + 1]) } else { None };
            let own = |all: [u32; 2], first: fn(&PaintRec) -> u32| all[0] as usize..child.as_ref().map_or(all[1], first) as usize;
            // Its text's own box, where the move puts it.
            let text = k.prev_text_cut[o + j];
            k.text_cut[i + j] = text.map(|t| sh.text_cut(&t));
            let of_text = n.text.as_ref().map(|_| text.as_ref());
            for q in &k.instances[own(p.inst, |c| c.inst[0])] { self.instances.push(sh.instance(q, of_text)?); }
            // Its hit rects: its box, moved, cut by the clip where that is now. Cut away whole
            // it gives no wheel or overlay hit, so one that is on only one side is no copy.
            let raw = k.prev_hit_raw[o + j];
            let [from, to] = sh.hit_clips(j == 0);
            let (was, now) = (cut(raw, from), cut(sh.raw(raw), to));
            if empty(was) != empty(now) { return None; }
            k.hit_raw[i + j] = sh.raw(raw);
            k.picture[i + j] = k.prev_picture[o + j];
            k.shown[i + j] = k.prev_shown[o + j];
            k.own_opacity[i + j] = k.prev_own_opacity[o + j];
            for at in own(p.hits, |c| c.hits[0]) {
                if k.hits[at].1 != was { return None; }
                self.hit.hits.push((k.hits[at].0, now));
            }
            for at in own(p.wheel, |c| c.wheel[0]) {
                if k.wheel[at].1 != was { return None; }
                self.hit.wheel_hits.push((k.wheel[at].0, now));
            }
            for at in own(p.overlay, |c| c.overlay[0]) {
                if k.overlay[at] != was { return None; }
                self.hit.overlay_hits.push(now);
            }
            for at in own(p.help, |c| c.help[0]) {
                let (id, rect, s, l) = k.help[at];
                if rect != was { return None; }
                self.hit.help_hits.push((id, now, s.wrapping_add(seq), l));
            }
            for at in own(p.drop, |c| c.drop[0]) {
                let (id, rect, s) = k.drop[at];
                if rect != was { return None; }
                self.hit.drop_hits.push((id, now, s.wrapping_add(seq)));
            }
        }
        // An instance after a node's children (a layer's end) belongs to none of them.
        if self.instances.len() - first != range(r.inst).len() { return None; }
        self.paint.drawn_scale[i..i + len].copy_from_slice(&k.drawn_scale[o..o + len]);
        self.paint.seen[i..i + len].copy_from_slice(&k.seen[o..o + len]);
        k.shifted += len;
        Some(())
    }

    /// Debug builds: what the subtree at `r` painted, against the moved copy `m` it would have
    /// got from last paint's index `o`. The names of what differs.
    pub(super) fn keep_check_moved(&self, r: usize, o: usize, m: &Moved, differ: &mut Vec<&'static str>) {
        let k = &self.paint.keep;
        let (now, was) = (k.recs[r], k.prev[o]);
        let range = |a: [u32; 2]| a[0] as usize..a[1] as usize;
        let len = k.end[r] as usize - r;
        let rest = |q: &Instance| Instance { rect: [0.0; 4], clip: [0.0; 4], ..*q };
        let got = &self.instances[range(now.inst)];
        if got.len() != m.instances.len() || got.iter().zip(&m.instances).any(|(g, w)| !near4(w.rect, g.rect) || !near4(w.clip, g.clip) || bytemuck::bytes_of(&rest(w)) != bytemuck::bytes_of(&rest(g))) {
            differ.push("moved instances");
        }
        let rects = |got: &[(NodeId, [f32; 4])], want: &[(NodeId, [f32; 4])]| got.len() == want.len() && got.iter().zip(want).all(|(g, w)| g.0 == w.0 && near4(g.1, w.1));
        if !rects(&self.hit.hits[range(now.hits)], &m.hits) { differ.push("moved hits"); }
        if !rects(&self.hit.wheel_hits[range(now.wheel)], &m.wheel) { differ.push("moved wheel hits"); }
        let got = &self.hit.overlay_hits[range(now.overlay)];
        if got.len() != m.overlay.len() || got.iter().zip(&m.overlay).any(|(g, w)| !near4(*g, *w)) { differ.push("moved overlay hits"); }
        let got = &self.hit.help_hits[range(now.help)];
        if got.len() != m.help.len() || got.iter().zip(&m.help).any(|(g, w)| g.0 != w.0 || !near4(g.1, w.1) || g.2 - now.hits[0] as usize != w.2 - m.hits_at || g.3 != w.3) { differ.push("moved help hits"); }
        let got = &self.hit.drop_hits[range(now.drop)];
        if got.len() != m.drop.len() || got.iter().zip(&m.drop).any(|(g, w)| g.0 != w.0 || !near4(g.1, w.1) || g.2 - now.hits[0] as usize != w.2 - m.hits_at) { differ.push("moved drop hits"); }
        if (0..len).any(|j| !near4(self.paint.drawn[r + j], m.drawn[j]) || self.paint.drawn_scale[r + j] != k.drawn_scale[o + j] || self.paint.seen[r + j] != k.seen[o + j]) {
            differ.push("moved drawn rects");
        }
        // A hover group's box is moved by the offset here and placed by it there: the two may
        // differ in the last bit, and by nothing else.
        let alike = |want: &Inh, got: &Inh| {
            let mut x = *want;
            if near4(x.group_rect, got.group_rect) { x.group_rect = got.group_rect; }
            inh_key(&x) == inh_key(got)
        };
        if (0..len).any(|j| !alike(&m.inh[j], &self.paint.inh_scratch[r + j])) { differ.push("moved inherited state"); }
        if self.tree.has_sticky && (0..len).any(|j| (0..8).any(|e| { let (g, w) = (self.scroll.sticky_frame[r + j][e], m.sticky[j][e]); g != w && !near(g, w) })) { differ.push("moved sticky frames"); }
        if self.track_marks && self.marks.get(range(now.inst)) != k.marks.get(range(was.inst)) { differ.push("marks"); }
        if now.glyphs[1] - now.glyphs[0] != was.glyphs[1] - was.glyphs[0] { differ.push("glyph count"); }
        if k.pages[range(now.pages)] != k.prev_pages[range(was.pages)] { differ.push("atlas pages"); }
    }
}
