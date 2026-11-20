//! Paint: a subtree that would paint what it painted last time copies it instead.
//!
//! Each paint records, for every node, what its subtree emitted (its instances and their marks,
//! its hits, the atlas pages its glyphs sample, its glyph count) and what that depended on: the
//! node's commit epoch (a node built again from its element gets a new one; a node a region kept
//! keeps its own), the per-node rows paint reads (`row_dy`, `row_h`, `row_mask`), and the state
//! it inherited (`Inh`). The next paint copies a subtree whose nodes are the same ones, in the
//! same shape, none of which the scene presented again for itself since (one presented only for
//! a node above it, with what it was presented with before, is as it was), under the same
//! inherited state, in a window whose shared paint state (scale, viewport, blending, the glyph
//! atlas's key, focus) is the same. Under inherited state that differs only by a move, the copy
//! is moved (`keep_shift.rs`); only by how far an ancestor has faded, it hands down the opacities
//! paint would (`keep_fade`). What paint would read elsewhere, or as time passes, keeps a
//! node from being copied (`static_ok`, `dynamic_ok`, and what the paint of it found).
//!
//! Debug builds paint a subtree they would copy and say on stderr where the copy would differ.

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};

use super::*;

static EPOCH: AtomicU32 = AtomicU32::new(1);

/// A new commit epoch for a node built from its element.
pub(in crate::runtime) fn next_epoch() -> u32 {
    EPOCH.fetch_add(1, Ordering::Relaxed)
}

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy)]
pub(in crate::runtime) struct PaintRec {
    id: NodeId,
    epoch: u32,
    /// Its index less its parent's (0 for a root).
    up: u32,
    own: u64,
    /// One past its subtree's last node.
    end: u32,
    inst: [u32; 2],
    hits: [u32; 2],
    /// Its range of `Runtime::wheel_hits`, kept with the hits so a reused subtree's wheel entries come back too.
    wheel: [u32; 2],
    overlay: [u32; 2],
    help: [u32; 2],
    drop: [u32; 2],
    pages: [u32; 2],
    glyphs: [u32; 2],
    /// What its own fields say (`static_ok`), kept while its epoch is.
    fits: bool,
    keep: bool,
    /// A glyph of its own text lay outside the clip and was left out (`keep_shift.rs`).
    cut: bool,
    /// It or a node under it draws a picture, whose place in the atlas a copy has to check.
    pictures: bool,
}

impl Default for PaintRec {
    fn default() -> Self {
        PaintRec { id: NodeId(u64::MAX), epoch: 0, up: 0, own: 0, end: 0, inst: [0; 2], hits: [0; 2], wheel: [0; 2], overlay: [0; 2], help: [0; 2], drop: [0; 2], pages: [0; 2], glyphs: [0; 2], fits: false, keep: false, cut: false, pictures: false }
    }
}

#[derive(Default)]
pub(in crate::runtime) struct PaintKeep {
    pub(in crate::runtime) cycles: cycle::CyclePaint,
    recs: Vec<PaintRec>,
    prev: Vec<PaintRec>,
    instances: Vec<Instance>,
    marks: Vec<u32>,
    hits: Vec<(NodeId, [f32; 4])>,
    wheel: Vec<(NodeId, [f32; 4])>,
    overlay: Vec<[f32; 4]>,
    help: Vec<(NodeId, [f32; 4], usize, Option<usize>)>,
    drop: Vec<(NodeId, [f32; 4], usize)>,
    drawn: Vec<[f32; 4]>,
    drawn_scale: Vec<f32>,
    seen: Vec<bool>,
    inh: Vec<Inh>,
    /// What each node inherited when it painted, this paint and last: a subtree that inherits
    /// the same but for a move is copied where it is now (`keep_shift.rs`).
    pinh: Vec<Inh>,
    prev_pinh: Vec<Inh>,
    /// The window in device pixels, as this paint culls against it.
    vp: [f32; 4],
    /// The node painting now left a glyph of its text out, for lying outside the clip.
    pub(in crate::runtime) glyph_out: bool,
    /// Each node's box as its hit rects are cut from it (points; x0, y0, x1, y1), this paint
    /// and last: a copy that moved cuts it again where the clip is now.
    pub(in crate::runtime) hit_raw: Vec<[f32; 4]>,
    prev_hit_raw: Vec<[f32; 4]>,
    /// Each node's own opacity, for paint and for hits, as it multiplied what it inherited by:
    /// a copy under a fade works out again what each node hands down (`keep_fade`).
    pub(in crate::runtime) own_opacity: Vec<[f32; 2]>,
    prev_own_opacity: Vec<[f32; 2]>,
    /// The opacities (paint, hits) each node of a subtree hands down under the fade it is
    /// being copied under.
    faded: Vec<[f32; 2]>,
    /// What the scene presented for each node when it painted, and where it said the node comes
    /// to rest, this paint and last. A node presented again since only because one above it was
    /// (`carried`), with the very same values, is as it was.
    pub(in crate::runtime) shown: Vec<Option<Shown>>,
    prev_shown: Vec<Option<Shown>>,
    /// Each picture's place in the image atlas as it was drawn, this paint and last: a copy
    /// of it holds while the atlas still has those pixels there (`keep_pictures_hold`).
    pub(in crate::runtime) picture: Vec<Option<Picture>>,
    prev_picture: Vec<Option<Picture>>,
    /// The box each text is clipped to, where it is clipped to its own, this paint and last.
    pub(in crate::runtime) text_cut: Vec<Option<shift::TextCut>>,
    prev_text_cut: Vec<Option<shift::TextCut>>,
    sticky_frame: Vec<[f32; 8]>,
    pages: Vec<(u16, u32)>,
    prev_pages: Vec<(u16, u32)>,
    /// Nodes presented again since the last paint: a value of their own changed.
    pub(in crate::runtime) moved: Vec<NodeId>,
    /// Nodes presented again since the last paint only because a node above them was.
    pub(in crate::runtime) carried: Vec<NodeId>,
    moved_at: Vec<bool>,
    /// The window's shared paint state at the last paint, when it could be kept from.
    key: Option<u64>,
    /// This paint copies what it can, and records what it paints (in tree order only).
    on: bool,
    recording: bool,
    /// Frames stepped, and the one the last paint ran in: a second paint in a frame paints all.
    pub(in crate::runtime) frame: u64,
    painted: u64,
    /// This paint's nodes: last paint's index of each, one past each subtree, whether each
    /// subtree is as it was.
    old: Vec<u32>,
    end: Vec<u32>,
    clean: Vec<bool>,
    fits: Vec<bool>,
    open: Vec<usize>,
    /// Last paint's index of each node by id, built when this paint's order leaves last paint's.
    by_id: HashMap<NodeId, u32>,
    /// Debug builds: the last paint's index of a subtree painted to check its copy.
    verify: Vec<u32>,
    verify_span: (usize, usize),
    /// Debug builds: the copy such a subtree would have got, moved.
    verify_moved: Vec<Option<Box<shift::Moved>>>,
    /// Debug builds: what each node of such a subtree would hand down, copied under a fade.
    verify_faded: Vec<Option<Box<[[f32; 2]]>>>,
    /// Tests: the nodes of subtrees checked against their paint as copies under a fade, and how
    /// many of those subtrees differed.
    #[cfg(test)]
    pub(in crate::runtime) fade_checks: (usize, usize),
    /// Tests: the nodes copied under a fade, copying as a release build does (`real_copies`).
    #[cfg(test)]
    pub(in crate::runtime) fade_copies: usize,
    /// Nodes copied this paint from where they were (`RUNE_PAINT_KEEP_LOG`).
    shifted: usize,
    /// Tests: the nodes of moved subtrees checked against their paint so far, and how many of
    /// those subtrees painted something else.
    #[cfg(test)]
    pub(in crate::runtime) moved_checks: (usize, usize),
    /// Tests: copy as a release build does, rather than paint what would be copied and check
    /// it, so a test can follow copies made from copies (`keep_shift.rs`).
    #[cfg(test)]
    pub(in crate::runtime) real_copies: bool,
    /// Tests: the nodes presented again only for one above them and found as they were, and
    /// how many subtrees, copied from where they were, painted something else.
    #[cfg(test)]
    pub(in crate::runtime) carried_checks: (usize, usize),
    /// Nodes copied this paint, and why the others weren't as they were (`RUNE_PAINT_KEEP_LOG`).
    pub(in crate::runtime) kept: usize,
    why: [usize; 6],
    /// The next paint paints everything (the raster check's fresh paint).
    pub(in crate::runtime) off_once: bool,
}

fn disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("RUNE_PAINT_KEEP").is_some_and(|v| v == "0"))
}

fn logging() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RUNE_PAINT_KEEP_LOG").is_some_and(|v| v != "0"))
}

#[inline(always)]
fn bits(h: &mut rune_motion::Fx, v: &[f32]) {
    // Two values a word: half the hasher's rounds.
    let mut pairs = v.chunks_exact(2);
    for p in &mut pairs {
        h.write_u64(u64::from(p[0].to_bits()) | u64::from(p[1].to_bits()) << 32);
    }
    if let [x] = pairs.remainder() {
        h.write_u32(x.to_bits());
    }
}

/// The bits of what a node inherits.
pub(in crate::runtime) fn inh_key(p: &Inh) -> u64 {
    let mut h = rune_motion::Fx::default();
    bits(&mut h, &[p.opacity, p.paint_opacity, p.first_fade, p.scale, p.blur, p.dy, p.dx, p.group, p.hit_opacity]);
    for a in [p.clip, p.fade, p.fade_origin, p.paint_clip, p.paint_fade, p.paint_fade_origin, p.group_rect] {
        bits(&mut h, &a);
    }
    bits(&mut h, &p.anchor);
    bits(&mut h, &p.anchor_to);
    bits(&mut h, &p.cover);
    match p.rounded {
        Some((a, b)) => { h.write_u8(1); bits(&mut h, &a); bits(&mut h, &b); }
        None => h.write_u8(0),
    }
    match p.outer {
        Some(o) => { h.write_u8(1); for a in o { bits(&mut h, &a); } }
        None => h.write_u8(0),
    }
    h.write_u32(p.mark);
    h.write_u8(p.scale_motion as u8);
    bits(&mut h, &[p.still.0]);
    bits(&mut h, &p.still.1);
    bits(&mut h, &p.still.2);
    bits(&mut h, &[p.held.0]);
    bits(&mut h, &p.held.1);
    bits(&mut h, &p.held.2);
    bits(&mut h, &[p.scale_target]);
    bits(&mut h, &p.press_off_by);
    (p.held_moving, p.scale_tracked, p.in_cache, p.skip_paint, p.anchor_hidden, p.turn).hash(&mut h);
    p.snap.hash_bits(&mut h);
    h.finish()
}

impl RuntimeCore {
    /// Before a paint clears them: last paint's output and side tables become the record copies
    /// come from.
    pub(in crate::runtime) fn keep_swap(&mut self) {
        let k = &mut self.paint.keep;
        k.cycles.begin(self.commit.commits);
        std::mem::swap(&mut self.instances, &mut k.instances);
        // Both sides of the swap retain the warmed paint budget, even if a reveal
        // temporarily emits fewer glyphs on the frames painted into one of them.
        self.instances.clear();
        self.instances.reserve_exact(k.instances.capacity());
        std::mem::swap(&mut self.marks, &mut k.marks);
        std::mem::swap(&mut self.hit.hits, &mut k.hits);
        std::mem::swap(&mut self.hit.wheel_hits, &mut k.wheel);
        std::mem::swap(&mut self.hit.overlay_hits, &mut k.overlay);
        std::mem::swap(&mut self.hit.help_hits, &mut k.help);
        std::mem::swap(&mut self.hit.drop_hits, &mut k.drop);
        std::mem::swap(&mut self.paint.drawn, &mut k.drawn);
        std::mem::swap(&mut self.paint.drawn_scale, &mut k.drawn_scale);
        std::mem::swap(&mut self.paint.seen, &mut k.seen);
        std::mem::swap(&mut self.paint.inh_scratch, &mut k.inh);
        std::mem::swap(&mut self.scroll.sticky_frame, &mut k.sticky_frame);
        std::mem::swap(&mut k.recs, &mut k.prev);
        std::mem::swap(&mut k.pinh, &mut k.prev_pinh);
        // Not cleared: a node reads last paint's only where its record says it was written.
        k.pinh.resize(self.tree.nodes.len(), Inh::default());
        std::mem::swap(&mut k.hit_raw, &mut k.prev_hit_raw);
        k.hit_raw.resize(self.tree.nodes.len(), [0.0; 4]);
        std::mem::swap(&mut k.own_opacity, &mut k.prev_own_opacity);
        k.own_opacity.resize(self.tree.nodes.len(), [f32::NAN; 2]);
        std::mem::swap(&mut k.shown, &mut k.prev_shown);
        k.shown.resize(self.tree.nodes.len(), None);
        std::mem::swap(&mut k.picture, &mut k.prev_picture);
        k.picture.resize(self.tree.nodes.len(), None);
        std::mem::swap(&mut k.text_cut, &mut k.prev_text_cut);
        k.text_cut.resize(self.tree.nodes.len(), None);
        std::mem::swap(&mut k.pages, &mut k.prev_pages);
        // The other pairs too, as the instances above: a paint that grows one buffer doesn't
        // leave the frame after it to grow the other.
        twin(&mut self.marks, &k.marks);
        twin(&mut self.hit.hits, &k.hits);
        twin(&mut self.hit.wheel_hits, &k.wheel);
        twin(&mut self.hit.overlay_hits, &k.overlay);
        twin(&mut self.hit.help_hits, &k.help);
        twin(&mut self.hit.drop_hits, &k.drop);
        twin(&mut k.pages, &k.prev_pages);
        k.pages.clear();
        k.recs.clear();
        k.recs.resize(self.tree.nodes.len(), PaintRec::default());
        k.open.clear();
        k.kept = 0;
        k.shifted = 0;
    }

    /// Whether this paint copies what it can, from the window's shared state, and if so which
    /// of its subtrees are as last paint left them.
    pub(in crate::runtime) fn keep_begin(&mut self, tree_order: bool, vp: [f32; 4], default_ring: Option<usize>, sel: &[(usize, usize, usize)], marks: &[(usize, usize, usize)]) {
        let second = self.paint.keep.painted == self.paint.keep.frame;
        self.paint.keep.painted = self.paint.keep.frame;
        // Paint goes depth first, siblings in paint order (`update_layer_paint_order`), so a subtree
        // paints in one run unless it holds a node lifted out to paint last.
        let lifted = !self.motion.flying.is_empty() || self.tree.has_raise || self.paint.has_lifted_anchors();
        let shareable = !lifted && self.motion.flights().is_empty() && self.paint.paint_rank.is_none()
            && self.kbd.modal_scope.is_none() && !self.commit.defaults.force_focus_rings && !disabled();
        let key = shareable.then(|| {
            let mut h = rune_motion::Fx::default();
            bits(&mut h, &[self.env.scale]);
            bits(&mut h, &vp);
            (self.paint.srgb_blend, self.track_marks, self.reduced_motion, self.env.system_dark, self.text.emit_key(), self.tree.has_sticky, tree_order).hash(&mut h);
            (self.kbd.focus.path(), self.kbd.focus.keyboard(), self.kbd.focus.shown()).hash(&mut h);
            default_ring.map(|i| self.tree.nodes[i].id).hash(&mut h);
            let ring = self.default_focus_ring();
            bits(&mut h, &[ring.width, ring.offset, ring.color.r, ring.color.g, ring.color.b, ring.color.a]);
            h.finish()
        });
        if logging() && !shareable {
            eprintln!("rune: paint keep off: raise {}, text flights {}, flying {}, rank {}, modal {}", self.tree.has_raise, self.motion.flights().len(), self.motion.flying.len(), self.paint.paint_rank.is_some(), self.kbd.modal_scope.is_some());
        }
        let k = &mut self.paint.keep;
        k.recording = !lifted;
        k.on = key.is_some() && key == k.key && !second && !std::mem::take(&mut k.off_once) && !k.prev.is_empty();
        k.key = key;
        let n = self.tree.nodes.len();
        k.verify.clear();
        k.verify.resize(n, NONE);
        k.verify_span = (0, 0);
        k.vp = vp;
        if cfg!(debug_assertions) {
            k.verify_moved.clear();
            k.verify_moved.resize_with(n, || None);
            k.verify_faded.clear();
            k.verify_faded.resize_with(n, || None);
        }
        if !k.recording {
            return;
        }
        // Last paint's index of each node: in order, as kept regions leave them.
        k.old.clear();
        k.old.resize(n, NONE);
        // Scratch kept from paint to paint: a frame allocates nothing here.
        k.by_id.clear();
        let mut cursor = 0usize;
        for i in 0..n {
            let id = self.tree.nodes[i].id;
            let o = if k.prev.get(cursor).is_some_and(|r| r.id == id) { Some(cursor) } else {
                if k.by_id.is_empty() {
                    k.by_id.extend(k.prev.iter().enumerate().map(|(o, r)| (r.id, o as u32)));
                }
                k.by_id.get(&id).map(|&o| o as usize)
            };
            if let Some(o) = o {
                k.old[i] = o as u32;
                cursor = o + 1;
            }
        }
        // What each node's own fields say, read again only for a node built again.
        k.fits.clear();
        for i in 0..n {
            let o = k.old[i];
            let fits = match k.prev.get(o as usize).filter(|r| o != NONE && r.epoch == self.tree.nodes[i].epoch) {
                Some(r) => r.fits,
                None => static_ok(&self.tree.nodes[i]),
            };
            k.fits.push(fits);
        }
        if !k.on {
            return;
        }
        k.moved_at.clear();
        k.moved_at.resize(n, false);
        // The scene presents a whole subtree again when the node above it changes at all. One
        // presented with what it painted from last time, nothing of its own having changed,
        // hasn't moved; what it inherits is compared as it paints.
        for id in &k.carried {
            if let Some(&i) = self.tree.node_index.get(id) && i < n {
                let was = k.prev_shown.get(k.old[i] as usize).copied().flatten();
                k.moved_at[i] = was.is_none() || was != shown(&self.scene, *id);
                #[cfg(test)]
                { k.carried_checks.0 += !k.moved_at[i] as usize; }
            }
        }
        for id in &k.moved {
            if let Some(&i) = self.tree.node_index.get(id) && i < n {
                k.moved_at[i] = true;
            }
        }
        // One past each subtree, and whether each node is as it was.
        k.end.clear();
        k.end.resize(n, n as u32);
        k.clean.clear();
        k.clean.resize(n, false);
        k.open.clear();
        for i in 0..n {
            let parent = self.tree.nodes[i].parent;
            let (own, fit) = (own_key(self, i), self.paint.keep.fits[i] && dynamic_ok(self, i, sel, marks));
            let k = &mut self.paint.keep;
            while let Some(&r) = k.open.last() && parent.is_none_or(|p| r > p) {
                k.end[r] = i as u32;
                k.open.pop();
            }
            k.open.push(i);
            let o = k.old[i];
            if logging() {
                let why = if !fit { 0 } else if o == NONE { 1 } else if k.moved_at[i] { 2 } else if k.prev[o as usize].epoch != self.tree.nodes[i].epoch { 3 } else if k.prev[o as usize].own != own { 4 } else { 5 };
                k.why[why] += 1;
                if (why == 2 || why == 3) && std::env::var_os("RUNE_PAINT_KEEP_LOG").is_some_and(|v| v == "2") {
                    eprintln!("rune: paint keep:   {} {}", if why == 2 { "moved" } else { "rebuilt" }, self.tree.nodes[i].path);
                }
            }
            k.clean[i] = fit && o != NONE && !k.moved_at[i] && {
                let r = &k.prev[o as usize];
                let up = parent.map_or(0, |p| (i - p) as u32);
                r.epoch == self.tree.nodes[i].epoch && r.up == up && r.own == own
                    && parent.is_none_or(|p| k.old[p] != NONE && k.old[p] + up == o)
            };
        }
        let k = &mut self.paint.keep;
        k.open.clear();
        for i in (1..n).rev() {
            if !k.clean[i] && let Some(p) = self.tree.nodes[i].parent {
                k.clean[p] = false;
            }
        }
    }

    /// Opens node `i`'s record, before it paints anything.
    pub(in crate::runtime) fn keep_open(&mut self, i: usize, glyphs: usize, sel: &[(usize, usize, usize)], marks: &[(usize, usize, usize)]) {
        self.paint.slide_node_opens(i, self.instances.len(), self.env.scale);
        if !self.paint.keep.recording { return; }
        let rec = PaintRec {
            id: self.tree.nodes[i].id,
            epoch: self.tree.nodes[i].epoch,
            up: self.tree.nodes[i].parent.map_or(0, |p| (i - p) as u32),
            own: own_key(self, i),
            end: 0,
            inst: [self.instances.len() as u32; 2],
            hits: [self.hit.hits.len() as u32; 2],
            wheel: [self.hit.wheel_hits.len() as u32; 2],
            overlay: [self.hit.overlay_hits.len() as u32; 2],
            help: [self.hit.help_hits.len() as u32; 2],
            drop: [self.hit.drop_hits.len() as u32; 2],
            pages: [self.paint.keep.pages.len() as u32; 2],
            glyphs: [glyphs as u32; 2],
            fits: self.paint.keep.fits[i],
            keep: self.paint.keep.fits[i] && dynamic_ok(self, i, sel, marks),
            cut: false,
            pictures: false,
        };
        self.paint.keep.glyph_out = false;
        self.paint.keep.picture[i] = None;
        self.paint.keep.text_cut[i] = None;
        self.paint.keep.shown[i] = shown(&self.scene, self.tree.nodes[i].id);
        self.paint.keep.own_opacity[i] = [f32::NAN; 2];
        self.paint.keep.recs[i] = rec;
        self.paint.keep.open.push(i);
    }

    /// Node `i` inherits `pinh`: copies its subtree from the last paint when that's what it would
    /// paint, and returns how many nodes it holds.
    pub(in crate::runtime) fn keep_or_paint(&mut self, i: usize, pinh: &Inh, glyph_count: &mut usize) -> Option<usize> {
        // Only a node that can be copied needs what it inherits.
        if !self.paint.keep.recording || !self.paint.keep.recs[i].keep { return None; }
        self.paint.keep.pinh[i] = *pinh;
        let k = &self.paint.keep;
        if !k.on || !k.clean[i] {
            return None;
        }
        let o = k.old[i] as usize;
        let r = k.prev[o];
        let end = k.end[i] as usize;
        if !r.keep || r.end as usize - o != end - i {
            return None;
        }
        // A picture is copied only while the atlas holds the pixels it drew, where it drew them.
        if r.pictures && !self.keep_pictures_hold(i, o, end - i) { return None; }
        // Inheriting what it did, it is copied as it was; inheriting that but for how far an
        // ancestor has faded, as it was but for the opacities it hands down; but for a move,
        // where it is now. What each node inherited is held for it, painted or copied, as the
        // paint left it: one copied where a scroll took it holds what it inherited there.
        let same = inh_key(&self.paint.keep.prev_pinh[o]) == inh_key(pinh);
        let starts = self.keep_starts();
        let fade = !same && self.keep_fade(o, end - i, pinh);
        let shift = if same || fade { None } else { Some(self.keep_move(i, o, end - i, pinh)?) };
        if cfg!(debug_assertions) && !real_copies(&self.paint.keep) {
            // The move was only tried: the subtree paints, and is checked against it.
            let moved = shift.map(|_| Box::new(self.keep_take_moved(i, end, starts)));
            // Checked as one: what it holds paints as it does, inside it.
            let (a, b) = self.paint.keep.verify_span;
            if !(a..b).contains(&i) {
                self.paint.keep.verify[i] = o as u32;
                self.paint.keep.verify_moved[i] = moved;
                self.paint.keep.verify_faded[i] = fade.then(|| self.paint.keep.faded.clone().into_boxed_slice());
                self.paint.keep.verify_span = (i, end);
                self.paint.keep.kept += end - i;
            }
            return None;
        }
        self.keep_copy(i, o, end, glyph_count, starts, shift.is_some());
        if fade {
            self.keep_faded(i, end - i);
            #[cfg(test)]
            { self.paint.keep.fade_copies += end - i; }
        }
        Some(end - i)
    }

    /// Whether the subtree of `len` nodes at last paint's `o` inherits under `pinh` what it did
    /// but for how far an ancestor has faded, and would paint what it painted: inside a fading
    /// layer a node draws at full strength, and the opacity it inherits only says whether it
    /// draws and takes hits at all. Leaves in `faded` what each node hands down now, worked
    /// out as paint does.
    fn keep_fade(&mut self, o: usize, len: usize, pinh: &Inh) -> bool {
        if fade_off() { return false; }
        let k = &mut self.paint.keep;
        let was = k.prev_pinh[o];
        if was.opacity == pinh.opacity && was.hit_opacity == pinh.hit_opacity { return false; }
        if inh_key(&Inh { opacity: pinh.opacity, hit_opacity: pinh.hit_opacity, ..was }) != inh_key(pinh) { return false; }
        k.faded.clear();
        for j in 0..len {
            let (down, own) = (k.inh[o + j], k.prev_own_opacity[o + j]);
            let (then, now) = if j == 0 { ([was.opacity, was.hit_opacity], [pinh.opacity, pinh.hit_opacity]) } else {
                let up = k.prev[o + j].up as usize;
                if up == 0 || up > j { return false; }
                // It inherited what its parent handed down, untouched on the way.
                let (given, held) = (k.prev_pinh[o + j], k.inh[o + j - up]);
                if given.opacity != held.opacity || given.hit_opacity != held.hit_opacity { return false; }
                ([given.opacity, given.hit_opacity], k.faded[j - up])
            };
            let of = |up: [f32; 2]| [(own[0] * up[0]).clamp(0.0, 1.0), (own[1] * up[1]).clamp(0.0, 1.0)];
            let (a, b) = (of(then), of(now));
            // What is on record is what these make of what it inherited then.
            if down.first_fade != 1.0 || a[0].to_bits() != down.opacity.to_bits() || a[1].to_bits() != down.hit_opacity.to_bits() { return false; }
            // It draws while its opacity is over a thousandth, and takes hits by that or, hidden,
            // by its hit opacity: over a thousandth, or over nothing.
            let hit = |v: [f32; 2]| if v[0] > 0.001 { v[0] } else { v[1] };
            if (a[0] > 0.001) != (b[0] > 0.001) || (hit(a) > 0.001) != (hit(b) > 0.001) || (hit(a) > 0.0) != (hit(b) > 0.0) { return false; }
            k.faded.push(b);
        }
        true
    }

    /// The subtree copied as this paint's `i..i + len` hands down what `keep_fade` worked out.
    fn keep_faded(&mut self, i: usize, len: usize) {
        let k = &mut self.paint.keep;
        for j in 0..len {
            let [opacity, hit_opacity] = k.faded[j];
            let down = &mut self.paint.inh_scratch[i + j];
            (down.opacity, down.hit_opacity) = (opacity, hit_opacity);
            if j > 0 {
                let [opacity, hit_opacity] = k.faded[j - k.recs[i + j].up as usize];
                let up = &mut k.pinh[i + j];
                (up.opacity, up.hit_opacity) = (opacity, hit_opacity);
            }
        }
    }

    /// Whether every picture in the subtree of `len` nodes at last paint's `o` (this paint's
    /// `i`) would be given the place in the image atlas it was drawn from: the same pixels, at
    /// the same level, where they were. Asking keeps them there, as painting them does.
    fn keep_pictures_hold(&mut self, i: usize, o: usize, len: usize) -> bool {
        for j in 0..len {
            let Some(was) = self.paint.keep.prev_picture[o + j] else { continue };
            let Some(pixels) = &self.tree.nodes[i + j].vis.pixels else { return false };
            if was.plays || pixels.signature() != was.signature { return false; }
            let now = self.images.place_at(pixels, was.drawn, Some(self.t));
            if now.is_some_and(|(_, next)| next.is_some()) || now.map(|(placed, _)| placed) != was.placed { return false; }
        }
        true
    }

    /// Copies the subtree at last paint's `o` as this paint's `i..end`. `moved`: `keep_move`
    /// has added what it painted already, from `starts` on.
    fn keep_copy(&mut self, i: usize, o: usize, end: usize, glyph_count: &mut usize, starts: shift::Starts, moved: bool) {
        let k = &mut self.paint.keep;
        k.open.pop();
        let r = k.prev[o];
        let range = |a: [u32; 2]| a[0] as usize..a[1] as usize;
        let at = [starts.instances, starts.hits, starts.overlay, k.pages.len()];
        let (at_wheel, at_help, at_drop) = (starts.wheel, starts.help, starts.drop);
        let len = end - i;
        match moved {
            true => {}
            false => {
                self.instances.extend_from_slice(&k.instances[range(r.inst)]);
                if self.track_marks {
                    self.marks.resize(at[0], 0);
                    self.marks.extend_from_slice(&k.marks[range(r.inst)]);
                }
                self.hit.hits.extend_from_slice(&k.hits[range(r.hits)]);
                self.hit.wheel_hits.extend_from_slice(&k.wheel[range(r.wheel)]);
                self.hit.overlay_hits.extend_from_slice(&k.overlay[range(r.overlay)]);
                let seq = |s: usize| s + at[1] - r.hits[0] as usize;
                self.hit.help_hits.extend(k.help[range(r.help)].iter().map(|&(id, rect, s, l)| (id, rect, seq(s), l)));
                self.hit.drop_hits.extend(k.drop[range(r.drop)].iter().map(|&(id, rect, s)| (id, rect, seq(s))));
                self.paint.drawn[i..end].copy_from_slice(&k.drawn[o..o + len]);
                self.paint.drawn_scale[i..end].copy_from_slice(&k.drawn_scale[o..o + len]);
                self.paint.seen[i..end].copy_from_slice(&k.seen[o..o + len]);
                self.paint.inh_scratch[i..end].copy_from_slice(&k.inh[o..o + len]);
                k.pinh[i + 1..end].copy_from_slice(&k.prev_pinh[o + 1..o + len]);
                k.hit_raw[i..end].copy_from_slice(&k.prev_hit_raw[o..o + len]);
                k.own_opacity[i..end].copy_from_slice(&k.prev_own_opacity[o..o + len]);
                k.text_cut[i..end].copy_from_slice(&k.prev_text_cut[o..o + len]);
                k.picture[i..end].copy_from_slice(&k.prev_picture[o..o + len]);
                k.shown[i..end].copy_from_slice(&k.prev_shown[o..o + len]);
            }
        }
        let k = &mut self.paint.keep;
        let pages = range(r.pages);
        k.pages.extend_from_slice(&k.prev_pages[pages.clone()]);
        for &(page, texels) in &k.prev_pages[pages] {
            self.text.touch_quad(page, texels);
        }
        *glyph_count += (r.glyphs[1] - r.glyphs[0]) as usize;
        if self.tree.has_sticky {
            self.scroll.sticky_frame[i..end].copy_from_slice(&k.sticky_frame[o..o + len]);
        }
        // Records for the next paint, as if it had painted them here.
        let shift = |a: [u32; 2], from: u32, to: usize| a.map(|v| v - from + to as u32);
        let glyphs = *glyph_count as u32 - (r.glyphs[1] - r.glyphs[0]);
        for d in 0..len {
            let p = k.prev[o + d];
            k.recs[i + d] = PaintRec {
                end: p.end - o as u32 + i as u32,
                inst: shift(p.inst, r.inst[0], at[0]),
                hits: shift(p.hits, r.hits[0], at[1]),
                wheel: shift(p.wheel, r.wheel[0], at_wheel),
                overlay: shift(p.overlay, r.overlay[0], at[2]),
                help: shift(p.help, r.help[0], at_help),
                drop: shift(p.drop, r.drop[0], at_drop),
                pages: shift(p.pages, r.pages[0], at[3]),
                glyphs: shift(p.glyphs, r.glyphs[0], glyphs as usize),
                ..p
            };
        }
        // A copy is never closed (`keep_close`), so it hands its pictures up here. Otherwise the
        // node above, painted around it, records none, and when that node is copied next it is
        // copied without asking the atlas for them: their pages go untouched, the atlas reuses
        // the space, and the copy shows whatever image lands there.
        if r.pictures && let Some(p) = self.tree.nodes[i].parent {
            k.recs[p].pictures = true;
        }
        k.kept += len;
    }

    /// Node `i` painted itself (not yet its children): what it emitted, and what painting it
    /// found, say whether it can be copied.
    pub(in crate::runtime) fn keep_painted(&mut self, i: usize, painted_ok: bool) {
        let k = &mut self.paint.keep;
        if !k.recording { return; }
        let start = k.recs[i].inst[0] as usize;
        // A picture's own image holds while its place in the atlas does; one that plays, or
        // any other image (a colour glyph, a colour icon), is drawn each paint.
        let picture = k.picture[i].is_some_and(|p| !p.plays);
        let mut ok = painted_ok && k.picture[i].is_none_or(|p| !p.plays);
        k.recs[i].pictures = k.picture[i].is_some();
        for q in &self.instances[start..] {
            let kind = q.params[1];
            if (kind == KIND_IMAGE && !picture) || kind == KIND_EFFECT || kind == crate::render::KIND_BACKDROP_BLUR || (kind == KIND_GLYPH && q.radii[3] > 0.0) {
                ok = false;
            } else if kind == KIND_GLYPH {
                let (page, texels) = (q.params[2] as u16, (q.rect[2] * if q.radii[0] == 1.0 { 3.0 } else { 1.0 } * q.rect[3]) as u32);
                let from = k.recs[i].pages[0] as usize;
                match k.pages[from..].iter_mut().find(|p| p.0 == page) {
                    Some(p) => p.1 += texels,
                    None => k.pages.push((page, texels)),
                }
            }
        }
        k.recs[i].keep &= ok;
        k.recs[i].cut = std::mem::take(&mut k.glyph_out);
    }

    /// Closes the records of the open subtrees node `next`'s parent isn't inside (all of them
    /// for None), with the layers they opened.
    pub(in crate::runtime) fn keep_close(&mut self, parent: Option<Option<usize>>, layers: &mut Vec<(usize, usize, usize)>, active_cache: &mut Option<ActiveCache>, glyph_count: &mut usize) {
        if !self.paint.keep.recording { return; }
        while let Some(&r) = self.paint.keep.open.last() {
            if let Some(Some(p)) = parent && r <= p {
                break;
            }
            while let Some(&(owner, start, surface_start)) = layers.last() {
                if owner < r { break; }
                layers.pop();
                let before = self.instances.len();
                if active_cache.is_some_and(|ac| ac.instance_start == start) {
                    let ac = active_cache.take().unwrap();
                    self.paint.close_cached_layer(&self.tree.nodes, &mut self.instances, &mut self.surfaces, &mut self.marks, self.track_marks, &mut self.text, self.commit.commit_gen(), ac, start, surface_start, glyph_count);
                } else {
                    crate::render::end_layer(&mut self.instances, &mut self.surfaces, &mut self.marks, start, surface_start);
                }
                // A scope folded into its one primitive took its marker out: what its subtree
                // recorded after the marker moved down one. The primitive took the scope's
                // opacity too, so a node inside doesn't hold what it painted, and isn't copied
                // on its own: the scope's node, which already holds what they said, is.
                if self.instances.len() < before {
                    for rec in &mut self.paint.keep.recs[owner + 1..] {
                        if rec.inst[0] as usize > start { rec.keep = false; }
                        for v in rec.inst.iter_mut() {
                            if *v as usize > start { *v -= 1; }
                        }
                    }
                } else if self.instances.len() != before + 1 {
                    // A cached scope elided or replaced what it drew.
                    self.paint.keep.recs[owner].keep = false;
                }
            }
            self.paint.keep.open.pop();
            let (help, drop) = (self.hit.help_hits.len() as u32, self.hit.drop_hits.len() as u32);
            let k = &mut self.paint.keep;
            let rec = &mut k.recs[r];
            (rec.help[1], rec.drop[1]) = (help, drop);
            rec.inst[1] = self.instances.len() as u32;
            rec.hits[1] = self.hit.hits.len() as u32;
            rec.wheel[1] = self.hit.wheel_hits.len() as u32;
            rec.overlay[1] = self.hit.overlay_hits.len() as u32;
            rec.pages[1] = k.pages.len() as u32;
            rec.glyphs[1] = *glyph_count as u32;
            let (keep, pictures) = (rec.keep, rec.pictures);
            if let Some(p) = self.tree.nodes[r].parent {
                k.recs[p].keep &= keep;
                k.recs[p].pictures |= pictures;
            }
            if cfg!(debug_assertions) && self.paint.keep.verify.get(r).is_some_and(|&o| o != NONE) {
                self.keep_check(r);
            }
        }
    }


    /// The end of a paint: what was moved is painted, and what wasn't kept is said.
    pub(in crate::runtime) fn keep_end(&mut self) {
        self.paint.close_slide(self.instances.len(), self.env.scale);
        let k = &mut self.paint.keep;
        k.moved.clear();
        k.carried.clear();
        if !k.recording {
            k.recs.clear();
            k.shifted = 0;
            return;
        }
        // Subtree ends, from the tree: a record holds one past its subtree.
        let n = self.tree.nodes.len();
        let open = &mut k.open;
        debug_assert!(open.is_empty(), "paint closed every subtree");
        for i in 0..n {
            let parent = self.tree.nodes[i].parent;
            while let Some(&r) = open.last() && parent.is_none_or(|p| r > p) {
                k.recs[r].end = i as u32;
                open.pop();
            }
            open.push(i);
        }
        for r in open.drain(..) { k.recs[r].end = n as u32; }
        k.moved.clear();
        k.carried.clear();
        if logging() {
            eprintln!("rune: paint keep: {} of {} nodes copied, {} of them from where they were (on: {}; unfit {}, new {}, moved {}, rebuilt {}, rows {}, as they were {})", k.kept, n, k.shifted, k.on, k.why[0], k.why[1], k.why[2], k.why[3], k.why[4], k.why[5]);
            k.why = [0; 6];
        }
    }

    /// Debug builds: subtree `r` painted what its copy from the last paint would have.
    fn keep_check(&mut self, r: usize) {
        let o = self.paint.keep.verify[r] as usize;
        let mut differ = Vec::new();
        if let Some(moved) = self.paint.keep.verify_moved[r].take() {
            self.keep_check_moved(r, o, &moved, &mut differ);
            #[cfg(test)]
            {
                let checked = self.paint.keep.end[r] as usize - r;
                let k = &mut self.paint.keep;
                k.moved_checks = (k.moved_checks.0 + checked, k.moved_checks.1 + !differ.is_empty() as usize);
            }
            if !differ.is_empty() {
                eprintln!("rune: paint keep: {} would have copied, moved, what differs: {}", self.tree.nodes[r].path, differ.join(", "));
            }
            return;
        }
        let k = &self.paint.keep;
        let (now, was) = (k.recs[r], k.prev[o]);
        let range = |a: [u32; 2]| a[0] as usize..a[1] as usize;
        let bytes = bytemuck::cast_slice::<Instance, u8>;
        if bytes(&self.instances[range(now.inst)]) != bytes(&k.instances[range(was.inst)]) { differ.push("instances"); }
        if self.hit.hits[range(now.hits)] != k.hits[range(was.hits)] { differ.push("hits"); }
        if self.hit.wheel_hits[range(now.wheel)] != k.wheel[range(was.wheel)] { differ.push("wheel hits"); }
        // Rebased hits compared in place, as the bytes are: a debug frame's check allocates nothing.
        let help = |v: &[(NodeId, [f32; 4], usize, Option<usize>)], from: u32, at: usize| { let (id, rect, s, l) = v[at]; (id, rect, s - from as usize, l) };
        let (a, b) = (&self.hit.help_hits[range(now.help)], &k.help[range(was.help)]);
        if a.len() != b.len() || (0..a.len()).any(|j| help(a, now.hits[0], j) != help(b, was.hits[0], j)) { differ.push("help hits"); }
        let drop = |v: &[(NodeId, [f32; 4], usize)], from: u32, at: usize| { let (id, rect, s) = v[at]; (id, rect, s - from as usize) };
        let (a, b) = (&self.hit.drop_hits[range(now.drop)], &k.drop[range(was.drop)]);
        if a.len() != b.len() || (0..a.len()).any(|j| drop(a, now.hits[0], j) != drop(b, was.hits[0], j)) { differ.push("drop hits"); }
        if self.track_marks && self.marks.get(range(now.inst)) != k.marks.get(range(was.inst)) { differ.push("marks"); }
        let len = k.end[r] as usize - r;
        if self.paint.drawn[r..r + len] != k.drawn[o..o + len] || self.paint.drawn_scale[r..r + len] != k.drawn_scale[o..o + len] || self.paint.seen[r..r + len] != k.seen[o..o + len] { differ.push("drawn rects"); }
        // Under a fade, a copy hands down what it did but for the opacities worked out for it.
        let faded = k.verify_faded[r].as_deref();
        let held = |d: usize| match faded {
            Some(f) => Inh { opacity: f[d][0], hit_opacity: f[d][1], ..k.inh[o + d] },
            None => k.inh[o + d],
        };
        if (0..len).any(|d| inh_key(&self.paint.inh_scratch[r + d]) != inh_key(&held(d))) { differ.push("inherited state"); }
        if self.tree.has_sticky && self.scroll.sticky_frame[r..r + len] != k.sticky_frame[o..o + len] { differ.push("sticky frames"); }
        if self.hit.overlay_hits[range(now.overlay)] != k.overlay[range(was.overlay)] { differ.push("overlay hits"); }
        if now.glyphs[1] - now.glyphs[0] != was.glyphs[1] - was.glyphs[0] { differ.push("glyph count"); }
        if k.pages[range(now.pages)] != k.prev_pages[range(was.pages)] { differ.push("atlas pages"); }
        #[cfg(test)]
        if faded.is_some() {
            let k = &mut self.paint.keep;
            k.fade_checks = (k.fade_checks.0 + len, k.fade_checks.1 + !differ.is_empty() as usize);
        }
        if !differ.is_empty() {
            #[cfg(test)]
            { self.paint.keep.carried_checks.1 += 1; }
            eprintln!("rune: paint keep: {} would have copied what differs: {}", self.tree.nodes[r].path, differ.join(", "));
        }
    }
}

#[path = "cycle.rs"]
pub(super) mod cycle;
#[path = "keep_shift.rs"]
mod shift;
pub(in crate::runtime) use shift::TextCut;

/// A picture as a paint drew it: what it asked the image atlas for and what it got. The atlas
/// gives the same again while it holds the same pixels in the same place, and so a copy of
/// the picture's instance still shows them.
#[derive(Clone, Copy, PartialEq)]
pub(in crate::runtime) struct Picture {
    /// The pixels' identity and generation (`Pixels::signature`).
    pub signature: (u64, u64),
    /// The drawn size it chose a mip level by.
    pub drawn: [f32; 2],
    /// Where the atlas put it: none while it has no pixels.
    pub placed: Option<crate::pixels::Placed>,
    /// It is animated, and shows another frame later.
    pub plays: bool,
}
#[path = "slide.rs"]
mod slide;

/// `now` with room for as much as `was` holds room for.
fn twin<T>(now: &mut Vec<T>, was: &Vec<T>) {
    if now.capacity() < was.capacity() {
        now.reserve_exact(was.capacity() - now.len());
    }
}

/// Whether a test asked for copies as a release build makes them (`PaintKeep::real_copies`).
#[inline]
fn real_copies(k: &PaintKeep) -> bool {
    #[cfg(test)]
    { k.real_copies }
    #[cfg(not(test))]
    { let _ = k; false }
}

/// The per-node rows paint reads besides the node.
fn own_key(rt: &RuntimeCore, i: usize) -> u64 {
    let mut h = rune_motion::Fx::default();
    bits(&mut h, &[rt.paint.row_dy[i], rt.paint.row_h[i], rt.paint.row_mask[i]]);
    h.write_u8((rt.tree.has_dissolve && rt.paint.dissolve_scratch.get(i).copied().unwrap_or(false)) as u8);
    h.finish()
}

/// `RUNE_PAINT_FADE=0`: paint what an ancestor's fade is all that changed for, again.
fn fade_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("RUNE_PAINT_FADE").is_some_and(|v| v == "0"))
}

/// What the scene presents for a node and the rect it says the node comes to rest at, which
/// paint snaps a gliding node by.
pub(in crate::runtime) type Shown = (rune_motion::Presented, Option<rune_motion::Rect>);

fn shown(scene: &rune_motion::Scene, id: NodeId) -> Option<Shown> {
    Some((scene.presented(id)?, scene.rest(id)))
}

/// What a node's own fields say: paint reads nothing for it that changes without a new epoch
/// or a scene value.
fn static_ok(n: &Node) -> bool {
    n.input.is_none() && !n.ime_cursor && n.scroll.is_none() && n.looping.is_none() && n.live.is_empty()
        && n.vis.surface.is_none() && n.material.is_none() && !n.vis_flags.cache() && !n.vis_flags.raise()
        && n.vis.backdrop_blur <= 0.0 && n.vis.effect.is_none()
        && n.sticky.is_none() && n.above.is_none() && n.matched.is_none()
        && n.static_scale == 1.0 && !n.scale_track && n.vis.mark == 0
        && n.text.as_ref().is_none_or(|t| t.cold.links.is_empty() && t.cold.live.is_none() && t.cold.advance_in.is_none()
            && !t.cold.pills.iter().any(crate::text::PillChip::has_image))
}

/// What the runtime holds for a node besides it: selection, marks, reveals, loops, scales
/// settling and folds.
fn dynamic_ok(rt: &RuntimeCore, i: usize, sel: &[(usize, usize, usize)], marks: &[(usize, usize, usize)]) -> bool {
    let id = rt.tree.nodes[i].id;
    !sel.iter().any(|r| r.0 == i) && !marks.iter().any(|r| r.0 == i)
        && !rt.motion.reveal_fading(id)
        && !rt.motion.has_loop(id)
        && (rt.paint.static_scales.is_empty() || !rt.paint.static_scales.contains_key(&id))
        && rt.paint.row_mask[i] < 0.0
}
