//! What part of a window a frame redraws. The renderer redraws only what changed into the frame
//! it keeps (`render::damaged`) and copies that out whole; on Windows the window also presents
//! only what changed (`Present1` dirty rectangles), which is all DWM composes again. A frame's GPU
//! time is mostly the pixels it shades, so a focus ring that moves, or a hover that fades, shades
//! a few rectangles instead of the window.
//!
//! What changed is the instances that differ from the last frame's. The lists' common start and
//! end are skipped; in what's left, an instance whose bytes are in both lists didn't change (its
//! place is in its bytes), so a typed glyph that pushes the rest of the list along changes only
//! itself and the caret. The bounds of the instances only one list has are the change, gathered
//! into at most `RECTS` rectangles. If the instances both lists have were drawn in another order,
//! everything changed. Lists of one length are compared in place instead, which needs no hashing:
//! the instances that differ there, old and new, bound every pixel that changed.
//!
//! A scroll changes every instance of a viewport and few of its pixels: `scrolled` pairs the
//! new list with the old one moved by a whole number of pixels, and the frame copies what moved
//! (`Shift`) and redraws the strip that came in, what changed and what stayed put over the copy.
//! A row the viewport's edge cuts is copied too: its clip, on whole pixels, lets through the same
//! copied pixels as before. Where there's no dirty-rectangle present (Linux, headless runs), a
//! frame that still has to draw whole draws into the kept frame, so the next one needn't.
//!
//! The whole window is drawn when anything else could have moved pixels (`Renderer::
//! redraws_in_place`), when the size or clear colour changed, after a new swap chain, and when
//! the change is most of the window anyway.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::render::{INSTANCE_WORDS, Instance};

/// Above this share of the window, a frame draws all of it.
const MOST: f32 = 0.6;
/// Past this many instances the new list has that the old doesn't, a frame draws all of it without
/// looking for where they are: a scroll moves every row, and checking the order of thousands of
/// instances and bounding them only to find they cover the window costs a list's frame hundreds of
/// microseconds. A typed glyph, a streamed line or a hover changes far fewer.
const MANY: usize = 256;
/// Rectangles a change is gathered into, at most.
pub(super) const RECTS: usize = crate::render::MAX_RECTS;
/// Rectangles a scroll's redraw is gathered into, at most.
const SCROLL_RECTS: usize = crate::render::MAX_ARMED;

/// The last frames' changes.
#[derive(Default)]
pub(crate) struct Damage {
    /// The last frame's instances, and its size, clear colour and swap chain (the surface's
    /// configuration, `configure::generation`).
    last: Vec<Instance>,
    shape: Option<((u32, u32), [f32; 4], u64)>,
    scratch: Scratch,
    /// The last change seen, however much was drawn (frame-profile).
    #[cfg_attr(not(feature = "frame-profile"), allow(dead_code))]
    pub(crate) seen: Option<[u32; 4]>,
}

/// Rectangles `[x0, y0, x1, y1]` in the window's pixels.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) enum Area {
    /// Everything (or not known).
    #[default]
    All,
    /// More than `MANY` instances.
    Many,
    Rects(Vec<[u32; 4]>),
    /// A scroll (`scrolled`): the kept frame's pixels move by a shift, and these redraw.
    Scroll(Shift, Vec<[u32; 4]>),
}

/// Pixels a partial frame copies within the kept frame before it redraws: the `[x, y, width,
/// height]` at `from` go to `to`. A scroll moves what it moved this way instead of shading it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shift {
    pub(crate) from: [u32; 4],
    pub(crate) to: [u32; 2],
}

/// What a frame draws.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Plan {
    /// Everything, and why.
    All(&'static str),
    /// Redraw `draw` (`[x, y, width, height]` each), present `present` (`[x0, y0, x1, y1]` each),
    /// after copying `shift` within the kept frame. `same`: nothing changed, and `draw` is one
    /// pixel only because a present needs one.
    Part { draw: Vec<[u32; 4]>, present: Vec<[i32; 4]>, shift: Option<Shift>, same: bool },
}

impl Damage {
    /// The frame about to draw `instances` into a `size` target cleared to `clear`, on swap chain
    /// `swap`; `in_place` when nothing but its instances draws there. With `scroll`, a frame that
    /// moved a viewport's content may copy it within the kept frame (`Plan::Part::shift`); a target
    /// presented in dirty rectangles (Windows) leaves it off.
    pub(crate) fn plan(&mut self, instances: &[Instance], size: (u32, u32), clear: [f32; 4], swap: u64, in_place: bool, scroll: bool) -> Plan {
        self.scratch.differ_all = false;
        let (now, why) = match (in_place && !whole_frames(), self.shape == Some((size, clear, swap))) {
            (false, _) => (Area::All, "not in place"),
            (_, false) => (Area::All, "new size, colour or swap chain"),
            _ => {
                let most = |r: &[[u32; 4]]| r.iter().map(|r| area(*r)).sum::<u32>() as f32 > MOST * (size.0 * size.1) as f32;
                // A scroll moves most of the list: asked first where a sample of it says so, since
                // finding what changed in such a list hashes all of it only to say most did.
                let first = if scroll && moved_mostly(&self.last, instances) { scrolled(&self.last, instances, size) } else { None };
                let mut now = match first {
                    Some((shift, rects)) => Area::Scroll(shift, rects),
                    None => changed(&self.last, instances, size, &mut self.scratch),
                };
                // What would draw whole may be a scroll: then most of it is copied, not shaded. So may
                // what redraws a good part of the window: a rail scrolled sideways moves one viewport's
                // instances, each of which differs, and their bounds are the rail. The scroll is taken
                // where it leaves less to redraw.
                let sum = |r: &[[u32; 4]]| r.iter().map(|r| area(*r)).sum::<u32>();
                let much = match &now { Area::Rects(r) => Some(sum(r)).filter(|a| *a > size.0 * size.1 / 16), Area::Scroll(..) => None, _ => Some(u32::MAX) };
                // A scroll moves many instances. A fade or a hover changes a few, or changes many where
                // they lie (their colours), however much of the window they cover, and isn't asked:
                // looking for a shift that isn't there walks both lists for nothing.
                let many = !self.scratch.differ_all
                    || (self.scratch.differ.len() >= 8 && self.scratch.differ.iter().take(16).any(|&i| self.last[i as usize].rect != instances[i as usize].rect));
                // A frame that would draw whole looks as far as a scroll does; one that redraws part
                // looks near, where a rail's cards are, since most such frames are no scroll at all.
                let reach = if much == Some(u32::MAX) || much.is_some_and(|a| a as f32 > MOST * (size.0 * size.1) as f32) { 256 } else { 24 };
                if scroll && many && let Some(redrawn) = much && !moved_mostly(&self.last, instances)
                    && let Some((shift, rects)) = scrolled_within(&self.last, instances, size, reach) && sum(&rects) < redrawn {
                    now = Area::Scroll(shift, rects);
                }
                self.seen = match &now { Area::Rects(r) | Area::Scroll(_, r) => r.iter().copied().reduce(join), Area::All | Area::Many => None };
                match now {
                    Area::Rects(r) | Area::Scroll(_, r) if most(&r) => (Area::All, "most of it changed"),
                    Area::All => (Area::All, "reordered"),
                    Area::Many => (Area::All, "many changed"),
                    other => (other, ""),
                }
            }
        };
        // Compared in place, the instances that differ are known: only they are copied, where
        // copying the list is a hundred kilobytes a frame, most frames to say the same again.
        if std::mem::take(&mut self.scratch.differ_all) && self.last.len() == instances.len() {
            for &i in &self.scratch.differ { self.last[i as usize] = instances[i as usize]; }
        } else {
            self.last.clear();
            self.last.extend_from_slice(instances);
        }
        self.shape = Some((size, clear, swap));
        match now {
            Area::All | Area::Many => Plan::All(why),
            // Nothing changed: one pixel keeps the present, which DWM composes again as is.
            Area::Rects(r) if r.is_empty() => Plan::Part { draw: vec![[0, 0, 1, 1]], present: vec![[0, 0, 1, 1]], shift: None, same: true },
            Area::Rects(r) => Plan::Part {
                draw: r.iter().map(|r| [r[0], r[1], r[2] - r[0], r[3] - r[1]]).collect(),
                present: r.iter().map(|r| r.map(|v| v as i32)).collect(),
                shift: None,
                same: false,
            },
            // The copy changes pixels all over the viewport: the whole target is presented.
            Area::Scroll(shift, r) => Plan::Part {
                draw: r.iter().map(|r| [r[0], r[1], r[2] - r[0], r[3] - r[1]]).collect(),
                present: vec![[0, 0, size.0 as i32, size.1 as i32]],
                shift: Some(shift),
                same: false,
            },
        }
    }

    /// A new swap chain: the next frame draws whole.
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    pub(crate) fn reset(&mut self) { self.shape = None; }

    /// The last frame drew into swap chain `swap` at this size and clear colour: a frame may draw
    /// and present only part of it. False after `reset` or a configure (a new swap chain), so its
    /// first frame, loop-only or not, goes through `plan` and draws whole.
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    pub(crate) fn current(&self, size: (u32, u32), clear: [f32; 4], swap: u64) -> bool { self.shape == Some((size, clear, swap)) }

    /// A loop-only frame changed only `patched` (ascending) and drew them where they were and
    /// are (`Renderer::arm_loop`): the next frame compares with them.
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    pub(crate) fn patched(&mut self, instances: &[Instance], patched: &[u32]) {
        if self.last.len() != instances.len() { self.shape = None; return; }
        for &i in patched { self.last[i as usize] = instances[i as usize]; }
    }
}
/// `RUNE_WHOLE_FRAMES=1`: every frame draws whole, to check partial frames against.
pub(crate) fn whole_frames() -> bool {
    static WHOLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WHOLE.get_or_init(|| std::env::var_os("RUNE_WHOLE_FRAMES").is_some_and(|v| v == "1"))
}

/// `RUNE_TARGET_KEEPS=1`: a headless run's target keeps its pixels between frames, and says so
/// (`Renderer::arm_kept_target`), as a host whose buffers come back with a known frame in them
/// would. Off, it draws as a window with a swap chain does, through the kept frame and a copy out.
pub(crate) fn target_keeps() -> bool {
    static KEEPS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KEEPS.get_or_init(|| std::env::var_os("RUNE_TARGET_KEEPS").is_some_and(|v| v == "1"))
}

/// `RUNE_TARGET_AGE=<n>`: a headless run draws into `n` targets in turn, as a window draws into
/// its swap chain's buffers, and tells the renderer each one holds the frame of `n` frames ago
/// (`Renderer::arm_aged_target`), as a host that knows its buffers' ages would. Unset: one target,
/// treated as holding nothing known, which is what wgpu says of a drawable.
#[cfg_attr(not(feature = "tooling"), allow(dead_code))]
pub(crate) fn target_age() -> u32 {
    static AGE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *AGE.get_or_init(|| rune_core::env::var("RUNE_TARGET_AGE").and_then(|v| v.parse().ok()).filter(|n| (1..=5).contains(n)).unwrap_or(0))
}

type Counts = HashMap<u64, (i32, usize), BuildHasherDefault<Hashed>>;

/// The map's keys are hashes already, mixed once more: FNV's multiply carries only upward, so its
/// low bits, which pick the bucket, come from the words' low bits alone, and most instances share
/// those (whole-pixel rects and clips). Unmixed, a scrolled list's thousands of instances fell in
/// a few long probe chains.
#[derive(Default)]
struct Hashed(u64);

impl Hasher for Hashed {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, _: &[u8]) { unreachable!("keys are u64") }
    fn write_u64(&mut self, v: u64) {
        // MurmurHash3's 64-bit finaliser.
        let mut h = v ^ (v >> 33);
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        self.0 = h ^ (h >> 33);
    }
}

/// Whether two instances are the same bytes: their double words compared inline, without a
/// branch. A `memcmp` call for each of a frame's instances cost as much as the rest of its plan.
#[inline(always)]
fn same(a: &Instance, b: &Instance) -> bool {
    let (a, b): ([u64; INSTANCE_WORDS / 2], [u64; INSTANCE_WORDS / 2]) = (bytemuck::cast(*a), bytemuck::cast(*b));
    a.iter().zip(&b).fold(0, |d, (x, y)| d | (x ^ y)) == 0
}

/// An instance's 16 double words, xor-multiplied in four lanes that the CPU runs side by side
/// (FNV's one chain of 32 multiplies cost a frame as much again), then folded. Each step is a
/// bijection of its lane, so instances that differ in one word never hash alike.
fn hash(i: &Instance) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let words: [u64; INSTANCE_WORDS / 2] = bytemuck::cast(*i);
    let mut lanes = [0xcbf2_9ce4_8422_2325u64; 4];
    for (n, w) in words.iter().enumerate() {
        let lane = &mut lanes[n % 4];
        *lane = (*lane ^ w).wrapping_mul(K);
    }
    lanes[0] ^ lanes[1].rotate_left(16) ^ lanes[2].rotate_left(32) ^ lanes[3].rotate_left(48)
}

/// What differs between `old` and `new`.
/// What `changed` keeps between frames so it allocates nothing.
#[derive(Default)]
pub(super) struct Scratch {
    old: Vec<u64>,
    new: Vec<u64>,
    counts: Counts,
    each: Vec<[u32; 4]>,
    /// Where the lists last compared in place differ, and whether that is every place they do
    /// (`changed` found them all): `plan` then brings its copy of the list up to date from these.
    differ: Vec<u32>,
    differ_all: bool,
}

fn changed(old: &[Instance], new: &[Instance], size: (u32, u32), s: &mut Scratch) -> Area {
    let start = old.iter().zip(new).take_while(|(a, b)| same(a, b)).count();
    let end = old[start..].iter().rev().zip(new[start..].iter().rev()).take_while(|(a, b)| same(a, b)).count();
    let (old, new) = (&old[start..old.len() - end], &new[start..new.len() - end]);
    // The same length: what differs in place is the change. A pixel whose instances, or their
    // order, changed lies in the bounds of one that differs there (old or new), so redrawing
    // those is exact, and it hashes nothing. A scroll, which moves thousands, draws whole.
    if old.len() == new.len() {
        // Counted first: a scroll gives up after `MANY` without bounding any of them.
        let mut differ = 0;
        for (a, b) in old.iter().zip(new) {
            if !same(a, b) {
                differ += 1;
                if differ > MANY { return Area::Many; }
            }
        }
        s.each.clear();
        s.differ.clear();
        for (i, (a, b)) in old.iter().zip(new).enumerate().filter(|(_, (a, b))| !same(a, b)) {
            s.each.extend(bounds(a, size));
            s.each.extend(bounds(b, size));
            s.differ.push((start + i) as u32);
        }
        s.differ_all = true;
        return gathered(&mut s.each);
    }
    // A scroll that mounted or let go of rows changes the length too: if most of a sample of
    // the new instances aren't anywhere near their place in the old list, it moved everything,
    // and the frame draws whole without hashing the lists. Inserted text keeps its neighbours
    // within the length's change of where they were.
    if new.len() > 2 * MANY {
        let reach = old.len().abs_diff(new.len()) + 2;
        let samples = 32;
        let away = (0..samples).map(|n| n * new.len() / samples).filter(|&at| {
            let lo = at.saturating_sub(reach);
            let hi = (at + reach + 1).min(old.len());
            !old[lo.min(hi)..hi].iter().any(|o| same(o, &new[at]))
        }).count();
        if away * 2 > samples { return Area::Many; }
    }
    // Per distinct instance: how many more the old list has than the new, and one of them.
    s.counts.clear();
    s.old.clear();
    s.old.extend(old.iter().map(hash));
    s.new.clear();
    s.new.extend(new.iter().map(hash));
    for (i, h) in s.old.iter().enumerate() { s.counts.entry(*h).or_insert((0, i)).0 += 1; }
    let mut added = 0;
    for (i, h) in s.new.iter().enumerate() {
        let count = &mut s.counts.entry(*h).or_insert((0, old.len() + i)).0;
        *count -= 1;
        if *count < 0 { added += 1; }
        if added > MANY { return Area::Many; }
    }
    // The instances both lists have, in order: drawn in another order, anything may have changed.
    let counts = &s.counts;
    let mut j = 0;
    for &h in &s.old {
        if counts[&h].0 != 0 { continue; }
        if j < s.new.len() && s.new[j] == h { j += 1; continue; }
        while j < s.new.len() && counts[&s.new[j]].0 != 0 { j += 1; }
        if j >= s.new.len() || s.new[j] != h { return Area::All; }
        j += 1;
    }
    if s.new[j..].iter().any(|h| counts[h].0 == 0) { return Area::All; }
    // In reading order, so neighbours gather before rectangles run out.
    s.each.clear();
    s.each.extend(counts.values().filter(|(n, _)| *n != 0)
        .filter_map(|(_, at)| bounds(if *at < old.len() { &old[*at] } else { &new[*at - old.len()] }, size)));
    // A scrolled list changes thousands of instances over most of the window: their union says
    // so (the plan draws whole above `MOST`) without gathering them into rectangles.
    if s.each.len() > 256 && let Some(all) = s.each.iter().copied().reduce(join) && area(all) as f32 > MOST * (size.0 * size.1) as f32 {
        return Area::Rects(vec![all]);
    }
    gathered(&mut s.each)
}

/// `each`, in reading order so neighbours gather before rectangles run out, as at most `RECTS`
/// rectangles.
fn gathered(each: &mut [[u32; 4]]) -> Area {
    each.sort_unstable_by_key(|r| (r[1], r[0]));
    let mut rects = Vec::new();
    for r in each.iter().copied() { add(&mut rects, r); }
    Area::Rects(rects)
}

/// `each` as at most `SCROLL_RECTS` rectangles, for a scroll: what redraws there is a strip at
/// one edge and whatever changed elsewhere, often far apart. Past the limit the two that grow
/// least as one are joined (not the newcomer into its nearest, which can take a strip at the
/// bottom and a label at the top into one rectangle the height of the window).
fn gathered_apart(each: &mut [[u32; 4]]) -> Vec<[u32; 4]> {
    each.sort_unstable_by_key(|r| (r[1], r[0]));
    let mut rects: Vec<[u32; 4]> = Vec::new();
    let absorb = |rects: &mut Vec<[u32; 4]>, mut r: [u32; 4]| {
        while let Some(i) = rects.iter().position(|q| near(*q, r)) { r = join(rects.swap_remove(i), r); }
        rects.push(r);
    };
    for r in each.iter().copied() {
        absorb(&mut rects, r);
        while rects.len() > SCROLL_RECTS {
            let mut best = (u64::MAX, 0, 1);
            for i in 0..rects.len() {
                for j in i + 1..rects.len() {
                    let grow = u64::from(area(join(rects[i], rects[j]))).saturating_sub(u64::from(area(rects[i])) + u64::from(area(rects[j])));
                    if grow < best.0 { best = (grow, i, j); }
                }
            }
            let b = rects.swap_remove(best.2);
            let a = rects.swap_remove(best.1);
            absorb(&mut rects, join(a, b));
        }
    }
    rects
}
/// Adds `r` to `rects`: into one it touches, or as its own until there are `RECTS`, then into
/// the one it grows least.
fn add(rects: &mut Vec<[u32; 4]>, r: [u32; 4]) {
    if let Some(i) = rects.iter().position(|q| near(*q, r)) {
        let merged = join(rects.swap_remove(i), r);
        return add(rects, merged);
    }
    if rects.len() < RECTS { rects.push(r); return; }
    let grow = |q: [u32; 4]| area(join(q, r)) - area(q);
    let i = (0..rects.len()).min_by_key(|&i| grow(rects[i])).unwrap_or(0);
    let merged = join(rects.swap_remove(i), r);
    add(rects, merged);
}

/// Whether two rectangles overlap or lie within 16 pixels of each other.
fn near(a: [u32; 4], b: [u32; 4]) -> bool {
    a[0] <= b[2] + 16 && b[0] <= a[2] + 16 && a[1] <= b[3] + 16 && b[1] <= a[3] + 16
}

fn join(a: [u32; 4], b: [u32; 4]) -> [u32; 4] { [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])] }

/// `i` shades the pixels of its rect inside its clip and no others: a plain rect (no border,
/// softened edge or rounded corner) on whole pixels, inside a clip on whole pixels that fades
/// nothing. The shader's coverage and clip ramps are exactly 0 at every pixel centre outside.
fn crisp(i: &Instance) -> bool {
    let [x, y, w, h] = i.rect;
    i.kind() == Some(crate::render::Kind::Rect) && i.params[0] == 0.0 && i.params[2] <= 0.5 && i.radii == [0.0; 4] && i.fade == [0; 4]
        && [x, y, x + w, y + h].iter().chain(&i.clip).all(|v| v.fract() == 0.0)
}

/// The pixels an instance can touch: its quad as the shader grows it, inside its clip.
fn bounds(i: &Instance, size: (u32, u32)) -> Option<[u32; 4]> {
    if crisp(i) {
        let ([x, y, w, h], c) = (i.rect, i.clip);
        let (x0, y0) = (x.max(c[0]).max(0.0), y.max(c[1]).max(0.0));
        let (x1, y1) = ((x + w).min(c[2]).min(size.0 as f32), (y + h).min(c[3]).min(size.1 as f32));
        return (x1 > x0 && y1 > y0).then_some([x0 as u32, y0 as u32, x1 as u32, y1 as u32]);
    }
    let (x0, y0, x1, y1) = i.damage_quad_bounds()?;
    if x1 <= x0 || y1 <= y0 { return None; }
    let c = i.clip;
    let (x0, y0) = ((x0.max(c[0]) - 1.0).floor().max(0.0), (y0.max(c[1]) - 1.0).floor().max(0.0));
    let (x1, y1) = ((x1.min(c[2]) + 1.0).ceil().min(size.0 as f32), (y1.min(c[3]) + 1.0).ceil().min(size.1 as f32));
    (x1 > x0 && y1 > y0).then(|| [x0 as u32, y0 as u32, x1 as u32, y1 as u32])
}

fn area(r: [u32; 4]) -> u32 { (r[2] - r[0]) * (r[3] - r[1]) }

/// Most of a sample of the new list isn't where it was in the old one, nor the same from the
/// end: what a scroll does to a list, and little else.
fn moved_mostly(old: &[Instance], new: &[Instance]) -> bool {
    const SAMPLES: usize = 16;
    if new.len() < 2 * SAMPLES || old.len() < 2 * SAMPLES { return false; }
    let differs = (0..SAMPLES).filter(|k| {
        let at = (2 * k + 1) * new.len() / (2 * SAMPLES);
        let from_end = new.len() - at;
        !old.get(at).is_some_and(|o| same(o, &new[at])) && !old.len().checked_sub(from_end).is_some_and(|o| same(&old[o], &new[at]))
    }).count();
    differs * 4 > SAMPLES * 3
}

/// A frame that moved a viewport's content by whole pixels, as a scroll does: the shift that
/// copies it within the kept frame, and what still redraws (`[x0, y0, x1, y1]` each). The new
/// list is walked in order beside the old, each instance paired with the old one it is, moved; a
/// copied pixel is exactly what the frame would shade there when every instance under it moved
/// by the shift (`moved`) or lies under all of them and shades it uniformly (`uniform_below`).
/// The rest redraws: what came into the viewport, what changed, and what moved otherwise.
fn scrolled(old: &[Instance], new: &[Instance], size: (u32, u32)) -> Option<(Shift, Vec<[u32; 4]>)> {
    scrolled_within(old, new, size, 256)
}

/// `scrolled`, looking no further than `reach` instances from a probe's place for the old one it
/// was: far for a frame that is a scroll or draws whole otherwise, near for one that only might be.
fn scrolled_within(old: &[Instance], new: &[Instance], size: (u32, u32), reach: usize) -> Option<(Shift, Vec<[u32; 4]>)> {
    let start = old.iter().zip(new).take_while(|(a, b)| same(a, b)).count();
    let end = old[start..].iter().rev().zip(new[start..].iter().rev()).take_while(|(a, b)| same(a, b)).count();
    let (om, nm) = (&old[start..old.len() - end], &new[start..new.len() - end]);
    // The shift, voted by new instances spread through the middle, each with every move that
    // takes an old one near its place to it: a glyph or a card like it elsewhere offers a wrong
    // one, which few agree on. (The first instances are often a row the viewport's edge cuts,
    // whose clip changed as it moved.) Then how far ahead the old middle runs there.
    let probes: Vec<usize> = (1..8).map(|k| k * nm.len() / 8).filter(|&j| j < nm.len()).collect();
    // Near its place first (a scroll alone leaves the list as long as it was), then further (rows
    // came into the list or left it): a wide look at every probe costs a frame thousands of
    // comparisons.
    // Every shift most probes agree on, the best supported first, with how far ahead the old middle
    // runs there. A card or a glyph like it elsewhere can win the near look when rows came into the
    // list or left it (what was near each probe is then another row's), so more than one is tried.
    let voted = |reach: usize| {
        let window = |j: usize| j.saturating_sub(reach)..(j + reach + 1).min(om.len());
        let mut votes: Vec<([f32; 2], usize)> = Vec::new();
        for &j in &probes {
            let mut offers: Vec<[f32; 2]> = Vec::new();
            for i in window(j) {
                if let Some(d) = moved(&om[i], &nm[j]) && !offers.contains(&d) { offers.push(d); }
            }
            for d in offers {
                match votes.iter_mut().find(|(v, _)| *v == d) { Some(e) => e.1 += 1, None => votes.push((d, 1)) }
            }
        }
        votes.retain(|(_, support)| support * 2 > probes.len());
        votes.sort_by_key(|(_, support)| std::cmp::Reverse(*support));
        votes.truncate(2);
        votes.into_iter().filter_map(|(d, _)| {
            let ahead = probes.iter().find_map(|&j| {
                window(j).filter(|&i| moved(&om[i], &nm[j]) == Some(d)).min_by_key(|&i| i.abs_diff(j)).map(|i| i as isize - j as isize)
            })?;
            Some((d, ahead))
        }).collect::<Vec<_>>()
    };
    // The first shift that leaves under half the window to redraw, or of those tried the one that
    // leaves least: a wrong one copies pixels that are right only where its few pairs lie, and
    // redraws nearly all the rest, where a fast scroll's own strip is seldom half the window. Each
    // further try is a walk of both lists, so a frame that scrolled as it seems pays for one.
    let little = (size.0 * size.1) / 2;
    let mut best: Option<(u32, (Shift, Vec<[u32; 4]>))> = None;
    let mut tried: Vec<[f32; 2]> = Vec::new();
    // Rows that came into the list or left it put what a probe was beside far from it, where the
    // near look finds only look-alikes: then the far look goes first, and its shift is tried
    // before theirs.
    let looks = if old.len().abs_diff(new.len()) > 24 { [256, 24] } else { [24, 256] };
    for reach in looks.into_iter().filter(|r| *r <= reach) {
        for (d, ahead) in voted(reach) {
            if tried.contains(&d) { continue; }
            tried.push(d);
            let Some(plan) = scrolled_by(old, new, start, end, size, d, ahead) else { continue };
            let redrawn = plan.1.iter().map(|r| area(*r)).sum::<u32>();
            if redrawn <= little { return Some(plan); }
            if best.as_ref().is_none_or(|(least, _)| redrawn < *least) { best = Some((redrawn, plan)); }
        }
    }
    best.map(|(_, plan)| plan)
}

/// `scrolled` for one shift `d`, with the old middle running `ahead` of the new at the probes.
fn scrolled_by(old: &[Instance], new: &[Instance], start: usize, end: usize, size: (u32, u32), d: [f32; 2], mut ahead: isize) -> Option<(Shift, Vec<[u32; 4]>)> {
    if d[0].fract() != 0.0 || d[1].fract() != 0.0 || !on_grid(d[0]) || !on_grid(d[1]) { return None; }
    let (om, nm) = (&old[start..old.len() - end], &new[start..new.len() - end]);
    // Moved first: it tells most pairs apart by their size and kind alone.
    let fits = |o: &Instance, n: &Instance| moved(o, n) == Some(d) || bytemuck::bytes_of(o) == bytemuck::bytes_of(n);
    let mut paired = vec![false; om.len()];
    let (mut un_new, mut stayed, mut fixed, mut along) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut first_moved = None;
    // A row that came into the list or left it is hundreds of instances in one place: found by
    // looking further (`FAR`) for where the lists meet again, a few times a frame at most.
    const FAR: usize = 512;
    let (mut came, mut searches, mut lost) = (0usize, 8, false);
    // The last old instance paired: pairs keep the lists' order, so what is drawn over what is
    // the same before and after, and a pair is only ever looked for after it.
    let mut last: Option<usize> = None;
    for (j, n) in nm.iter().enumerate() {
        // One of a run that came in: the old one due waits for the new one after them.
        if came > 0 {
            came -= 1;
            ahead -= 1;
            un_new.push(j);
            continue;
        }
        let old_at = |k: isize| usize::try_from(j as isize + ahead + k).ok().filter(|&o| o < om.len() && last.is_none_or(|l| o > l));
        // Paired where it's due, or a few further on when old ones went.
        let mut found = (0..=8).find_map(|k| old_at(k).filter(|&o| fits(&om[o], n)).map(|o| (k, o)));
        // Or it came in (the old one due here pairs a later new one), or it changed.
        let mut after = if found.is_some() { None } else { old_at(0).and_then(|o| (1..=8).find(|&m| nm.get(j + m).is_some_and(|later| fits(&om[o], later)))) };
        // Once where the pairing stops, not for each instance of a run that changed.
        if found.is_none() && after.is_none() && searches > 0 && !std::mem::replace(&mut lost, true) {
            searches -= 1;
            found = (9..=FAR as isize).find_map(|k| old_at(k).filter(|&o| fits(&om[o], n)).map(|o| (k, o)));
            after = old_at(0).and_then(|o| (9..=FAR).find(|&m| nm.get(j + m).is_some_and(|later| fits(&om[o], later))));
            // The nearer meeting: the fewer instances given up as gone or new.
            if let (Some((k, _)), Some(m)) = (found, after) { if (k as usize) <= m { after = None; } else { found = None; } }
            // Or the old one is behind: `ahead` was read at a probe further on, past where rows came
            // into the list or left it, and before them the lists run as they did at their start.
            if found.is_none() && after.is_none() {
                found = (1..=FAR as isize).find_map(|k| old_at(-k).filter(|&o| fits(&om[o], n)).map(|o| (-k, o)));
            }
        }
        let Some((k, o)) = found else {
            if let Some(m) = after {
                ahead -= 1;
                came = m - 1;
            }
            un_new.push(j);
            continue;
        };
        lost = false;
        ahead += k;
        last = Some(o);
        paired[o] = true;
        if om[o].rect == n.rect { stayed.push(start + j); continue; }
        if !moves_exactly(&om[o], n) { paired[o] = false; un_new.push(j); continue; }
        first_moved.get_or_insert(start + j);
        if om[o].clip == n.clip { fixed.push((o, j)); } else { along.push((o, j)); }
    }
    let first_moved = first_moved?;
    // The viewport: the clip most moved instances keep, on whole pixels. Those keeping another
    // move in a viewport of their own, which the copy doesn't follow: they redraw.
    let mut clips: Vec<([f32; 4], usize)> = Vec::new();
    for &(_, j) in &fixed {
        match clips.iter_mut().find(|(c, _)| *c == nm[j].clip) { Some(e) => e.1 += 1, None => clips.push((nm[j].clip, 1)) }
    }
    // None keeps one (every row in a clip of its own): what their clips reach, before and after.
    let reach = || along.iter().flat_map(|&(o, j)| [om[o].clip, nm[j].clip])
        .reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]);
    let v = clips.iter().max_by_key(|(_, n)| *n).map(|(c, _)| *c).or_else(reach)?;
    for &(o, j) in &fixed {
        if nm[j].clip != v { paired[o] = false; un_new.push(j); }
    }
    let (w, h) = (size.0 as f32, size.1 as f32);
    let v = [v[0].max(0.0), v[1].max(0.0), v[2].min(w), v[3].min(h)];
    if v.iter().any(|c| c.fract() != 0.0) || v[2] <= v[0] || v[3] <= v[1] { return None; }
    let inside = |c: [f32; 4]| c[0] >= v[0] && c[1] >= v[1] && c[2] <= v[2] && c[3] <= v[3];
    let r = [v[0].max(v[0] + d[0]), v[1].max(v[1] + d[1]), v[2].min(v[2] + d[0]), v[3].min(v[3] + d[1])];
    if r[2] <= r[0] || r[3] <= r[1] { return None; }
    // A clip of its own (a row's, cut by the viewport's edge, or one that moved along): it must
    // lie in the viewport, or the instance draws outside it, and it must let through the same
    // copied pixels as it did where they came from. Its edges on whole pixels, a pixel is in or
    // out, so a pixel the copy fills holds exactly what the instance gave it before the move.
    let meet = |c: [f32; 4]| {
        let m = [c[0].max(r[0]), c[1].max(r[1]), c[2].min(r[2]), c[3].min(r[3])];
        (m[2] > m[0] && m[3] > m[1]).then_some(m)
    };
    for &(o, j) in &along {
        let (was, is) = (om[o].clip, nm[j].clip);
        let whole = was.map(f32::fract) == [0.0; 4] && is.map(f32::fract) == [0.0; 4];
        let same_copied = meet([was[0] + d[0], was[1] + d[1], was[2] + d[0], was[3] + d[1]]) == meet(is);
        if !whole || !same_copied || !inside(was) || !inside(is) { paired[o] = false; un_new.push(j); }
    }
    let (from, r) = (r.map(|c| c as u32), r.map(|c| c as u32));
    let from = [(from[0] as f32 - d[0]) as u32, (from[1] as f32 - d[1]) as u32];
    let shift = Shift { from: [from[0], from[1], r[2] - r[0], r[3] - r[1]], to: [r[0], r[1]] };
    // Where the copy puts the pixels of `b`: moved, inside what it fills.
    let moved_by = |b: [u32; 4]| {
        let x = |c: u32| (c as f32 + d[0]).clamp(r[0] as f32, r[2] as f32) as u32;
        let y = |c: u32| (c as f32 + d[1]).clamp(r[1] as f32, r[3] as f32) as u32;
        [x(b[0]), y(b[1]), x(b[2]), y(b[3])]
    };
    // What redraws: the viewport the copy doesn't fill, what's new, what's gone (where it was and
    // where its copied pixels land), and what stayed put over the copy (where it is, and where
    // the copy put its pixels) unless it shades all of the copy alike from under what moved.
    let vi = v.map(|c| c as u32);
    let mut each = vec![[vi[0], vi[1], vi[2], r[1]], [vi[0], r[3], vi[2], vi[3]], [vi[0], vi[1], r[0], vi[3]], [r[2], vi[1], vi[2], vi[3]]];
    each.extend(un_new.iter().filter_map(|&j| bounds(&nm[j], size)));
    for (o, _) in paired.iter().enumerate().filter(|(_, p)| !**p) {
        let Some(b) = bounds(&om[o], size) else { continue };
        // In the viewport, where it was is copied over or redrawn as a strip; outside, it redraws.
        if !(b[0] >= vi[0] && b[1] >= vi[1] && b[2] <= vi[2] && b[3] <= vi[3]) { each.push(b); }
        each.push(moved_by(b));
    }
    let copied = [r[0].min(from[0]), r[1].min(from[1]), r[2].max(from[0] + shift.from[2]), r[3].max(from[1] + shift.from[3])];
    let meets = |b: [u32; 4]| b[0] < copied[2] && copied[0] < b[2] && b[1] < copied[3] && copied[1] < b[3];
    for at in (0..start).chain(stayed).chain(new.len() - end..new.len()) {
        let Some(b) = bounds(&new[at], size) else { continue };
        if !meets(b) || (at < first_moved && uniform_below(&new[at], copied)) { continue; }
        each.push(b);
        each.push(moved_by(b));
    }
    each.retain(|b| b[2] > b[0] && b[3] > b[1]);
    let rects = gathered_apart(&mut each);
    Some((shift, rects))
}

/// `n` is `o` moved: every byte the same but the rect's place and the clip (which `scrolled`
/// checks once it knows what the copy fills). The move, or `None`. Compared where they lie: a
/// frame asks this of thousands of pairs.
fn moved(o: &Instance, n: &Instance) -> Option<[f32; 2]> {
    // Most pairs asked about differ in size or in what they are: told apart before the rest.
    if o.rect[2].to_bits() != n.rect[2].to_bits() || o.rect[3].to_bits() != n.rect[3].to_bits() || o.params[1].to_bits() != n.params[1].to_bits() { return None; }
    let (a, b) = (bytemuck::bytes_of(o), bytemuck::bytes_of(n));
    // Bytes: 0..8 the rect's place, 8..16 its size, 96..112 the clip; the rest must match.
    if a[8..96] != b[8..96] || a[112..] != b[112..] || a[..8] == b[..8] { return None; }
    Some([n.rect[0] - o.rect[0], n.rect[1] - o.rect[1]])
}

/// A moved instance whose pixels move with it exactly: its kind shades a pixel from where it lies
/// relative to the rect (a turned glyph samples the interpolated position, which moves with
/// rounding), it fades nothing, and its rect and clip lie on a quarter-pixel grid near the origin,
/// so the shader's sums, half-sizes and differences round nothing.
fn moves_exactly(o: &Instance, n: &Instance) -> bool {
    use crate::render::Kind;
    let kind = matches!(n.kind(), Some(Kind::Rect | Kind::Shadow | Kind::Inset | Kind::Gradient | Kind::Image | Kind::Dashed | Kind::Radial))
        || (n.kind() == Some(Kind::Glyph) && n.params[0] == 0.0);
    let grid = |v: [f32; 4]| v.map(on_grid) == [true; 4];
    let clip_ok = |c: [f32; 4]| c == crate::render::NO_CLIP || grid(c);
    kind && n.fade == [0; 4] && grid(o.rect) && grid(n.rect) && clip_ok(o.clip) && (o.clip == n.clip || clip_ok(n.clip))
}

/// On the quarter-pixel grid, under 2^19: sums of two such values and their halves are exact in f32.
fn on_grid(v: f32) -> bool { (v * 4.0).fract() == 0.0 && v.abs() < 524_288.0 }

/// `i`, drawn under everything that moved, shades every pixel of `region` (`[x0, y0, x1, y1]`)
/// alike: a plain rect on whole pixels with nothing but its fill, covering it, inside its clip.
/// The copy then keeps what it gives each pixel.
fn uniform_below(i: &Instance, region: [u32; 4]) -> bool {
    let [x, y, w, h] = i.rect;
    let c = i.clip;
    let [x0, y0, x1, y1] = region.map(|v| v as f32);
    i.kind() == Some(crate::render::Kind::Rect) && i.params[0] == 0.0 && i.params[2] <= 0.5 && i.radii == [0.0; 4] && i.fade == [0; 4]
        && [x, y, x + w, y + h].iter().all(|v| v.fract() == 0.0)
        && x <= x0 && y <= y0 && x + w >= x1 && y + h >= y1
        && c[0] <= x0 && c[1] <= y0 && c[2] >= x1 && c[3] >= y1
}

#[cfg(test)]
#[path = "damage_tests.rs"]
mod tests;
