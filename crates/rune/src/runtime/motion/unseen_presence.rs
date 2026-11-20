//! Presence nobody would see. A removed node that drew nothing last frame has no exit to show: it
//! faded out already, sits under a parent that hid at once, or lies out of the window or its
//! clips with everything it holds (a filtered list's rows scrolled out of view). It goes at
//! once rather than fading where nothing shows; what mounts out of sight enters at once, and a
//! node out of sight before and after (or transparent, or under a clip with no area) lands its
//! layout and colours. Likewise a virtual row that
//! a change mounted with its entrance but that landed out of view lands its entrance at once,
//! and one nobody saw that a filter brings into view enters like the rows new to it.
use super::*;
use crate::el::Flow;

impl RuntimeCore {
    /// Per node of the last build: neither it nor anything under it drew. Empty before the first
    /// build and after a commit, until the next one draws.
    pub(in crate::runtime) fn drew_nothing(&self) -> Vec<bool> {
        if !self.tree.built || self.paint.seen().len() != self.tree.nodes.len() || self.paint.inh_scratch.len() != self.tree.nodes.len() {
            return Vec::new();
        }
        let mut seen = self.paint.seen().to_vec();
        mark_ancestors(&self.tree.nodes, &mut seen);
        seen.iter().zip(&self.paint.inh_scratch).map(|(&seen, h)| !seen || h.opacity < 1.0 / 512.0).collect()
    }

    /// How tall growth row `id` can be, in its own points, before its bottom passes the bottom
    /// of its clip (a scroll's viewport, the window), from where and how big it last drew
    /// (`El::fold_glide`); unbounded before it draws.
    pub(in crate::runtime) fn fold_limit(&self, id: NodeId) -> f32 {
        if !self.tree.built || self.paint.drawn.len() != self.tree.nodes.len() { return f32::INFINITY; }
        let Some(i) = self.tree.nodes.iter().position(|n| n.id == id) else { return f32::INFINITY };
        let Some(h) = self.motion.grows.get(&id).map(|g| g.h).filter(|&h| h > 0.5) else { return f32::INFINITY };
        let d = self.paint.drawn[i];
        let k = d[3] / h;
        if !(k > 0.01) { return f32::INFINITY; }
        let bottom = (self.screen_clip_of_parent(self.tree.nodes[i].parent)[3] / self.env.scale).min(self.viewport.1);
        h + (bottom - d[1] - d[3]) / k
    }

    /// Node `i` of the commit is out of sight before and after (`unseen_before_and_after`), so
    /// its entrance, colours and blur land. A matched element still flies to its match.
    pub(in crate::runtime) fn enters_unseen(&self, nodes: &[Node], unseen: &[bool], i: usize) -> bool {
        unseen.get(i).copied().unwrap_or(false) && nodes[i].matched.is_none()
    }

    /// Ends the exits that drew nothing last frame. A flight paints outside its clips, so it stays.
    pub(in crate::runtime) fn end_unseen_exits(&mut self) {
        if !self.tree.built || !self.scene.any_exiting() { return; }
        let mut hid: Option<Vec<bool>> = None;
        for i in 0..self.tree.nodes.len() {
            let id = self.tree.nodes[i].id;
            let faded = self.paint.inh_scratch.get(i).is_some_and(|h| h.opacity < 1.0 / 512.0);
            if (!faded && self.paint.was_seen(i, true)) || !self.is_exiting(id) || self.motion.flying.iter().any(|f| f.0 == id) {
                continue;
            }
            if hid.get_or_insert_with(|| self.drew_nothing()).get(i).copied().unwrap_or(false) {
                self.scene.unmount(id);
            }
        }
    }

    /// Row `i` of the commit's `nodes`, kept by a change that replaced what the reader saw
    /// ([`crate::el::virtual_list::VirtualList::replaced_view`]) but drawn nowhere before, enters
    /// like the rows new to the view instead of appearing whole.
    pub(in crate::runtime) fn reenter_unseen_row(&mut self, nodes: &[Node], i: usize, unseen: &HashSet<NodeId>) {
        if !unseen.contains(&nodes[i].id) { return; }
        let Some(p) = nodes[i].parent else { return };
        let Some(v) = self.commit.virtuals.get_mut(&nodes[p].path) else { return };
        let Some(key) = nodes[i].path.strip_prefix(nodes[p].path.as_str()).and_then(|k| k.strip_prefix('/')) else { return };
        if !v.replaced_view() { return; }
        v.entering.insert(key.to_owned());
        self.scene.reenter(nodes[i].id);
    }

    /// Virtual rows mounted with their entrance have just drawn once: the ones out of view land.
    /// Its own fade starts a row at no opacity, so only where it drew counts, strictly: a row
    /// touching its viewport's edge shows nothing. Runs after each draw, since a list filling its
    /// window commits again on the frames that follow.
    pub(in crate::runtime) fn land_unseen_entrances(&mut self) {
        if self.commit.virtuals.values().all(|v| v.entering.is_empty()) { return; }
        let s = self.env.scale;
        let (w, h) = self.viewport;
        let mut land = Vec::new();
        for (path, v) in &mut self.commit.virtuals {
            for key in v.entering.drain() {
                let Some(&id) = self.tree.path_ids.get(&format!("{path}/{key}")) else { continue };
                let Some(&i) = self.tree.node_index.get(&id) else { continue };
                let Some(&[x, y, rw, rh]) = self.paint.drawn.get(i) else { continue };
                let clip = screen_clip_of(self.tree.nodes[i].parent, &self.paint.inh_scratch, &self.paint.turn_scratch);
                let shows = (x + rw).min(clip[2] / s).min(w) > x.max(clip[0] / s).max(0.0)
                    && (y + rh).min(clip[3] / s).min(h) > y.max(clip[1] / s).max(0.0);
                if !shows { land.push(id); }
            }
        }
        for id in land { self.scene.enter_now(id); }
    }

    /// Per node of the new tree `nodes`: neither it nor anything under it drew in the last
    /// build (out of sight or transparent), and none of its swept path lands in sight in this
    /// layout (in the window, inside its clips, not transparent). Empty when nothing non-skipped
    /// qualifies. `drew_nothing` is [`Self::drew_nothing`] if the commit worked it out already,
    /// or empty. Only the nodes apply doesn't skip (`skip`) are worked out; the rest read false.
    pub(in crate::runtime) fn unseen_before_and_after(&self, nodes: &[Node], skip: &[memo_reuse::CleanApplySpan], drew_nothing: &[bool]) -> Vec<bool> {
        let worked;
        let drew_nothing = if drew_nothing.is_empty() { worked = self.drew_nothing(); worked.as_slice() } else { drew_nothing };
        // New nodes may mount out of sight however much drew.
        let mounts = nodes.len() > self.tree.nodes.len();
        if drew_nothing.is_empty() || (!mounts && drew_nothing.iter().all(|&nothing| !nothing)) {
            return Vec::new();
        }
        let any_active_unseen = self.tree.nodes.iter().zip(drew_nothing).any(|(old, &nothing)| {
            nothing && self.tree.node_index.get(&old.id).is_some_and(|&ni| !memo_reuse::is_in_clean_spans(skip, ni))
        });
        if !any_active_unseen && !mounts { return Vec::new(); }
        let unseen = self.unseen_outside(nodes, skip, drew_nothing).unwrap_or_else(|| self.unseen_outside(nodes, &[], drew_nothing).unwrap_or_default());
        // Debug builds work out every node too, and hold the nodes apply reads to it.
        if cfg!(debug_assertions) && !skip.is_empty() && let Some(all) = self.unseen_outside(nodes, &[], drew_nothing) {
            let differ = (0..nodes.len()).find(|&i| !memo_reuse::is_in_clean_spans(skip, i) && unseen[i] != all[i]);
            debug_assert!(differ.is_none(), "unseen outside the skipped spans differs from unseen worked out whole at {:?}", differ.map(|i| &nodes[i].path));
        }
        unseen
    }

    /// [`Self::unseen_before_and_after`] for the nodes outside the spans apply skips, and the
    /// ancestors they need from inside one: a span may hold a list's node alone, or run it
    /// together with rows that stayed, while a row that mounted under it is outside. What a
    /// span holds can only show a node above it, though: None when a node apply reads shows
    /// nothing itself but holds a skipped node.
    fn unseen_outside(&self, nodes: &[Node], skip: &[memo_reuse::CleanApplySpan], drew_nothing: &[bool]) -> Option<Vec<bool>> {
        let len = nodes.len();
        let window = [0.0, 0.0, self.viewport.0, self.viewport.1];
        // Per node: the swept clip and scroll-offset range its children are under, in points.
        let mut inner = vec![(window, [0.0f32; 2], [0.0f32; 2]); len];
        let mut after = vec![false; len];
        // Per node: the opacity it rests at, its ancestors' included. A live opacity or a loop
        // may show it whatever it rests at.
        let mut alpha = vec![1.0f32; len];
        let mut olds: Vec<Option<usize>> = vec![None; len];
        let mut done = vec![false; len];
        let mut skipped = vec![false; len];
        for s in skip { skipped[s.new_start..s.new_start + s.len].fill(true); }
        // A skipped node's place in the last build: its span holds the same nodes in order.
        let old_in_span = |k: usize| { let s = &skip[skip.partition_point(|s| s.new_start <= k) - 1]; s.old_start + (k - s.new_start) };
        // Each other new node's place in the last build, found walking both trees in step.
        let mut last = paint_tracks::Finder::default();
        let mut chain = Vec::new();
        let (mut i, mut cursor) = (0usize, 0usize);
        while i < len {
            while cursor < skip.len() && skip[cursor].new_start + skip[cursor].len <= i { cursor += 1; }
            if let Some(span) = skip.get(cursor).filter(|s| s.new_start == i) {
                last.advance(span.old_start + span.len);
                i += span.len;
                continue;
            }
            olds[i] = last.find(&self.tree.nodes, nodes[i].id);
            // Its ancestors not worked out yet are skipped ones: top down, before it.
            chain.clear();
            let mut p = nodes[i].parent;
            while let Some(j) = p.filter(|&j| j < i && !done[j]) {
                chain.push(j);
                p = nodes[j].parent;
            }
            for &j in chain.iter().rev() {
                olds[j] = Some(old_in_span(j));
                self.unseen_node(nodes, j, olds[j], window, &mut alpha, &mut after, &mut inner);
                done[j] = true;
            }
            self.unseen_node(nodes, i, olds[i], window, &mut alpha, &mut after, &mut inner);
            done[i] = true;
            i += 1;
        }
        mark_ancestors(nodes, &mut after);
        if !skip.is_empty() {
            // Every ancestor of a skipped node, found from each span's nodes whose parent lies
            // before the span.
            let mut holds = vec![false; len];
            for s in skip {
                for k in s.new_start..s.new_start + s.len {
                    let mut p = nodes[k].parent.filter(|&p| p < s.new_start);
                    while let Some(j) = p {
                        if holds[j] { break; }
                        holds[j] = true;
                        p = nodes[j].parent;
                    }
                }
            }
            if (0..len).any(|j| holds[j] && !skipped[j] && !after[j]) { return None; }
        }
        // A node new out of sight (a ticking digit in a row scrolled out of view, a row mounting
        // past its list's viewport) enters where nothing shows, so it lands at once too.
        Some((0..len).map(|i| !skipped[i] && !after[i] && olds[i].is_none_or(|o| drew_nothing[o])).collect())
    }

    /// Node `i`'s part of [`Self::unseen_outside`], its parent's worked out already: the opacity
    /// it rests at, whether its swept path lands in sight, and the clip and scroll-offset range
    /// its children are under. `old` is its place in the last build.
    #[allow(clippy::too_many_arguments)]
    fn unseen_node(&self, nodes: &[Node], i: usize, old: Option<usize>, window: [f32; 4], alpha: &mut [f32], after: &mut [bool], inner: &mut [([f32; 4], [f32; 2], [f32; 2])]) {
        let n = &nodes[i];
        // A hover group's hidden actions (`El::group_opacity`) rest at their rest level.
        let own = if n.looping.is_some() || n.live.iter().any(|b| b.property == crate::live::Property::Opacity) { 1.0 } else { n.vis.opacity * n.vis.group_opacity.map_or(1.0, |g| g[0].max(0.0)) };
        alpha[i] = own * n.parent.filter(|&p| p < i).map_or(1.0, |p| alpha[p]);
        let (clip, shift_min, shift_max) = match n.parent {
            Some(p) if n.above.is_none() && p < i => inner[p],
            Some(_) if n.above.is_none() => { after[i] = true; (window, [0.0; 2], [0.0; 2]) }
            _ => (window, [0.0; 2], [0.0; 2]),
        };
        // A stuck element draws where it's held, not where scrolling puts it: its shift last
        // frame brings it, and everything it holds, back into its viewport (a header's
        // ticking number mounts in sight there).
        let (shift_min, shift_max) = match old.filter(|_| self.tree.has_sticky).and_then(|o| self.scroll.stuck.get(o).filter(|s| **s != 0.0).map(|&s| (o, s))) {
            Some((o, s)) => {
                let axis = usize::from(!matches!(self.sticky_of(o), Some((crate::el::StickyEdge::Left, ..))));
                let (mut lo, mut hi) = (shift_min, shift_max);
                lo[axis] -= s;
                hi[axis] -= s;
                (lo, hi)
            }
            None => (shift_min, shift_max),
        };
        let t = n.target;
        let r = [t[0] - shift_max[0], t[1] - shift_max[1], t[2] - shift_min[0], t[3] - shift_min[1]];
        let drew = old.filter(|&o| { let t = self.tree.nodes[o].target; t[2] > t[0] || t[3] > t[1] });
        let swept = match drew.map(|o| self.drawn_edges(o)) {
            Some(b) => [b[0].min(r[0]), b[1].min(r[1]), b[2].max(r[2]), b[3].max(r[3])],
            None => r,
        };
        // Under a clip with no area nothing draws, shadows included.
        // Room for its shadows, blur and a focus ring, which draw past its rect.
        let margin = n.vis.shadows.iter().filter(|s| !s.inset).map(|s| s.x.abs() + s.y.abs() + s.spread.max(0.0) + s.blur * 1.5).fold(0.0, f32::max) + n.vis.blur * 3.0
            + if n.focus.enabled { 4.0 } else { 0.0 };
        // A rect that only touches the clip's edge draws nothing inside it.
        after[i] |= alpha[i] >= 1.0 / 512.0 && clip[2] > clip[0] && clip[3] > clip[1]
            && swept[2] > clip[0] - margin && swept[0] < clip[2] + margin
            && swept[3] > clip[1] - margin && swept[1] < clip[3] + margin;
        let clip = if n.vis.clip.is_some() || n.scroll.is_some() {
            intersect(clip, swept)
        } else { clip };
        let (shift_min, shift_max) = match (self.scroll.scrolls.get(&n.id), n.scroll) {
            (Some(st), Some(sp)) => {
                let children = || (i + 1..nodes.len()).take_while(|&j| nodes[j].parent >= Some(i)).filter(|&j| nodes[j].parent == Some(i));
                let max_x = match (n.scroll_axes.x(), st.x.flow) {
                    (false, _) => 0.0,
                    (true, Flow::Ltr) => children().fold(0.0_f32, |m, c| m.max(nodes[c].target[2] - n.target[2])),
                    (true, Flow::Rtl) => children().fold(0.0_f32, |m, c| m.max(n.target[0] - nodes[c].target[0])),
                };
                let max_y = if n.scroll_axes.y() { children().fold(0.0_f32, |m, c| m.max(nodes[c].target[3] - n.target[3])) } else { 0.0 };
                let ox0 = st.x.shift() as f32;
                let ox1 = match st.x.flow { Flow::Ltr => ox0.clamp(0.0, max_x), Flow::Rtl => ox0.clamp(-max_x, 0.0) };
                let oy0 = st.offset as f32 + st.slack.height();
                let oy1 = if st.pinned() && sp.stick_bottom { max_y + st.slack.height() } else { oy0.clamp(0.0, max_y + st.slack.height()) };
                (
                    [shift_min[0] + ox0.min(ox1), shift_min[1] + oy0.min(oy1)],
                    [shift_max[0] + ox0.max(ox1), shift_max[1] + oy0.max(oy1)],
                )
            }
            _ => (shift_min, shift_max),
        };
        inner[i] = (clip, shift_min, shift_max);
    }
}
