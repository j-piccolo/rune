//! The current nodes' scroll viewports, each with its direct children, found once for each
//! commit's nodes. Every frame steps every viewport (`step_scroll`) and paint asks each where
//! its content ends (`bottom_offset`): without the index, each of those walks the viewport's
//! whole subtree, hundreds of nodes under a page of rails, several times a frame.

use super::*;
use crate::runtime::key_index::NodesId;

/// Which nodes have `Node::scroll`, and the children `RuntimeCore::scroll_children` walks to
/// for each, before it leaves out the ones exiting or unmounted (which change between commits).
#[derive(Default)]
pub(in crate::runtime) struct Viewports {
    /// The nodes it describes. `None` once they change under the same commit (`forget`).
    built: Option<NodesId>,
    /// Each viewport's node index, in tree order.
    at: Vec<u32>,
    /// Viewport `at[k]`'s direct children are `kids[starts[k]..starts[k + 1]]`, in tree order.
    starts: Vec<u32>,
    kids: Vec<u32>,
    /// Build scratch: (viewport, child) pairs in tree order, and the viewports still open.
    pairs: Vec<(u32, u32)>,
    open: Vec<u32>,
}

impl Viewports {
    /// Indexes `nodes`, the runtime's current nodes as of `commit`, unless it already has.
    pub(in crate::runtime) fn prepare(&mut self, nodes: &[Node], commit: u64) {
        let id = NodesId::of(nodes, commit);
        if self.built == Some(id) { return; }
        self.build(nodes);
        self.built = Some(id);
    }

    /// The nodes changed under the same commit.
    pub(in crate::runtime) fn forget(&mut self) {
        self.built = None;
    }

    /// The index, when it describes `nodes` as of `commit`.
    pub(in crate::runtime) fn get(&self, nodes: &[Node], commit: u64) -> Option<&Self> {
        (self.built == Some(NodesId::of(nodes, commit))).then_some(self)
    }

    /// The viewports' node indexes, in tree order.
    pub(in crate::runtime) fn nodes(&self) -> &[u32] {
        &self.at
    }

    /// Viewport `i`'s direct children, `None` when node `i` has no `Node::scroll`.
    pub(in crate::runtime) fn kids(&self, i: usize) -> Option<&[u32]> {
        let k = self.at.binary_search(&u32::try_from(i).ok()?).ok()?;
        Some(&self.kids[self.starts[k] as usize..self.starts[k + 1] as usize])
    }

    /// One pass in tree order. A viewport's walk (`scroll_children`) runs until the first node
    /// whose parent comes before it, so the open viewports are a stack: a node closes every one
    /// after its parent, and is a child of the one left on top only if that is its parent.
    fn build(&mut self, nodes: &[Node]) {
        self.at.clear();
        self.pairs.clear();
        self.open.clear();
        for (j, n) in nodes.iter().enumerate() {
            while let Some(&k) = self.open.last() && n.parent < Some(self.at[k as usize] as usize) {
                self.open.pop();
            }
            if let Some(&k) = self.open.last() && n.parent == Some(self.at[k as usize] as usize) {
                self.pairs.push((k, j as u32));
            }
            if n.scroll.is_some() {
                self.open.push(self.at.len() as u32);
                self.at.push(j as u32);
            }
        }
        // By viewport, each one's children still in tree order.
        self.starts.clear();
        self.starts.resize(self.at.len() + 1, 0);
        for &(k, _) in &self.pairs { self.starts[k as usize + 1] += 1; }
        for k in 0..self.at.len() { self.starts[k + 1] += self.starts[k]; }
        self.kids.clear();
        self.kids.resize(self.pairs.len(), 0);
        self.open.clear();
        self.open.extend_from_slice(&self.starts[..self.at.len()]);
        for &(k, j) in &self.pairs {
            let next = &mut self.open[k as usize];
            self.kids[*next as usize] = j;
            *next += 1;
        }
    }
}

impl RuntimeCore {
    /// Indexes the current nodes' scroll viewports, unless they already are.
    #[inline]
    pub(in crate::runtime) fn prepare_viewports(&mut self) {
        let commit = self.commit.commit_gen();
        self.tree.viewports.prepare(&self.tree.nodes, commit);
    }

    /// The scroll viewport index, when it describes the current nodes.
    #[inline]
    pub(in crate::runtime) fn viewports(&self) -> Option<&Viewports> {
        self.tree.viewports.get(&self.tree.nodes, self.commit.commit_gen())
    }

    /// Scroll viewport `i`'s direct children from the index, before `scroll_children` leaves
    /// any out; `None` when the index is out of date or `i` isn't a viewport. Debug builds walk
    /// the subtree as well and check the two agree.
    #[inline]
    pub(in crate::runtime) fn indexed_scroll_children(&self, i: usize) -> Option<&[u32]> {
        let kids = self.viewports()?.kids(i)?;
        #[cfg(debug_assertions)]
        {
            let walked: Vec<u32> = (i + 1..self.tree.nodes.len())
                .take_while(|&j| self.tree.nodes[j].parent >= Some(i))
                .filter(|&j| self.tree.nodes[j].parent == Some(i))
                .map(|j| j as u32)
                .collect();
            debug_assert_eq!(kids, &walked[..], "scroll viewport {i}'s indexed children differ from its subtree's");
        }
        Some(kids)
    }
}
