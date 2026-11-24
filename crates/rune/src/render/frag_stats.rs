//! Frame-profile builds: the GPU's own count of the fragment shader invocations a frame's passes ran
//! (`wgpu::Features::PIPELINE_STATISTICS_QUERY`), for `tooling::FrameLog`. A frame judge counts the
//! fragments it expects from the draw list; this is what the GPU says it shaded, so the two can be
//! checked against each other. Each pass that draws instances (the frame, its re-opened halves
//! around a backdrop, isolated groups) runs one query; the log sums them. Other builds compile the
//! calls to nothing.

#[cfg(feature = "frame-profile")]
mod on {
    use std::cell::RefCell;

    thread_local! {
        /// The query set a frame is counted into, its next free index, and how many it holds.
        static ACTIVE: RefCell<Option<(wgpu::QuerySet, u32, u32)>> = const { RefCell::new(None) };
    }

    /// Counts the passes recorded on this thread into `queries` until [`finish`].
    pub(crate) fn start(queries: wgpu::QuerySet, count: u32) {
        ACTIVE.with_borrow_mut(|a| *a = Some((queries, 0, count)));
    }

    /// Stops counting; the queries used, and whether the frame had more passes than the set holds.
    pub(crate) fn finish() -> (u32, bool) {
        ACTIVE.with_borrow_mut(|a| match a.take() {
            Some((_, next, count)) => (next.min(count), next > count),
            None => (0, false),
        })
    }

    /// Begins a query in `pass` when a frame is being counted; true when it did. End it with [`end`]
    /// before the pass ends.
    pub(crate) fn begin(pass: &mut wgpu::RenderPass<'_>) -> bool {
        ACTIVE.with_borrow_mut(|a| match a {
            Some((queries, next, count)) => {
                let at = *next;
                *next += 1;
                if at >= *count { return false; }
                pass.begin_pipeline_statistics_query(queries, at);
                true
            }
            None => false,
        })
    }

    pub(crate) fn end(pass: &mut wgpu::RenderPass<'_>, began: bool) {
        if began { pass.end_pipeline_statistics_query(); }
    }
}

#[cfg(feature = "frame-profile")]
pub(crate) use on::*;

#[cfg(not(feature = "frame-profile"))]
#[inline(always)]
pub(crate) fn begin(_: &mut wgpu::RenderPass<'_>) -> bool { false }

#[cfg(not(feature = "frame-profile"))]
#[inline(always)]
pub(crate) fn end(_: &mut wgpu::RenderPass<'_>, _: bool) {}

/// A pass a renderer owns, whose query ends as it does.
pub(crate) struct Counted<'p> {
    pass: wgpu::RenderPass<'p>,
    began: bool,
}

impl<'p> Counted<'p> {
    pub(crate) fn new(mut pass: wgpu::RenderPass<'p>) -> Self {
        let began = begin(&mut pass);
        Self { pass, began }
    }
}

impl Drop for Counted<'_> {
    fn drop(&mut self) { end(&mut self.pass, self.began); }
}

impl<'p> std::ops::Deref for Counted<'p> {
    type Target = wgpu::RenderPass<'p>;
    fn deref(&self) -> &Self::Target { &self.pass }
}

impl std::ops::DerefMut for Counted<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.pass }
}
