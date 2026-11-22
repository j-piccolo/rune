//! A Wayland shell's surfaces: overlays placed with wlr-layer-shell as a bar, a panel or a
//! launcher is (any layer, anchors, margins, exclusive zone, keyboard), opened and closed while
//! the app runs; session-lock surfaces; the outputs they go on; and blur behind them.
//!
//! An overlay given a [`ShellSurface`] (`ConfiguredOverlay::shell`) is that surface on
//! Wayland instead of rune's own edge placement. [`open`] adds one to the running event loop
//! from the UI thread, and `OverlayHandle::close` takes it away again. [`watch_outputs`] hears
//! the outputs (by connector name) as they come and go, and [`SessionLock`] holds the session's
//! lock (`ext-session-lock-v1`) while lock surfaces ([`ShellSurface::Lock`]) show on it.
//!
//! Elsewhere than Wayland these do nothing: overlays keep rune's own placement, no outputs are
//! heard, and a lock is [`LockState::Unsupported`].

use crate::overlay::OverlayHandle;
use crate::ConfiguredOverlay;

/// Which layer-shell layer a surface sits in, bottom to top.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Layer {
    Background,
    Bottom,
    Top,
    #[default]
    Overlay,
}

/// When a layer surface takes the keyboard.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LayerKeyboard {
    /// Never.
    #[default]
    None,
    /// Whenever it shows, from everything else (a launcher).
    Exclusive,
    /// When the compositor gives it focus, as a click does.
    OnDemand,
}

/// A wlr-layer-shell surface's placement: where and how the compositor puts it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayerShell {
    pub layer: Layer,
    /// Top, right, bottom, left.
    pub anchors: [bool; 4],
    /// Top, right, bottom, left, in points.
    pub margin: [i32; 4],
    /// -1: placed over other surfaces' zones; 0: kept clear of them; `n`: reserves `n` points
    /// along the edge it is anchored to.
    pub exclusive_zone: i32,
    pub keyboard: LayerKeyboard,
    /// The output by its connector name (`DP-1`); `None` leaves the pick to the compositor.
    pub output: Option<String>,
}

/// What a Wayland overlay is.
#[derive(Clone, Debug, PartialEq)]
pub enum ShellSurface {
    /// A layer-shell surface placed as it says. The overlay's size is the size it asks for; 0
    /// on an axis it is anchored across stretches it.
    Layer(LayerShell),
    /// A session-lock surface on the named output, while a [`SessionLock`] holds the lock. The
    /// compositor sizes it to the output.
    Lock { output: String },
}

/// Blur the compositor draws behind a surface (`ext-background-effect-v1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlurRegion {
    /// `[x, y, width, height]` in surface points; `None` is the whole surface, as it is sized.
    pub rect: Option<[f32; 4]>,
    /// Rounds the region's corners, in points.
    pub radius: f32,
}

/// An output, as the compositor describes it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Output {
    /// Its connector name (`DP-1`), or empty when the compositor gives none.
    pub name: String,
    /// Its size in points (logical pixels).
    pub logical: (i32, i32),
    /// Where it sits in the layout, in points.
    pub position: (i32, i32),
    /// Panel pixels per point; fractional where the compositor scales so.
    pub scale: f64,
    /// Its mode's size in panel pixels.
    pub physical: (i32, i32),
}

/// Where a session lock stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockState {
    /// Asked for; the compositor hasn't said.
    Pending,
    /// The session is locked: every output shows a lock surface or nothing.
    Locked,
    /// The lock ended: refused before it held, or given up by the compositor after.
    Finished,
    /// There is no `ext-session-lock-v1` here.
    Unsupported,
}

/// Calls `f` on the UI thread with the outputs as soon as they are known, and again whenever
/// one comes, goes or changes. Call it on the UI thread, before or while the app runs.
pub fn watch_outputs(f: impl FnMut(&[Output]) + 'static) {
    #[cfg(all(unix, not(target_os = "macos"), not(target_os = "ios"), not(target_os = "android")))]
    crate::platform::wayland_shell::watch_outputs(Box::new(f));
    #[cfg(not(all(unix, not(target_os = "macos"), not(target_os = "ios"), not(target_os = "android"))))]
    drop(f);
}

/// The outputs as last heard; empty before the event loop runs or off Wayland.
pub fn outputs() -> Vec<Output> {
    #[cfg(all(unix, not(target_os = "macos"), not(target_os = "ios"), not(target_os = "android")))]
    return crate::platform::wayland_shell::outputs();
    #[cfg(not(all(unix, not(target_os = "macos"), not(target_os = "ios"), not(target_os = "android"))))]
    Vec::new()
}

/// The session's lock. Make it on the UI thread; it is taken on the event loop's next turn.
/// Dropping it does not unlock: only [`SessionLock::unlock`] does, so a locker that dies
/// leaves the session locked, as the protocol means it to.
#[derive(Clone)]
pub struct SessionLock {
    #[cfg(all(unix, not(target_os = "macos"), not(target_os = "ios"), not(target_os = "android")))]
    inner: std::rc::Rc<crate::platform::wayland_shell::Lock>,
}

impl SessionLock {
    /// Asks the compositor to lock the session; `on_change` hears each state the lock passes
    /// through, on the UI thread.
    pub fn lock(on_change: impl FnMut(LockState) + 'static) -> SessionLock {
        #[cfg(all(unix, not(target_os = "macos"), not(target_os = "ios"), not(target_os = "android")))]
        return SessionLock { inner: crate::platform::wayland_shell::lock(Box::new(on_change)) };
        #[cfg(not(all(unix, not(target_os = "macos"), not(target_os = "ios"), not(target_os = "android"))))]
        {
            let mut on_change = on_change;
            on_change(LockState::Unsupported);
            SessionLock {}
        }
    }

    pub fn state(&self) -> LockState {
        #[cfg(all(unix, not(target_os = "macos"), not(target_os = "ios"), not(target_os = "android")))]
        return self.inner.state();
        #[cfg(not(all(unix, not(target_os = "macos"), not(target_os = "ios"), not(target_os = "android"))))]
        LockState::Unsupported
    }

    /// Unlocks the session and ends the lock. The compositor has read the request before this
    /// returns, so the process may exit at once.
    pub fn unlock(&self) {
        #[cfg(all(unix, not(target_os = "macos"), not(target_os = "ios"), not(target_os = "android")))]
        self.inner.unlock();
    }
}

/// Adds an overlay to the running app from the UI thread; it joins the event loop on its next
/// turn, hidden until `show`. Its fonts are `ConfiguredOverlay::fonts`, else rune's own.
pub fn open<T: 'static>(configured: impl Into<ConfiguredOverlay<T>>, state: T) -> OverlayHandle<T> {
    crate::overlay::open_at_runtime(configured.into(), state)
}
