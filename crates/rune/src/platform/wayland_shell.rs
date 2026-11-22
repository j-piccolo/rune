//! The Wayland side of [`crate::wayland`]: one connection on winit's display that every layer
//! and lock surface shares, the outputs as the compositor describes them, and the session lock.
//!
//! The windows host installs it when the loop starts (`install`) and turns it at the top and
//! end of every loop turn (`turn`): what the compositor sent is dispatched there, and output
//! and lock changes reach their callbacks on the UI thread, outside any window's handlers.
//! While it is connected, winit wakes on any read of the display (`window::wayland_wake`, as
//! for a live layer surface), so nothing polls. Nothing is connected until something asks: a
//! layer or lock surface, an output watcher or a lock.

use std::cell::{Cell, RefCell};
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use wayland_backend::client::Backend;
use wayland_client::globals::{GlobalListContents, registry_queue_init};
use wayland_client::protocol::wl_output::{self, WlOutput};
use wayland_client::protocol::wl_registry::{self, WlRegistry};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum};
use wayland_protocols::ext::session_lock::v1::client::ext_session_lock_manager_v1::ExtSessionLockManagerV1;
use wayland_protocols::ext::session_lock::v1::client::ext_session_lock_v1::{self, ExtSessionLockV1};
use wayland_protocols::xdg::xdg_output::zv1::client::zxdg_output_manager_v1::ZxdgOutputManagerV1;
use wayland_protocols::xdg::xdg_output::zv1::client::zxdg_output_v1::{self, ZxdgOutputV1};

use crate::runtime::Waker;
use crate::wayland::{LockState, Output};

/// What a lock's owner hears.
type OnLock = Box<dyn FnMut(LockState)>;
/// What an output watcher hears.
type OnOutputs = Box<dyn FnMut(&[Output])>;

thread_local! {
    static SHELL: RefCell<Shell> = RefCell::new(Shell::default());
}

#[derive(Default)]
struct Shell {
    /// winit's display and the loop's wake, once the loop runs on Wayland.
    display: Option<NonNull<std::ffi::c_void>>,
    wake: Option<Waker>,
    services: Option<Services>,
    /// Watchers, each with whether it has heard the outputs yet.
    watchers: Vec<(OnOutputs, bool)>,
    /// Locks asked for and not yet taken, and those held.
    locks: Vec<Rc<Lock>>,
}

/// The shared connection and what it binds.
struct Services {
    conn: Connection,
    queue: EventQueue<State>,
    state: State,
}

struct OutputEntry {
    global: u32,
    proxy: WlOutput,
    xdg: Option<ZxdgOutputV1>,
    /// What the compositor said before its last `done`, and after.
    pending: Output,
    shown: Output,
    integer_scale: i32,
    transform_swaps: bool,
    /// It has had a `done`: it is complete enough to report.
    complete: bool,
}

#[derive(Default)]
struct State {
    xdg: Option<ZxdgOutputManagerV1>,
    lock_manager: Option<ExtSessionLockManagerV1>,
    outputs: Vec<OutputEntry>,
    /// An output came, went or changed since the watchers last heard.
    outputs_changed: bool,
    /// Lock state changes by lock id, in order.
    lock_events: Vec<(u64, LockState)>,
}

/// One session lock, from `crate::wayland::SessionLock`.
pub(crate) struct Lock {
    id: u64,
    state: Cell<LockState>,
    on_change: RefCell<OnLock>,
    object: RefCell<Option<ExtSessionLockV1>>,
}

impl Lock {
    pub(crate) fn state(&self) -> LockState {
        self.state.get()
    }

    fn set(&self, state: LockState) {
        if self.state.replace(state) == state {
            return;
        }
        // The callback may ask things of the shell: it runs outside its borrow.
        let mut on_change = self.on_change.borrow_mut();
        (*on_change)(state);
    }

    pub(crate) fn unlock(&self) {
        let Some(object) = self.object.borrow_mut().take() else { return };
        object.unlock_and_destroy();
        // Read by the compositor before this returns, not only sent: a client that hangs up loses
        // what the compositor has yet to read (libwayland-server destroys it on the hang-up), and
        // a locker exiting right after left the session locked on its red screen. The roundtrip
        // waits for no frame: the compositor answers it as it reads it.
        SHELL.with_borrow_mut(|shell| if let Some(services) = &mut shell.services {
            services.queue.roundtrip(&mut services.state).ok();
        });
        SHELL.with_borrow_mut(|shell| shell.locks.retain(|lock| lock.id != self.id));
    }
}

/// Called by the windows host when the loop starts on Wayland: what connecting needs.
pub(crate) fn install(display: NonNull<std::ffi::c_void>, wake: Waker) {
    SHELL.with_borrow_mut(|shell| {
        shell.display = Some(display);
        shell.wake = Some(wake);
    });
}

/// The shared connection, made on first use. `None` off Wayland or before the loop runs.
pub(crate) fn connection() -> Option<Connection> {
    if let Some(conn) = SHELL.with_borrow(|shell| shell.services.as_ref().map(|s| s.conn.clone())) {
        return Some(conn);
    }
    let display = SHELL.with_borrow(|shell| shell.display)?;
    let services = Services::connect(display)?;
    let conn = services.conn.clone();
    SHELL.with_borrow_mut(|shell| shell.services = Some(services));
    Some(conn)
}

impl Services {
    #[allow(unsafe_code)]
    fn connect(display: NonNull<std::ffi::c_void>) -> Option<Services> {
        // SAFETY: winit owns this `wl_display` for the life of the event loop; `shutdown` drops
        // the connection before the loop ends, and a foreign backend never closes it.
        let backend = unsafe { Backend::from_foreign_display(display.as_ptr().cast()) };
        let conn = Connection::from_backend(backend);
        let (globals, queue) = registry_queue_init::<State>(&conn).ok()?;
        let qh = queue.handle();
        let mut state = State {
            xdg: globals.bind::<ZxdgOutputManagerV1, _, _>(&qh, 1..=3, ()).ok(),
            lock_manager: globals.bind::<ExtSessionLockManagerV1, _, _>(&qh, 1..=1, ()).ok(),
            ..State::default()
        };
        let registry = globals.registry().clone();
        globals.contents().with_list(|list| {
            for global in list.iter().filter(|global| global.interface == "wl_output") {
                state.add_output(&registry, global.name, global.version, &qh);
            }
        });
        Some(Services { conn, queue, state })
    }
}

/// Connected: the loop wakes on every read of the display, for this queue's events.
pub(crate) fn live() -> bool {
    SHELL.with_borrow(|shell| shell.services.is_some())
}

/// A read during the turn (a surface's roundtrip, the GPU's present) left events in this queue
/// that no read will wake the loop for: the loop turns again at once to dispatch them.
pub(crate) fn has_pending_events() -> bool {
    SHELL.with_borrow(|shell| shell.services.as_ref().is_some_and(|services| services.queue.prepare_read().is_none()))
}

/// Runs at the top of every loop turn: dispatches what the compositor sent and tells the
/// watchers and lock owners what changed.
pub(crate) fn turn() {
    let wants = SHELL.with_borrow(|shell| shell.display.is_some()
        && (shell.services.is_some() || !shell.watchers.is_empty() || !shell.locks.is_empty()));
    if !wants || connection().is_none() { return; }
    let (outputs, events, unsupported, take) = SHELL.with_borrow_mut(|shell| {
        let shell = &mut *shell;
        let services = shell.services.as_mut().expect("connected above");
        services.queue.dispatch_pending(&mut services.state).ok();
        services.queue.flush().ok();
        // Locks asked for since the last turn are taken now.
        let qh = services.queue.handle();
        let mut unsupported = Vec::new();
        for lock in &shell.locks {
            if lock.object.borrow().is_some() || lock.state.get() != LockState::Pending { continue; }
            match &services.state.lock_manager {
                Some(manager) => *lock.object.borrow_mut() = Some(manager.lock(&qh, lock.id)),
                None => unsupported.push(lock.clone()),
            }
        }
        if !shell.locks.is_empty() { services.conn.flush().ok(); }
        let fresh = shell.watchers.iter().any(|(_, heard)| !heard);
        let changed = std::mem::take(&mut services.state.outputs_changed);
        let outputs = (changed || fresh).then(|| services.state.list());
        let events = std::mem::take(&mut services.state.lock_events);
        // Watchers run outside the borrow: they may open surfaces or watch more.
        let take = outputs.as_ref().map(|_| std::mem::take(&mut shell.watchers));
        (outputs, events, unsupported, take)
    });
    for lock in unsupported { lock.set(LockState::Unsupported); }
    for (id, state) in events {
        let lock = SHELL.with_borrow(|shell| shell.locks.iter().find(|lock| lock.id == id).cloned());
        if let Some(lock) = lock {
            if state == LockState::Finished {
                let object = lock.object.borrow_mut().take();
                if let Some(object) = object { object.destroy(); }
                SHELL.with_borrow_mut(|shell| shell.locks.retain(|held| held.id != id));
            }
            lock.set(state);
        }
    }
    if let (Some(outputs), Some(mut watchers)) = (outputs, take) {
        for (watch, heard) in &mut watchers {
            watch(outputs.as_slice());
            *heard = true;
        }
        SHELL.with_borrow_mut(|shell| {
            // Watchers added by a watcher come after the ones that ran.
            let added = std::mem::take(&mut shell.watchers);
            shell.watchers = watchers;
            shell.watchers.extend(added);
        });
    }
}

/// Drops the connection while winit's display is still open.
pub(crate) fn shutdown() {
    let services = SHELL.with_borrow_mut(|shell| {
        shell.locks.clear();
        shell.watchers.clear();
        shell.services.take()
    });
    drop(services);
}

pub(crate) fn watch_outputs(f: OnOutputs) {
    SHELL.with_borrow_mut(|shell| shell.watchers.push((f, false)));
    wake_loop();
}

pub(crate) fn outputs() -> Vec<Output> {
    SHELL.with_borrow(|shell| shell.services.as_ref().map(|s| s.state.list()).unwrap_or_default())
}

static NEXT_LOCK: AtomicU64 = AtomicU64::new(1);

pub(crate) fn lock(on_change: OnLock) -> Rc<Lock> {
    let lock = Rc::new(Lock {
        id: NEXT_LOCK.fetch_add(1, Ordering::Relaxed),
        state: Cell::new(LockState::Pending),
        on_change: RefCell::new(on_change),
        object: RefCell::new(None),
    });
    SHELL.with_borrow_mut(|shell| shell.locks.push(lock.clone()));
    wake_loop();
    lock
}

/// The lock held now, for a lock surface to be made on.
pub(crate) fn held_lock() -> Option<ExtSessionLockV1> {
    SHELL.with_borrow(|shell| shell.locks.iter().find_map(|lock| lock.object.borrow().clone()))
}

/// The output named `name`: its registry name, its proxy on the shared connection, and its
/// integer buffer scale.
pub(crate) fn output_named(name: &str) -> Option<(u32, WlOutput, i32)> {
    SHELL.with_borrow(|shell| {
        let services = shell.services.as_ref()?;
        services.state.outputs.iter().find(|out| out.complete && out.shown.name == name)
            .map(|out| (out.global, out.proxy.clone(), out.integer_scale.max(1)))
    })
}

/// Asks the loop for a turn, so a watcher or lock asked for now is served then.
fn wake_loop() {
    let wake = SHELL.with_borrow(|shell| shell.wake.clone());
    if let Some(wake) = wake { wake(); }
}

impl State {
    fn add_output(&mut self, registry: &WlRegistry, global: u32, version: u32, qh: &QueueHandle<State>) {
        let proxy: WlOutput = registry.bind(global, version.min(4), qh, global);
        let xdg = self.xdg.as_ref().map(|manager| manager.get_xdg_output(&proxy, qh, global));
        self.outputs.push(OutputEntry {
            global, proxy, xdg,
            pending: Output { scale: 1.0, ..Output::default() },
            shown: Output::default(),
            integer_scale: 1,
            transform_swaps: false,
            complete: false,
        });
    }

    fn remove_output(&mut self, global: u32) {
        let Some(at) = self.outputs.iter().position(|out| out.global == global) else { return };
        let gone = self.outputs.remove(at);
        if let Some(xdg) = gone.xdg { xdg.destroy(); }
        if gone.proxy.version() >= 3 { gone.proxy.release(); }
        self.outputs_changed |= gone.complete;
    }

    fn entry(&mut self, global: u32) -> Option<&mut OutputEntry> {
        self.outputs.iter_mut().find(|out| out.global == global)
    }

    fn list(&self) -> Vec<Output> {
        self.outputs.iter().filter(|out| out.complete).map(|out| out.shown.clone()).collect()
    }
}

impl OutputEntry {
    /// The compositor finished describing it: what it said becomes what it is.
    fn done(&mut self) -> bool {
        let mut next = self.pending.clone();
        let (pw, ph) = if self.transform_swaps { (next.physical.1, next.physical.0) } else { next.physical };
        if next.logical.0 <= 0 || next.logical.1 <= 0 {
            // No xdg-output: the mode at the integer scale.
            let scale = self.integer_scale.max(1);
            next.logical = (pw / scale, ph / scale);
        }
        next.scale = if next.logical.0 > 0 { f64::from(pw) / f64::from(next.logical.0) } else { f64::from(self.integer_scale.max(1)) };
        self.pending = next.clone();
        let changed = !self.complete || self.shown != next;
        self.shown = next;
        self.complete = true;
        changed
    }
}

impl Dispatch<WlRegistry, GlobalListContents> for State {
    fn event(state: &mut Self, registry: &WlRegistry, event: wl_registry::Event, _: &GlobalListContents, _: &Connection, qh: &QueueHandle<Self>) {
        match event {
            wl_registry::Event::Global { name, interface, version } if interface == "wl_output" => state.add_output(registry, name, version, qh),
            wl_registry::Event::GlobalRemove { name } => state.remove_output(name),
            _ => {}
        }
    }
}

impl Dispatch<WlOutput, u32> for State {
    fn event(state: &mut Self, _: &WlOutput, event: wl_output::Event, global: &u32, _: &Connection, _: &QueueHandle<Self>) {
        let Some(out) = state.entry(*global) else { return };
        match event {
            wl_output::Event::Geometry { transform, .. } => {
                out.transform_swaps = matches!(transform, WEnum::Value(
                    wl_output::Transform::_90 | wl_output::Transform::_270
                        | wl_output::Transform::Flipped90 | wl_output::Transform::Flipped270
                ));
            }
            wl_output::Event::Mode { flags: WEnum::Value(flags), width, height, .. } if flags.contains(wl_output::Mode::Current) => {
                out.pending.physical = (width, height);
            }
            wl_output::Event::Scale { factor } => out.integer_scale = factor.max(1),
            wl_output::Event::Name { name } => out.pending.name = name,
            wl_output::Event::Done => {
                let changed = out.done();
                state.outputs_changed |= changed;
            }
            _ => {}
        }
    }
}

impl Dispatch<ZxdgOutputV1, u32> for State {
    fn event(state: &mut Self, xdg: &ZxdgOutputV1, event: zxdg_output_v1::Event, global: &u32, _: &Connection, _: &QueueHandle<Self>) {
        let Some(out) = state.entry(*global) else { return };
        match event {
            zxdg_output_v1::Event::LogicalPosition { x, y } => out.pending.position = (x, y),
            zxdg_output_v1::Event::LogicalSize { width, height } => out.pending.logical = (width, height),
            // Before wl_output v4 the name comes from here.
            zxdg_output_v1::Event::Name { name } if out.pending.name.is_empty() => out.pending.name = name,
            // From version 3 wl_output's `done` closes the description.
            zxdg_output_v1::Event::Done if xdg.version() < 3 => {
                let changed = out.done();
                state.outputs_changed |= changed;
            }
            _ => {}
        }
    }
}

impl Dispatch<ZxdgOutputManagerV1, ()> for State {
    fn event(_: &mut Self, _: &ZxdgOutputManagerV1, _: <ZxdgOutputManagerV1 as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<ExtSessionLockManagerV1, ()> for State {
    fn event(_: &mut Self, _: &ExtSessionLockManagerV1, _: <ExtSessionLockManagerV1 as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<ExtSessionLockV1, u64> for State {
    fn event(state: &mut Self, _: &ExtSessionLockV1, event: ext_session_lock_v1::Event, id: &u64, _: &Connection, _: &QueueHandle<Self>) {
        match event {
            ext_session_lock_v1::Event::Locked => state.lock_events.push((*id, LockState::Locked)),
            ext_session_lock_v1::Event::Finished => state.lock_events.push((*id, LockState::Finished)),
            _ => {}
        }
    }
}
