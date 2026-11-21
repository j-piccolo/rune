//! A Wayland layer-shell or session-lock surface for overlays: `zwlr_layer_shell_v1` (or
//! `ext_session_lock_v1`) on winit's own display, through the connection every such surface
//! shares (`wayland_shell`), in a private event queue with its own seat. Placed by rune's edge
//! placement, or as its `OverlayOptions::shell` says (any layer, anchors, margins, exclusive
//! zone and keyboard). Input regions, blur regions and the rendered buffer share a commit.
//! Global pointer observation and capture exclusion are unavailable. wgpu draws through the
//! raw `wl_surface`.

use std::os::fd::{AsRawFd, RawFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use winit::dpi::PhysicalPosition;
use winit::event::MouseScrollDelta;
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle};
use wayland_backend::client::Backend;
use wayland_client::globals::{GlobalListContents, registry_queue_init};
use wayland_client::protocol::wl_callback::{self, WlCallback};
use wayland_client::protocol::wl_compositor::WlCompositor;
use wayland_client::protocol::wl_region::WlRegion;
use wayland_client::protocol::wl_registry::WlRegistry;
use wayland_client::protocol::wl_surface::{self, WlSurface};
use wayland_client::protocol::{wl_output::{self, WlOutput}, wl_seat::{self, WlSeat}, wl_pointer::{self, WlPointer}, wl_keyboard::{self, WlKeyboard}, wl_touch::{self, WlTouch}};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle};
use wayland_protocols::ext::background_effect::v1::client::ext_background_effect_manager_v1::ExtBackgroundEffectManagerV1;
use wayland_protocols::ext::background_effect::v1::client::ext_background_effect_surface_v1::ExtBackgroundEffectSurfaceV1;
use wayland_protocols::ext::session_lock::v1::client::ext_session_lock_surface_v1::{self, ExtSessionLockSurfaceV1};
use wayland_protocols::wp::cursor_shape::v1::client::wp_cursor_shape_device_v1::{Shape, WpCursorShapeDeviceV1};
use wayland_protocols::wp::cursor_shape::v1::client::wp_cursor_shape_manager_v1::WpCursorShapeManagerV1;
use wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_shell_v1::{Layer, ZwlrLayerShellV1};
use wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::{self, Anchor, KeyboardInteractivity, ZwlrLayerSurfaceV1};

use super::layer_shell_machine::{frame, keyboard, readiness, touch};
use crate::el::Align;
use crate::overlay::{Placement, ScreenEdge};
use crate::runtime::Waker;
use crate::wayland::{BlurRegion, LayerKeyboard, LayerShell, ShellSurface};
use crate::{OverlayInput, OverlayOptions, OverlayPlacement};

pub(crate) enum Input {
    Move(f32, f32), Leave, Button(bool), Secondary(bool), Focus(bool), Key(crate::KeyEvent), KeyUp(crate::KeyEvent),
    Physical(crate::PhysicalInput, Option<crate::KeyEvent>),
    /// A wheel or touchpad frame, as winit would report it.
    Wheel { delta: MouseScrollDelta, phase: crate::GesturePhase },
    /// The middle, back or forward button.
    Other(crate::PointerButton, bool),
}

/// Whether the pointer is over the surface: the serial of its enter, which setting a cursor
/// needs, and the cursor the surface set since (none: the one it arrived with).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Over {
    #[default]
    Outside,
    Inside { serial: u32, cursor: Option<crate::el::Cursor> },
}

impl Over {
    fn inside(self) -> bool { matches!(self, Over::Inside { .. }) }
}

/// One `wl_pointer` frame of scrolling, collected until its `frame` event.
#[derive(Clone, Copy, Debug, Default)]
struct AxisFrame {
    /// Surface units, Wayland's sign (down and right positive).
    x: f64,
    y: f64,
    /// Wheel detents in 120ths, from `axis_value120` (or `axis_discrete` before version 8).
    x120: i32,
    y120: i32,
    discrete: bool,
    source: Option<wl_pointer::AxisSource>,
    stop: bool,
    any: bool,
}

#[derive(Default)]
struct State {
    /// How far the compositor has got with the surface: its first configure is what makes it
    /// drawable, and closing it is final.
    readiness: readiness::State,
    /// Whether the compositor has drawn the last frame presented.
    frame: frame::State,
    input: Vec<Input>,
    surface: Option<WlSurface>,
    over: Over,
    keyboard_focus: keyboard::FocusState,
    want_pointer: bool,
    want_keys: bool,
    capabilities: Option<wl_seat::Capability>,
    pointer: Option<WlPointer>,
    /// A finger on the surface acts as the primary button, so a tap is a click.
    touch: Option<WlTouch>,
    finger: touch::State,
    /// `wp_cursor_shape_manager_v1` when the compositor has it, and its device for `pointer`.
    shapes: Option<WpCursorShapeManagerV1>,
    shape: Option<WpCursorShapeDeviceV1>,
    keyboard: Option<WlKeyboard>,
    xkb: Option<xkbcommon::xkb::State>,
    mod_state: Option<keyboard::ModState>,
    /// The scroll being collected, and whether a touchpad scroll is under way.
    axis: AxisFrame,
    scrolling: bool,
    /// The buffer scale, for scroll deltas in physical pixels as winit gives them.
    scale: f64,
}

impl Dispatch<WlOutput, ()> for State {
    fn event(_: &mut Self, _: &WlOutput, _: wl_output::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<WlSeat, ()> for State {
    fn event(state: &mut Self, seat: &WlSeat, event: wl_seat::Event, _: &(), _: &Connection, qh: &QueueHandle<Self>) {
        if let wl_seat::Event::Capabilities { capabilities: wayland_client::WEnum::Value(caps) } = event {
            state.capabilities = Some(caps);
            state.update_seat(seat, qh);
        }
    }
}
impl State {
    fn step_keyboard_focus(&mut self, event: keyboard::FocusEvent) {
        for effect in keyboard::step(&mut self.keyboard_focus, event) {
            let keyboard::FocusEffect::Focus(focused) = effect;
            self.input.push(Input::Focus(focused));
        }
    }

    fn update_seat(&mut self, seat: &WlSeat, qh: &QueueHandle<Self>) {
        let caps = self.capabilities.unwrap_or_else(wl_seat::Capability::empty);
        let pointer = self.want_pointer && caps.contains(wl_seat::Capability::Pointer);
        let keyboard = self.want_keys && caps.contains(wl_seat::Capability::Keyboard);
        if pointer && self.pointer.is_none() {
            let p = seat.get_pointer(qh, ());
            self.shape = self.shapes.as_ref().map(|shapes| shapes.get_pointer(&p, qh, ()));
            self.pointer = Some(p);
        }
        let touch = self.want_pointer && caps.contains(wl_seat::Capability::Touch);
        if touch && self.touch.is_none() { self.touch = Some(seat.get_touch(qh, ())); }
        if keyboard && self.keyboard.is_none() { self.keyboard = Some(seat.get_keyboard(qh, ())); }
        if !pointer && let Some(p) = self.pointer.take() {
            if let Some(shape) = self.shape.take() { shape.destroy(); }
            if p.version() >= 3 { p.release(); }
            self.over = Over::Outside;
        }
        if !touch && let Some(t) = self.touch.take() {
            if t.version() >= 3 { t.release(); }
            touch::step(&mut self.finger, touch::Event::Gone);
        }
        if !keyboard && let Some(k) = self.keyboard.take() {
            if k.version() >= 3 { k.release(); }
            self.xkb = None;
            self.mod_state = None;
            self.step_keyboard_focus(keyboard::FocusEvent::KeyboardGone);
        }
    }

    /// The scroll collected since the last frame, as winit would report it: wheel detents as
    /// lines, a touchpad's travel as pixels with Chromium's gain, Wayland's sign turned.
    fn flush_axis(&mut self) {
        let frame = std::mem::take(&mut self.axis);
        if !frame.any || !self.over.inside() { return; }
        let has_detents = frame.discrete && (frame.x120 != 0 || frame.y120 != 0);
        let phase = if frame.stop {
            self.scrolling = false;
            crate::GesturePhase::Ended
        } else if has_detents || self.scrolling {
            crate::GesturePhase::Moved
        } else {
            self.scrolling = true;
            crate::GesturePhase::Began
        };
        let delta = if has_detents {
            MouseScrollDelta::LineDelta(-(frame.x120 as f32) / 120.0, -(frame.y120 as f32) / 120.0)
        } else {
            let gain = match frame.source {
                Some(wl_pointer::AxisSource::Finger | wl_pointer::AxisSource::Continuous) => 120.0 / 10.0,
                _ => 1.0,
            };
            let scale = if self.scale > 0.0 { self.scale } else { 1.0 };
            MouseScrollDelta::PixelDelta(PhysicalPosition::new(-frame.x * gain * scale, -frame.y * gain * scale))
        };
        self.input.push(Input::Wheel { delta, phase });
    }
}
impl Dispatch<WlPointer, ()> for State {
    fn event(state: &mut Self, pointer: &WlPointer, event: wl_pointer::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        use wayland_client::WEnum;
        match event {
            wl_pointer::Event::Enter { serial, surface, surface_x, surface_y } => {
                // Until the surface sets a cursor, the pointer keeps the one it arrived with.
                let inside = state.surface.as_ref() == Some(&surface);
                state.over = if inside { Over::Inside { serial, cursor: None } } else { Over::Outside };
                if inside { state.input.push(Input::Move(surface_x as f32, surface_y as f32)); }
            }
            wl_pointer::Event::Motion { surface_x, surface_y, .. } if state.over.inside() => state.input.push(Input::Move(surface_x as f32, surface_y as f32)),
            wl_pointer::Event::Leave { surface, .. } if state.surface.as_ref() == Some(&surface) => {
                state.over = Over::Outside;
                state.scrolling = false;
                state.input.push(Input::Leave);
            }
            wl_pointer::Event::Button { button: 0x110, state: WEnum::Value(button), .. } if state.over.inside() => {
                let pressed = button == wl_pointer::ButtonState::Pressed;
                state.input.push(Input::Button(pressed));
                if pressed && let Some(xkb) = &state.xkb {
                    state.input.push(Input::Physical(crate::PhysicalInput::Modifiers(key_mods(xkb)), None));
                }
            }
            // BTN_RIGHT: the secondary gesture, as a window's right button is.
            wl_pointer::Event::Button { button: 0x111, state: WEnum::Value(button), .. } if state.over.inside() => {
                state.input.push(Input::Secondary(button == wl_pointer::ButtonState::Pressed));
            }
            // BTN_MIDDLE, BTN_SIDE and BTN_EXTRA: middle, back and forward.
            wl_pointer::Event::Button { button: code @ 0x112..=0x114, state: WEnum::Value(button), .. } if state.over.inside() => {
                let which = match code {
                    0x112 => crate::PointerButton::Middle,
                    0x113 => crate::PointerButton::Back,
                    _ => crate::PointerButton::Forward,
                };
                state.input.push(Input::Other(which, button == wl_pointer::ButtonState::Pressed));
            }
            wl_pointer::Event::Axis { axis: WEnum::Value(axis), value, .. } => {
                match axis {
                    wl_pointer::Axis::HorizontalScroll => state.axis.x += value,
                    _ => state.axis.y += value,
                }
                state.axis.any = true;
                // Before version 5 there are no frames: each event stands alone.
                if pointer.version() < 5 { state.flush_axis(); }
            }
            wl_pointer::Event::AxisSource { axis_source: WEnum::Value(source) } => state.axis.source = Some(source),
            wl_pointer::Event::AxisStop { .. } => {
                state.axis.stop = true;
                state.axis.any = true;
            }
            wl_pointer::Event::AxisDiscrete { axis: WEnum::Value(axis), discrete } => {
                match axis {
                    wl_pointer::Axis::HorizontalScroll => state.axis.x120 += discrete * 120,
                    _ => state.axis.y120 += discrete * 120,
                }
                state.axis.discrete = true;
                state.axis.any = true;
            }
            wl_pointer::Event::AxisValue120 { axis: WEnum::Value(axis), value120 } => {
                match axis {
                    wl_pointer::Axis::HorizontalScroll => state.axis.x120 += value120,
                    _ => state.axis.y120 += value120,
                }
                state.axis.discrete = true;
                state.axis.any = true;
            }
            wl_pointer::Event::Frame => state.flush_axis(),
            _ => {}
        }
    }
}
impl Dispatch<WlTouch, ()> for State {
    fn event(state: &mut Self, _: &WlTouch, event: wl_touch::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        use touch::Event as Touch;
        let event = match event {
            wl_touch::Event::Down { surface, id, x, y, .. } if state.surface.as_ref() == Some(&surface) => Touch::Down { id, x: x as f32, y: y as f32 },
            wl_touch::Event::Motion { id, x, y, .. } => Touch::Motion { id, x: x as f32, y: y as f32 },
            wl_touch::Event::Up { id, .. } => Touch::Up { id },
            wl_touch::Event::Cancel => Touch::Cancel,
            _ => return,
        };
        for effect in touch::step(&mut state.finger, event) {
            state.input.push(match effect {
                touch::Effect::Move(x, y) => Input::Move(x, y),
                touch::Effect::Press => Input::Button(true),
                touch::Effect::Release => Input::Button(false),
                touch::Effect::Leave => Input::Leave,
            });
        }
    }
}
impl Dispatch<WlKeyboard, ()> for State {
    fn event(state: &mut Self, _: &WlKeyboard, event: wl_keyboard::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        use xkbcommon::xkb;
        match event {
            wl_keyboard::Event::Enter { surface, .. } => {
                let focus_event = if state.surface.as_ref() == Some(&surface) {
                    keyboard::FocusEvent::EnterOurs
                } else {
                    keyboard::FocusEvent::EnterOther
                };
                state.step_keyboard_focus(focus_event);
            }
            wl_keyboard::Event::Leave { surface, .. } if state.surface.as_ref() == Some(&surface) => {
                state.step_keyboard_focus(keyboard::FocusEvent::LeaveOurs);
            }
            wl_keyboard::Event::Keymap { format: wayland_client::WEnum::Value(wl_keyboard::KeymapFormat::XkbV1), fd, size } => {
                use std::os::unix::fs::FileExt;
                // Read at offset 0, not from the fd's position: a compositor may hand every
                // keyboard the one sealed memfd (smithay does), whose position they all share, so
                // a second surface's `read` started where the first left off and parsed nothing.
                const MAX: u32 = 16 * 1024 * 1024;
                let mut bytes = vec![0; size.min(MAX) as usize];
                if size <= MAX && std::fs::File::from(fd).read_exact_at(&mut bytes, 0).is_ok() {
                    let map = String::from_utf8_lossy(&bytes);
                    let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
                    state.xkb = xkb::Keymap::new_from_string(&context, map.trim_end_matches('\0').to_owned(), xkb::KEYMAP_FORMAT_TEXT_V1, xkb::COMPILE_NO_FLAGS).map(|map| {
                        let mut xkb_state = xkb::State::new(&map);
                        if let Some(m) = state.mod_state {
                            xkb_state.update_mask(m.depressed, m.latched, m.locked, 0, 0, m.group);
                        }
                        xkb_state
                    });
                }
            }
            wl_keyboard::Event::Modifiers { mods_depressed, mods_latched, mods_locked, group, .. } => {
                state.mod_state = Some(keyboard::ModState { depressed: mods_depressed, latched: mods_latched, locked: mods_locked, group });
                if let Some(xkb) = &mut state.xkb {
                    xkb.update_mask(mods_depressed, mods_latched, mods_locked, 0, 0, group);
                    state.input.push(Input::Physical(crate::PhysicalInput::Modifiers(key_mods(xkb)), None));
                }
            }
            wl_keyboard::Event::Key { key, state: wayland_client::WEnum::Value(pressed), .. } if state.keyboard_focus.focused() => {
                use winit::platform::scancode::PhysicalKeyExtScancode;
                if let Some(xkb) = &state.xkb {
                    let pressed = pressed == wl_keyboard::KeyState::Pressed;
                    let physical = match crate::PhysicalKeyCode::from_scancode(key) {
                        winit::keyboard::PhysicalKey::Code(code) => Some(code),
                        _ => None,
                    };
                    let fallback = key_event(xkb, key, physical);
                    if let Some(code) = physical {
                        state.input.push(Input::Physical(crate::PhysicalInput::Key { key: code, mods: key_mods(xkb), pressed, repeat: false }, fallback));
                    } else if let Some(key) = fallback {
                        state.input.push(if pressed { Input::Key(key) } else { Input::KeyUp(key) });
                    }
                }
            }
            _ => {}
        }
    }
}

fn key_mods(state: &xkbcommon::xkb::State) -> crate::Mods {
    use xkbcommon::xkb;
    let active = |name: &str| state.mod_name_is_active(name, xkb::STATE_MODS_EFFECTIVE);
    crate::input::key_table::mods_from_keys(active(xkb::MOD_NAME_SHIFT), active(xkb::MOD_NAME_CTRL), active(xkb::MOD_NAME_ALT), active(xkb::MOD_NAME_LOGO))
}

fn key_event(state: &xkbcommon::xkb::State, key: u32, physical: Option<crate::PhysicalKeyCode>) -> Option<crate::KeyEvent> {
    use xkbcommon::xkb;
    let code = xkb::Keycode::new(key + 8);
    let sym = state.key_get_one_sym(code);
    let mods = key_mods(state);
    let text = state.key_get_utf8(code);
    keyboard::key_event_from(named_keysym(sym.raw()), xkb::keysym_to_utf32(sym), text, mods, physical)
}

/// The named key an XKB keysym is, or None for one that types (or nothing Rune names).
fn named_keysym(symbol: u32) -> Option<crate::Key> {
    Some(match symbol {
        0xff1b => crate::Key::Escape, 0xff0d => crate::Key::Enter, 0xff08 => crate::Key::Backspace,
        0xffff => crate::Key::Delete, 0xff09 | 0xfe20 => crate::Key::Tab,
        0xff50 => crate::Key::Home, 0xff57 => crate::Key::End, 0xff55 => crate::Key::PageUp, 0xff56 => crate::Key::PageDown,
        0xff51 => crate::Key::Left, 0xff52 => crate::Key::Up, 0xff53 => crate::Key::Right, 0xff54 => crate::Key::Down,
        0xffbe..=0xffd5 => crate::Key::Function((symbol - 0xffbe + 1) as u8),
        0xff67 => crate::Key::ContextMenu,
        0xff63 => crate::Key::Insert, 0xff61 | 0xff15 => crate::Key::PrintScreen,
        0xff14 => crate::Key::ScrollLock, 0xff13 | 0xff6b => crate::Key::Pause,
        _ => return None,
    })
}

impl Dispatch<WlRegistry, GlobalListContents> for State {
    fn event(_: &mut Self, _: &WlRegistry, _: <WlRegistry as Proxy>::Event, _: &GlobalListContents, _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<WlCompositor, ()> for State {
    fn event(_: &mut Self, _: &WlCompositor, _: <WlCompositor as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<WlRegion, ()> for State {
    fn event(_: &mut Self, _: &WlRegion, _: <WlRegion as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<ZwlrLayerShellV1, ()> for State {
    fn event(_: &mut Self, _: &ZwlrLayerShellV1, _: <ZwlrLayerShellV1 as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<WpCursorShapeManagerV1, ()> for State {
    fn event(_: &mut Self, _: &WpCursorShapeManagerV1, _: <WpCursorShapeManagerV1 as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<WpCursorShapeDeviceV1, ()> for State {
    fn event(_: &mut Self, _: &WpCursorShapeDeviceV1, _: <WpCursorShapeDeviceV1 as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<ExtBackgroundEffectManagerV1, ()> for State {
    fn event(_: &mut Self, _: &ExtBackgroundEffectManagerV1, _: <ExtBackgroundEffectManagerV1 as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
impl Dispatch<ExtBackgroundEffectSurfaceV1, ()> for State {
    fn event(_: &mut Self, _: &ExtBackgroundEffectSurfaceV1, _: <ExtBackgroundEffectSurfaceV1 as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

/// The cursor-shape-v1 shape for `c`, as `os_cursor` maps winit's. An image is the arrow.
fn shape(c: crate::el::Cursor) -> Shape {
    use crate::el::Cursor;
    match c {
        Cursor::Default | Cursor::Hidden | Cursor::Image(_) => Shape::Default,
        Cursor::Pointer => Shape::Pointer,
        Cursor::Text => Shape::Text,
        Cursor::Grab => Shape::Grab,
        Cursor::Grabbing => Shape::Grabbing,
        Cursor::NotAllowed => Shape::NotAllowed,
        Cursor::Move => Shape::Move,
        Cursor::ResizeColumn => Shape::ColResize,
        Cursor::ResizeRow => Shape::RowResize,
        Cursor::Crosshair => Shape::Crosshair,
        Cursor::Cell => Shape::Cell,
        Cursor::ZoomIn => Shape::ZoomIn,
        Cursor::ZoomOut => Shape::ZoomOut,
    }
}
impl Dispatch<WlSurface, ()> for State {
    fn event(state: &mut Self, _: &WlSurface, event: wl_surface::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        if let wl_surface::Event::PreferredBufferScale { factor } = event {
            readiness::step(&mut state.readiness, readiness::Event::Scale(factor.max(1)));
        }
    }
}
/// A frame callback, with the number its present asked as.
impl Dispatch<WlCallback, u64> for State {
    fn event(state: &mut Self, _: &WlCallback, event: wl_callback::Event, ask: &u64, _: &Connection, _: &QueueHandle<Self>) {
        if let wl_callback::Event::Done { .. } = event {
            frame::step(&mut state.frame, frame::Event::Answered { ask: *ask });
        }
    }
}
impl Dispatch<ZwlrLayerSurfaceV1, ()> for State {
    fn event(state: &mut Self, layer: &ZwlrLayerSurfaceV1, event: zwlr_layer_surface_v1::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        match event {
            zwlr_layer_surface_v1::Event::Configure { serial, width, height } => {
                layer.ack_configure(serial);
                readiness::step(&mut state.readiness, readiness::Event::Configure { width, height });
            }
            zwlr_layer_surface_v1::Event::Closed => readiness::step(&mut state.readiness, readiness::Event::Closed),
            _ => {}
        }
    }
}
impl Dispatch<ExtSessionLockSurfaceV1, ()> for State {
    fn event(state: &mut Self, lock: &ExtSessionLockSurfaceV1, event: ext_session_lock_surface_v1::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        // A lock surface is the output's size, always: the buffer must match what it says.
        if let ext_session_lock_surface_v1::Event::Configure { serial, width, height } = event {
            lock.ack_configure(serial);
            readiness::step(&mut state.readiness, readiness::Event::Configure { width, height });
        }
    }
}

/// Wakes the event loop when the compositor writes to the display. winit turns its loop only
/// for its own queue's events, so input for this surface's queue (a click on an idle overlay)
/// would otherwise wait for an unrelated wake. An XWayland overlay watches its X connection
/// the same way. Layer surfaces share one watcher (`wayland_shell`); this one is for a
/// connection of its own.
pub(crate) struct Watch {
    stop: Arc<AtomicBool>,
    /// How many times the overlay has dispatched, so the watcher waits for the loop to read
    /// what woke it instead of spinning on a socket that stays readable until then.
    turns: Arc<(Mutex<u64>, Condvar)>,
}

impl Watch {
    #[allow(unsafe_code)]
    pub(crate) fn spawn(fd: RawFd, wake: Waker) -> Watch {
        let stop = Arc::new(AtomicBool::new(false));
        let turns = Arc::new((Mutex::new(0u64), Condvar::new()));
        let (stopped, seen) = (stop.clone(), turns.clone());
        std::thread::Builder::new().name("rune-layer-watch".into()).spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                let mut poll = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
                // SAFETY: one pollfd for a display socket. winit's outlives the event loop; an
                // XWayland surface drops its watch before closing its connection, so at worst a
                // reused descriptor wakes the loop once more before this thread stops.
                if unsafe { libc::poll(&mut poll, 1, 250) } <= 0 { continue; }
                let (count, turned) = &*seen;
                let before = *count.lock().unwrap();
                wake();
                turned.wait_timeout_while(count.lock().unwrap(), Duration::from_millis(100), |n| *n == before).ok();
            }
        }).ok();
        Watch { stop, turns }
    }

    pub(crate) fn turned(&self) {
        let (count, turned) = &*self.turns;
        *count.lock().unwrap() += 1;
        turned.notify_all();
    }
}

impl Drop for Watch {
    /// The thread notices within one poll timeout; joining would stall hiding.
    fn drop(&mut self) { self.stop.store(true, Ordering::Relaxed); }
}

/// How the surface wakes the loop: the shared watcher, or a thread of its own on a connection
/// of its own (no shared one: the shell services weren't installed).
enum Wakes {
    /// Held for its `Drop`, which takes the wake off the shared watcher.
    Shared(#[allow(dead_code)] super::wayland_shell::Registration),
    Own(Watch),
}

/// What the overlay needs to know after the compositor talked to the surface.
pub(crate) struct Changes {
    pub size: Option<(u32, u32)>,
    pub input: Vec<Input>,
    pub scale: Option<i32>,
    pub closed: bool,
}

/// The surface's role.
enum Role {
    Layer(ZwlrLayerSurfaceV1),
    Lock(ExtSessionLockSurfaceV1),
}

pub(crate) struct LayerSurface {
    registry: WlRegistry,
    seat_global: Option<(u32, u32)>,
    regions: std::cell::RefCell<Option<Vec<[i32; 4]>>>,
    compositor: WlCompositor,
    /// An output bound here by registry name, released with the surface.
    output: Option<WlOutput>,
    seat: Option<WlSeat>,
    options: OverlayOptions,
    queue: EventQueue<State>,
    state: State,
    role: Role,
    surface: WlSurface,
    /// `ext-background-effect-v1` on this surface, where the compositor has it, and the
    /// region last sent (none: no blur).
    blur: Option<ExtBackgroundEffectSurfaceV1>,
    blur_sent: Option<Vec<[i32; 4]>>,
    /// Frames presented, to tell each one's answer from an earlier one's.
    asks: u64,
    display: NonNull<std::ffi::c_void>,
    wakes: Option<Wakes>,
}

impl LayerSurface {
    /// None when the display isn't Wayland, the compositor lacks layer-shell (or, for a lock
    /// surface, no lock is held or the output is unknown).
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn new(display: RawDisplayHandle, namespace: &str, at: Placement, interactive: bool, options: &OverlayOptions, output_id: Option<u32>, named: Option<WlOutput>, wake: Option<Waker>) -> Option<LayerSurface> {
        let RawDisplayHandle::Wayland(display) = display else { return None };
        // Every shell surface shares one connection, so a lock surface can be made on the lock
        // the shell holds. Without one (the services weren't installed), a connection of its own.
        let shared = super::wayland_shell::connection();
        let conn = match shared.clone() {
            Some(conn) => conn,
            // SAFETY: winit owns this `wl_display` for the life of the event loop, which
            // outlives every overlay; a foreign backend never closes it.
            None => Connection::from_backend(unsafe { Backend::from_foreign_display(display.display.as_ptr().cast()) }),
        };
        let (globals, queue) = registry_queue_init::<State>(&conn).ok()?;
        let qh = queue.handle();
        let compositor: WlCompositor = globals.bind(&qh, 1..=6, ()).ok()?;
        let lock = matches!(options.shell, Some(ShellSurface::Lock { .. }));
        let shell: Option<ZwlrLayerShellV1> = if lock { None } else { Some(globals.bind(&qh, 1..=4, ()).ok()?) };
        let output = if named.is_some() { None } else {
            output_id.and_then(|id| {
                globals.contents().with_list(|list| list.iter().find(|g| g.name == id && g.interface == "wl_output").map(|g| {
                    globals.registry().bind::<WlOutput, _, _>(g.name, g.version.min(4), &qh, ())
                }))
            })
        };
        let layer_spec = match &options.shell { Some(ShellSurface::Layer(spec)) => Some(spec.clone()), _ => None };
        let want_pointer = options.input.resolved(interactive) != OverlayInput::None || lock;
        let want_keys = options.keys == crate::KeyPolicy::OnDemand || lock
            || layer_spec.as_ref().is_some_and(|spec| spec.keyboard != LayerKeyboard::None);
        let registry = globals.registry().clone();
        let seat_global = globals.contents().with_list(|list| list.iter().find(|g| g.interface == "wl_seat").map(|g| (g.name, g.version.min(8))));
        let seat = if want_pointer || want_keys { seat_global.map(|(name, version)| registry.bind::<WlSeat, _, _>(name, version, &qh, ())) } else { None };
        let surface = compositor.create_surface(&qh, ());
        let region = compositor.create_region(&qh, ());
        surface.set_input_region(Some(&region));
        region.destroy();
        let role = if lock {
            let held = super::wayland_shell::held_lock()?;
            let on = named.as_ref()?;
            Role::Lock(held.get_lock_surface(&surface, on, &qh, ()))
        } else {
            let on = named.as_ref().or(output.as_ref());
            let level = layer_spec.as_ref().map_or(Layer::Overlay, |spec| layer_of(spec.layer));
            // Preserve existing overlay-layer semantics without a shell spec. There is no layer
            // above Overlay; relative ordering of peers is compositor policy.
            let layer = shell?.get_layer_surface(&surface, on, level, namespace.to_owned(), &qh, ());
            let keys = layer_spec.as_ref().map_or(KeyboardInteractivity::None, |spec| interactivity(spec.keyboard, layer.version()));
            layer.set_keyboard_interactivity(keys);
            layer.set_exclusive_zone(layer_spec.as_ref().map_or(0, |spec| spec.exclusive_zone));
            Role::Layer(layer)
        };
        let blur = globals.bind::<ExtBackgroundEffectManagerV1, _, _>(&qh, 1..=1, ()).ok()
            .map(|manager| manager.get_background_effect(&surface, &qh, ()));
        let shapes = globals.bind::<WpCursorShapeManagerV1, _, _>(&qh, 1..=1, ()).ok();
        let state = State { surface: Some(surface.clone()), want_pointer, want_keys, shapes, scale: 1.0, ..State::default() };
        let wakes = wake.map(|wake| match shared {
            Some(_) => super::wayland_shell::register_wake(wake.clone()).map_or_else(|| Wakes::Own(Watch::spawn(conn.backend().poll_fd().as_raw_fd(), wake)), Wakes::Shared),
            None => Wakes::Own(Watch::spawn(conn.backend().poll_fd().as_raw_fd(), wake)),
        });
        let mut this = LayerSurface {
            registry, seat_global, regions: std::cell::RefCell::new(Some(Vec::new())), compositor, output, seat,
            options: options.clone(), queue, state, role, surface, blur, blur_sent: None, asks: 0, display: display.display, wakes,
        };
        if let Some(seat) = &this.seat { this.state.update_seat(seat, &this.queue.handle()); }
        this.place(at);
        // A layer surface's first commit asks for its configure. A lock surface is configured
        // unasked, and a commit before it acks that is a protocol error that ends the client.
        if matches!(this.role, Role::Layer(_)) {
            this.surface.commit();
        }
        this.queue.flush().ok();
        while !this.state.readiness.configured() {
            this.queue.blocking_dispatch(&mut this.state).ok()?;
        }
        (!this.state.readiness.closed()).then_some(this)
    }

    pub fn place(&mut self, at: Placement) {
        let Role::Layer(layer) = &self.role else { return };
        if let Some(ShellSurface::Layer(spec)) = &self.options.shell {
            place_shell(layer, spec, at.size);
            return;
        }
        match self.options.placement {
            Some(OverlayPlacement::FullScreen) => {
                layer.set_size(0, 0);
                layer.set_anchor(Anchor::Top | Anchor::Bottom | Anchor::Left | Anchor::Right);
                layer.set_margin(0, 0, 0, 0);
                return;
            }
            Some(OverlayPlacement::Rect { x, y, width, height }) => {
                layer.set_size(width.ceil().max(1.0) as u32, height.ceil().max(1.0) as u32);
                layer.set_anchor(Anchor::Top | Anchor::Left);
                layer.set_margin(y.round() as i32, 0, 0, x.round() as i32);
                return;
            }
            _ => {}
        }
        let at = match self.options.placement {
            Some(OverlayPlacement::Edge { edge, margin, align, inset }) => Placement { size: at.size, edge, margin, align, inset },
            _ => at,
        };
        let (margin, inset) = (at.margin.round() as i32, at.inset.round() as i32);
        let (vertical, top, bottom) = match at.edge {
            ScreenEdge::Top => (Anchor::Top, margin, 0),
            ScreenEdge::Bottom => (Anchor::Bottom, 0, margin),
        };
        let (horizontal, left, right) = match at.align {
            Align::Start => (Anchor::Left, inset, 0),
            Align::Center => (Anchor::empty(), 0, 0),
            Align::End => (Anchor::Right, 0, inset),
        };
        layer.set_size(at.size.0.ceil() as u32, at.size.1.ceil() as u32);
        layer.set_anchor(vertical | horizontal);
        layer.set_margin(top, right, bottom, left);
    }

    /// Takes `options`' shell placement (`OverlayHandle::set_shell`) on the live surface: layer,
    /// anchors, margins, zone and keyboard, sent with the next commit. The output stays.
    pub fn reshape(&mut self, options: &OverlayOptions, at: Placement) {
        self.options.shell = options.shell.clone();
        let keys = match &self.options.shell { Some(ShellSurface::Layer(spec)) => spec.keyboard != LayerKeyboard::None, _ => false };
        let keys = keys || self.options.keys == crate::KeyPolicy::OnDemand;
        if keys != self.state.want_keys {
            self.state.want_keys = keys;
            if keys && self.seat.is_none() {
                self.seat = self.seat_global.map(|(name, version)| self.registry.bind(name, version, &self.queue.handle(), ()));
            }
            if let Some(seat) = &self.seat { self.state.update_seat(seat, &self.queue.handle()); }
        }
        if let (Role::Layer(layer), Some(ShellSurface::Layer(spec))) = (&self.role, &self.options.shell) {
            if layer.version() >= 2 { layer.set_layer(layer_of(spec.layer)); }
            layer.set_keyboard_interactivity(interactivity(spec.keyboard, layer.version()));
            layer.set_exclusive_zone(spec.exclusive_zone);
        }
        self.place(at);
    }

    pub fn configured_size(&self) -> Option<(u32, u32)> { self.state.readiness.size() }

    /// Sets the surface's pending input region; true when it changed, so it waits for a commit.
    pub fn input_regions(&self, regions: &[[f32; 4]], interactive: bool) -> bool {
        let next = match self.options.input.resolved(interactive) {
            OverlayInput::All => None,
            OverlayInput::Regions => Some(regions.iter().filter_map(|r| {
                let (x, y) = (r[0].floor() as i32, r[1].floor() as i32);
                let (w, h) = (r[2].ceil() as i32 - x, r[3].ceil() as i32 - y);
                (w > 0 && h > 0).then_some([x, y, w, h])
            }).collect()),
            _ => Some(Vec::new()),
        };
        if *self.regions.borrow() == next { return false; }
        if let Some(rects) = &next {
            let region = self.compositor.create_region(&self.queue.handle(), ());
            for &[x, y, w, h] in rects { region.add(x, y, w, h); }
            self.surface.set_input_region(Some(&region));
            region.destroy();
        } else {
            self.surface.set_input_region(None);
        }
        *self.regions.borrow_mut() = next;
        // No commit here: a drawn frame's presentation commits input and pixels together, and a
        // frame that draws nothing commits it alone (`Overlay::unchanged_frame`).
        true
    }

    /// The compositor's blur behind the surface, `size` points big (for a region of the whole
    /// surface); sent with the next commit. Nothing where the compositor has no blur.
    pub fn set_blur(&mut self, blur: Option<BlurRegion>, size: (f32, f32)) {
        let Some(effect) = &self.blur else { return };
        let next = blur.map(|blur| {
            let [x, y, w, h] = blur.rect.unwrap_or([0.0, 0.0, size.0, size.1]);
            rounded(x, y, w, h, blur.radius)
        });
        if self.blur_sent == next { return; }
        match &next {
            Some(rects) => {
                let region = self.compositor.create_region(&self.queue.handle(), ());
                for &[x, y, w, h] in rects { region.add(x, y, w, h); }
                effect.set_blur_region(Some(&region));
                region.destroy();
            }
            None => effect.set_blur_region(None),
        }
        self.blur_sent = next;
    }

    pub fn set_input(&mut self, input: OverlayInput) {
        self.options.input = input;
        let lock = matches!(self.role, Role::Lock(_));
        self.state.want_pointer = input != OverlayInput::None || lock;
        if self.seat.is_none() && self.state.want_pointer {
            self.seat = self.seat_global.map(|(name, version)| self.registry.bind(name, version, &self.queue.handle(), ()));
        }
        if let Some(seat) = &self.seat { self.state.update_seat(seat, &self.queue.handle()); }
    }

    pub fn request_keys(&self) -> bool {
        let Role::Layer(layer) = &self.role else { return false };
        if !keyboard::can_request_keys(self.options.keys, layer.version()) { return false; }
        layer.set_keyboard_interactivity(KeyboardInteractivity::OnDemand);
        self.surface.commit();
        self.queue.flush().ok();
        true
    }

    pub fn supports_keys(&self) -> bool {
        match &self.role {
            Role::Lock(_) => true,
            Role::Layer(layer) => keyboard::supports_keys(self.options.keys, layer.version(), self.seat.is_some())
                || matches!(&self.options.shell, Some(ShellSurface::Layer(spec)) if spec.keyboard != LayerKeyboard::None),
        }
    }

    /// Shows `want` while the pointer is over the surface, once per change. Without
    /// cursor-shape-v1 only `Cursor::Hidden` applies; others keep the cursor the pointer
    /// arrived with, such as a terminal's I-beam.
    pub fn set_cursor(&mut self, want: crate::el::Cursor) {
        let (Over::Inside { serial, cursor }, Some(pointer)) = (self.state.over, &self.state.pointer) else { return };
        if cursor == Some(want) { return; }
        match (want, &self.state.shape) {
            (crate::el::Cursor::Hidden, _) => pointer.set_cursor(serial, None, 0, 0),
            (want, Some(device)) => device.set_shape(serial, shape(want)),
            (_, None) => return,
        }
        self.state.over = Over::Inside { serial, cursor: Some(want) };
        self.queue.flush().ok();
    }

    /// Commit role state (placement) without a new buffer; wgpu's present commits buffers.
    pub fn commit(&mut self) {
        self.surface.commit();
        self.queue.flush().ok();
    }

    /// Asks the compositor to say when it has drawn the frame about to be presented. The request
    /// rides the commit wgpu's present makes; no frame is presented before the answer
    /// (`awaits_frame`).
    pub fn ask_frame(&mut self) {
        self.asks += 1;
        self.surface.frame(&self.queue.handle(), self.asks);
        frame::step(&mut self.state.frame, frame::Event::Presents { ask: self.asks, now: crate::time::now() });
    }

    /// When the frame the compositor has yet to draw was presented, as of the last `dispatch`.
    pub fn awaits_frame(&self) -> Option<crate::time::Instant> { self.state.frame.asked() }

    pub fn set_buffer_scale(&mut self, scale: i32) {
        if self.surface.version() >= 3 {
            self.surface.set_buffer_scale(scale);
        }
        self.state.scale = f64::from(scale.max(1));
    }

    /// Handles what the compositor sent since the last call (winit reads the socket).
    pub fn dispatch(&mut self) -> Changes {
        self.turn();
        Changes {
            size: self.state.readiness.take_size(),
            input: std::mem::take(&mut self.state.input),
            scale: self.state.readiness.take_scale(),
            closed: self.state.readiness.closed(),
        }
    }

    /// Handles what the compositor sent to a hidden surface and drops its input, keeping size
    /// and scale for the next `dispatch`; true once the compositor closed it. libwayland frees
    /// an undispatched event without closing its fd, so a queue left alone leaks every
    /// `wl_keyboard.keymap` (each virtual keyboard a paste tool makes sends a few).
    pub fn drain(&mut self) -> bool {
        self.turn();
        self.state.input.clear();
        self.state.readiness.closed()
    }

    fn turn(&mut self) {
        self.queue.dispatch_pending(&mut self.state).ok();
        self.queue.flush().ok();
        match &self.wakes {
            Some(Wakes::Own(watch)) => watch.turned(),
            Some(Wakes::Shared(_)) => super::wayland_shell::turned(),
            None => {}
        }
    }

    pub fn raw_handles(&self) -> (RawDisplayHandle, RawWindowHandle) {
        let surface = NonNull::new(self.surface.id().as_ptr().cast()).expect("live wl_surface");
        (
            RawDisplayHandle::Wayland(WaylandDisplayHandle::new(self.display)),
            RawWindowHandle::Wayland(WaylandWindowHandle::new(surface)),
        )
    }
}

/// A layer surface placed as `spec` says, asking for `size` points (0 stretches an axis it is
/// anchored across).
fn place_shell(layer: &ZwlrLayerSurfaceV1, spec: &LayerShell, size: (f32, f32)) {
    let [top, right, bottom, left] = spec.anchors;
    let mut anchor = Anchor::empty();
    if top { anchor |= Anchor::Top; }
    if right { anchor |= Anchor::Right; }
    if bottom { anchor |= Anchor::Bottom; }
    if left { anchor |= Anchor::Left; }
    layer.set_size(size.0.max(0.0).ceil() as u32, size.1.max(0.0).ceil() as u32);
    layer.set_anchor(anchor);
    let [mt, mr, mb, ml] = spec.margin;
    layer.set_margin(mt, mr, mb, ml);
    layer.set_exclusive_zone(spec.exclusive_zone);
}

fn layer_of(layer: crate::wayland::Layer) -> Layer {
    match layer {
        crate::wayland::Layer::Background => Layer::Background,
        crate::wayland::Layer::Bottom => Layer::Bottom,
        crate::wayland::Layer::Top => Layer::Top,
        crate::wayland::Layer::Overlay => Layer::Overlay,
    }
}

/// The keyboard interactivity `keys` asks for, on a layer surface of `version`: on demand needs
/// version 4, where earlier ones only know exclusive or none.
fn interactivity(keys: LayerKeyboard, version: u32) -> KeyboardInteractivity {
    match keys {
        LayerKeyboard::None => KeyboardInteractivity::None,
        LayerKeyboard::Exclusive => KeyboardInteractivity::Exclusive,
        LayerKeyboard::OnDemand if version >= 4 => KeyboardInteractivity::OnDemand,
        LayerKeyboard::OnDemand => KeyboardInteractivity::None,
    }
}

/// A rounded rect as region rects, in whole points: one per point row through each corner,
/// inset to the curve, and one for the straight middle. A blur region is rects only, and a
/// square one shows past a rounded panel's corners.
fn rounded(x: f32, y: f32, width: f32, height: f32, radius: f32) -> Vec<[i32; 4]> {
    let rect = |x: f32, y: f32, w: f32, h: f32| [x.round() as i32, y.round() as i32, w.round() as i32, h.round() as i32];
    let r = radius.min(width / 2.0).min(height / 2.0).max(0.0).floor();
    if r < 1.0 { return vec![rect(x, y, width, height)]; }
    let rows = r as usize;
    let mut out = Vec::with_capacity(2 * rows + 1);
    for row in 0..rows {
        // How far in the curve is at the middle of this row.
        let dy = r - row as f32 - 0.5;
        let inset = (r - (r * r - dy * dy).max(0.0).sqrt()).round();
        let span = (width - 2.0 * inset).max(0.0);
        out.push(rect(x + inset, y + row as f32, span, 1.0));
        out.push(rect(x + inset, y + height - row as f32 - 1.0, span, 1.0));
    }
    out.push(rect(x, y + r, width, height - 2.0 * r));
    out
}

impl Drop for LayerSurface {
    fn drop(&mut self) {
        if let Some(shape) = self.state.shape.take() { shape.destroy(); }
        if let Some(shapes) = self.state.shapes.take() { shapes.destroy(); }
        if let Some(pointer) = self.state.pointer.take() && pointer.version() >= 3 { pointer.release(); }
        if let Some(touch) = self.state.touch.take() && touch.version() >= 3 { touch.release(); }
        if let Some(keyboard) = self.state.keyboard.take() && keyboard.version() >= 3 { keyboard.release(); }
        if let Some(seat) = self.seat.take() && seat.version() >= 5 { seat.release(); }
        if let Some(output) = self.output.take() && output.version() >= 3 { output.release(); }
        if let Some(blur) = self.blur.take() { blur.destroy(); }
        match &self.role {
            Role::Layer(layer) => layer.destroy(),
            Role::Lock(lock) => lock.destroy(),
        }
        self.surface.destroy();
        self.queue.flush().ok();
    }
}

#[cfg(test)]
#[path = "layer_shell_tests.rs"]
mod tests;
