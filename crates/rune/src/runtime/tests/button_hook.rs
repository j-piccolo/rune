//! `ConfiguredOverlay::on_button`'s hook hears a press before the view does and a release after
//! the view ran what its click queued, so a shell's click-away can tell whether anything in the
//! view took the click, however many clicks wait for the next frame.
use super::*;
use crate::{Cx, El, Env, row};

#[derive(Default)]
struct Clicks {
    heard: Vec<&'static str>,
}

/// A button whose handler runs on the release, and one whose handler runs on the way down.
fn view(_: &Clicks, _: &mut Cx<Clicks>) -> El {
    row()
        .size(200.0, 40.0)
        .child(row().key("up").size(80.0, 40.0).on_press(|s: &mut Clicks| s.heard.push("up handler")))
        .child(row().key("down").size(80.0, 40.0).press_on_down().on_press(|s: &mut Clicks| s.heard.push("down handler")))
}

fn hook(s: &mut Clicks, down: bool) {
    s.heard.push(if down { "hook press" } else { "hook release" });
}

fn runtime() -> Runtime<Clicks> {
    let mut rt = Runtime::new(Clicks::default(), view, (200.0, 40.0), Env::new(true, 1.0));
    rt.hooks.on_button = Some(hook);
    rt.frame(0.0);
    rt
}

fn centre(rt: &Runtime<Clicks>, key: &str) -> (f32, f32) {
    let r = rt.elements().into_iter().find(|e| e.key.as_deref() == Some(key)).expect("the button").rect;
    (r[0] + r[2] / 2.0, r[1] + r[3] / 2.0)
}

/// A click as an overlay host dispatches it: the hook around the view's own press and release.
fn click(rt: &mut Runtime<Clicks>, (x, y): (f32, f32)) {
    let (core, mut app) = rt.split_mut();
    core.button_hook(&mut app, true);
    core.pointer_down(&mut app, x, y);
    core.pointer_up(&app, x, y);
    core.button_hook(&mut app, false);
}

#[test]
fn a_release_reaches_the_hook_after_the_handlers_of_its_click() {
    for (key, handler) in [("up", "up handler"), ("down", "down handler")] {
        let mut rt = runtime();
        let at = centre(&rt, key);
        click(&mut rt, at);
        assert_eq!(rt.state.heard, ["hook press"], "a press reaches the hook before the view hears it");
        rt.frame(1.0 / 60.0);
        assert_eq!(rt.state.heard, ["hook press", handler, "hook release"]);
    }
}

#[test]
fn a_release_on_nothing_reaches_the_hook_with_the_next_frame() {
    let mut rt = runtime();
    click(&mut rt, (190.0, 20.0));
    rt.frame(1.0 / 60.0);
    assert_eq!(rt.state.heard, ["hook press", "hook release"]);
}

#[test]
fn clicks_waiting_for_one_frame_reach_the_hook_whole_and_in_order() {
    let mut rt = runtime();
    let (up, down) = (centre(&rt, "up"), centre(&rt, "down"));
    click(&mut rt, up);
    click(&mut rt, down);
    assert_eq!(rt.state.heard, ["hook press"], "the second press waits behind the first release");
    rt.frame(1.0 / 60.0);
    assert_eq!(
        rt.state.heard,
        ["hook press", "up handler", "hook release", "hook press", "down handler", "hook release"]
    );
}

#[test]
fn a_runtime_without_the_hook_queues_nothing_for_it() {
    let mut rt = Runtime::new(Clicks::default(), view, (200.0, 40.0), Env::new(true, 1.0));
    rt.frame(0.0);
    let (core, mut app) = rt.split_mut();
    core.button_hook(&mut app, true);
    core.button_hook(&mut app, false);
    assert!(!rt.core().keys_queued(), "no button waits for a hook that isn't there");
}
