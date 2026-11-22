//! Layout feedback gone quiet reopens when anything else commits, a commit landed before the
//! frame's input runs (the tree a key routes through, a host settle) among them: `on_layout`
//! then hears what that commit laid out, as a sheet an update opens hears its own size.
use super::*;
use crate::{Cx, El, Env, Key, KeyEvent, Mods, col, el};

#[derive(Default)]
struct S {
    /// What `echo`'s own `on_layout` grew it by, until feedback fell quiet.
    grow: u32,
    panel: bool,
    heard: Vec<[f32; 4]>,
}

fn view(s: &S, _: &mut Cx<S>) -> El {
    col()
        .size(300.0, 300.0)
        .child(el().key("echo").size(10.0 + s.grow as f32, 10.0).on_layout(|s: &mut S, _| s.grow += 1))
        .children(s.panel.then(|| el().key("panel").size(120.0, 80.0).on_layout(|s: &mut S, r| s.heard.push(r))))
}

/// A runtime whose layout feedback fell quiet: `echo` grew itself until it did.
fn quiet() -> Runtime<S> {
    let mut rt = Runtime::new(S::default(), view, (300.0, 300.0), Env::new(true, 1.0));
    for _ in 0..20 {
        rt.frame(1.0 / 60.0);
        if !rt.core().layout_feedback_open() { return rt; }
    }
    panic!("layout feedback never fell quiet");
}

/// An update mounts the panel, and a key comes with it.
fn open_panel_with_a_key(rt: &mut Runtime<S>) {
    rt.update_if(|s, _| {
        s.panel = true;
        true
    });
    rt.key(KeyEvent { key: Key::Left, mods: Mods::default() });
}

#[test]
fn an_update_a_key_routes_through_reopens_layout_feedback() {
    let mut rt = quiet();
    open_panel_with_a_key(&mut rt);
    rt.frame(1.0 / 60.0);
    rt.frame(1.0 / 60.0);
    assert!(rt.core().layout_feedback_open());
    assert_eq!(rt.state.heard.len(), 1, "the panel the update mounted hears its layout");
}

#[test]
fn an_update_a_host_settle_lands_reopens_layout_feedback() {
    let mut rt = quiet();
    open_panel_with_a_key(&mut rt);
    rt.settle_all();
    rt.frame(1.0 / 60.0);
    rt.frame(1.0 / 60.0);
    assert!(rt.core().layout_feedback_open());
    assert_eq!(rt.state.heard.len(), 1, "the panel the update mounted hears its layout");
}
