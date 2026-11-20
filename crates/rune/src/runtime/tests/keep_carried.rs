use super::*;

const DT: f64 = 1.0 / 120.0;

/// A panel that grows taller when `s.0`, holding rows that each hold a plate; row 5's is
/// lighter when `s.1`.
fn view(s: &(bool, bool), _: &mut Cx<(bool, bool)>) -> El {
    let shade = |k: usize| if k == 5 && s.1 { 0.8 } else { 0.2 };
    let rows = (0..10).map(|k| crate::row().key(format!("r{k}")).w_full().h(20.0).gap(8.0)
        .child(crate::row().key(format!("b{k}")).size(60.0, 16.0).radius(4.0).fill(Color { r: shade(k), g: shade(k), b: shade(k), a: 1.0 })
            .child(crate::text(format!("Row {k}")).font(crate::TextStyle::new(12.0, 500.0)).color(Color::WHITE)))
        .child(crate::text(format!("Note {k}")).font(crate::TextStyle::new(11.0, 400.0)).color(Color::WHITE)));
    let panel = col().key("panel").w_full().h(if s.0 { 380.0 } else { 220.0 }).fill(Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 });
    col().size(200.0, 400.0).child(panel.children(rows))
}

/// While the panel grows, the scene presents everything in it again each frame, though only the
/// panel changed: what it holds is as it was and is copied, and a debug build paints each copy
/// and finds what it holds. A plate crossfading to another colour at the same time is presented
/// for itself, with the values it had (how far it has faded isn't one of them), and paints.
#[test]
fn what_a_growing_panel_holds_is_as_it_was_but_for_a_plate_changing_colour() {
    for scale in [1.0, 1.25, 2.0] {
        let mut rt = Runtime::new((false, false), view, (200.0, 400.0), Env::new(true, scale));
        rt.frame(0.0);
        rt.frame(DT);
        rt.update(|s, _| s.0 = true);
        let (mut grew, mut lit) = (0, 0);
        let height = |rt: &Runtime<(bool, bool)>| rt.elements().into_iter().find(|e| e.key.as_deref() == Some("panel")).unwrap().rect[3];
        let mut h = height(&rt);
        for f in 0..60 {
            if f == 4 { rt.update(|s, _| s.1 = true); }
            let before = rt.instances.clone();
            rt.frame(DT);
            let now = height(&rt);
            grew += (now != h) as usize;
            // The panel's own box is one instance: another changing is the plate's.
            let changed = rt.instances.iter().zip(&before).filter(|(a, b)| bytemuck::bytes_of(*a) != bytemuck::bytes_of(*b)).count();
            lit += (now != h && changed > 1) as usize;
            h = now;
        }
        let (still, differed) = rt.paint.carried_checks();
        assert!(grew > 10, "the panel grew over the frames at {scale}x ({grew})");
        assert!(lit > 4, "the plate crossfaded while the panel grew at {scale}x ({lit})");
        assert!(still > 200, "what was presented only for the panel was found as it was at {scale}x ({still})");
        assert_eq!(differed, 0, "a copy in place differs from its paint at {scale}x");
    }
}

/// A panel shown in full when `s.0`, and else faint or (`s.1`) not at all, holding rows of a
/// pressable plate, a note and a mark shown only under the pointer, and a row so faint (0.002)
/// that it draws only once the panel is past half.
fn faded(s: &(bool, bool), _: &mut Cx<(bool, bool)>) -> El {
    let rows = (0..10).map(|k| crate::row().key(format!("r{k}")).w_full().h(20.0).gap(8.0).hover_group()
        .child(crate::row().key(format!("b{k}")).size(60.0, 16.0).radius(4.0).fill(Color { r: 0.2, g: 0.2, b: 0.2, a: 1.0 }).on_press(|_: &mut (bool, bool)| {})
            .child(crate::text(format!("Row {k}")).font(crate::TextStyle::new(12.0, 500.0)).color(Color::WHITE)))
        .child(crate::text(format!("Note {k}")).font(crate::TextStyle::new(11.0, 400.0)).color(Color::WHITE))
        .child(el().key(format!("a{k}")).size(10.0, 10.0).fill(Color::WHITE).group_opacity(0.0, 1.0)));
    let panel = col().key("panel").w_full().fill(Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }).opacity(if s.0 { 1.0 } else if s.1 { 0.0 } else { 0.25 });
    let faint = crate::row().key("faint").w_full().h(20.0).opacity(0.002)
        .child(el().key("faint-plate").size(60.0, 16.0).fill(Color::WHITE));
    col().size(200.0, 400.0).child(panel.children(rows).child(faint))
}

/// A panel fading in draws what it holds at full strength into a layer, and fades the layer:
/// each frame of the fade, what it holds inherits a new opacity and draws what it drew. Those
/// are copied, with the opacity each hands down worked out again, and a debug build paints each
/// copy and finds what it holds, hits and hidden marks too. What it holds starts to draw and to
/// take hits as it passes a thousandth, from nothing or for opacity of its own, and paints then.
#[test]
fn what_a_fading_panel_holds_is_copied_with_the_opacity_it_hands_down() {
    for (scale, hidden) in [(1.0, false), (1.25, false), (2.0, false), (1.0, true), (2.0, true)] {
        let mut rt = Runtime::new((false, hidden), faded, (200.0, 400.0), Env::new(true, scale));
        rt.frame(0.0);
        rt.frame(DT);
        rt.pointer_move(30.0, 110.0);
        rt.frame(DT);
        rt.update(|s, _| s.0 = true);
        let opacity = |rt: &Runtime<(bool, bool)>| rt.elements().into_iter().find(|e| e.key.as_deref() == Some("b3")).unwrap().opacity;
        let (mut fading, mut last) = (0, opacity(&rt));
        for _ in 0..90 {
            rt.frame(DT);
            let now = opacity(&rt);
            fading += (now != last) as usize;
            last = now;
        }
        let (checked, differed) = rt.paint.fade_checks();
        assert!(fading > 10 && last == 1.0, "the panel faded in over the frames at {scale}x ({fading} to {last})");
        assert!(checked > 200, "what the panel holds was checked as copies under its fade at {scale}x ({checked})");
        assert_eq!(differed, 0, "a copy under the fade differs from its paint at {scale}x");
        assert_eq!(rt.paint.carried_checks().1, 0, "a copy differs from its paint at {scale}x");
    }
}

/// A card shown faint, or in full when `s`, holding one dot and nothing else.
fn folded(full: &bool, _: &mut Cx<bool>) -> El {
    col().size(200.0, 100.0)
        .child(el().key("card").size(80.0, 40.0).opacity(if *full { 1.0 } else { 0.25 })
            .child(el().key("dot").size(10.0, 10.0).fill(Color::WHITE)))
}

/// A fading card holding one thing folds its layer into that thing, which draws at the card's
/// opacity rather than at full strength: the dot, inheriting a new opacity each frame of the
/// fade, isn't copied as it was, but painted.
#[test]
fn a_fading_card_folded_into_its_one_dot_paints_the_dot() {
    for scale in [1.0, 2.0] {
        let mut rt = Runtime::new(false, folded, (200.0, 100.0), Env::new(true, scale));
        rt.frame(0.0);
        rt.frame(DT);
        rt.update(|full, _| *full = true);
        let opacity = |rt: &Runtime<bool>| rt.elements().into_iter().find(|e| e.key.as_deref() == Some("dot")).unwrap().opacity;
        let (mut fading, mut last) = (0, opacity(&rt));
        for _ in 0..60 {
            rt.frame(DT);
            // One primitive, and no layer around it: the card folded into the dot.
            assert_eq!(rt.instances.len(), 1, "at {scale}x");
            let now = opacity(&rt);
            fading += (now != last) as usize;
            last = now;
        }
        assert!(fading > 10, "the card faded in at {scale}x ({fading})");
        assert_eq!(rt.paint.carried_checks().1, 0, "a copy differs from its paint at {scale}x");
    }
}

/// Copying as a release build does, from copies made under the fade as well, the fading panel
/// paints what a runtime painting all of it does, frame by frame, as the pointer crosses it:
/// what it holds hands down the opacities its copies worked out, which its elements report and
/// what paints under them draws and takes hits by.
#[test]
fn a_fading_panel_copied_for_real_paints_what_a_full_paint_does() {
    for (scale, hidden) in [(1.0, false), (1.25, false), (2.0, false), (1.0, true), (2.0, true)] {
        let mut copied = Runtime::new((false, hidden), faded, (200.0, 400.0), Env::new(true, scale));
        copied.paint.copy_for_real();
        let mut painted = Runtime::new((false, hidden), faded, (200.0, 400.0), Env::new(true, scale));
        let mut differed = Vec::new();
        for f in 0..90 {
            for rt in [&mut copied, &mut painted] {
                match f {
                    2 => rt.update(|s, _| s.0 = true),
                    8 => rt.pointer_move(30.0, 110.0),
                    20 => rt.pointer_move(30.0, 150.0),
                    _ => {}
                }
                rt.frame(if f == 0 { 0.0 } else { DT });
            }
            let opacities = |rt: &Runtime<(bool, bool)>| rt.elements().into_iter().map(|e| (e.key, e.opacity.to_bits())).collect::<Vec<_>>();
            if !super::keep_shift::same_paint(&copied.instances, &painted.instances) || opacities(&copied) != opacities(&painted) { differed.push(f); }
        }
        assert!(copied.paint.fade_copies() > 200, "what the panel holds was copied under its fade at {scale}x ({})", copied.paint.fade_copies());
        assert!(differed.is_empty(), "frames painted otherwise than a full paint at {scale}x (from nothing: {hidden}): {differed:?}");
    }
}
