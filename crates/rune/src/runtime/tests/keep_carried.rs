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
