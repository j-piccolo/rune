use super::*;
use rune_motion::Transition;

const DT: f64 = 1.0 / 120.0;

/// A label that enters, once shown, with a fade (0), a slide (1) or a scale (2).
fn entering(s: &(bool, u8), _: &mut Cx<(bool, u8)>) -> El {
    let t = match s.1 { 0 => Transition::opacity(), 1 => Transition::offset(0.0, 12.0), _ => Transition::scale(0.9) };
    col().size(200.0, 100.0).children(s.0.then(|| {
        crate::text("Entering label").font(crate::TextStyle::new(14.0, 500.0)).color(Color::WHITE).key("label").transition(t)
    }))
}

/// The glyphs drawn this frame, and those drawn from their outlines rather than the glyph atlas
/// (an outline glyph holds its pen in `radii[3]`).
fn glyphs(rt: &Runtime<(bool, u8)>) -> (usize, usize) {
    let glyph = rt.instances.iter().filter(|q| q.params[1] == crate::render::KIND_GLYPH);
    (glyph.clone().count(), glyph.filter(|q| q.radii[3] > 0.0).count())
}

/// Text entering with a fade or a slide scales nothing, and draws from the glyph atlas as it
/// does at rest, so nothing about it changes on the frame it lands. Text entering with a scale
/// draws from its outlines while it scales; under reduced motion, where an entrance only fades,
/// from the atlas again.
#[test]
fn entering_text_draws_from_its_outlines_only_while_it_scales() {
    for (how, reduced, outlines) in [(0, false, false), (1, false, false), (2, false, true), (2, true, false)] {
        let mut rt = Runtime::new((false, how), entering, (200.0, 100.0), Env::new(true, 1.0));
        rt.set_system_reduced_motion(reduced);
        rt.frame(0.0);
        rt.update(|s, _| s.0 = true);
        let (mut drawn, mut outlined, mut moving) = (0, 0, 0);
        for _ in 0..90 {
            rt.frame(DT);
            let (all, from_outlines) = glyphs(&rt);
            (drawn, outlined, moving) = (drawn + all, outlined + from_outlines, moving + rt.animating as usize);
        }
        assert!(moving > 5 && drawn > 0, "entrance {how} (reduced motion: {reduced}) ran ({moving} frames) and drew text");
        assert_eq!(outlined > 0, outlines, "entrance {how} (reduced motion: {reduced}): {outlined} of {drawn} glyphs drawn from outlines");
        assert_eq!(glyphs(&rt).1, 0, "entrance {how} (reduced motion: {reduced}) rests on the atlas");
    }
}
