use super::*;
use crate::el::ScrollSpec;

/// A page of rows of cards, as a media app's: each row a title over a rail that scrolls
/// sideways, each card a gradient plate with initials, a truncated title and a note.
fn page(_: &(), _: &mut Cx<()>) -> El {
    let card = |row: usize, at: usize| {
        col().key(format!("card-{row}-{at}")).w(120.0).gap(3.0).hover_group()
            .child(crate::el().size(120.0, 68.0).radius(6.0).center().fill(ink(0.2, 0.3, 0.2))
                .linear_gradient(2.4, [(0.0, ink(0.4, 0.5, 0.3)), (1.0, ink(0.1, 0.2, 0.1))])
                .child(crate::text(format!("C{at}")).font(crate::TextStyle::new(22.0, 700.0)).color(crate::Color::WHITE)))
            .child(crate::text(format!("Channel {row} {at} with a name too long for its card")).font(crate::TextStyle::new(12.5, 600.0)).color(crate::Color::WHITE).truncate().shrink())
            .child(crate::text(format!("Category {row}")).font(crate::TextStyle::new(11.0, 400.0)).color(crate::Color::WHITE))
    };
    col().key("page").size(640.0, 400.0).scroll(ScrollSpec { stick_bottom: false }).child(
        col().w_full().gap(9.5).padding(13.25).children((0..12).map(|row| {
            col().key(format!("row-{row}")).w_full()
                .child(crate::text(format!("Category {row}")).font(crate::TextStyle::new(17.0, 700.0)).color(crate::Color::WHITE))
                .child(row_el().key("rail").scroll_x().child(row_el().gap(7.0).py(10.0).children((0..9).map(move |at| card(row, at)))))
        })),
    )
}

fn row_el() -> El {
    crate::row()
}

fn ink(r: f32, g: f32, b: f32) -> crate::Color {
    crate::Color { r, g, b, a: 1.0 }
}

/// The nodes of moved subtrees checked against their paint, and the subtrees that differed.
fn checks(rt: &Runtime<()>) -> (usize, usize) {
    rt.paint.moved_checks()
}

/// A debug build paints every subtree it would have copied and compares the two
/// (`keep_check`), so scrolling here checks each copy a moved subtree would have got.
fn scrolled(scale: f32, steps: &[(f32, f32, f32, f32)]) -> (usize, usize) {
    let mut rt = Runtime::new((), page, (640.0, 400.0), Env::new(true, scale));
    rt.frame(0.0);
    rt.frame(1.0 / 120.0);
    for &(x, y, dx, dy) in steps {
        rt.wheel_xy(x, y, dx, dy);
        rt.frame(1.0 / 120.0);
        rt.frame(1.0 / 120.0);
    }
    checks(&rt)
}

/// Scrolling a page moves its rows by whole pixels: what a row painted last frame, moved, is
/// what it paints now, at any display scale, for offsets that land on and off the pixel grid.
#[test]
fn a_scrolled_page_paints_what_its_moved_copy_holds() {
    let steps: Vec<_> = [-7.0, -13.5, -40.25, -3.125, 22.0, -61.75, 9.5, -120.0, 55.0].iter().map(|&dy| (320.0, 200.0, 0.0, dy)).collect();
    for scale in [1.0, 1.25, 1.5, 2.0] {
        let (checked, differed) = scrolled(scale, &steps);
        assert!(checked > 200, "moved subtrees were checked at {scale}x ({checked} nodes)");
        assert_eq!(differed, 0, "a moved copy differs from its paint at {scale}x");
    }
}

/// A rail scrolled sideways moves its cards the same way, under a clip that stays put.
#[test]
fn a_rail_scrolled_sideways_paints_what_its_moved_copy_holds() {
    let steps: Vec<_> = [-9.0, -30.5, -64.25, 12.0, -150.0, 41.75].iter().map(|&dx| (320.0, 120.0, dx, 0.0)).collect();
    for scale in [1.0, 1.25, 2.0] {
        let (checked, differed) = scrolled(scale, &steps);
        assert!(checked > 20, "moved cards were checked at {scale}x ({checked} nodes)");
        assert_eq!(differed, 0, "a moved copy differs from its paint at {scale}x");
    }
}

/// Both at once, and back again: rows leave and come back in, cards cross the rail's edge.
#[test]
fn a_page_and_a_rail_scrolled_together_paint_what_their_moved_copies_hold() {
    let mut steps = Vec::new();
    for k in 0..24 {
        let dy = if k % 8 < 5 { -37.5 } else { 44.25 };
        steps.push((320.0, 200.0, 0.0, dy));
        steps.push((320.0, 150.0, if k % 3 == 0 { -53.0 } else { 17.5 }, 0.0));
    }
    let (checked, differed) = scrolled(1.0, &steps);
    assert!(checked > 500, "moved subtrees were checked ({checked} nodes)");
    assert_eq!(differed, 0, "a moved copy differs from its paint");
}

const PROSE: &str = "Amber lantern orchard quiet harbor, copper meadow signal drift. Pebble willow \
    ember tide under a juniper kestrel; nimbus opal prism and a quiet harbor beneath the lantern. ";

/// A document: paragraphs that wrap over many lines (some taller than the window), bordered
/// and shadowed panels, a dimmed panel, and a list that scrolls inside the page.
fn document(_: &(), _: &mut Cx<()>) -> El {
    let white = crate::Color::WHITE;
    let para = |i: usize| crate::text(format!("{i}. {}", PROSE.repeat(1 + i % 7))).key(format!("p{i}")).font(crate::TextStyle::new(13.0, 400.0)).color(white).w_full().wrap();
    let panel = |i: usize| {
        col().key(format!("panel{i}")).w_full().padding(11.5).radius(8.0).fill(ink(0.15, 0.15, 0.2)).border(1.0, ink(0.4, 0.4, 0.5))
            .shadow(6.0, 14.0, crate::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.6 })
            .child(crate::text(format!("Panel {i}")).font(crate::TextStyle::new(15.0, 700.0)).color(white))
            .child(crate::text(PROSE).font(crate::TextStyle::new(12.0, 400.0)).color(white).truncate().shrink())
    };
    let inner = col().key("inner").w_full().h(120.0).scroll(ScrollSpec { stick_bottom: false })
        .children((0..30).map(|i| crate::text(format!("Inner line {i}")).key(format!("in{i}")).font(crate::TextStyle::new(12.0, 400.0)).color(white)));
    col().key("doc").size(480.0, 360.0).scroll(ScrollSpec { stick_bottom: false }).child(
        col().w_full().gap(8.25).padding(10.0).children((0..40).map(|i| match i % 5 {
            1 => panel(i),
            3 if i == 13 => inner.clone(),
            4 => panel(i).layer_opacity(0.6),
            _ => para(i),
        })),
    )
}

fn scrolled_document(scale: f32, steps: &[(f32, f32, f32, f32)]) -> (usize, usize) {
    let mut rt = Runtime::new((), document, (480.0, 360.0), Env::new(true, scale));
    rt.frame(0.0);
    rt.frame(1.0 / 120.0);
    for &(x, y, dx, dy) in steps {
        rt.wheel_xy(x, y, dx, dy);
        rt.frame(1.0 / 120.0);
        rt.frame(1.0 / 120.0);
    }
    rt.paint.moved_checks()
}

/// A small generator, so a run of offsets is the same every time.
fn offsets(seed: u32, count: usize, reach: f32) -> Vec<f32> {
    let mut x = seed;
    (0..count).map(|_| {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Sixty-fourths of a point, as a trackpad's deltas land anywhere.
        ((x >> 8) % (2 * (reach * 64.0) as u32)) as f32 / 64.0 - reach
    }).collect()
}

/// Long texts are drawn a band of lines at a time, panels carry borders and shadows, and one
/// list scrolls inside the page: whatever a moved copy would hold is what paints, through a
/// long run of offsets down and back up.
#[test]
fn a_scrolled_document_paints_what_its_moved_copies_hold() {
    for (scale, seed) in [(1.0, 7), (1.25, 11), (2.0, 13)] {
        let mut steps: Vec<_> = offsets(seed, 120, 90.0).into_iter().map(|dy| (240.0, 300.0, 0.0, dy - 20.0)).collect();
        // Over the inner list, which takes the wheel until it runs out.
        steps.extend(offsets(seed + 1, 60, 40.0).into_iter().map(|dy| (240.0, 180.0, 0.0, dy)));
        steps.extend(offsets(seed + 2, 120, 90.0).into_iter().map(|dy| (240.0, 300.0, 0.0, dy + 20.0)));
        let (checked, differed) = scrolled_document(scale, &steps);
        assert!(checked > 300, "moved subtrees were checked at {scale}x ({checked} nodes)");
        assert_eq!(differed, 0, "a moved copy differs from its paint at {scale}x");
    }
}

/// The page of cards through a long run of offsets on both axes.
#[test]
fn a_page_of_cards_paints_what_its_moved_copies_hold_through_many_offsets() {
    for (scale, seed) in [(1.0, 3), (1.5, 5)] {
        let mut steps = Vec::new();
        for (k, (dy, dx)) in offsets(seed, 200, 70.0).into_iter().zip(offsets(seed + 9, 200, 50.0)).enumerate() {
            steps.push((320.0, 200.0, 0.0, dy - if k < 100 { 15.0 } else { -15.0 }));
            if k % 4 == 0 { steps.push((320.0, 60.0 + (k % 5) as f32 * 70.0, dx, 0.0)); }
        }
        let (checked, differed) = scrolled(scale, &steps);
        assert!(checked > 2000, "moved subtrees were checked at {scale}x ({checked} nodes)");
        assert_eq!(differed, 0, "a moved copy differs from its paint at {scale}x");
    }
}

/// A wall of posters: each card a picture (some filling their box, some fitted inside it)
/// over a title. One picture has no pixels yet.
fn posters(pictures: &Vec<crate::Pixels>, _: &mut Cx<Vec<crate::Pixels>>) -> El {
    let white = crate::Color::WHITE;
    col().key("wall").size(520.0, 380.0).scroll(ScrollSpec { stick_bottom: false }).child(
        col().w_full().gap(10.5).padding(9.25).children((0..10).map(|row| {
            row_el().key(format!("row-{row}")).gap(8.0).children((0..5).map(|at| {
                let picture = &pictures[(row * 5 + at) % pictures.len()];
                let fit = if at % 2 == 0 { crate::SurfaceFit::Fill } else { crate::SurfaceFit::Fit };
                col().key(format!("card-{row}-{at}")).w(92.0).gap(3.0)
                    .child(crate::el().size(92.0, 130.0).radius(5.0).center().fill(ink(0.2, 0.2, 0.3))
                        .child(crate::el().w_full().h_full().radius(5.0).pixels_fit(picture, fit)))
                    .child(crate::text(format!("Title {row} {at}")).font(crate::TextStyle::new(12.0, 600.0)).color(white).truncate().shrink())
            }))
        })),
    )
}

fn picture(w: u32, h: u32, shade: u8) -> crate::Pixels {
    let px = crate::Pixels::new();
    px.set(w, h, (0..w * h).flat_map(|k| [shade, (k % 251) as u8, 90, 255]).collect());
    px
}

/// Cards with pictures move as the rest do, while the image atlas holds what they drew. A
/// picture that arrives, or is replaced by one of another size, is painted, not copied.
#[test]
fn scrolled_pictures_paint_what_their_moved_copies_hold() {
    for scale in [1.0, 1.5, 2.0] {
        let late = crate::Pixels::new();
        let pictures = vec![picture(60, 90, 40), picture(120, 80, 90), late.clone(), picture(48, 48, 160), picture(200, 300, 220)];
        let swapped = pictures[1].clone();
        let mut rt = Runtime::new(pictures, posters, (520.0, 380.0), Env::new(true, scale));
        rt.frame(0.0);
        rt.frame(1.0 / 120.0);
        for (k, dy) in offsets(21, 160, 60.0).into_iter().enumerate() {
            // Part way through, the missing picture arrives and another changes shape.
            if k == 50 { late.set(64, 64, vec![200; 64 * 64 * 4]); }
            if k == 90 { swapped.set(70, 140, vec![120; 70 * 140 * 4]); }
            rt.wheel_xy(260.0, 190.0, 0.0, dy - if k < 80 { 12.0 } else { -12.0 });
            rt.frame(1.0 / 120.0);
            rt.frame(1.0 / 120.0);
        }
        let (checked, differed) = rt.paint.moved_checks();
        assert!(checked > 1000, "moved subtrees were checked at {scale}x ({checked} nodes)");
        assert_eq!(differed, 0, "a moved copy differs from its paint at {scale}x");
    }
}

/// Sections under headers that stick to the top of the page as it scrolls.
fn sections(_: &(), _: &mut Cx<()>) -> El {
    let white = crate::Color::WHITE;
    col().key("list").size(420.0, 320.0).scroll(ScrollSpec { stick_bottom: false }).children((0..8).map(|section| {
        col().key(format!("section-{section}")).w_full()
            .child(crate::row().key("header").w_full().h(28.0).padding(6.0).fill(ink(0.1, 0.1, 0.15)).sticky_top(0.0)
                .child(crate::text(format!("Section {section}")).font(crate::TextStyle::new(13.0, 700.0)).color(white)))
            .children((0..6).map(move |at| {
                crate::row().key(format!("item-{at}")).w_full().gap(8.0).padding(7.5)
                    .child(crate::el().size(24.0, 24.0).radius(12.0).fill(ink(0.3, 0.5, 0.4)))
                    .child(crate::text(format!("Item {section}.{at} with a long enough name to be cut off at the edge")).font(crate::TextStyle::new(12.5, 400.0)).color(white).truncate().shrink())
            }))
    }))
}

/// Under sticky headers, rows still move as one with the scroll: where each may stick moves
/// with it, and a header's own section paints.
#[test]
fn rows_under_sticky_headers_paint_what_their_moved_copies_hold() {
    for (scale, seed) in [(1.0, 31), (1.25, 37), (2.0, 41)] {
        let mut rt = Runtime::new((), sections, (420.0, 320.0), Env::new(true, scale));
        rt.frame(0.0);
        rt.frame(1.0 / 120.0);
        for (k, dy) in offsets(seed, 240, 55.0).into_iter().enumerate() {
            rt.wheel_xy(210.0, 160.0, 0.0, dy - if k < 120 { 14.0 } else { -14.0 });
            rt.frame(1.0 / 120.0);
            rt.frame(1.0 / 120.0);
        }
        let (checked, differed) = rt.paint.moved_checks();
        assert!(checked > 1000, "moved subtrees were checked at {scale}x ({checked} nodes)");
        assert_eq!(differed, 0, "a moved copy differs from its paint at {scale}x");
    }
}

/// Whether two paints drew the same, but for a moved copy's rects and clips, which may differ
/// from a fresh paint's in the last bit (`keep_check_moved`).
fn same_paint(a: &[crate::render::Instance], b: &[crate::render::Instance]) -> bool {
    let near = |x: [f32; 4], y: [f32; 4]| (0..4).all(|k| x[k] == y[k] || (x[k] - y[k]).abs() < 1e-3);
    let rest = |q: &crate::render::Instance| crate::render::Instance { rect: [0.0; 4], clip: [0.0; 4], ..*q };
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| near(x.rect, y.rect) && near(x.clip, y.clip) && bytemuck::bytes_of(&rest(x)) == bytemuck::bytes_of(&rest(y)))
}

/// A page scrolled down, then straight back as a card in its first row takes the pointer:
/// copied where the scroll took them, the rows are then copied again from those copies, each
/// by what it inherited there. Copying as a release build does, the page paints what a runtime
/// painting all of it does, frame by frame.
#[test]
fn a_page_scrolled_down_and_back_copies_its_rows_from_where_they_were() {
    for scale in [1.0, 1.25, 2.0] {
        let mut copied = Runtime::new((), page, (640.0, 400.0), Env::new(true, scale));
        copied.paint.copy_for_real();
        let mut painted = Runtime::new((), page, (640.0, 400.0), Env::new(true, scale));
        let mut differed = Vec::new();
        for f in 0..16 {
            for rt in [&mut copied, &mut painted] {
                match f {
                    2 => rt.wheel_xy(320.0, 200.0, 0.0, -40.0),
                    3 => { rt.wheel_xy(320.0, 200.0, 0.0, 40.0); rt.pointer_move(70.0, 80.0); }
                    _ => {}
                }
                rt.frame(if f == 0 { 0.0 } else { 1.0 / 120.0 });
            }
            if !same_paint(&copied.instances, &painted.instances) { differed.push(f); }
        }
        assert!(differed.is_empty(), "frames painted otherwise than a full paint at {scale}x: {differed:?}");
    }
}
