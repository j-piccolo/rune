//! A virtual list whose window shifts: the rows that stay mounted keep their place in the
//! scene, and are copied rather than painted again when the window moved by whole device
//! pixels; a row leaving as it shifts fades where it was.
use super::keep_shift::same_paint;
use super::*;
use crate::el::GrowthSpec;
use crate::{ScrollSpec, row, text};
use std::borrow::Cow;

fn ink(r: f32, g: f32, b: f32) -> crate::Color {
    crate::Color { r, g, b, a: 1.0 }
}

fn card(row: usize, at: usize) -> El {
    col().key(format!("card-{at}")).w(110.0).gap(3.0)
        .child(el().size(110.0, 62.0).radius(6.0).fill(ink(0.2, 0.3, 0.2)))
        .child(text(format!("Title {row}.{at} with a name too long for its card")).font(crate::TextStyle::new(12.5, 600.0)).color(crate::Color::WHITE).truncate().shrink())
        .child(text(format!("Note {at}")).font(crate::TextStyle::new(11.0, 400.0)).color(crate::Color::WHITE))
}

/// A media page as a virtual list: forty rows `height` tall and 9 pt apart, each a title over
/// a rail of cards, of which a window of a few is mounted.
fn shelf(_: &f32, _: &mut Cx<f32>) -> El {
    col().key("page").size(640.0, 400.0).scroll(ScrollSpec { stick_bottom: false }).child(
        col().key("rows").w_full().gap(9.0).padding(12.0).growth(GrowthSpec::default()).virtual_rows(
            40,
            120.0,
            |_: &f32, i| Cow::Owned(format!("row-{i}")),
            |height: &f32, r| {
                col().w_full().h(*height).gap(6.0)
                    .child(text(format!("Category {r}")).font(crate::TextStyle::new(17.0, 700.0)).color(crate::Color::WHITE))
                    .child(row().gap(7.0).children((0..6).map(|at| card(r, at))))
            },
        ),
    )
}

/// The rows mounted now: each row's key and how many nodes it holds.
fn mounted(rt: &Runtime<f32>) -> Vec<(String, usize)> {
    let nodes = &rt.tree.nodes;
    let list = rt.tree.node_index[&rt.tree.path_ids["page/rows"]];
    let inside = |mut i: usize, top: usize| loop {
        match nodes[i].parent {
            Some(p) if p == top => return true,
            Some(p) if p > top => i = p,
            _ => return false,
        }
    };
    (list + 1..nodes.len())
        .filter(|&i| nodes[i].parent == Some(list))
        .map(|r| (nodes[r].key.clone().unwrap_or_default(), 1 + (r + 1..nodes.len()).filter(|&j| inside(j, r)).count()))
        .collect()
}

/// Scrolling the page shifts its window: rows leave at one end and enter at the other. With
/// rows 128 pt apart, the window moves by whole pixels at 1x, 1.25x and 2x, and the rows that
/// stay are copied where the scroll took them. What paints is the page, the list, the rows that
/// entered, and what the scroll took across the window's top or bottom edge, which a moved copy
/// gives back to paint: a small part of what stayed.
#[test]
fn rows_that_stay_as_a_virtual_window_shifts_are_copied() {
    for scale in [1.0, 1.25, 2.0] {
        let mut rt = Runtime::new(119.0f32, shelf, (640.0, 400.0), Env::new(true, scale));
        rt.frame(0.0);
        rt.frame(1.0 / 120.0);
        let mut was = mounted(&rt);
        let mut shifts = 0;
        for k in 0..70 {
            rt.wheel_xy(320.0, 200.0, 0.0, if k < 40 { -45.5 } else { 57.25 });
            rt.frame(1.0 / 120.0);
            let now = mounted(&rt);
            if now.iter().map(|r| &r.0).ne(was.iter().map(|r| &r.0)) {
                shifts += 1;
                let entered: usize = now.iter().filter(|r| !was.iter().any(|w| w.0 == r.0)).map(|r| r.1).sum();
                let (n, painted) = (rt.tree.nodes.len(), rt.tree.nodes.len() - rt.paint.kept());
                assert!(painted - entered <= (n - entered) / 4, "at {scale}x, step {k}: {painted} of {n} nodes painted, {entered} in rows that entered");
            }
            rt.frame(1.0 / 120.0);
            was = mounted(&rt);
        }
        assert!(shifts > 10, "the window shifted at {scale}x ({shifts} times)");
        assert_eq!(rt.paint.moved_checks().1, 0, "a moved copy differs from its paint at {scale}x");
        assert_eq!(rt.paint.carried_checks().1, 0, "a copy in place differs from its paint at {scale}x");
    }
}

/// Copying as a release build does, from copies made under earlier windows, the page paints
/// what a runtime painting all of it does, frame by frame, down through the rows and back. Rows
/// 126.5 pt apart move the window by half a pixel at an odd count of them, where the rows that
/// stay round otherwise and are painted again, and by whole pixels at an even count.
#[test]
fn a_virtual_list_scrolled_down_and_back_paints_what_a_full_paint_does() {
    for height in [119.0f32, 117.5] {
        for scale in [1.0, 1.25, 2.0] {
            let mut copied = Runtime::new(height, shelf, (640.0, 400.0), Env::new(true, scale));
            copied.paint.copy_for_real();
            let mut painted = Runtime::new(height, shelf, (640.0, 400.0), Env::new(true, scale));
            let mut differed = Vec::new();
            for f in 0..90 {
                for rt in [&mut copied, &mut painted] {
                    if (2..80).contains(&f) && f % 2 == 0 {
                        rt.wheel_xy(320.0, 200.0, 0.0, if f < 44 { -61.25 } else { 64.5 });
                    }
                    rt.frame(if f == 0 { 0.0 } else { 1.0 / 120.0 });
                }
                if !same_paint(&copied.instances, &painted.instances) { differed.push(f); }
            }
            assert!(differed.is_empty(), "frames painted otherwise than a full paint for rows {height} pt at {scale}x: {differed:?}");
        }
    }
}

/// Eighty rows 68 pt apart, scrolled through, with a press that removes two of them.
struct Rows(Vec<u32>);

fn removable(s: &Rows, _: &mut Cx<Rows>) -> El {
    col().key("page").size(480.0, 600.0)
        .child(el().key("remove").size(100.0, 30.0).fill(ink(0.6, 0.2, 0.2)).on_press(|s: &mut Rows| s.0.retain(|&id| id != 5 && id != 23)))
        .child(col().key("scroll").w_full().h(560.0).scroll(ScrollSpec { stick_bottom: false }).child(
            col().key("rows").w_full().gap(8.0).padding(12.0).growth(GrowthSpec::default()).virtual_overscan(0.25).virtual_rows(
                s.0.len(),
                60.0,
                |s: &Rows, i| Cow::Owned(format!("r{}", s.0[i])),
                |s: &Rows, i| row().w_full().h(60.0).fill(ink(0.3, 0.3, 0.5)).child(text(format!("Row {}", s.0[i])).color(crate::Color::WHITE)),
            ),
        ))
}

/// One press removes a row culled above the window and one in view: the room the window
/// leaves above it shrinks by a row as the row in view fades out. It fades where it was, not
/// a row higher over the row above it, where the window's old room left it.
#[test]
fn a_row_removed_as_the_window_shifts_fades_where_it_was() {
    for scale in [1.0, 1.25, 2.0] {
        let mut rt = Runtime::new(Rows((0..80).collect()), removable, (480.0, 600.0), Env::new(true, scale));
        rt.frame(0.0);
        rt.frame(1.0 / 120.0);
        rt.wheel_xy(240.0, 300.0, 0.0, -1400.0);
        for _ in 0..40 { rt.frame(1.0 / 60.0); }
        let before = rt.commit.virtuals.values().next().unwrap().before;
        assert!(before > 0.0, "rows above the window are culled at {scale}x");
        let (was, shown) = rt.presented("r23").unwrap();
        assert!(shown > 0.99 && was[1] > 40.0 && was[1] < 560.0, "row 23 shows at {scale}x: {was:?}");
        let (x, y) = rt.find("remove").unwrap();
        rt.pointer_down(x, y);
        rt.pointer_up(x, y);
        let mut drawn = Vec::new();
        for _ in 0..8 {
            rt.frame(1.0 / 60.0);
            if let Some((r, o)) = rt.presented("r23").filter(|(_, o)| *o > 0.0) { drawn.push((r, o)); }
        }
        assert!(rt.commit.virtuals.values().next().unwrap().before < before, "the window's room above shrank at {scale}x");
        assert!(drawn.iter().any(|&(_, o)| o < 0.99), "row 23 fades at {scale}x: {drawn:?}");
        for (r, _) in &drawn {
            assert!((r[1] - was[1]).abs() < 0.5, "row 23 fades where it was at {scale}x: from {was:?}, {drawn:?}");
        }
    }
}
