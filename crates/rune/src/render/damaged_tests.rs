use super::*;

/// The window's rows of a viewport as the kept frame holds them under `roll`, read back.
fn straightened(kept: &[u32], roll: Option<Roll>, size: [u32; 2]) -> Vec<u32> {
    let mut out = vec![0; (size[0] * size[1]) as usize];
    let mut pieces = Vec::new();
    Roll::pieces(roll, [0, 0, size[0], size[1]], &mut pieces);
    let mut covered = 0;
    for ([x, y, w, h], [kx, ky]) in pieces {
        covered += w * h;
        for dy in 0..h {
            for dx in 0..w {
                out[((y + dy) * size[0] + x + dx) as usize] = kept[((ky + dy) * size[0] + kx + dx) as usize];
            }
        }
    }
    assert_eq!(covered, size[0] * size[1], "the pieces are the window, once");
    out
}

#[test]
fn a_rolled_frame_reads_back_as_the_scrolled_window() {
    let size = [12u32, 16];
    let band = [2u32, 3, 8, 10];
    // The window: every pixel its own number. The kept frame starts as the window.
    let mut window: Vec<u32> = (0..size[0] * size[1]).collect();
    let mut kept = window.clone();
    let mut roll = None;
    let mut next = 1000;
    for (dx, dy) in [(0i32, -3i32), (0, -4), (0, 2), (0, -9), (3, 0), (-5, 1), (0, 9)] {
        // The viewport's pixels move by (dx, dy); what comes in is new.
        let (from, to) = {
            let (x0, y0) = (band[0] as i32 + dx.max(0), band[1] as i32 + dy.max(0));
            let (w, h) = (band[2] - dx.unsigned_abs(), band[3] - dy.unsigned_abs());
            ([(x0 - dx) as u32, (y0 - dy) as u32, w, h], [x0 as u32, y0 as u32])
        };
        let before = window.clone();
        let mut came = Vec::new();
        for y in band[1]..band[1] + band[3] {
            for x in band[0]..band[0] + band[2] {
                let (sx, sy) = (x as i32 - dx, y as i32 - dy);
                let inside = sx >= band[0] as i32 && sx < (band[0] + band[2]) as i32 && sy >= band[1] as i32 && sy < (band[1] + band[3]) as i32;
                window[(y * size[0] + x) as usize] = if inside { before[(sy as u32 * size[0] + sx as u32) as usize] } else { next += 1; came.push([x, y, 1, 1]); next };
            }
        }
        roll = Roll::moved(roll, from, to);
        assert!(roll.is_some(), "one viewport, one roll");
        // What came in is redrawn, each pixel where the kept frame holds it.
        let mut pieces = Vec::new();
        for r in came { Roll::pieces(roll, r, &mut pieces); }
        for ([x, y, w, h], [kx, ky]) in pieces {
            assert_eq!([w, h], [1, 1]);
            kept[(ky * size[0] + kx) as usize] = window[(y * size[0] + x) as usize];
        }
        assert_eq!(straightened(&kept, roll, size), window, "after moving by ({dx}, {dy})");
    }
    // Another viewport's scroll doesn't continue this roll.
    assert_eq!(Roll::moved(roll, [0, 4, 12, 10], [0, 2]), None);
}

#[test]
fn a_rect_across_the_seam_is_kept_in_pieces() {
    let roll = Some(Roll { band: [0, 10, 100, 50], off: [0, 20] });
    let mut pieces = Vec::new();
    // Rows 5..55: five above the viewport, then the viewport's rows 10..40 (kept from row 30 on)
    // and 40..55 (kept from row 10 on: the roll wrapped).
    Roll::pieces(roll, [10, 5, 30, 50], &mut pieces);
    pieces.sort_unstable();
    assert_eq!(pieces, vec![([10, 5, 30, 5], [10, 5]), ([10, 10, 30, 30], [10, 30]), ([10, 40, 30, 15], [10, 10])]);
}
