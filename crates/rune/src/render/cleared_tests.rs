use super::super::{Gpu, Instance, KIND_GLYPH, NO_CLIP, OFFSCREEN_FORMAT, Offscreen, Renderer};
use super::Cleared;

/// The clear colour: the root's fill.
const C: [f32; 4] = [0.078, 0.078, 0.078, 1.0];

fn rect(rect: [f32; 4], fill: [f32; 4]) -> Instance {
    Instance { rect, fill, clip: NO_CLIP, ..bytemuck::Zeroable::zeroed() }
}

/// The root's fill over the whole target, a bar in the same colour along its top, a red panel.
fn page() -> Vec<Instance> {
    vec![rect([0.0, 0.0, 64.0, 48.0], C), rect([0.0, 0.0, 64.0, 8.0], C), rect([8.0, 16.0, 24.0, 16.0], [1.0, 0.0, 0.0, 1.0])]
}

fn clear(c: [f32; 4]) -> wgpu::LoadOp<wgpu::Color> {
    wgpu::LoadOp::Clear(wgpu::Color { r: c[0] as f64, g: c[1] as f64, b: c[2] as f64, a: c[3] as f64 })
}

#[test]
fn the_run_ends_at_the_first_instance_that_could_change_a_pixel_over_the_clear() {
    assert_eq!(Cleared::of(&page(), (64, 48)).len, 2);
    assert_eq!(Cleared::of(&[], (64, 48)).len, 0);
    assert_eq!(Cleared::of(&[rect([0.0, 0.0, 64.0, 48.0], [0.5, 0.5, 0.5, 0.5])], (64, 48)).len, 0, "a translucent root clears to another colour");
    let mut clipped_inside = page();
    clipped_inside[1].clip = [0.0, 0.0, 64.0, 48.0];
    assert_eq!(Cleared::of(&clipped_inside, (64, 48)).len, 2, "a clip holding the whole rect clips nothing");
    type Change = fn(&mut Instance);
    let cases: [(&str, Change); 9] = [
        ("a rect off whole pixels", |i| i.rect[1] = 0.5),
        ("a width off whole pixels", |i| i.rect[2] = 7.25),
        ("a rounded rect", |i| i.radii = [4.0; 4]),
        ("a bordered rect", |i| i.params[0] = 1.0),
        ("a softened rect", |i| i.params[2] = 2.0),
        ("a clip cutting the rect", |i| i.clip = [0.0, 0.0, 32.0, 32.0]),
        ("a faded rect", |i| i.fade = [1, 0, 0, 0]),
        ("another colour", |i| i.fill = [0.1, 0.078, 0.078, 1.0]),
        ("a glyph", |i| i.params[1] = KIND_GLYPH),
    ];
    for (what, change) in cases {
        let mut list = page();
        change(&mut list[1]);
        assert_eq!(Cleared::of(&list, (64, 48)).len, 1, "{what} ends the run");
    }
}

#[test]
fn a_pass_starts_past_the_run_only_when_it_clears_to_its_colour_and_short_of_any_surface() {
    let cleared = Cleared::of(&page(), (64, 48));
    assert_eq!(cleared.start(clear(C), u32::MAX), 2);
    assert_eq!(cleared.start(clear([0.0, 0.0, 0.0, 1.0]), u32::MAX), 0, "another clear colour");
    assert_eq!(cleared.start(wgpu::LoadOp::Load, u32::MAX), 0, "a pass that keeps what the target holds");
    assert_eq!(cleared.start(clear(C), 1), 1, "a video drawn after the root's fill: the bar draws over it");
    assert_eq!(cleared.start(clear(C), 0), 0);
}

/// The run left out, the target holds the same bytes as with every instance drawn over the same
/// clear: through `draw` and through `draw_with`, and for lists whose run ends early.
#[test]
fn leaving_the_run_out_draws_the_same_bytes() {
    let gpu = Gpu::headless();
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut renderer = Renderer::new(device, OFFSCREEN_FORMAT);
    let mut text = crate::text::TextSystem::new(&[], ("Inter", "Inter"));
    let mut off_pixels = page();
    off_pixels.insert(2, rect([3.5, 20.0, 9.25, 7.5], C));
    let mut rounded = page();
    rounded[1].radii = [3.0; 4];
    rounded[1].fill = [0.2, 0.4, 0.6, 1.0];
    for (name, list, left_out) in [("page", page(), 2), ("off whole pixels", off_pixels, 2), ("rounded", rounded, 1)] {
        renderer.prepare(device, queue, &mut text.atlas, &list, (64, 48));
        assert_eq!(renderer.cleared_start(clear(C)), left_out, "{name}");
        let draw = |how: &str| {
            let target = Offscreen::new(device, 64, 48);
            let mut encoder = device.create_command_encoder(&Default::default());
            match how {
                "draw" => renderer.draw(&mut encoder, &target.view, C),
                "draw_with" => renderer.draw_with(&mut encoder, &target.view, C, &mut |_, _| {}),
                _ => {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &target.view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations { load: clear(C), store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    renderer.draw_range(&mut pass, OFFSCREEN_FORMAT, 0..list.len() as u32);
                }
            }
            target.copy(&mut encoder);
            queue.submit([encoder.finish()]);
            target.read(device)
        };
        let every = draw("every instance");
        assert_eq!(draw("draw"), every, "{name}: draw");
        assert_eq!(draw("draw_with"), every, "{name}: draw_with");
    }
}
