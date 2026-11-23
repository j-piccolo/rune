//! Offscreen render targets with CPU readback and PNG writing.

/// An offscreen sRGB target with CPU readback.
pub struct Offscreen {
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    readback: rune_staging::Readback,
    pub width: u32,
    pub height: u32,
}

pub const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// What a window's surface takes on Linux, where Vulkan and Mesa offer BGRA first: a frame budget's
/// offscreen frames draw into it, as the window would (`remove_srgb_suffix` with sRGB blending).
pub const WINDOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

impl Offscreen {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Offscreen {
        Self::with_format(device, width, height, OFFSCREEN_FORMAT)
    }

    pub fn with_format(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Offscreen {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // COPY_DST: a partial frame copies the renderer's kept frame in (`damaged`).
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let readback = rune_staging::Readback::new(device, "readback", width, height, 4);
        Offscreen { texture, view, readback, width, height }
    }

    pub fn copy(&self, encoder: &mut wgpu::CommandEncoder) {
        self.readback.copy_from(encoder, &self.texture);
    }

    /// Underlying offscreen texture, for readback rings (`tooling::record`).
    pub(crate) fn texture(&self) -> &wgpu::Texture { &self.texture }

    /// Tightly packed sRGB RGBA8 after `copy` was submitted.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(&self, device: &wgpu::Device) -> Vec<u8> {
        // A callback never panics: one that did while a failed wait unwinds (wgpu aborts the map
        // then) would abort the process. A failed map shows as `take_mapped`'s panic.
        self.readback.buffer().slice(..).map_async(wgpu::MapMode::Read, |_| {});
        rune_gpu::health::poll(device, wgpu::PollType::wait_indefinitely()).expect("poll");
        self.readback.take_mapped()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn read(&self, _: &wgpu::Device) -> Vec<u8> {
        panic!("blocking GPU readback is unsupported on web")
    }
}

pub fn write_png(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, width, height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut w = enc.write_header().map_err(std::io::Error::other)?;
    w.write_image_data(rgba).map_err(std::io::Error::other)?;
    Ok(())
}

/// Draws `rt`'s last frame into a new offscreen target `size` device pixels big and reads it
/// back: tightly packed RGBA8, as the renderer's target format encodes it. For an app that
/// renders its own views to images (a screenshot tool, a golden): run `rt.frame(..)` first.
#[cfg(not(target_arch = "wasm32"))]
pub fn snapshot<S: 'static>(gpu: &super::Gpu, renderer: &mut super::Renderer, rt: &mut crate::Runtime<S>, size: (u32, u32)) -> Vec<u8> {
    let target = Offscreen::with_format(&gpu.device, size.0.max(1), size.1.max(1), renderer.format());
    let core = rt.core_mut();
    renderer.attach_text(&mut core.text.atlas);
    crate::frame::host_step::prepare_gpu_resources(renderer, gpu, core, size, false);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    crate::frame::host_step::draw_ui(renderer, gpu, &mut encoder, &target.view, core, size, [0.0; 4]);
    target.copy(&mut encoder);
    renderer.submit(&gpu.queue, encoder);
    target.read(&gpu.device)
}
