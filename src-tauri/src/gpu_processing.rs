use std::sync::{Arc, MutexGuard};
use std::time::Instant;

use half::f16;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use std::num::NonZero;

#[cfg(not(any(target_os = "android", target_os = "linux")))]
use tauri::Manager;
use wgpu::util::{DeviceExt, TextureDataOrder};

use crate::app_state::SharedMaskBitmap;
use crate::image_processing::{AllAdjustments, GpuContext, MAX_MASKS};
use crate::lut_processing::Lut;
use crate::render_strategy::{
    GPU_INPUT_UPLOAD_BAND_ROWS, GPU_TILE_OVERLAP, GPU_TILE_SIZE, GpuInputUploadPlan,
    GpuProcessorTexturePlan, MaskTexturePlan, MaskTileUploadPlan, StreamingExportBufferPlan,
    should_reclaim_gpu_resources,
};
use crate::{AppState, GpuImageCache};

#[derive(Clone, Copy, Debug)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct RenderRequest<'a> {
    pub adjustments: AllAdjustments,
    pub mask_bitmaps: &'a [SharedMaskBitmap],
    pub lut: Option<Arc<Lut>>,
    pub roi: Option<Roi>,
}

type RgbaRowSink<'a> = dyn FnMut(&[u8]) -> Result<(), String> + 'a;

enum GpuRenderOutput<'a> {
    CpuImage,
    NativeDisplay,
    StreamRows(&'a mut RgbaRowSink<'a>),
}

type GpuProcessorGuard<'a> = MutexGuard<'a, Option<crate::GpuProcessorState>>;
type GpuImageCacheGuard<'a> = MutexGuard<'a, Option<GpuImageCache>>;
type LockedGpuResources<'a> = (GpuProcessorGuard<'a>, GpuImageCacheGuard<'a>);

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DisplayTransform {
    pub rect: [f32; 4],
    pub clip: [f32; 4],
    pub window: [f32; 2],
    pub image_size: [f32; 2],
    pub texture_size: [f32; 2],
    pub pixelated: f32,
    pub _pad: f32,
    pub bg_primary: [f32; 4],
    pub bg_secondary: [f32; 4],
}

pub struct WgpuDisplay {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    pub transform_buffer: wgpu::Buffer,
    pub latest_transform: DisplayTransform,
    pub current_bind_group: Option<wgpu::BindGroup>,
}

impl WgpuDisplay {
    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if let Some(bind_group) = &self.current_bind_group {
            let output = match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(tex)
                | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    self.surface.configure(device, &self.config);
                    match self.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(tex)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
                        _ => panic!("Failed to acquire surface texture"),
                    }
                }
                _ => return,
            };
            let view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: self.latest_transform.bg_primary[0] as f64,
                                g: self.latest_transform.bg_primary[1] as f64,
                                b: self.latest_transform.bg_primary[2] as f64,
                                a: self.latest_transform.bg_primary[3] as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: NonZero::new(0),
                });
                let clip_x1 = self.latest_transform.clip[0].max(0.0);
                let clip_y1 = self.latest_transform.clip[1].max(0.0);
                let clip_x2 =
                    (self.latest_transform.clip[0] + self.latest_transform.clip[2]).max(0.0);
                let clip_y2 =
                    (self.latest_transform.clip[1] + self.latest_transform.clip[3]).max(0.0);

                let final_clip_x = clip_x1.floor() as u32;
                let final_clip_y = clip_y1.floor() as u32;
                let final_clip_w = (clip_x2.ceil() as u32).saturating_sub(final_clip_x);
                let final_clip_h = (clip_y2.ceil() as u32).saturating_sub(final_clip_y);

                let max_x = self.config.width;
                let max_y = self.config.height;

                if final_clip_x < max_x && final_clip_y < max_y {
                    let clamped_width = final_clip_w.min(max_x - final_clip_x);
                    let clamped_height = final_clip_h.min(max_y - final_clip_y);

                    if clamped_width > 0 && clamped_height > 0 {
                        rpass.set_scissor_rect(
                            final_clip_x,
                            final_clip_y,
                            clamped_width,
                            clamped_height,
                        );

                        rpass.set_pipeline(&self.pipeline);
                        rpass.set_bind_group(0, bind_group, &[]);
                        rpass.draw(0..4, 0..1);
                    }
                }
            }
            queue.submit(Some(encoder.finish()));
            queue.present(output);
        }
    }
}

pub fn get_or_init_gpu_context(
    state: &tauri::State<AppState>,
    _app_handle: &tauri::AppHandle,
) -> Result<GpuContext, String> {
    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    let app_handle = _app_handle;

    let mut context_lock = state.gpu_context.lock().unwrap();
    if let Some(context) = &*context_lock {
        return Ok(context.clone());
    }

    #[allow(unused_mut)]
    let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();

    #[cfg(target_os = "windows")]
    if std::env::var("WGPU_BACKEND").is_err() {
        instance_desc.backends = wgpu::Backends::PRIMARY;
    }

    let flag_path = state.gpu_crash_flag_path.lock().unwrap().clone();
    if let Some(p) = &flag_path {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(p, "initializing_gpu");
    }

    let instance = wgpu::Instance::new(instance_desc);

    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    let surface_opt = {
        let settings = crate::app_settings::load_settings(app_handle.clone()).unwrap_or_default();
        let use_wgpu_renderer = settings.use_wgpu_renderer.unwrap_or(true);

        if use_wgpu_renderer {
            if let Some(window) = app_handle.get_webview_window("main") {
                match instance.create_surface(window) {
                    Ok(surface) => Some(surface),
                    Err(e) => {
                        log::warn!(
                            "Failed to create surface, falling back to compute-only: {}",
                            e
                        );
                        if let Some(p) = &flag_path {
                            let _ = std::fs::remove_file(p);
                        }
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        }
    };

    #[cfg(any(target_os = "android", target_os = "linux"))]
    let surface_opt: Option<wgpu::Surface> = None;

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: surface_opt.as_ref(),
        ..Default::default()
    }))
    .map_err(|e| {
        if let Some(p) = &flag_path {
            let _ = std::fs::remove_file(p);
        }
        format!("Failed to find a wgpu adapter: {}", e)
    })?;

    let mut required_features = wgpu::Features::empty();
    if adapter
        .features()
        .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
    {
        required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    }

    let limits = adapter.limits();

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Processing Device"),
        required_features,
        required_limits: limits.clone(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| {
        if let Some(p) = &flag_path {
            let _ = std::fs::remove_file(p);
        }
        e.to_string()
    })?;

    if let Some(p) = &flag_path {
        let _ = std::fs::remove_file(p);
    }

    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    let display_opt = if let Some(surface) = surface_opt {
        let window = app_handle
            .get_webview_window("main")
            .ok_or("Failed to get main window")?;

        let swapchain_caps = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(swapchain_caps.formats[0]);

        let alpha_mode = if cfg!(target_os = "windows")
            && swapchain_caps
                .alpha_modes
                .contains(&wgpu::CompositeAlphaMode::Opaque)
        {
            wgpu::CompositeAlphaMode::Opaque
        } else if swapchain_caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
        {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else if swapchain_caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PostMultiplied)
        {
            wgpu::CompositeAlphaMode::PostMultiplied
        } else {
            swapchain_caps.alpha_modes[0]
        };

        let size = window
            .inner_size()
            .unwrap_or(tauri::PhysicalSize::new(1280, 720));
        let config = wgpu::SurfaceConfiguration {
            width: size.width.max(1),
            height: size.height.max(1),
            format: swapchain_format,
            color_space: wgpu::SurfaceColorSpace::Srgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Display Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Display BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Display Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Display Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: swapchain_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: NonZero::new(0),
            cache: None,
        });

        let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transform Buffer"),
            size: std::mem::size_of::<DisplayTransform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Display Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Some(WgpuDisplay {
            surface,
            config,
            pipeline,
            bind_group_layout,
            transform_buffer,
            latest_transform: DisplayTransform {
                rect: [0.0, 0.0, 100.0, 100.0],
                clip: [0.0, 0.0, 10000.0, 10000.0],
                window: [1280.0, 720.0],
                image_size: [100.0, 100.0],
                texture_size: [100.0, 100.0],
                pixelated: 0.0,
                _pad: 0.0,
                bg_primary: [24.0 / 255.0, 24.0 / 255.0, 24.0 / 255.0, 1.0],
                bg_secondary: [35.0 / 255.0, 35.0 / 255.0, 35.0 / 255.0, 1.0],
            },
            sampler,
            current_bind_group: None,
        })
    } else {
        None
    };

    #[cfg(any(target_os = "android", target_os = "linux"))]
    let display_opt = None;

    let new_context = GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        limits,
        display: Arc::new(std::sync::Mutex::new(display_opt)),
    };
    *context_lock = Some(new_context.clone());
    Ok(new_context)
}

fn read_texture_data_roi(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    origin: wgpu::Origin3d,
    size: wgpu::Extent3d,
) -> Result<Vec<u8>, String> {
    let unpadded_bytes_per_row = 4 * size.width;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);
    let output_buffer_size = (padded_bytes_per_row * size.height) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );

    queue.submit(Some(encoder.finish()));
    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(60)),
        })
        .map_err(|e| format!("Failed while polling mapped GPU buffer: {}", e))?;
    let map_result = rx
        .recv()
        .map_err(|e| format!("Failed receiving GPU map result: {}", e))?;
    map_result.map_err(|e| e.to_string())?;

    let padded_data = buffer_slice
        .get_mapped_range()
        .map_err(|e| format!("Failed to get mapped GPU buffer range: {}", e))?
        .to_vec();
    output_buffer.unmap();

    if padded_bytes_per_row == unpadded_bytes_per_row {
        Ok(padded_data)
    } else {
        let mut unpadded_data = Vec::with_capacity((unpadded_bytes_per_row * size.height) as usize);
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }
        Ok(unpadded_data)
    }
}

#[cfg(test)]
fn to_rgba_f16(img: &DynamicImage) -> Vec<f16> {
    match img {
        DynamicImage::ImageRgb32F(rgb) => {
            let mut output = Vec::with_capacity(rgb.as_raw().len() / 3 * 4);
            for pixel in rgb.as_raw().chunks_exact(3) {
                output.extend_from_slice(&[
                    f16::from_f32(pixel[0]),
                    f16::from_f32(pixel[1]),
                    f16::from_f32(pixel[2]),
                    f16::ONE,
                ]);
            }
            output
        }
        DynamicImage::ImageRgba32F(rgba) => {
            rgba.as_raw().iter().copied().map(f16::from_f32).collect()
        }
        _ => img
            .to_rgba32f()
            .into_raw()
            .into_iter()
            .map(f16::from_f32)
            .collect(),
    }
}

#[inline]
fn write_rgba_f16_pixel(output: &mut [f16], pixel_index: usize, rgba: [f32; 4]) {
    let offset = pixel_index * 4;
    output[offset] = f16::from_f32(rgba[0]);
    output[offset + 1] = f16::from_f32(rgba[1]);
    output[offset + 2] = f16::from_f32(rgba[2]);
    output[offset + 3] = f16::from_f32(rgba[3]);
}

fn write_rgba_f16_row_from_samples<T>(
    source: &[T],
    y: u32,
    output: &mut [f16],
    width: usize,
    channels: usize,
    max_value: f32,
) -> Result<(), String>
where
    T: Copy + Into<f32>,
{
    let row_samples = width
        .checked_mul(4)
        .ok_or_else(|| "GPU input row exceeds the addressable buffer size".to_string())?;
    if output.len() < row_samples {
        return Err("GPU input row staging buffer is too small".to_string());
    }
    if !(1..=4).contains(&channels) {
        return Err("GPU input row has an unsupported channel count".to_string());
    }

    let source_row_samples = width
        .checked_mul(channels)
        .ok_or_else(|| "GPU input source row exceeds the addressable buffer size".to_string())?;
    let start = (y as usize)
        .checked_mul(source_row_samples)
        .ok_or_else(|| "GPU input source row offset overflow".to_string())?;
    let end = start
        .checked_add(source_row_samples)
        .ok_or_else(|| "GPU input source row end overflow".to_string())?;
    let row = source
        .get(start..end)
        .ok_or_else(|| "GPU input source row is outside the source image".to_string())?;

    for (pixel_index, pixel) in row.chunks_exact(channels).enumerate() {
        let red = pixel[0].into() / max_value;
        let green = if channels >= 3 {
            pixel[1].into() / max_value
        } else {
            red
        };
        let blue = if channels >= 3 {
            pixel[2].into() / max_value
        } else {
            red
        };
        let alpha = match channels {
            2 => pixel[1].into() / max_value,
            4 => pixel[3].into() / max_value,
            _ => 1.0,
        };
        write_rgba_f16_pixel(output, pixel_index, [red, green, blue, alpha]);
    }

    Ok(())
}

fn write_rgba_f16_row(
    image: &DynamicImage,
    y: u32,
    output: &mut [f16],
    width: usize,
) -> Result<(), String> {
    let max_u8 = u8::MAX as f32;
    let max_u16 = u16::MAX as f32;

    match image {
        DynamicImage::ImageLuma8(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 1, max_u8)?;
        }
        DynamicImage::ImageLumaA8(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 2, max_u8)?;
        }
        DynamicImage::ImageRgb8(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 3, max_u8)?;
        }
        DynamicImage::ImageRgba8(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 4, max_u8)?;
        }
        DynamicImage::ImageLuma16(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 1, max_u16)?;
        }
        DynamicImage::ImageLumaA16(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 2, max_u16)?;
        }
        DynamicImage::ImageRgb16(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 3, max_u16)?;
        }
        DynamicImage::ImageRgba16(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 4, max_u16)?;
        }
        DynamicImage::ImageRgb32F(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 3, 1.0)?;
        }
        DynamicImage::ImageRgba32F(source) => {
            write_rgba_f16_row_from_samples(source.as_raw(), y, output, width, 4, 1.0)?;
        }
        _ => {
            for x in 0..width as u32 {
                let pixel = image.get_pixel(x, y);
                write_rgba_f16_pixel(
                    output,
                    x as usize,
                    [
                        f32::from(pixel[0]) / max_u8,
                        f32::from(pixel[1]) / max_u8,
                        f32::from(pixel[2]) / max_u8,
                        f32::from(pixel[3]) / max_u8,
                    ],
                );
            }
        }
    }

    Ok(())
}

fn upload_image_to_texture_bounded(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    image: &DynamicImage,
) -> Result<GpuInputUploadPlan, String> {
    let (width, height) = image.dimensions();
    let plan = GpuInputUploadPlan::new(width, height)
        .ok_or_else(|| format!("Cannot upload invalid GPU input dimensions {width}x{height}"))?;
    let padded_row_samples = plan.padded_bytes_per_row as usize / std::mem::size_of::<f16>();
    let max_staging_samples = plan.max_staging_bytes / std::mem::size_of::<f16>();
    let mut staging = vec![f16::ZERO; max_staging_samples];

    for band_start in (0..height).step_by(GPU_INPUT_UPLOAD_BAND_ROWS as usize) {
        let band_height = (height - band_start).min(plan.band_rows);
        let band_samples = padded_row_samples
            .checked_mul(band_height as usize)
            .ok_or_else(|| "GPU input band exceeds the addressable buffer size".to_string())?;
        staging[..band_samples].fill(f16::ZERO);
        for row in 0..band_height {
            let row_start = row as usize * padded_row_samples;
            write_rgba_f16_row(
                image,
                band_start + row,
                &mut staging[row_start..row_start + padded_row_samples],
                width as usize,
            )?;
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: band_start,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&staging[..band_samples]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(plan.padded_bytes_per_row),
                rows_per_image: Some(band_height),
            },
            wgpu::Extent3d {
                width,
                height: band_height,
                depth_or_array_layers: 1,
            },
        );
    }

    Ok(plan)
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    radius: u32,
    tile_offset_x: u32,
    tile_offset_y: u32,
    input_width: u32,
    input_height: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FlareParams {
    amount: f32,
    is_raw: u32,
    exposure: f32,
    brightness: f32,
    contrast: f32,
    whites: f32,
    aspect_ratio: f32,
    _pad: f32,
}

pub struct GpuProcessor {
    context: GpuContext,
    blur_bgl: wgpu::BindGroupLayout,
    h_blur_pipeline: wgpu::ComputePipeline,
    v_blur_pipeline: wgpu::ComputePipeline,
    blur_params_buffer: wgpu::Buffer,

    flare_bgl_0: wgpu::BindGroupLayout,
    flare_bgl_1: wgpu::BindGroupLayout,
    flare_threshold_pipeline: wgpu::ComputePipeline,
    flare_ghosts_pipeline: wgpu::ComputePipeline,
    flare_params_buffer: wgpu::Buffer,
    flare_threshold_view: wgpu::TextureView,
    flare_ghosts_view: wgpu::TextureView,
    flare_final_view: wgpu::TextureView,
    flare_sampler: wgpu::Sampler,

    main_bgl: wgpu::BindGroupLayout,
    main_pipeline: wgpu::ComputePipeline,
    adjustments_buffer: wgpu::Buffer,
    dummy_blur_view: wgpu::TextureView,
    dummy_mask_view: wgpu::TextureView,
    dummy_lut_view: wgpu::TextureView,
    dummy_lut_sampler: wgpu::Sampler,
    ping_pong_view: wgpu::TextureView,
    sharpness_blur_view: wgpu::TextureView,
    tonal_blur_view: wgpu::TextureView,
    clarity_blur_view: wgpu::TextureView,
    structure_blur_view: wgpu::TextureView,

    pub tile_output_texture: wgpu::Texture,
    pub tile_output_texture_view: wgpu::TextureView,
    pub working_texture: wgpu::Texture,
    pub working_texture_view: wgpu::TextureView,
    pub output_texture: wgpu::Texture,
    pub output_texture_view: wgpu::TextureView,
}

const FLARE_MAP_SIZE: u32 = 512;

impl GpuProcessor {
    pub fn new(
        context: GpuContext,
        width: u32,
        height: u32,
        needs_native_display: bool,
    ) -> Result<Self, String> {
        let texture_plan = GpuProcessorTexturePlan::new(width, height, needs_native_display);
        let max_width = texture_plan.processing_width;
        let max_height = texture_plan.processing_height;
        let device = &context.device;
        const MAX_MASK_BINDINGS: u32 = 1;

        let blur_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur.wgsl").into()),
        });

        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blur BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur Pipeline Layout"),
            bind_group_layouts: &[Some(&blur_bgl)],
            immediate_size: 0,
        });

        let h_blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Horizontal Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_shader_module,
            entry_point: Some("horizontal_blur"),
            compilation_options: Default::default(),
            cache: None,
        });

        let v_blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vertical Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_shader_module,
            entry_point: Some("vertical_blur"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blur_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blur Params Buffer"),
            size: std::mem::size_of::<BlurParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flare Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flare.wgsl").into()),
        });

        let flare_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flare BGL 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let flare_bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flare BGL 1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let flare_threshold_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Flare Threshold Layout"),
                bind_group_layouts: &[Some(&flare_bgl_0)],
                immediate_size: 0,
            });

        let flare_ghosts_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flare Ghosts Layout"),
            bind_group_layouts: &[Some(&flare_bgl_0), Some(&flare_bgl_1)],
            immediate_size: 0,
        });

        let flare_threshold_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Flare Threshold Pipeline"),
                layout: Some(&flare_threshold_layout),
                module: &flare_shader,
                entry_point: Some("threshold_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let flare_ghosts_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Flare Ghosts Pipeline"),
                layout: Some(&flare_ghosts_layout),
                module: &flare_shader,
                entry_point: Some("ghosts_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let flare_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flare Params Buffer"),
            size: std::mem::size_of::<FlareParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flare_tex_desc = wgpu::TextureDescriptor {
            label: Some("Flare Tex"),
            size: wgpu::Extent3d {
                width: FLARE_MAP_SIZE,
                height: FLARE_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let flare_threshold_texture = device.create_texture(&flare_tex_desc);
        let flare_threshold_view = flare_threshold_texture.create_view(&Default::default());
        let flare_ghosts_texture = device.create_texture(&flare_tex_desc);
        let flare_ghosts_view = flare_ghosts_texture.create_view(&Default::default());
        let flare_final_texture = device.create_texture(&flare_tex_desc);
        let flare_final_view = flare_final_texture.create_view(&Default::default());

        let flare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Flare Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Image Processing Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
        });

        let mut bind_group_layout_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 6 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 7 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 10 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });

        let main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main BGL"),
            entries: &bind_group_layout_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&main_bgl)],
            immediate_size: 0,
        });

        let main_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let adjustments_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adjustments Buffer"),
            size: std::mem::size_of::<AllAdjustments>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dummy_texture_desc = wgpu::TextureDescriptor {
            label: Some("Dummy Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let dummy_blur_texture = device.create_texture(&dummy_texture_desc);
        let dummy_blur_view = dummy_blur_texture.create_view(&Default::default());

        let dummy_mask_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Mask Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_mask_view = dummy_mask_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let dummy_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            dimension: wgpu::TextureDimension::D3,
            ..dummy_texture_desc
        });
        let dummy_lut_view = dummy_lut_texture.create_view(&Default::default());
        let dummy_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let clamped_tile_width = max_width.min(GPU_TILE_SIZE + GPU_TILE_OVERLAP * 2);
        let clamped_tile_height = max_height.min(GPU_TILE_SIZE + GPU_TILE_OVERLAP * 2);

        let clamped_tile_size = wgpu::Extent3d {
            width: clamped_tile_width,
            height: clamped_tile_height,
            depth_or_array_layers: 1,
        };

        let display_image_size = wgpu::Extent3d {
            width: texture_plan.display_width,
            height: texture_plan.display_height,
            depth_or_array_layers: 1,
        };

        let reusable_texture_desc = wgpu::TextureDescriptor {
            label: None,
            size: clamped_tile_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let ping_pong_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Ping Pong Texture"),
            ..reusable_texture_desc
        });
        let ping_pong_view = ping_pong_texture.create_view(&Default::default());

        let sharpness_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Sharpness Blur Texture"),
            ..reusable_texture_desc
        });
        let sharpness_blur_view = sharpness_blur_texture.create_view(&Default::default());

        let tonal_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Tonal Blur Texture"),
            ..reusable_texture_desc
        });
        let tonal_blur_view = tonal_blur_texture.create_view(&Default::default());

        let clarity_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Clarity Blur Texture"),
            ..reusable_texture_desc
        });
        let clarity_blur_view = clarity_blur_texture.create_view(&Default::default());

        let structure_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Structure Blur Texture"),
            ..reusable_texture_desc
        });
        let structure_blur_view = structure_blur_texture.create_view(&Default::default());

        let tile_output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Tile Output Texture"),
            size: clamped_tile_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let tile_output_texture_view = tile_output_texture.create_view(&Default::default());

        let working_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Working Output Texture"),
            size: display_image_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let working_texture_view = working_texture.create_view(&Default::default());

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Full Output Texture"),
            size: display_image_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_texture_view = output_texture.create_view(&Default::default());

        Ok(Self {
            context,
            blur_bgl,
            h_blur_pipeline,
            v_blur_pipeline,
            blur_params_buffer,
            flare_bgl_0,
            flare_bgl_1,
            flare_threshold_pipeline,
            flare_ghosts_pipeline,
            flare_params_buffer,
            flare_threshold_view,
            flare_ghosts_view,
            flare_final_view,
            flare_sampler,
            main_bgl,
            main_pipeline,
            adjustments_buffer,
            dummy_blur_view,
            dummy_mask_view,
            dummy_lut_view,
            dummy_lut_sampler,
            ping_pong_view,
            sharpness_blur_view,
            tonal_blur_view,
            clarity_blur_view,
            structure_blur_view,
            tile_output_texture,
            tile_output_texture_view,
            working_texture,
            working_texture_view,
            output_texture,
            output_texture_view,
        })
    }

    fn run(
        &self,
        input_texture_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        request: RenderRequest,
        mut output: GpuRenderOutput<'_>,
    ) -> Result<(Vec<u8>, u32, u32, u32, u32), String> {
        let skip_cpu_readback = matches!(&output, GpuRenderOutput::NativeDisplay);
        let output_to_display = skip_cpu_readback;
        let stream_rows = matches!(&output, GpuRenderOutput::StreamRows(_));

        let device = &self.context.device;
        let queue = &self.context.queue;
        let scale = (width.min(height) as f32) / 1080.0;
        const MAX_MASK_BINDINGS: u32 = 1;

        let bounds = request.roi.unwrap_or(Roi {
            x: 0,
            y: 0,
            width,
            height,
        });
        let out_width = bounds.width;
        let out_height = bounds.height;
        if out_width == 0 || out_height == 0 {
            return Err("GPU render bounds must have non-zero dimensions".to_string());
        }
        let bounds_right = bounds
            .x
            .checked_add(bounds.width)
            .filter(|right| *right <= width)
            .ok_or_else(|| {
                format!(
                    "GPU render bounds x={} width={} exceed input width={width}",
                    bounds.x, bounds.width
                )
            })?;
        let bounds_bottom = bounds
            .y
            .checked_add(bounds.height)
            .filter(|bottom| *bottom <= height)
            .ok_or_else(|| {
                format!(
                    "GPU render bounds y={} height={} exceed input height={height}",
                    bounds.y, bounds.height
                )
            })?;
        if stream_rows {
            let stream_plan = StreamingExportBufferPlan::new(out_width, out_height);
            log::info!(
                "Streaming GPU output in bands of at most {} rows ({} logical bytes instead of {} full-frame bytes)",
                stream_plan.band_rows,
                stream_plan.band_rgba_bytes(),
                stream_plan.legacy_full_rgba_bytes()
            );
        }
        let mask_plan = MaskTexturePlan::new(
            width,
            height,
            request.adjustments.mask_count as usize,
            request.mask_bitmaps.len(),
            MAX_MASKS,
        );
        for (layer, mask_bitmap) in request
            .mask_bitmaps
            .iter()
            .take(mask_plan.upload_layers as usize)
            .enumerate()
        {
            if mask_bitmap.dimensions() != (width, height) {
                return Err(format!(
                    "Mask layer {} dimensions {}x{} do not match render input {}x{}",
                    layer,
                    mask_bitmap.width(),
                    mask_bitmap.height(),
                    width,
                    height
                ));
            }
        }
        let mask_texture = if mask_plan.use_dummy {
            None
        } else {
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Active Mask Tile Texture Array"),
                size: wgpu::Extent3d {
                    width: mask_plan.texture_width,
                    height: mask_plan.texture_height,
                    depth_or_array_layers: mask_plan.layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }))
        };
        let mask_texture_view = mask_texture
            .as_ref()
            .map(|texture| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                })
            })
            .unwrap_or_else(|| self.dummy_mask_view.clone());
        log::debug!(
            "Mask texture: {} layer(s), {}x{} tile / {}x{} source, {} logical byte(s), {} byte(s) avoided, {} uploaded layer(s)",
            mask_plan.layers,
            mask_plan.texture_width,
            mask_plan.texture_height,
            mask_plan.source_width,
            mask_plan.source_height,
            mask_plan.logical_texture_bytes(),
            mask_plan.saved_texture_bytes(),
            mask_plan.upload_layers
        );

        let (lut_texture_view, lut_sampler) = if let Some(lut_arc) = &request.lut {
            let lut_data = &lut_arc.data;
            let size = lut_arc.size;
            let mut rgba_lut_data_f16 = Vec::with_capacity(lut_data.len() / 3 * 4);
            for chunk in lut_data.chunks_exact(3) {
                rgba_lut_data_f16.push(f16::from_f32(chunk[0]));
                rgba_lut_data_f16.push(f16::from_f32(chunk[1]));
                rgba_lut_data_f16.push(f16::from_f32(chunk[2]));
                rgba_lut_data_f16.push(f16::ONE);
            }
            let lut_texture = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("LUT 3D Texture"),
                    size: wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: size,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                TextureDataOrder::MipMajor,
                bytemuck::cast_slice(&rgba_lut_data_f16),
            );
            let view = lut_texture.create_view(&Default::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            (view, sampler)
        } else {
            (self.dummy_lut_view.clone(), self.dummy_lut_sampler.clone())
        };

        let adjustments = request.adjustments;
        if adjustments.global.flare_amount > 0.0 {
            let mut encoder = device.create_command_encoder(&Default::default());

            let aspect_ratio = if height > 0 {
                width as f32 / height as f32
            } else {
                1.0
            };
            let f_params = FlareParams {
                amount: adjustments.global.flare_amount,
                is_raw: adjustments.global.is_raw_image,
                exposure: adjustments.global.exposure,
                brightness: adjustments.global.brightness,
                contrast: adjustments.global.contrast,
                whites: adjustments.global.whites,
                aspect_ratio,
                _pad: 0.0,
            };
            queue.write_buffer(&self.flare_params_buffer, 0, bytemuck::bytes_of(&f_params));

            let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG0"),
                layout: &self.flare_bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_threshold_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.flare_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.flare_sampler),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&self.flare_threshold_pipeline);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.dispatch_workgroups(FLARE_MAP_SIZE / 16, FLARE_MAP_SIZE / 16, 1);
            }

            let bg0_ghosts = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG0 Ghosts"),
                layout: &self.flare_bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_final_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.flare_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.flare_sampler),
                    },
                ],
            });

            let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG1"),
                layout: &self.flare_bgl_1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.flare_threshold_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_ghosts_view),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&self.flare_ghosts_pipeline);
                cpass.set_bind_group(0, &bg0_ghosts, &[]);
                cpass.set_bind_group(1, &bg1, &[]);
                cpass.dispatch_workgroups(FLARE_MAP_SIZE / 16, FLARE_MAP_SIZE / 16, 1);
            }

            queue.submit(Some(encoder.finish()));
        }

        let final_pixel_bytes = (out_width as usize)
            .saturating_mul(out_height as usize)
            .saturating_mul(4);
        let mut final_pixels = vec![
            0u8;
            if skip_cpu_readback || stream_rows {
                0
            } else {
                final_pixel_bytes
            }
        ];

        let start_tile_x = bounds.x / GPU_TILE_SIZE;
        let start_tile_y = bounds.y / GPU_TILE_SIZE;
        let end_tile_x = bounds_right.div_ceil(GPU_TILE_SIZE);
        let end_tile_y = bounds_bottom.div_ceil(GPU_TILE_SIZE);

        for tile_y in start_tile_y..end_tile_y {
            let band_y_start = (tile_y * GPU_TILE_SIZE).max(bounds.y);
            let band_y_end = ((tile_y + 1) * GPU_TILE_SIZE)
                .min(bounds_bottom)
                .min(height);
            let band_height = band_y_end.saturating_sub(band_y_start);
            let mut streamed_band = vec![
                0u8;
                if stream_rows {
                    (out_width as usize)
                        .saturating_mul(band_height as usize)
                        .saturating_mul(4)
                } else {
                    0
                }
            ];

            for tile_x in start_tile_x..end_tile_x {
                let x_start_unclamped = tile_x * GPU_TILE_SIZE;
                let y_start_unclamped = tile_y * GPU_TILE_SIZE;

                let x_start = x_start_unclamped.max(bounds.x);
                let y_start = y_start_unclamped.max(bounds.y);
                let x_end = (x_start_unclamped + GPU_TILE_SIZE)
                    .min(bounds_right)
                    .min(width);
                let y_end = (y_start_unclamped + GPU_TILE_SIZE)
                    .min(bounds_bottom)
                    .min(height);

                let tile_width = x_end - x_start;
                let tile_height = y_end - y_start;

                let input_x_start = (x_start as i32 - GPU_TILE_OVERLAP as i32).max(0) as u32;
                let input_y_start = (y_start as i32 - GPU_TILE_OVERLAP as i32).max(0) as u32;
                let input_x_end = (x_end + GPU_TILE_OVERLAP).min(width);
                let input_y_end = (y_end + GPU_TILE_OVERLAP).min(height);
                let input_width = input_x_end - input_x_start;
                let input_height = input_y_end - input_y_start;

                let input_texture_size = wgpu::Extent3d {
                    width: input_width,
                    height: input_height,
                    depth_or_array_layers: 1,
                };

                if let Some(mask_texture) = &mask_texture {
                    // Queue writes and dispatches are ordered. Refill the reusable mask array with
                    // this input tile's ROI, then let WGSL sample it with `id.xy` local coordinates.
                    if input_width > mask_plan.texture_width
                        || input_height > mask_plan.texture_height
                    {
                        return Err(format!(
                            "Mask tile {}x{} exceeds reusable texture {}x{}",
                            input_width,
                            input_height,
                            mask_plan.texture_width,
                            mask_plan.texture_height
                        ));
                    }
                    let upload = MaskTileUploadPlan::new(
                        width,
                        height,
                        input_x_start,
                        input_y_start,
                        input_width,
                        input_height,
                    )
                    .ok_or_else(|| {
                        format!(
                            "Mask tile x={} y={} {}x{} exceeds source {}x{}",
                            input_x_start, input_y_start, input_width, input_height, width, height
                        )
                    })?;
                    for (layer, mask_bitmap) in request
                        .mask_bitmaps
                        .iter()
                        .take(mask_plan.upload_layers as usize)
                        .enumerate()
                    {
                        queue.write_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: mask_texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d {
                                    x: 0,
                                    y: 0,
                                    z: layer as u32,
                                },
                                aspect: wgpu::TextureAspect::All,
                            },
                            mask_bitmap.as_raw(),
                            wgpu::TexelCopyBufferLayout {
                                offset: upload.source_offset_bytes,
                                bytes_per_row: Some(upload.bytes_per_row),
                                rows_per_image: Some(upload.rows_per_image),
                            },
                            wgpu::Extent3d {
                                width: upload.width,
                                height: upload.height,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                }

                let run_blur = |base_radius: f32, output_view: &wgpu::TextureView| -> bool {
                    let radius = (base_radius * scale).ceil().max(1.0) as u32;
                    if radius == 0 {
                        return false;
                    }

                    let params = BlurParams {
                        radius,
                        tile_offset_x: input_x_start,
                        tile_offset_y: input_y_start,
                        input_width,
                        input_height,
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
                    };
                    queue.write_buffer(&self.blur_params_buffer, 0, bytemuck::bytes_of(&params));

                    let mut blur_encoder = device.create_command_encoder(&Default::default());

                    let h_blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("H-Blur BG"),
                        layout: &self.blur_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(input_texture_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&self.ping_pong_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.blur_params_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    {
                        let mut cpass = blur_encoder.begin_compute_pass(&Default::default());
                        cpass.set_pipeline(&self.h_blur_pipeline);
                        cpass.set_bind_group(0, &h_blur_bg, &[]);
                        cpass.dispatch_workgroups(input_width.div_ceil(256), input_height, 1);
                    }

                    let v_blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("V-Blur BG"),
                        layout: &self.blur_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&self.ping_pong_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(output_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.blur_params_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    {
                        let mut cpass = blur_encoder.begin_compute_pass(&Default::default());
                        cpass.set_pipeline(&self.v_blur_pipeline);
                        cpass.set_bind_group(0, &v_blur_bg, &[]);
                        cpass.dispatch_workgroups(input_width, input_height.div_ceil(256), 1);
                    }

                    queue.submit(Some(blur_encoder.finish()));
                    true
                };

                let did_create_sharpness_blur = run_blur(1.0, &self.sharpness_blur_view);
                let did_create_tonal_blur = run_blur(3.5, &self.tonal_blur_view);
                let did_create_clarity_blur = run_blur(8.0, &self.clarity_blur_view);
                let did_create_structure_blur = run_blur(40.0, &self.structure_blur_view);

                let mut main_encoder = device.create_command_encoder(&Default::default());

                let mut tile_adjustments = adjustments;
                tile_adjustments.tile_offset_x = input_x_start;
                tile_adjustments.tile_offset_y = input_y_start;
                queue.write_buffer(
                    &self.adjustments_buffer,
                    0,
                    bytemuck::bytes_of(&tile_adjustments),
                );

                let mut bind_group_entries = vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.tile_output_texture_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.adjustments_buffer.as_entire_binding(),
                    },
                ];
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&mask_texture_view),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 3 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(&lut_texture_view),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 4 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::Sampler(&lut_sampler),
                });

                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 5 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(if did_create_sharpness_blur {
                        &self.sharpness_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 6 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(if did_create_tonal_blur {
                        &self.tonal_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 7 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(if did_create_clarity_blur {
                        &self.clarity_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 8 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(if did_create_structure_blur {
                        &self.structure_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });

                let use_flare = adjustments.global.flare_amount > 0.0;
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 9 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(if use_flare {
                        &self.flare_ghosts_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 10 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::Sampler(&self.flare_sampler),
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Tile Bind Group"),
                    layout: &self.main_bgl,
                    entries: &bind_group_entries,
                });

                {
                    let mut compute_pass = main_encoder.begin_compute_pass(&Default::default());
                    compute_pass.set_pipeline(&self.main_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups(
                        input_width.div_ceil(8),
                        input_height.div_ceil(8),
                        1,
                    );
                }

                let crop_x_start = x_start - input_x_start;
                let crop_y_start = y_start - input_y_start;

                if output_to_display {
                    main_encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.tile_output_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: crop_x_start,
                                y: crop_y_start,
                                z: 0,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.working_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: x_start,
                                y: y_start,
                                z: 0,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: tile_width,
                            height: tile_height,
                            depth_or_array_layers: 1,
                        },
                    );
                }

                queue.submit(Some(main_encoder.finish()));

                if !skip_cpu_readback {
                    let processed_tile_data = read_texture_data_roi(
                        device,
                        queue,
                        &self.tile_output_texture,
                        wgpu::Origin3d::ZERO,
                        input_texture_size,
                    )?;

                    for row in 0..tile_height {
                        let final_y = y_start + row - bounds.y;
                        let final_x = x_start - bounds.x;
                        let final_row_offset = if stream_rows {
                            ((y_start + row - band_y_start) as usize * out_width as usize
                                + final_x as usize)
                                * 4
                        } else {
                            (final_y as usize * out_width as usize + final_x as usize) * 4
                        };
                        let source_y = crop_y_start + row;
                        let source_row_offset =
                            (source_y * input_width + crop_x_start) as usize * 4;
                        let copy_bytes = (tile_width * 4) as usize;

                        let destination = if stream_rows {
                            &mut streamed_band
                        } else {
                            &mut final_pixels
                        };
                        destination[final_row_offset..final_row_offset + copy_bytes]
                            .copy_from_slice(
                                &processed_tile_data
                                    [source_row_offset..source_row_offset + copy_bytes],
                            );
                    }
                }
            }

            if let GpuRenderOutput::StreamRows(sink) = &mut output {
                let row_bytes = out_width as usize * 4;
                for row in streamed_band.chunks_exact(row_bytes) {
                    sink(row)?;
                }
            }
        }

        Ok((final_pixels, out_width, out_height, bounds.x, bounds.y))
    }
}

fn lock_gpu_resources<'a>(
    context: &GpuContext,
    state: &'a AppState,
    base_image: &DynamicImage,
    transform_hash: u64,
    output_to_display: bool,
) -> Result<LockedGpuResources<'a>, String> {
    let (width, height) = base_image.dimensions();
    let device = &context.device;
    let queue = &context.queue;
    let requested_texture_plan = GpuProcessorTexturePlan::new(width, height, output_to_display);

    let mut processor_lock = state.gpu_processor.lock().unwrap();
    let needs_new_processor = match processor_lock.as_ref() {
        Some(processor) => {
            processor
                .texture_plan()
                .should_rebuild_for(width, height, output_to_display)
        }
        None => true,
    };

    if needs_new_processor {
        let previous_bytes = processor_lock
            .as_ref()
            .map(|processor| processor.texture_plan().logical_texture_bytes())
            .unwrap_or(0);
        log::info!(
            "Creating GPU processor for {}x{} (native display: {}, logical textures: {} -> {} bytes)",
            requested_texture_plan.processing_width,
            requested_texture_plan.processing_height,
            output_to_display,
            previous_bytes,
            requested_texture_plan.logical_texture_bytes()
        );

        drop(processor_lock.take());
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(500)),
        });

        let new_processor = GpuProcessor::new(context.clone(), width, height, output_to_display)?;
        *processor_lock = Some(crate::GpuProcessorState {
            processor: new_processor,
            width: requested_texture_plan.processing_width,
            height: requested_texture_plan.processing_height,
            display_width: requested_texture_plan.display_width,
            display_height: requested_texture_plan.display_height,
        });
    }

    let mut cache_lock = state.gpu_image_cache.lock().unwrap();
    let needs_new_cache = match cache_lock.as_ref() {
        Some(cache) => {
            cache.transform_hash != transform_hash || cache.width != width || cache.height != height
        }
        None => true,
    };

    if needs_new_cache {
        drop(cache_lock.take());
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(500)),
        });

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Input Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let upload_plan = upload_image_to_texture_bounded(queue, &texture, base_image)?;
        log::debug!(
            "GPU input upload {}x{} uses {} row bands of at most {} B (saved {} B vs padded full-frame staging)",
            upload_plan.width,
            upload_plan.height,
            upload_plan.height.div_ceil(upload_plan.band_rows),
            upload_plan.max_staging_bytes,
            upload_plan.saved_staging_bytes(),
        );
        let texture_view = texture.create_view(&Default::default());

        *cache_lock = Some(GpuImageCache {
            texture,
            texture_view,
            width,
            height,
            transform_hash,
        });
    }

    Ok((processor_lock, cache_lock))
}

pub fn process_and_get_dynamic_image(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
) -> Result<DynamicImage, String> {
    let shared = process_and_get_dynamic_image_inner(
        context,
        state,
        base_image,
        transform_hash,
        request,
        caller_id,
        false,
        None,
    )?;
    Arc::try_unwrap(shared)
        .map_err(|_| "Unexpected shared GPU result in non-analytics render".to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn process_and_get_dynamic_image_with_analytics(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
    output_to_display: bool,
    analytics_config: Option<crate::AnalyticsConfig>,
) -> Result<Arc<DynamicImage>, String> {
    process_and_get_dynamic_image_inner(
        context,
        state,
        base_image,
        transform_hash,
        request,
        caller_id,
        output_to_display,
        analytics_config,
    )
}

pub fn process_and_stream_rgba_rows(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
    row_sink: &mut dyn FnMut(&[u8]) -> Result<(), String>,
) -> Result<(u32, u32), String> {
    let start_time = Instant::now();
    let (width, height) = base_image.dimensions();
    let max_dim = context.limits.max_texture_dimension_2d;
    if width > max_dim || height > max_dim {
        return Err(format!(
            "Cannot stream adjusted export {}x{} because the GPU limit is {}",
            width, height, max_dim
        ));
    }

    let (processor_lock, cache_lock) =
        lock_gpu_resources(context, state.inner(), base_image, transform_hash, false)?;
    let processor_state = processor_lock.as_ref().unwrap();
    let cache = cache_lock.as_ref().unwrap();
    let (_, out_width, out_height, _, _) = processor_state.processor.run(
        &cache.texture_view,
        cache.width,
        cache.height,
        request,
        GpuRenderOutput::StreamRows(row_sink),
    )?;

    log::info!(
        "[{}] {}x{} streamed from GPU tiles in {:?}",
        caller_id,
        out_width,
        out_height,
        start_time.elapsed()
    );
    Ok((out_width, out_height))
}

/// Drop export-sized processor and input textures once all batch workers have
/// joined. The native display bind group owns its last presented output view,
/// so the visible frame remains valid until the next preview replaces it.
pub fn reclaim_gpu_resources_after_export(context: &GpuContext, state: &AppState) -> usize {
    let mut processor_lock = state.gpu_processor.lock().unwrap();
    let processor_bytes = processor_lock
        .as_ref()
        .map(|processor| processor.texture_plan().logical_texture_bytes())
        .unwrap_or(0);

    let mut cache_lock = state.gpu_image_cache.lock().unwrap();
    let input_bytes = cache_lock
        .as_ref()
        .map(|cache| {
            (cache.width as usize)
                .saturating_mul(cache.height as usize)
                .saturating_mul(8)
        })
        .unwrap_or(0);

    if !should_reclaim_gpu_resources(processor_bytes, input_bytes) {
        return 0;
    }

    drop(processor_lock.take());
    drop(cache_lock.take());
    drop(cache_lock);
    drop(processor_lock);

    let released_bytes = processor_bytes.saturating_add(input_bytes);
    if released_bytes > 0 {
        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(500)),
        });
        log::info!(
            "Released {} logical bytes of export GPU high-water resources",
            released_bytes
        );
    }
    released_bytes
}

#[allow(clippy::too_many_arguments)]
fn process_and_get_dynamic_image_inner(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
    output_to_display: bool,
    analytics_config: Option<crate::AnalyticsConfig>,
) -> Result<Arc<DynamicImage>, String> {
    let start_time = Instant::now();
    let (width, height) = base_image.dimensions();
    let device = &context.device;
    let queue = &context.queue;

    let max_dim = context.limits.max_texture_dimension_2d;
    if width > max_dim || height > max_dim {
        log::warn!(
            "Image dimensions ({}x{}) exceed GPU limits ({}). Bypassing GPU processing and returning unprocessed image to prevent a crash. Try upgrading your GPU :)",
            width,
            height,
            max_dim
        );
        return Ok(Arc::new(base_image.clone()));
    }

    let (processor_lock, cache_lock) = lock_gpu_resources(
        context,
        state.inner(),
        base_image,
        transform_hash,
        output_to_display,
    )?;
    let processor_state = processor_lock.as_ref().unwrap();
    let processor = &processor_state.processor;
    let cache = cache_lock.as_ref().unwrap();

    let skip_readback = output_to_display;

    let render_output = if output_to_display {
        GpuRenderOutput::NativeDisplay
    } else {
        GpuRenderOutput::CpuImage
    };
    let (processed_pixels, out_w, out_h, out_x, out_y) = processor.run(
        &cache.texture_view,
        cache.width,
        cache.height,
        request,
        render_output,
    )?;

    let mut final_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Final Passes Encoder"),
    });
    let mut submit_final_encoder = false;

    let mut async_readback_buffer: Option<wgpu::Buffer> = None;
    let mut async_padded_bpr: u32 = 0;
    let mut async_unpadded_bpr: u32 = 0;

    if analytics_config.is_some() && skip_readback {
        let unpadded_bytes_per_row = 4 * out_w;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);
        let output_buffer_size = (padded_bytes_per_row * out_h) as u64;

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Async Analytics Readback Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        final_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &processor.working_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: out_x,
                    y: out_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(out_h),
                },
            },
            wgpu::Extent3d {
                width: out_w,
                height: out_h,
                depth_or_array_layers: 1,
            },
        );

        async_readback_buffer = Some(output_buffer);
        async_padded_bpr = padded_bytes_per_row;
        async_unpadded_bpr = unpadded_bytes_per_row;
        submit_final_encoder = true;
    }

    if output_to_display {
        final_encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &processor.working_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: out_x,
                    y: out_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &processor.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: out_x,
                    y: out_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: out_w,
                height: out_h,
                depth_or_array_layers: 1,
            },
        );
        submit_final_encoder = true;
    }

    if submit_final_encoder {
        queue.submit(Some(final_encoder.finish()));
    }

    let mut immediate_analytics = None;
    if let Some(analytics) = analytics_config {
        if let Some(buffer) = async_readback_buffer {
            let output_buffer: wgpu::Buffer = buffer;
            let padded_bytes_per_row: u32 = async_padded_bpr;
            let unpadded_bytes_per_row: u32 = async_unpadded_bpr;
            let device_clone = context.device.clone();

            std::thread::spawn(move || {
                let buffer_slice = output_buffer.slice(..);
                let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();

                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });

                if let Err(e) = device_clone.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(60)),
                }) {
                    log::error!("Async analytics readback poll failed: {}", e);
                    return;
                }

                if let Ok(Ok(())) = rx.recv() {
                    let padded_data = match buffer_slice.get_mapped_range() {
                        Ok(range) => range.to_vec(),
                        Err(e) => {
                            log::error!("Failed to get mapped GPU buffer range: {}", e);
                            return;
                        }
                    };
                    output_buffer.unmap();

                    let mut unpadded_data =
                        Vec::with_capacity((unpadded_bytes_per_row * out_h) as usize);
                    if padded_bytes_per_row == unpadded_bytes_per_row {
                        unpadded_data = padded_data;
                    } else {
                        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
                            unpadded_data
                                .extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
                        }
                    }

                    if let Some(img_buf) =
                        ImageBuffer::<Rgba<u8>, _>::from_raw(out_w, out_h, unpadded_data)
                    {
                        let dynamic_img = DynamicImage::ImageRgba8(img_buf);
                        let _ = analytics.sender.send(crate::AnalyticsJob {
                            path: analytics.path,
                            image: std::sync::Arc::new(dynamic_img),
                            compute_waveform: analytics.compute_waveform,
                            active_waveform_channel: analytics.active_waveform_channel,
                        });
                    }
                }
            });
        } else {
            immediate_analytics = Some(analytics);
        }
    }

    if output_to_display
        && let Ok(mut display_lock) = context.display.lock()
        && let Some(display) = display_lock.as_mut()
    {
        display.latest_transform.image_size = [width as f32, height as f32];
        display.latest_transform.texture_size = [
            processor_state.display_width as f32,
            processor_state.display_height as f32,
        ];

        queue.write_buffer(
            &display.transform_buffer,
            0,
            bytemuck::bytes_of(&display.latest_transform),
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &display.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: display.transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&processor.output_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&display.sampler),
                },
            ],
            label: None,
        });
        display.current_bind_group = Some(bind_group);
        display.render(device, queue);
    }

    if skip_readback {
        let duration = start_time.elapsed();
        let fps = 1.0 / duration.as_secs_f64();
        log::info!(
            "[{}] {}x{} native WGPU display updated in {:?} ({:.2} FPS)",
            caller_id,
            width,
            height,
            duration,
            fps
        );
        return Ok(Arc::new(DynamicImage::new_rgba8(0, 0)));
    }

    let duration = start_time.elapsed();
    let fps = 1.0 / duration.as_secs_f64();
    log::info!(
        "[{}] {}x{} processed (ROI: {}x{}) on GPU in {:?} ({:.2} FPS)",
        caller_id,
        width,
        height,
        out_w,
        out_h,
        duration,
        fps
    );

    let img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(out_w, out_h, processed_pixels)
        .ok_or("Failed to create image buffer from GPU data")?;
    let shared_image = Arc::new(DynamicImage::ImageRgba8(img_buf));

    if let Some(analytics) = immediate_analytics {
        let _ = analytics.sender.send(crate::AnalyticsJob {
            path: analytics.path,
            image: Arc::clone(&shared_image),
            compute_waveform: analytics.compute_waveform,
            active_waveform_channel: analytics.active_waveform_channel,
        });
    }

    Ok(shared_image)
}

#[cfg(test)]
mod shader_tests {
    use std::mem::size_of;
    use std::sync::{Arc, Mutex};

    use half::f16;
    use image::{DynamicImage, GrayImage, Luma, Rgb32FImage, Rgba32FImage};
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    use wgpu::util::{DeviceExt, TextureDataOrder};

    use super::{GpuProcessor, GpuRenderOutput, RenderRequest, to_rgba_f16, write_rgba_f16_row};
    use crate::image_processing::{AllAdjustments, GpuContext};

    #[test]
    fn bundled_wgsl_modules_parse_and_validate() {
        for (name, source) in [
            ("display", include_str!("shaders/display.wgsl")),
            ("blur", include_str!("shaders/blur.wgsl")),
            ("flare", include_str!("shaders/flare.wgsl")),
            ("image-processing", include_str!("shaders/shader.wgsl")),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name} WGSL failed to parse: {error}"));
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|error| panic!("{name} WGSL failed validation: {error}"));
        }
    }

    #[test]
    fn float_images_convert_directly_to_the_existing_rgba_f16_contract() {
        let images = [
            DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(7, 5, |x, y| {
                image::Rgb([
                    (x * 3 + y) as f32 / 31.0,
                    (x + y * 5) as f32 / 37.0,
                    (x * 7 + y * 11) as f32 / 127.0,
                ])
            })),
            DynamicImage::ImageRgba32F(Rgba32FImage::from_fn(7, 5, |x, y| {
                image::Rgba([
                    (x * 3 + y) as f32 / 31.0,
                    (x + y * 5) as f32 / 37.0,
                    (x * 7 + y * 11) as f32 / 127.0,
                    (x * 13 + y * 2) as f32 / 97.0,
                ])
            })),
        ];

        for image in images {
            let expected = image
                .to_rgba32f()
                .into_raw()
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>();
            assert_eq!(to_rgba_f16(&image), expected);
        }
    }

    fn assert_bounded_upload_matches_reference(image: &DynamicImage) {
        let plan = crate::render_strategy::GpuInputUploadPlan::new(image.width(), image.height())
            .expect("upload plan");
        let padded_row_samples = plan.padded_bytes_per_row as usize / std::mem::size_of::<f16>();
        let row_samples = image.width() as usize * 4;
        let mut streamed = Vec::with_capacity(row_samples * image.height() as usize);
        for y in 0..image.height() {
            let mut row = vec![f16::ZERO; padded_row_samples];
            write_rgba_f16_row(image, y, &mut row, image.width() as usize)
                .expect("convert bounded upload row");
            streamed.extend_from_slice(&row[..row_samples]);
        }

        assert_eq!(streamed, to_rgba_f16(image));
    }

    #[test]
    fn bounded_upload_rows_match_the_full_float_conversion_reference() {
        let images = [
            DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(7, 5, |x, y| {
                image::Rgb([
                    (x * 3 + y) as f32 / 31.0,
                    (x + y * 5) as f32 / 37.0,
                    (x * 7 + y * 11) as f32 / 127.0,
                ])
            })),
            DynamicImage::ImageRgba32F(Rgba32FImage::from_fn(7, 5, |x, y| {
                image::Rgba([
                    (x * 3 + y) as f32 / 31.0,
                    (x + y * 5) as f32 / 37.0,
                    (x * 7 + y * 11) as f32 / 127.0,
                    (x * 13 + y * 2) as f32 / 97.0,
                ])
            })),
        ];

        for image in images {
            assert_bounded_upload_matches_reference(&image);
        }
    }

    #[test]
    fn bounded_upload_rows_preserve_integer_image_conversion_reference() {
        let images = [
            DynamicImage::ImageLuma8(image::GrayImage::from_fn(7, 5, |x, y| {
                image::Luma([(x * 17 + y * 5) as u8])
            })),
            DynamicImage::ImageLumaA8(image::GrayAlphaImage::from_fn(7, 5, |x, y| {
                image::LumaA([(x * 17 + y * 5) as u8, (x * 23 + y * 7) as u8])
            })),
            DynamicImage::ImageRgb8(image::RgbImage::from_fn(7, 5, |x, y| {
                image::Rgb([
                    (x * 17 + y * 5) as u8,
                    (x * 23 + y * 7) as u8,
                    (x * 29 + y * 11) as u8,
                ])
            })),
            DynamicImage::ImageRgba8(image::RgbaImage::from_fn(7, 5, |x, y| {
                image::Rgba([
                    (x * 17 + y * 5) as u8,
                    (x * 23 + y * 7) as u8,
                    (x * 29 + y * 11) as u8,
                    (x * 31 + y * 13) as u8,
                ])
            })),
            DynamicImage::ImageLuma16(image::ImageBuffer::from_fn(7, 5, |x, y| {
                image::Luma([(x * 1_003 + y * 257) as u16])
            })),
            DynamicImage::ImageLumaA16(image::ImageBuffer::from_fn(7, 5, |x, y| {
                image::LumaA([(x * 1_003 + y * 257) as u16, (x * 1_307 + y * 313) as u16])
            })),
            DynamicImage::ImageRgb16(image::ImageBuffer::from_fn(7, 5, |x, y| {
                image::Rgb([
                    (x * 1_003 + y * 257) as u16,
                    (x * 1_307 + y * 313) as u16,
                    (x * 1_711 + y * 401) as u16,
                ])
            })),
            DynamicImage::ImageRgba16(image::ImageBuffer::from_fn(7, 5, |x, y| {
                image::Rgba([
                    (x * 1_003 + y * 257) as u16,
                    (x * 1_307 + y * 313) as u16,
                    (x * 1_711 + y * 401) as u16,
                    (x * 1_913 + y * 433) as u16,
                ])
            })),
        ];

        for image in images {
            assert_bounded_upload_matches_reference(&image);
        }
    }

    #[test]
    fn direct_float_upload_removes_60mp_rgba32f_cpu_staging() {
        const WIDTH: u64 = 9_504;
        const HEIGHT: u64 = 6_336;
        const RGBA32F_BYTES: u64 = 963_477_504;
        const RGBA16F_BYTES: u64 = 481_738_752;

        let pixels = WIDTH * HEIGHT;
        assert_eq!(pixels * 4 * size_of::<f32>() as u64, RGBA32F_BYTES);
        assert_eq!(pixels * 4 * size_of::<f16>() as u64, RGBA16F_BYTES);
        assert_eq!(RGBA32F_BYTES + RGBA16F_BYTES, 1_445_216_256);
        assert_eq!(RGBA32F_BYTES + RGBA16F_BYTES - RGBA16F_BYTES, RGBA32F_BYTES);
    }

    #[test]
    #[ignore = "manual local-GPU mask tile pixel regression"]
    fn tiled_mask_gpu_sampling_matches_source_pixels_across_tile_boundary() {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            ..Default::default()
        }))
        .expect("local GPU adapter for mask tile regression");
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Mask Tile Regression Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .expect("request local GPU device for mask tile regression");
        let context = GpuContext {
            device: Arc::new(device),
            queue: Arc::new(queue),
            limits,
            display: Arc::new(Mutex::new(None)),
        };

        for (width, height) in [(2_305_u32, 8_u32), (8, 2_050)] {
            let processor = GpuProcessor::new(context.clone(), width, height, false)
                .expect("create GPU processor for mask tile regression");

            let mut source = Vec::with_capacity(width as usize * height as usize * 4);
            for _ in 0..u64::from(width) * u64::from(height) {
                source.extend([
                    f16::from_f32(0.18),
                    f16::from_f32(0.22),
                    f16::from_f32(0.26),
                    f16::ONE,
                ]);
            }
            let input_texture = context.device.create_texture_with_data(
                &context.queue,
                &wgpu::TextureDescriptor {
                    label: Some("Mask Tile Regression Input"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                TextureDataOrder::MipMajor,
                bytemuck::cast_slice(&source),
            );
            let input_view = input_texture.create_view(&Default::default());

            let mut base_adjustments = AllAdjustments::default();
            base_adjustments.global.is_raw_image = 1;
            let (base, ..) = processor
                .run(
                    &input_view,
                    width,
                    height,
                    RenderRequest {
                        adjustments: base_adjustments,
                        mask_bitmaps: &[],
                        lut: None,
                        roi: None,
                    },
                    GpuRenderOutput::CpuImage,
                )
                .expect("render unmasked GPU reference");

            let horizontal_case = width > height;
            let mask = Arc::new(GrayImage::from_fn(width, height, |x, y| {
                let axis = if horizontal_case { x } else { y };
                Luma([if (axis / 64) % 2 == 0 { 255 } else { 0 }])
            }));
            let mut masked_adjustments = base_adjustments;
            masked_adjustments.mask_count = 1;
            masked_adjustments.mask_adjustments[0].exposure = 1.0;
            let (masked, ..) = processor
                .run(
                    &input_view,
                    width,
                    height,
                    RenderRequest {
                        adjustments: masked_adjustments,
                        mask_bitmaps: std::slice::from_ref(&mask),
                        lut: None,
                        roi: None,
                    },
                    GpuRenderOutput::CpuImage,
                )
                .expect("render tiled GPU mask");

            assert_eq!(base.len(), masked.len());
            for y in 0..height {
                for x in 0..width {
                    let offset = (y as usize * width as usize + x as usize) * 4;
                    let base_pixel = &base[offset..offset + 4];
                    let masked_pixel = &masked[offset..offset + 4];
                    if mask.get_pixel(x, y)[0] == 0 {
                        assert_eq!(
                            masked_pixel, base_pixel,
                            "zero mask changed pixel ({x}, {y}) in {width}x{height} case"
                        );
                    } else {
                        assert!(
                            masked_pixel[0] > base_pixel[0],
                            "white mask did not apply exposure at ({x}, {y}) in {width}x{height} case: {base_pixel:?} -> {masked_pixel:?}"
                        );
                        assert_eq!(masked_pixel[3], base_pixel[3]);
                    }
                }
            }

            for axis in [2_047, 2_048, 2_049] {
                let (x, y) = if horizontal_case {
                    (axis, 0)
                } else {
                    (0, axis)
                };
                let offset = (y as usize * width as usize + x as usize) * 4;
                let changed = masked[offset] > base[offset];
                assert_eq!(
                    changed,
                    mask.get_pixel(x, y)[0] == 255,
                    "tile seam at ({x}, {y}) in {width}x{height} case"
                );
            }
        }
    }
}
