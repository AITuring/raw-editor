//! Render entry points over the shared GPU processor.
//!
//! Locks (and rebuilds when needed) the processor and the cached input
//! texture, renders one request to a CPU image, the native display or
//! streamed export rows, and reclaims export-sized resources afterwards.

use crate::AppState;
use crate::GpuImageCache;
use crate::image_processing::GpuContext;
use crate::render_strategy::GpuProcessorTexturePlan;
use crate::render_strategy::should_reclaim_gpu_resources;
use image::DynamicImage;
use image::ImageBuffer;
use image::Rgba;
use std::sync::Arc;
use std::sync::MutexGuard;
use std::time::Instant;

use super::{
    GpuInputSource, GpuProcessor, GpuRenderOutput, RenderRequest, upload_source_to_texture_bounded,
};

pub(super) type GpuProcessorGuard<'a> = MutexGuard<'a, Option<crate::GpuProcessorState>>;

pub(super) type GpuImageCacheGuard<'a> = MutexGuard<'a, Option<GpuImageCache>>;

pub(super) type LockedGpuResources<'a> = (GpuProcessorGuard<'a>, GpuImageCacheGuard<'a>);

pub(super) fn lock_gpu_resources<'a>(
    context: &GpuContext,
    state: &'a AppState,
    base_image: &dyn GpuInputSource,
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
        let upload_plan = upload_source_to_texture_bounded(queue, &texture, base_image)?;
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
    base_image: &dyn GpuInputSource,
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
    base_image: &dyn GpuInputSource,
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
    base_image: &dyn GpuInputSource,
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
pub(super) fn process_and_get_dynamic_image_inner(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &dyn GpuInputSource,
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
        return base_image
            .materialize_cpu_fallback()
            .map(Arc::new)
            .ok_or_else(|| {
                format!(
                    "Cannot fall back to a materialized image for streamed GPU input {}x{}",
                    width, height
                )
            });
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
