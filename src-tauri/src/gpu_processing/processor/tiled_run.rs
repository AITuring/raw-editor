//! `GpuProcessor::run`: the tiled render pass.
//!
//! Validates the render bounds, uploads masks and the LUT, renders the flare map,
//! then renders the bounds tile by tile into CPU pixels, the native display
//! texture or streamed rows.

use crate::gpu_processing::{GpuRenderOutput, RenderRequest, Roi, read_texture_data_roi};
use crate::image_processing::MAX_MASKS;
use crate::render_strategy::GPU_TILE_OVERLAP;
use crate::render_strategy::GPU_TILE_SIZE;
use crate::render_strategy::MaskTexturePlan;
use crate::render_strategy::MaskTileUploadPlan;
use crate::render_strategy::StreamingExportBufferPlan;
use half::f16;
use wgpu::util::DeviceExt;
use wgpu::util::TextureDataOrder;

use super::{BlurParams, FLARE_MAP_SIZE, FlareParams, GpuProcessor};

impl GpuProcessor {
    pub(in crate::gpu_processing) fn run(
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
