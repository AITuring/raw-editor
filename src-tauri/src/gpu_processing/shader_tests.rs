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
        ("display", include_str!("../shaders/display.wgsl")),
        ("blur", include_str!("../shaders/blur.wgsl")),
        ("flare", include_str!("../shaders/flare.wgsl")),
        ("image-processing", include_str!("../shaders/shader.wgsl")),
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

#[test]
#[ignore = "manual local-GPU color transfer parity regression"]
fn gpu_srgb_and_raw_color_transfer_match_cpu_reference() {
    use crate::color_management::{linear_to_srgb_channel, srgb_to_linear_channel};

    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        ..Default::default()
    }))
    .expect("local GPU adapter for color transfer parity");
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Color Transfer Parity Device"),
        required_features: wgpu::Features::empty(),
        required_limits: limits.clone(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .expect("request local GPU device for color transfer parity");
    let context = GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        limits,
        display: Arc::new(Mutex::new(None)),
    };

    let patches: [[u8; 3]; 8] = [
        [0, 0, 0],
        [24, 24, 24],
        [118, 118, 118],
        [255, 255, 255],
        [204, 52, 48],
        [54, 172, 76],
        [48, 104, 212],
        [192, 148, 96],
    ];
    let width = 8_u32;
    let height = 2_u32;
    let processor = GpuProcessor::new(context.clone(), width, height, false)
        .expect("create GPU processor for color parity");

    let make_view = |pixels: &[f16], label: &str| {
        let texture = context.device.create_texture_with_data(
            &context.queue,
            &wgpu::TextureDescriptor {
                label: Some(label),
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
            bytemuck::cast_slice(pixels),
        );
        texture.create_view(&Default::default())
    };

    let mut srgb_texels = Vec::with_capacity(width as usize * height as usize * 4);
    let mut linear_texels = Vec::with_capacity(width as usize * height as usize * 4);
    for _row in 0..height {
        for &[r, g, b] in &patches {
            let sr = r as f32 / 255.0;
            let sg = g as f32 / 255.0;
            let sb = b as f32 / 255.0;
            srgb_texels.extend([
                f16::from_f32(sr),
                f16::from_f32(sg),
                f16::from_f32(sb),
                f16::ONE,
            ]);
            linear_texels.extend([
                f16::from_f32(srgb_to_linear_channel(sr)),
                f16::from_f32(srgb_to_linear_channel(sg)),
                f16::from_f32(srgb_to_linear_channel(sb)),
                f16::ONE,
            ]);
        }
    }
    let srgb_view = make_view(&srgb_texels, "Color Parity sRGB Input");
    let linear_view = make_view(&linear_texels, "Color Parity Linear RAW Input");

    // 1. Neutral non-RAW sRGB round-trip must match input within 1 LSB.
    let mut neutral_srgb = AllAdjustments::default();
    neutral_srgb.global.is_raw_image = 0;
    neutral_srgb.global.tonemapper_mode = 0;
    let (gpu_neutral, ..) = processor
        .run(
            &srgb_view,
            width,
            height,
            RenderRequest {
                adjustments: neutral_srgb,
                mask_bitmaps: &[],
                lut: None,
                roi: None,
            },
            GpuRenderOutput::CpuImage,
        )
        .expect("render neutral sRGB");
    for (idx, &[r, g, b]) in patches.iter().enumerate() {
        let px = &gpu_neutral[idx * 4..idx * 4 + 4];
        assert!(
            px[0].abs_diff(r) <= 1,
            "neutral R patch {idx}: got {}, expected {r}",
            px[0]
        );
        assert!(
            px[1].abs_diff(g) <= 1,
            "neutral G patch {idx}: got {}, expected {g}",
            px[1]
        );
        assert!(
            px[2].abs_diff(b) <= 1,
            "neutral B patch {idx}: got {}, expected {b}",
            px[2]
        );
    }

    // 2. +1 EV linear exposure on non-RAW sRGB must match CPU EOTF -> *2 -> OETF within 1 LSB.
    let mut exposed_srgb = neutral_srgb;
    exposed_srgb.global.exposure = 1.0;
    let (gpu_exposed, ..) = processor
        .run(
            &srgb_view,
            width,
            height,
            RenderRequest {
                adjustments: exposed_srgb,
                mask_bitmaps: &[],
                lut: None,
                roi: None,
            },
            GpuRenderOutput::CpuImage,
        )
        .expect("render +1 EV sRGB");
    for (idx, &rgb) in patches.iter().enumerate() {
        let px = &gpu_exposed[idx * 4..idx * 4 + 4];
        for c in 0..3 {
            let expected = (linear_to_srgb_channel(
                (srgb_to_linear_channel(rgb[c] as f32 / 255.0) * 2.0).clamp(0.0, 1.0),
            ) * 255.0)
                .round() as u8;
            assert!(
                px[c].abs_diff(expected) <= 1,
                "+1 EV patch {idx} ch {c}: GPU {}, CPU {expected}",
                px[c]
            );
        }
    }

    // 3. Under AgX tonemapper, is_raw=0 (sRGB input) and is_raw=1 (CPU-linearized input)
    // must produce identical RGBA8Unorm output within 1 LSB.
    let mut agx_srgb = AllAdjustments::default();
    agx_srgb.global.is_raw_image = 0;
    agx_srgb.global.tonemapper_mode = 1;
    let mut agx_raw = agx_srgb;
    agx_raw.global.is_raw_image = 1;

    let (gpu_agx_srgb, ..) = processor
        .run(
            &srgb_view,
            width,
            height,
            RenderRequest {
                adjustments: agx_srgb,
                mask_bitmaps: &[],
                lut: None,
                roi: None,
            },
            GpuRenderOutput::CpuImage,
        )
        .expect("render AgX sRGB");
    let (gpu_agx_raw, ..) = processor
        .run(
            &linear_view,
            width,
            height,
            RenderRequest {
                adjustments: agx_raw,
                mask_bitmaps: &[],
                lut: None,
                roi: None,
            },
            GpuRenderOutput::CpuImage,
        )
        .expect("render AgX RAW");
    for idx in 0..patches.len() {
        for c in 0..3 {
            let srgb_out = gpu_agx_srgb[idx * 4 + c];
            let raw_out = gpu_agx_raw[idx * 4 + c];
            assert!(
                srgb_out.abs_diff(raw_out) <= 1,
                "AgX sRGB vs RAW mismatch patch {idx} ch {c}: sRGB={srgb_out}, RAW={raw_out}"
            );
        }
    }
}
