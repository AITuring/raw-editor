//! The GPU processor: pipelines, textures and the tiled render pass.
//!
//! `GpuProcessor::new` builds the blur, flare and main compute pipelines with
//! their working textures; `run` renders one input tile by tile into a CPU
//! image, the native display or streamed rows. Each method has its own
//! `impl GpuProcessor` block: `new` in `pipelines`, `run` in `tiled_run`.

use crate::image_processing::GpuContext;

mod pipelines;
mod tiled_run;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BlurParams {
    pub(super) radius: u32,
    pub(super) tile_offset_x: u32,
    pub(super) tile_offset_y: u32,
    pub(super) input_width: u32,
    pub(super) input_height: u32,
    pub(super) _pad1: u32,
    pub(super) _pad2: u32,
    pub(super) _pad3: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FlareParams {
    pub(super) amount: f32,
    pub(super) is_raw: u32,
    pub(super) exposure: f32,
    pub(super) brightness: f32,
    pub(super) contrast: f32,
    pub(super) whites: f32,
    pub(super) aspect_ratio: f32,
    pub(super) _pad: f32,
}

pub struct GpuProcessor {
    pub(super) context: GpuContext,
    pub(super) blur_bgl: wgpu::BindGroupLayout,
    pub(super) h_blur_pipeline: wgpu::ComputePipeline,
    pub(super) v_blur_pipeline: wgpu::ComputePipeline,
    pub(super) blur_params_buffer: wgpu::Buffer,

    pub(super) flare_bgl_0: wgpu::BindGroupLayout,
    pub(super) flare_bgl_1: wgpu::BindGroupLayout,
    pub(super) flare_threshold_pipeline: wgpu::ComputePipeline,
    pub(super) flare_ghosts_pipeline: wgpu::ComputePipeline,
    pub(super) flare_params_buffer: wgpu::Buffer,
    pub(super) flare_threshold_view: wgpu::TextureView,
    pub(super) flare_ghosts_view: wgpu::TextureView,
    pub(super) flare_final_view: wgpu::TextureView,
    pub(super) flare_sampler: wgpu::Sampler,

    pub(super) main_bgl: wgpu::BindGroupLayout,
    pub(super) main_pipeline: wgpu::ComputePipeline,
    pub(super) adjustments_buffer: wgpu::Buffer,
    pub(super) dummy_blur_view: wgpu::TextureView,
    pub(super) dummy_mask_view: wgpu::TextureView,
    pub(super) dummy_lut_view: wgpu::TextureView,
    pub(super) dummy_lut_sampler: wgpu::Sampler,
    pub(super) ping_pong_view: wgpu::TextureView,
    pub(super) sharpness_blur_view: wgpu::TextureView,
    pub(super) tonal_blur_view: wgpu::TextureView,
    pub(super) clarity_blur_view: wgpu::TextureView,
    pub(super) structure_blur_view: wgpu::TextureView,

    pub tile_output_texture: wgpu::Texture,
    pub tile_output_texture_view: wgpu::TextureView,
    pub working_texture: wgpu::Texture,
    pub working_texture_view: wgpu::TextureView,
    pub output_texture: wgpu::Texture,
    pub output_texture_view: wgpu::TextureView,
}

pub(super) const FLARE_MAP_SIZE: u32 = 512;
