//! Sky, foreground and depth model passes.
//!
//! Each model runs on a letterboxed square input and its output is normalised
//! back to a grayscale mask or depth map.

use anyhow::Result;
use image::DynamicImage;
use image::GenericImageView;
use image::GrayImage;
use image::imageops;
use image::imageops::FilterType;
use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;

use super::{DEPTH_INPUT_SIZE, SKYSEG_INPUT_SIZE, U2NETP_INPUT_SIZE};

pub fn run_sky_seg_model(
    image: &DynamicImage,
    sky_seg_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let (orig_width, orig_height) = image.dimensions();

    let resized_image = image.resize(SKYSEG_INPUT_SIZE, SKYSEG_INPUT_SIZE, FilterType::Triangle);
    let (resized_w, resized_h) = resized_image.dimensions();
    let resized_rgb = resized_image.into_rgb8();
    let raw_pixels = resized_rgb.as_raw();

    let paste_x = ((SKYSEG_INPUT_SIZE - resized_w) / 2) as usize;
    let paste_y = ((SKYSEG_INPUT_SIZE - resized_h) / 2) as usize;

    let mut input_tensor: Array<f32, _> =
        Array::zeros((1, 3, SKYSEG_INPUT_SIZE as usize, SKYSEG_INPUT_SIZE as usize));

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let rw = resized_w as usize;
    let rh = resized_h as usize;

    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            let dest_y = y + paste_y;
            let dest_x = x + paste_x;

            input_tensor[[0, 0, dest_y, dest_x]] =
                (raw_pixels[idx] as f32 / 255.0 - mean[0]) / std[0];
            input_tensor[[0, 1, dest_y, dest_x]] =
                (raw_pixels[idx + 1] as f32 / 255.0 - mean[1]) / std[1];
            input_tensor[[0, 2, dest_y, dest_x]] =
                (raw_pixels[idx + 2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let t_input = Tensor::from_array(input_tensor_dyn.as_standard_layout().into_owned())?;

    let mut session = sky_seg_session.lock().unwrap();
    let outputs = session.run(ort::inputs![t_input])?;
    let output_tensor = outputs[0].try_extract_array::<f32>()?.to_owned();
    let out_slice = output_tensor.as_slice().unwrap();

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in out_slice {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }

    let range = max_val - min_val;
    let scale = if range > 1e-6 { 255.0 / range } else { 0.0 };

    let usize_size = SKYSEG_INPUT_SIZE as usize;
    let mut cropped_mask_data = Vec::with_capacity(rw * rh);

    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            let pixel = if range > 1e-6 {
                ((val - min_val) * scale) as u8
            } else {
                0
            };
            cropped_mask_data.push(pixel);
        }
    }

    let cropped_mask = GrayImage::from_raw(resized_w, resized_h, cropped_mask_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask from Sky Segmentation output"))?;

    let final_mask = imageops::resize(&cropped_mask, orig_width, orig_height, FilterType::Triangle);

    Ok(final_mask)
}

pub fn run_u2netp_model(
    image: &DynamicImage,
    u2netp_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let (orig_width, orig_height) = image.dimensions();

    let resized_image = image.resize(U2NETP_INPUT_SIZE, U2NETP_INPUT_SIZE, FilterType::Triangle);
    let (resized_w, resized_h) = resized_image.dimensions();
    let resized_rgb = resized_image.into_rgb8();
    let raw_pixels = resized_rgb.as_raw();

    let paste_x = ((U2NETP_INPUT_SIZE - resized_w) / 2) as usize;
    let paste_y = ((U2NETP_INPUT_SIZE - resized_h) / 2) as usize;

    let mut input_tensor: Array<f32, _> =
        Array::zeros((1, 3, U2NETP_INPUT_SIZE as usize, U2NETP_INPUT_SIZE as usize));

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let rw = resized_w as usize;
    let rh = resized_h as usize;

    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            let dest_y = y + paste_y;
            let dest_x = x + paste_x;

            input_tensor[[0, 0, dest_y, dest_x]] =
                (raw_pixels[idx] as f32 / 255.0 - mean[0]) / std[0];
            input_tensor[[0, 1, dest_y, dest_x]] =
                (raw_pixels[idx + 1] as f32 / 255.0 - mean[1]) / std[1];
            input_tensor[[0, 2, dest_y, dest_x]] =
                (raw_pixels[idx + 2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let t_input = Tensor::from_array(input_tensor_dyn.as_standard_layout().into_owned())?;

    let mut session = u2netp_session.lock().unwrap();
    let outputs = session.run(ort::inputs![t_input])?;
    let output_tensor = outputs[0].try_extract_array::<f32>()?.to_owned();
    let out_slice = output_tensor.as_slice().unwrap();

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in out_slice {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }

    let range = max_val - min_val;
    let scale = if range > 1e-6 { 255.0 / range } else { 0.0 };

    let usize_size = U2NETP_INPUT_SIZE as usize;
    let mut cropped_mask_data = Vec::with_capacity(rw * rh);

    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            let pixel = if range > 1e-6 {
                ((val - min_val) * scale) as u8
            } else {
                0
            };
            cropped_mask_data.push(pixel);
        }
    }

    let cropped_mask = GrayImage::from_raw(resized_w, resized_h, cropped_mask_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask from U-2-Netp output"))?;

    let final_mask = imageops::resize(&cropped_mask, orig_width, orig_height, FilterType::Triangle);

    Ok(final_mask)
}

pub fn run_depth_anything_model(
    image: &DynamicImage,
    depth_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let resized_image = image.resize(DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE, FilterType::Triangle);
    let (resized_w, resized_h) = resized_image.dimensions();
    let resized_rgb = resized_image.into_rgb8();
    let raw_pixels = resized_rgb.as_raw();

    let paste_x = ((DEPTH_INPUT_SIZE - resized_w) / 2) as usize;
    let paste_y = ((DEPTH_INPUT_SIZE - resized_h) / 2) as usize;

    let mut input_tensor: Array<f32, _> =
        Array::zeros((1, 3, DEPTH_INPUT_SIZE as usize, DEPTH_INPUT_SIZE as usize));

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let rw = resized_w as usize;
    let rh = resized_h as usize;

    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            let dest_y = y + paste_y;
            let dest_x = x + paste_x;

            input_tensor[[0, 0, dest_y, dest_x]] =
                (raw_pixels[idx] as f32 / 255.0 - mean[0]) / std[0];
            input_tensor[[0, 1, dest_y, dest_x]] =
                (raw_pixels[idx + 1] as f32 / 255.0 - mean[1]) / std[1];
            input_tensor[[0, 2, dest_y, dest_x]] =
                (raw_pixels[idx + 2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let t_input = Tensor::from_array(input_tensor_dyn.as_standard_layout().into_owned())?;

    let mut session = depth_session.lock().unwrap();
    let outputs = session.run(ort::inputs![t_input])?;
    let output_tensor = outputs[0].try_extract_array::<f32>()?.to_owned();
    let out_slice = output_tensor.as_slice().unwrap();

    let usize_size = DEPTH_INPUT_SIZE as usize;

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    let range = max_val - min_val;
    let scale = if range > 1e-6 { 255.0 / range } else { 0.0 };

    let mut cropped_depth_data = Vec::with_capacity(rw * rh);

    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            let pixel = if range > 1e-6 {
                ((val - min_val) * scale) as u8
            } else {
                0
            };
            cropped_depth_data.push(pixel);
        }
    }

    let depth_map = GrayImage::from_raw(resized_w, resized_h, cropped_depth_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask from Depth output"))?;

    Ok(depth_map)
}
