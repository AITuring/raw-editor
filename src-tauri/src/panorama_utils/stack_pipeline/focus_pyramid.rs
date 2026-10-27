//! Offline FocusPyramid reference provider for task 21.3.
//!
//! This is intentionally a value-level comparison provider.  It operates on
//! proxy luma planes, keeps the winner map and pyramid bands observable, and
//! never changes the native hard-ownership compositor.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FocusPyramidConfig {
    pub levels: usize,
    pub energy_radius: usize,
}

impl Default for FocusPyramidConfig {
    fn default() -> Self {
        Self {
            levels: 5,
            energy_radius: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyramidPlane {
    pub width: usize,
    pub height: usize,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FocusPyramidResult {
    pub width: usize,
    pub height: usize,
    pub levels: Vec<Vec<PyramidPlane>>,
    pub energy_maps: Vec<Vec<f32>>,
    pub winner_map: Vec<u16>,
}

fn checked_len(width: usize, height: usize) -> Result<usize, String> {
    width
        .checked_mul(height)
        .ok_or_else(|| "focus pyramid dimensions overflow".to_string())
}

fn downsample(plane: &PyramidPlane) -> PyramidPlane {
    let width = plane.width.div_ceil(2).max(1);
    let height = plane.height.div_ceil(2).max(1);
    let mut samples = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut count: f32 = 0.0;
            for source_y in (y * 2)..((y * 2 + 2).min(plane.height)) {
                for source_x in (x * 2)..((x * 2 + 2).min(plane.width)) {
                    sum += plane.samples[source_y * plane.width + source_x];
                    count += 1.0;
                }
            }
            samples[y * width + x] = sum / count.max(1.0);
        }
    }
    PyramidPlane {
        width,
        height,
        samples,
    }
}

fn laplacian_energy(plane: &PyramidPlane, radius: usize) -> Vec<f32> {
    let mut energy = vec![0.0; plane.samples.len()];
    for y in 0..plane.height {
        for x in 0..plane.width {
            let center = plane.samples[y * plane.width + x];
            let left = plane.samples[y * plane.width + x.saturating_sub(1)];
            let right = plane.samples[y * plane.width + (x + 1).min(plane.width - 1)];
            let top = plane.samples[y.saturating_sub(1) * plane.width + x];
            let bottom = plane.samples[(y + 1).min(plane.height - 1) * plane.width + x];
            let laplacian = (left + right + top + bottom) - 4.0 * center;
            let mut sum = 0.0;
            let mut count: f32 = 0.0;
            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius + 1).min(plane.height);
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(plane.width);
            for sample_y in y0..y1 {
                for sample_x in x0..x1 {
                    let neighbour = plane.samples[sample_y * plane.width + sample_x];
                    let local = ((neighbour - center).abs() + laplacian.abs()) * 0.5;
                    sum += local;
                    count += 1.0;
                }
            }
            energy[y * plane.width + x] = sum / count.max(1.0);
        }
    }
    energy
}

fn upsample_add(
    target: &mut [f32],
    width: usize,
    height: usize,
    source: &[f32],
    source_width: usize,
    source_height: usize,
    weight: f32,
) {
    for y in 0..height {
        let source_y = (y * source_height / height).min(source_height - 1);
        for x in 0..width {
            let source_x = (x * source_width / width).min(source_width - 1);
            target[y * width + x] += source[source_y * source_width + source_x] * weight;
        }
    }
}

/// Compute a deterministic five-level reference winner map from proxy luma
/// planes.  `frames` must be aligned in the same proxy coordinate frame.
pub(crate) fn compute_focus_pyramid(
    width: usize,
    height: usize,
    frames: &[Vec<f32>],
    config: FocusPyramidConfig,
) -> Result<FocusPyramidResult, String> {
    let expected = checked_len(width, height)?;
    if width == 0 || height == 0 || frames.is_empty() {
        return Err("focus pyramid requires a non-empty proxy frame set".to_string());
    }
    if config.levels == 0 || config.energy_radius == 0 {
        return Err("focus pyramid levels and energy_radius must be non-zero".to_string());
    }
    if frames.iter().any(|frame| frame.len() != expected) {
        return Err("focus pyramid frames have inconsistent dimensions".to_string());
    }

    let mut levels = Vec::with_capacity(frames.len());
    for frame in frames {
        let mut pyramid = vec![PyramidPlane {
            width,
            height,
            samples: frame.clone(),
        }];
        for _ in 1..config.levels {
            let next = downsample(pyramid.last().expect("base pyramid level"));
            pyramid.push(next);
        }
        levels.push(pyramid);
    }

    let mut energy_maps = Vec::with_capacity(frames.len());
    for pyramid in &levels {
        let mut energy = vec![0.0; expected];
        for (level, plane) in pyramid.iter().enumerate() {
            let level_energy = laplacian_energy(plane, config.energy_radius);
            upsample_add(
                &mut energy,
                width,
                height,
                &level_energy,
                plane.width,
                plane.height,
                1.0 / (level as f32 + 1.0),
            );
        }
        energy_maps.push(energy);
    }

    let mut winner_map = vec![0u16; expected];
    for index in 0..expected {
        let mut winner = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (frame_index, energy) in energy_maps.iter().enumerate() {
            let value = energy[index];
            if value > best {
                best = value;
                winner = frame_index;
            }
        }
        winner_map[index] = winner.min(u16::MAX as usize) as u16;
    }

    Ok(FocusPyramidResult {
        width,
        height,
        levels,
        energy_maps,
        winner_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_keeps_five_levels_and_stable_winner_ties() {
        let frame = vec![0.5; 8 * 6];
        let result =
            compute_focus_pyramid(8, 6, &[frame.clone(), frame], FocusPyramidConfig::default())
                .expect("valid focus pyramid");
        assert_eq!(result.levels[0].len(), 5);
        assert!(result.winner_map.iter().all(|&winner| winner == 0));
        assert_eq!(result.energy_maps.len(), 2);
    }

    #[test]
    fn pyramid_prefers_the_frame_with_a_high_frequency_edge() {
        let smooth = vec![0.5; 16 * 8];
        let mut edge = smooth.clone();
        for y in 0..8 {
            edge[y * 16 + 8] = 0.0;
            edge[y * 16 + 9] = 1.0;
        }
        let result = compute_focus_pyramid(16, 8, &[smooth, edge], FocusPyramidConfig::default())
            .expect("valid focus pyramid");
        assert!(result.winner_map.iter().any(|&winner| winner == 1));
    }
}
