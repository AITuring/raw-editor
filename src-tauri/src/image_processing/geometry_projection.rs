//! Geometry transform matrices and projection mapping.
//!
//! Builds the perspective/rotation/scale matrices, remaps points through the
//! projection strength and inverts the generic distortion model.

use nalgebra::Matrix3 as NaMatrix3;

use super::GeometryParams;

pub(super) fn build_transform_matrices(
    params: &GeometryParams,
    width: f32,
    height: f32,
) -> (NaMatrix3<f32>, f32, f32, f64) {
    let cx = width / 2.0;
    let cy = height / 2.0;
    let ref_dim = 2000.0;

    let p_vert = (params.vertical / 100000.0) * (ref_dim / height);
    let p_horiz = (-params.horizontal / 100000.0) * (ref_dim / width);
    let theta = params.rotate.to_radians();

    let aspect_factor = if params.aspect >= 0.0 {
        1.0 + params.aspect / 100.0
    } else {
        1.0 / (1.0 + params.aspect.abs() / 100.0)
    };

    let scale_factor = params.scale / 100.0;
    let off_x = (params.x_offset / 100.0) * width;
    let off_y = (params.y_offset / 100.0) * height;

    let t_center = NaMatrix3::new(1.0, 0.0, cx, 0.0, 1.0, cy, 0.0, 0.0, 1.0);
    let t_uncenter = NaMatrix3::new(1.0, 0.0, -cx, 0.0, 1.0, -cy, 0.0, 0.0, 1.0);
    let m_perspective = NaMatrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, p_horiz, p_vert, 1.0);

    let (sin_t, cos_t) = theta.sin_cos();
    let m_rotate = NaMatrix3::new(cos_t, -sin_t, 0.0, sin_t, cos_t, 0.0, 0.0, 0.0, 1.0);
    let m_scale = NaMatrix3::new(
        scale_factor * aspect_factor,
        0.0,
        0.0,
        0.0,
        scale_factor,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let m_offset = NaMatrix3::new(1.0, 0.0, off_x, 0.0, 1.0, off_y, 0.0, 0.0, 1.0);

    let forward = t_center * m_offset * m_perspective * m_rotate * m_scale * t_uncenter;
    let half_diagonal =
        ((width as f64 * width as f64 + height as f64 * height as f64).sqrt()) / 2.0;

    (forward, cx, cy, half_diagonal)
}

pub(super) const MAX_PROJECTION_STRENGTH: f64 = 1.1;

#[inline(always)]
pub(super) fn remap_projection_point(
    x: f64,
    y: f64,
    cx: f64,
    cy: f64,
    half_diagonal: f64,
    projection: f32,
    inverse: bool,
) -> (f64, f64) {
    if projection.abs() < 1e-5 || half_diagonal < 1e-6 {
        return (x, y);
    }

    let dx = x - cx;
    let dy = y - cy;
    let radius = (dx * dx + dy * dy).sqrt();
    if radius < 1e-6 {
        return (x, y);
    }

    let normalized_radius = radius / half_diagonal;
    let strength = (projection.abs() as f64 / 100.0) * MAX_PROJECTION_STRENGTH;
    let compress = if inverse {
        projection < 0.0
    } else {
        projection > 0.0
    };
    let mapped_radius = if compress {
        (strength * normalized_radius).atan() / strength
    } else {
        let angle = (strength * normalized_radius).min(std::f64::consts::FRAC_PI_2 - 1e-4);
        angle.tan() / strength
    };
    let scale = mapped_radius / normalized_radius;
    (cx + dx * scale, cy + dy * scale)
}

#[inline(always)]
pub(super) fn project_geometry_point(
    x: f64,
    y: f64,
    cx: f64,
    cy: f64,
    half_diagonal: f64,
    projection: f32,
) -> (f64, f64) {
    remap_projection_point(x, y, cx, cy, half_diagonal, projection, false)
}

#[inline(always)]
pub(super) fn unproject_geometry_point(
    x: f64,
    y: f64,
    cx: f64,
    cy: f64,
    half_diagonal: f64,
    projection: f32,
) -> (f64, f64) {
    remap_projection_point(x, y, cx, cy, half_diagonal, projection, true)
}

pub(super) fn solve_generic_distortion_inv(r_target: f64, k_scaled: f64) -> f64 {
    if k_scaled.abs() < 1e-9 {
        return r_target;
    }

    let mut r = r_target;
    for _ in 0..10 {
        let r2 = r * r;
        let val = k_scaled * r2 * r + r - r_target;
        let slope = 3.0 * k_scaled * r2 + 1.0;

        if slope.abs() < 1e-9 {
            break;
        }
        let delta = val / slope;
        r -= delta;
        if delta.abs() < 1e-6 {
            break;
        }
    }
    r
}
