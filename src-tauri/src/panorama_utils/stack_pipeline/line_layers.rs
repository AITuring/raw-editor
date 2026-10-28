//! Native, one-dimensional contour evidence for a separate near-field layer.
//! These observations never create a camera relation or a two-dimensional
//! point match. Only the normal position of repeated long bright ridges moves.
use image::Rgb32FImage;
use nalgebra::{Matrix3, Point2, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

#[derive(Debug, Clone)]
pub(crate) struct ContourTrack {
    pub station: usize,
    /// Ordered (world y, world x) measurements, from native image profiles.
    pub points: Vec<[f64; 2]>,
}

fn luma(p: &image::Rgb<f32>) -> f64 {
    f64::from(p[0]) * 0.2126 + f64::from(p[1]) * 0.7152 + f64::from(p[2]) * 0.0722
}

/// A ridge has contrast on both sides; a broad white canvas or a coverage
/// transition therefore cannot become a foreground object.
pub(crate) fn detect_native_contours(
    image: &Rgb32FImage,
    covered: impl Fn(u32, u32) -> bool,
) -> Vec<Vec<Point2<f64>>> {
    let (width, height) = image.dimensions();
    if width < 160 || height < 512 {
        return Vec::new();
    }
    let mut bright = Vec::new();
    for y in (8..height - 8).step_by(16) {
        for x in 64..width - 64 {
            if !covered(x, y) || !covered(x - 16, y) || !covered(x + 16, y) {
                continue;
            }
            let v = luma(image.get_pixel(x, y));
            let base = luma(image.get_pixel(x - 8, y)).max(luma(image.get_pixel(x + 8, y)));
            if v > 0.65 && v - base >= 0.15 {
                bright.push((x as f64, y as f64, v - base));
            }
        }
    }
    // Dense bright texture is not an isolated long foreground contour.
    if bright.len() < 24 || bright.len() > 200_000 {
        return Vec::new();
    }
    let mut available = vec![true; bright.len()];
    let mut tracks = Vec::new();
    let centre = f64::from(height) * 0.5;
    let pad = f64::from(height) * 0.16;
    let bins = ((f64::from(width) + 2.0 * pad) / 8.0).ceil() as usize + 2;
    for _ in 0..8 {
        let mut best: Option<(f64, f64, f64)> = None;
        for slope_index in -60..=60 {
            let slope = f64::from(slope_index) * 0.0025;
            let mut mass = vec![0.0f64; bins];
            let mut ymin = vec![f64::INFINITY; bins];
            let mut ymax = vec![f64::NEG_INFINITY; bins];
            for (i, &(x, y, w)) in bright.iter().enumerate() {
                if !available[i] {
                    continue;
                }
                let bin = ((x - slope * (y - centre) + pad) / 8.0).floor() as usize;
                if bin >= bins {
                    continue;
                }
                mass[bin] += w;
                ymin[bin] = ymin[bin].min(y);
                ymax[bin] = ymax[bin].max(y);
            }
            for bin in 0..bins {
                if ymax[bin] - ymin[bin] < f64::from(height) * 0.20 {
                    continue;
                }
                if best.is_none_or(|b| mass[bin] > b.0) {
                    best = Some((
                        mass[bin],
                        slope,
                        (bin as f64 + 0.5) * 8.0 - pad - slope * centre,
                    ));
                }
            }
        }
        let Some((mass, slope, intercept)) = best else {
            break;
        };
        if mass < 8.0 {
            break;
        }
        for (i, &(x, y, _)) in bright.iter().enumerate() {
            if (x - slope * y - intercept).abs() <= 32.0 {
                available[i] = false;
            }
        }
        let mut points = Vec::new();
        for y in (8..height - 8).step_by(16) {
            let prediction = slope * f64::from(y) + intercept;
            let lo = (prediction.floor() as i64 - 24).max(64) as u32;
            let hi = (prediction.ceil() as i64 + 24)
                .min(i64::from(width) - 65)
                .max(i64::from(lo)) as u32;
            if hi - lo < 32 {
                continue;
            }
            let Some(peak) = (lo..=hi).filter(|&x| covered(x, y)).max_by(|&a, &b| {
                luma(image.get_pixel(a, y))
                    .total_cmp(&luma(image.get_pixel(b, y)))
                    .then_with(|| b.cmp(&a))
            }) else {
                continue;
            };
            let wing = |a: u32, b: u32| {
                let mut values = (a..=b)
                    .filter(|&x| covered(x, y))
                    .map(|x| luma(image.get_pixel(x, y)))
                    .collect::<Vec<_>>();
                values.sort_by(f64::total_cmp);
                (!values.is_empty()).then(|| values[values.len() / 2])
            };
            let (Some(left), Some(right)) = (wing(peak - 16, peak - 9), wing(peak + 9, peak + 16))
            else {
                continue;
            };
            let base = left.max(right);
            if luma(image.get_pixel(peak, y)) < 0.65 || luma(image.get_pixel(peak, y)) - base < 0.15
            {
                continue;
            }
            let mut sum = 0.0;
            let mut sx = 0.0;
            let mut sx2 = 0.0;
            let mut valid = true;
            for x in peak - 8..=peak + 8 {
                if !covered(x, y) {
                    valid = false;
                    break;
                }
                let w = (luma(image.get_pixel(x, y)) - base).max(0.0);
                sum += w;
                sx += w * f64::from(x);
                sx2 += w * f64::from(x).powi(2);
            }
            if !valid || sum < 0.3 {
                continue;
            }
            let x = sx / sum;
            if (sx2 / sum - x * x).max(0.0).sqrt() > 6.0 {
                continue;
            }
            points.push(Point2::new(x, f64::from(y)));
        }
        if points.len() >= 30 && points.last().unwrap().y - points[0].y >= f64::from(height) * 0.2 {
            tracks.push(points);
        }
    }
    tracks
}

fn at(points: &[[f64; 2]], y: f64) -> Option<f64> {
    if points.len() < 2 || y < points[0][0] || y > points.last()?[0] {
        return None;
    }
    let hi = points.partition_point(|p| p[0] < y).min(points.len() - 1);
    if hi == 0 {
        return Some(points[0][1]);
    }
    let a = points[hi - 1];
    let b = points[hi];
    if b[0] - a[0] > 128.0 {
        return None;
    }
    let t = ((y - a[0]) / (b[0] - a[0]).max(1e-9)).clamp(0.0, 1.0);
    Some(a[1] + (b[1] - a[1]) * t)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct NormalAlignmentReport {
    pub reference_station: usize,
    pub station: usize,
    pub training_samples: usize,
    pub held_out_samples: usize,
    pub normal_median_px: f64,
    pub normal_p95_px: f64,
    pub uncorrected_normal_p95_px: f64,
    pub maximum_displacement_px: f64,
    pub normal_only: bool,
}
#[derive(Debug, Clone)]
struct Edge {
    a: usize,
    b: usize,
    origin: f64,
    scale: f64,
    coeff: Vector3<f64>,
    range: (f64, f64),
    report: NormalAlignmentReport,
}
impl Edge {
    fn delta(&self, y: f64) -> f64 {
        let z = (y.clamp(self.range.0, self.range.1) - self.origin) / self.scale;
        self.coeff.dot(&Vector3::new(1.0, z, z * z))
    }
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn fit_edge(a: usize, b: usize, tracks: &[ContourTrack]) -> Option<Edge> {
    let (pa, pb) = (&tracks[a].points, &tracks[b].points);
    let start = pa.first()?[0].max(pb.first()?[0]);
    let end = pa.last()?[0].min(pb.last()?[0]);
    if end - start < 768.0 {
        return None;
    }
    let samples = ((start / 32.0).ceil() as i64..=(end / 32.0).floor() as i64)
        .filter_map(|i| {
            let y = i as f64 * 32.0;
            Some((y, at(pa, y)? - at(pb, y)?))
        })
        .collect::<Vec<_>>();
    if samples.len() < 30 {
        return None;
    }
    let origin = (start + end) * 0.5;
    let scale = ((end - start) * 0.5).max(1.0);
    let mut coeff = Vector3::zeros();
    for iteration in 0..12 {
        let mut normal = Matrix3::zeros();
        let mut rhs = Vector3::zeros();
        for (i, &(y, d)) in samples.iter().enumerate() {
            if i % 5 == 0 {
                continue;
            }
            let z = (y - origin) / scale;
            let row = Vector3::new(1.0, z, z * z);
            let w = if iteration == 0 {
                1.0
            } else {
                (1.5 / (d - row.dot(&coeff)).abs().max(1e-8)).min(1.0)
            };
            normal += row * row.transpose() * w;
            rhs += row * (d * w);
        }
        coeff = normal.try_inverse()? * rhs;
    }
    let mut held = samples
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 5 == 0)
        .map(|(_, &(y, d))| {
            let z = (y - origin) / scale;
            (d - coeff.dot(&Vector3::new(1.0, z, z * z))).abs()
        })
        .collect::<Vec<_>>();
    if held.len() < 6 {
        return None;
    }
    let med = median(&mut held);
    let p95 = held[(held.len() * 95).div_ceil(100) - 1];
    let mut raw = samples.iter().map(|&(_, d)| d.abs()).collect::<Vec<_>>();
    raw.sort_by(f64::total_cmp);
    let raw_p95 = raw[(raw.len() * 95).div_ceil(100) - 1];
    let max = samples
        .iter()
        .map(|&(y, _)| {
            let z = (y - origin) / scale;
            coeff.dot(&Vector3::new(1.0, z, z * z)).abs()
        })
        .fold(0.0, f64::max);
    let neighbour_ok = samples.windows(2).all(|p| {
        let eval = |y: f64| {
            let z = (y - origin) / scale;
            coeff.dot(&Vector3::new(1.0, z, z * z))
        };
        (eval(p[1].0) - eval(p[0].0)).abs() <= 8.0 * ((p[1].0 - p[0].0) / 64.0).max(1.0)
    });
    if med > 1.5 || p95 > 3.0 || max > 32.0 || !neighbour_ok {
        return None;
    }
    Some(Edge {
        a,
        b,
        origin,
        scale,
        coeff,
        range: (start, end),
        report: NormalAlignmentReport {
            reference_station: tracks[a].station,
            station: tracks[b].station,
            training_samples: samples.len() - held.len(),
            held_out_samples: held.len(),
            normal_median_px: med,
            normal_p95_px: p95,
            uncorrected_normal_p95_px: raw_p95,
            maximum_displacement_px: max,
            normal_only: true,
        },
    })
}

#[derive(Debug, Clone, Default)]
pub(crate) struct LineLayers {
    pub tracks: Vec<ContourTrack>,
    corrections: Vec<Vec<(Edge, f64)>>,
    bounds: Vec<[f64; 4]>,
    pub reports: Vec<NormalAlignmentReport>,
}
impl LineLayers {
    pub(crate) fn fit(mut tracks: Vec<ContourTrack>) -> Self {
        tracks.retain(|t| {
            t.points.len() >= 30
                && t.points
                    .iter()
                    .all(|p| p[0].is_finite() && p[1].is_finite())
                && t.points.windows(2).all(|p| p[1][0] > p[0][0])
        });
        let mut edges = Vec::new();
        for a in 0..tracks.len() {
            for b in a + 1..tracks.len() {
                if tracks[a].station == tracks[b].station {
                    continue;
                }
                if let Some(edge) = fit_edge(a, b, &tracks) {
                    edges.push(edge);
                }
            }
        }
        // Mutually nearest curves prevent nearby parallel glints from being
        // paired with different physical rails in different views.
        edges.retain(|edge| {
            let y = (edge.range.0 + edge.range.1) * 0.5;
            let Some(ax) = at(&tracks[edge.a].points, y) else {
                return false;
            };
            let Some(bx) = at(&tracks[edge.b].points, y) else {
                return false;
            };
            let nearest = |target: usize, x: f64, station: usize| {
                tracks
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.station == station)
                    .filter_map(|(i, t)| Some((i, (at(&t.points, y)? - x).abs())))
                    .min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)))
                    .is_some_and(|p| p.0 == target)
            };
            nearest(edge.b, ax, tracks[edge.b].station)
                && nearest(edge.a, bx, tracks[edge.a].station)
        });
        edges.sort_by(|a, b| {
            b.report
                .training_samples
                .cmp(&a.report.training_samples)
                .then_with(|| a.report.normal_p95_px.total_cmp(&b.report.normal_p95_px))
                .then_with(|| a.a.cmp(&b.a))
                .then_with(|| a.b.cmp(&b.b))
        });
        let bounds = tracks
            .iter()
            .map(|track| {
                let xmin = track
                    .points
                    .iter()
                    .map(|p| p[1])
                    .fold(f64::INFINITY, f64::min)
                    - 48.0;
                let xmax = track
                    .points
                    .iter()
                    .map(|p| p[1])
                    .fold(f64::NEG_INFINITY, f64::max)
                    + 48.0;
                [
                    xmin,
                    xmax,
                    track.points[0][0],
                    track.points.last().unwrap()[0],
                ]
            })
            .collect();
        let mut result = Self {
            corrections: vec![Vec::new(); tracks.len()],
            bounds,
            tracks,
            reports: Vec::new(),
        };
        let mut visited = vec![false; result.tracks.len()];
        for seed in 0..result.tracks.len() {
            if visited[seed] {
                continue;
            }
            let mut component = vec![seed];
            let mut queue = VecDeque::from([seed]);
            visited[seed] = true;
            while let Some(node) = queue.pop_front() {
                for edge in &edges {
                    let next = if edge.a == node {
                        edge.b
                    } else if edge.b == node {
                        edge.a
                    } else {
                        continue;
                    };
                    if !visited[next] {
                        visited[next] = true;
                        component.push(next);
                        queue.push_back(next);
                    }
                }
            }
            if component
                .iter()
                .map(|&i| result.tracks[i].station)
                .collect::<BTreeSet<_>>()
                .len()
                < 3
            {
                continue;
            }
            if !edges.iter().any(|e| {
                component.contains(&e.a)
                    && component.contains(&e.b)
                    && e.report.uncorrected_normal_p95_px > 3.0
            }) {
                continue;
            }
            let root = *component
                .iter()
                .min_by_key(|&&i| (result.tracks[i].station, i))
                .unwrap();
            let mut placed = BTreeSet::from([root]);
            let mut reports = Vec::new();
            loop {
                let Some(edge) = edges
                    .iter()
                    .find(|e| placed.contains(&e.a) ^ placed.contains(&e.b))
                else {
                    break;
                };
                let (parent, child, sign) = if placed.contains(&edge.a) {
                    (edge.a, edge.b, 1.0)
                } else {
                    (edge.b, edge.a, -1.0)
                };
                if !component.contains(&child) {
                    break;
                }
                result.corrections[child] = result.corrections[parent].clone();
                result.corrections[child].push((edge.clone(), sign));
                placed.insert(child);
                let mut report = edge.report.clone();
                report.reference_station = result.tracks[parent].station;
                report.station = result.tracks[child].station;
                reports.push(report);
            }
            let valid = edges
                .iter()
                .filter(|e| component.contains(&e.a) && component.contains(&e.b))
                .all(|e| {
                    let mut errors = Vec::new();
                    for k in 0..16 {
                        let y = e.range.0 + (e.range.1 - e.range.0) * k as f64 / 15.0;
                        if let (Some(a), Some(b)) = (
                            at(&result.tracks[e.a].points, y),
                            at(&result.tracks[e.b].points, y),
                        ) {
                            errors
                                .push((a + result.shift(e.a, y) - b - result.shift(e.b, y)).abs());
                        }
                    }
                    errors.len() >= 6
                        && median(&mut errors) <= 1.5
                        && errors.iter().copied().fold(0.0, f64::max) <= 3.0
                });
            if valid {
                result.reports.extend(reports)
            } else {
                for i in component {
                    result.corrections[i].clear();
                }
            }
        }
        result
    }
    fn shift(&self, i: usize, y: f64) -> f64 {
        self.corrections[i]
            .iter()
            .map(|(e, s)| e.delta(y) * s)
            .sum::<f64>()
            .clamp(-32.0, 32.0)
    }
    pub(crate) fn has_station(&self, station: usize) -> bool {
        self.tracks
            .iter()
            .enumerate()
            .any(|(i, t)| t.station == station && !self.corrections[i].is_empty())
    }
    pub(crate) fn inverse(&self, world: Point2<f64>, station: usize) -> Point2<f64> {
        let mut best: Option<(f64, f64)> = None;
        for (i, track) in self.tracks.iter().enumerate() {
            if track.station != station || self.corrections[i].is_empty() {
                continue;
            }
            let bounds = self.bounds[i];
            if world.x < bounds[0]
                || world.x > bounds[1]
                || world.y < bounds[2]
                || world.y > bounds[3]
            {
                continue;
            }
            let Some(x) = at(&track.points, world.y) else {
                continue;
            };
            let shift = self.shift(i, world.y);
            let target = x + shift;
            let distance = (world.x - target).abs();
            // Cover both the original and relocated ridge, including its
            // photographic halo. This is a depth boundary, not an image blur.
            if world.x < x.min(target) - 16.0 || world.x > x.max(target) + 16.0 {
                continue;
            }
            if best.is_none_or(|b| distance < b.0) {
                best = Some((distance, shift));
            }
        }
        best.map_or(world, |(_, d)| Point2::new(world.x - d, world.y))
    }
    pub(crate) fn affects(&self, world: Point2<f64>, station: usize) -> bool {
        (self.inverse(world, station) - world).norm() > 1e-8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    #[test]
    fn a_contour_layer_needs_three_views_and_preserves_the_tangent_coordinate() {
        let track = |station, shift| ContourTrack {
            station,
            points: (0..160)
                .map(|i| [i as f64 * 32.0, 400.0 + i as f64 * 0.4 + shift])
                .collect(),
        };
        assert!(!LineLayers::fit(vec![track(0, 0.0), track(1, 17.0)]).has_station(1));
        let model = LineLayers::fit(vec![track(0, 0.0), track(1, 17.0), track(2, -7.0)]);
        assert!(model.has_station(1));
        let p = Point2::new(408.0, 640.0);
        let q = model.inverse(p, 1);
        assert!((q.x - (p.x + 17.0)).abs() < 1e-6);
        assert_eq!(q.y, p.y);
        assert_eq!(
            model.inverse(Point2::new(100.0, 640.0), 1),
            Point2::new(100.0, 640.0)
        );
        assert!(
            model
                .reports
                .iter()
                .all(|r| r.held_out_samples >= 6 && r.normal_p95_px < 1e-6)
        );
    }
    #[test]
    fn native_ridges_are_measured_in_supported_pixels_not_broad_white_or_padding() {
        let image = Rgb32FImage::from_fn(600, 1600, |x, y| {
            let center = 250.0 + f64::from(y) * 0.025;
            let v = 0.1 + 0.85 * (-0.5 * ((f64::from(x) - center) / 1.5).powi(2)).exp();
            Rgb([v as f32; 3])
        });
        let tracks = detect_native_contours(&image, |_, _| true);
        assert_eq!(tracks.len(), 1);
        assert!(
            tracks[0]
                .iter()
                .all(|p| (p.x - (250.0 + p.y * 0.025)).abs() < 0.15)
        );
        assert!(detect_native_contours(&image, |_, _| false).is_empty());
        assert!(
            detect_native_contours(
                &Rgb32FImage::from_pixel(600, 1600, Rgb([0.95; 3])),
                |_, _| true
            )
            .is_empty()
        );
    }
}

/// Resampled detail is used only to plan ownership. The compositor rebuilds
/// every affected tone field through its original RAW sampler before fitting
/// the final radiometry; these estimates never supply output pixels.
pub(crate) fn warp_planning_detail(
    tile: &mut super::tone::ToneTile,
    layers: &LineLayers,
    offset: (f64, f64),
) {
    if !layers.has_station(tile.station_index) {
        return;
    }
    let Some(native) = tile.native_detail.as_ref() else {
        return;
    };
    let stride = tile.world_stride / 4.0;
    let shifted = Rgb32FImage::from_fn(native.width(), native.height(), |x, y| {
        let world = Point2::new(
            tile.world_origin.0 + x as f64 * stride + (stride - 1.0) * 0.5 - offset.0,
            tile.world_origin.1 + y as f64 * stride + (stride - 1.0) * 0.5 - offset.1,
        );
        let source = layers.inverse(world, tile.station_index);
        if source == world {
            return *native.get_pixel(x, y);
        }
        let u = (source.x + offset.0 - tile.world_origin.0 - (stride - 1.0) * 0.5) / stride;
        let a = u.floor() as i64;
        let b = a + 1;
        if a < 0
            || b >= native.width() as i64
            || tile.validity.get_pixel(a as u32 / 4, y / 4)[0] != super::tone::TONE_CELL_SAMPLE
            || tile.validity.get_pixel(b as u32 / 4, y / 4)[0] != super::tone::TONE_CELL_SAMPLE
        {
            return *native.get_pixel(x, y);
        }
        let t = (u - u.floor()) as f32;
        image::Rgb(std::array::from_fn(|c| {
            native.get_pixel(a as u32, y)[c] * (1.0 - t) + native.get_pixel(b as u32, y)[c] * t
        }))
    });
    tile.native_detail = Some(shifted);
}
