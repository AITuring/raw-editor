//! Coverage-safe edge planes and the edge chains traced through them.

use super::*;

pub(super) const MAX_CHAIN_STEPS: usize = 512;

#[derive(Clone)]
pub(super) struct EdgePlane {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) edge: Vec<bool>,
    pub(super) orientation: Vec<f64>,
}

#[derive(Clone)]
pub(super) struct EdgeChain {
    pub(super) id: u32,
    pub(super) points: Vec<(u32, u32)>,
}

pub(super) fn edge_index(width: u32, x: u32, y: u32) -> usize {
    y as usize * width as usize + x as usize
}

pub(super) fn edge_neighbours(plane: &EdgePlane, x: u32, y: u32) -> Vec<(u32, u32)> {
    let mut result = Vec::with_capacity(8);
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0
                && ny >= 0
                && nx < plane.width as i32
                && ny < plane.height as i32
                && plane.edge[edge_index(plane.width, nx as u32, ny as u32)]
            {
                result.push((nx as u32, ny as u32));
            }
        }
    }
    result.sort_unstable();
    result
}

pub(super) fn coverage_safe_edges(
    gray: &GrayImage,
    coverage: &stitching::CoverageMask,
) -> Option<EdgePlane> {
    let (width, height) = gray.dimensions();
    if coverage.dimensions() != (width, height) || width < 5 || height < 5 {
        return None;
    }
    let count = width as usize * height as usize;
    let mut strength = vec![0u16; count];
    let mut orientation = vec![0.0; count];
    let mut nonzero = Vec::new();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let safe = (y - 1..=y + 1).all(|sample_y| {
                (x - 1..=x + 1).all(|sample_x| coverage.is_covered(sample_x, sample_y))
            });
            if !safe {
                continue;
            }
            let p = |sx: u32, sy: u32| i32::from(gray.get_pixel(sx, sy)[0]);
            let gx = -p(x - 1, y - 1) + p(x + 1, y - 1) - 2 * p(x - 1, y) + 2 * p(x + 1, y)
                - p(x - 1, y + 1)
                + p(x + 1, y + 1);
            let gy = -p(x - 1, y - 1) - 2 * p(x, y - 1) - p(x + 1, y - 1)
                + p(x - 1, y + 1)
                + 2 * p(x, y + 1)
                + p(x + 1, y + 1);
            let magnitude = (((gx * gx + gy * gy) as f64).sqrt().round() as u16).min(1443);
            if magnitude > 0 {
                let index = edge_index(width, x, y);
                strength[index] = magnitude;
                orientation[index] = (gy as f64).atan2(gx as f64);
                nonzero.push(magnitude);
            }
        }
    }
    if nonzero.len() < MIN_CHAIN_POINTS {
        return None;
    }
    nonzero.sort_unstable();
    let threshold = nonzero[nonzero.len() * 3 / 4].max(32);
    let mut edge = vec![false; count];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let index = edge_index(width, x, y);
            if strength[index] < threshold {
                continue;
            }
            let angle = orientation[index];
            let horizontal = angle.cos().abs() >= angle.sin().abs();
            let (a, b) = if horizontal {
                (
                    strength[edge_index(width, x - 1, y)],
                    strength[edge_index(width, x + 1, y)],
                )
            } else {
                (
                    strength[edge_index(width, x, y - 1)],
                    strength[edge_index(width, x, y + 1)],
                )
            };
            edge[index] = strength[index] >= a && strength[index] >= b;
        }
    }
    Some(EdgePlane {
        width,
        height,
        edge,
        orientation,
    })
}

pub(super) fn edge_key(width: u32, a: (u32, u32), b: (u32, u32)) -> (usize, usize) {
    let a = edge_index(width, a.0, a.1);
    let b = edge_index(width, b.0, b.1);
    (a.min(b), a.max(b))
}

pub(super) fn trace_edge_chains(plane: &EdgePlane) -> Vec<EdgeChain> {
    let mut visited = BTreeSet::new();
    let mut starts = Vec::new();
    for y in 0..plane.height {
        for x in 0..plane.width {
            if plane.edge[edge_index(plane.width, x, y)] {
                let degree = edge_neighbours(plane, x, y).len();
                if degree != 2 {
                    starts.push((x, y));
                }
            }
        }
    }
    // Remaining degree-two loops are reached by the second pass.
    starts.extend((0..plane.height).flat_map(|y| (0..plane.width).map(move |x| (x, y))));
    let mut chains = Vec::new();
    for start in starts {
        if !plane.edge[edge_index(plane.width, start.0, start.1)] {
            continue;
        }
        for first in edge_neighbours(plane, start.0, start.1) {
            if visited.contains(&edge_key(plane.width, start, first)) {
                continue;
            }
            let mut points = vec![start, first];
            visited.insert(edge_key(plane.width, start, first));
            while points.len() < MAX_CHAIN_STEPS {
                let previous = points[points.len() - 2];
                let current = *points.last().unwrap();
                let neighbours = edge_neighbours(plane, current.0, current.1);
                if neighbours.len() != 2 {
                    break;
                }
                let Some(next) = neighbours.into_iter().find(|&point| point != previous) else {
                    break;
                };
                let key = edge_key(plane.width, current, next);
                if visited.contains(&key) {
                    break;
                }
                visited.insert(key);
                points.push(next);
            }
            if points.len() >= MIN_CHAIN_POINTS {
                chains.push(EdgeChain {
                    id: chains.len() as u32,
                    points,
                });
            }
        }
    }
    chains
}
