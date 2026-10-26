//! Shared pure helpers used by the panorama matching, pose, tile, and execution paths.
//!
//! These functions deliberately avoid owning a pipeline stage; they provide
//! deterministic evidence keys, robust medians, and cross-stage diagnostics.

use super::*;

pub(super) fn single_layer_station_evidence_discard(
    independent_support: usize,
    left_station: usize,
    right_station: usize,
) -> Option<serde_json::Value> {
    (independent_support < 2).then(|| {
        serde_json::json!({
            "count": 1,
            "stations": [left_station, right_station],
            "independent_support": independent_support,
        })
    })
}

/// Total, content-only ordering key of a station relation candidate.
///
/// Two focus layer relations of the same station pair can measure the same
/// quality and the same support; the bit pattern of the relation they propose is
/// then the only thing left that still comes from the measurement rather than
/// from the import order (需求 14.6).
pub(super) fn focus_group_candidate_key(candidate: &FocusGroupEdgeCandidate) -> [u64; 9] {
    let mut key = [0u64; 9];
    for (slot, value) in key.iter_mut().zip(candidate.transform.iter()) {
        *slot = value.to_bits();
    }
    key
}

pub(super) fn focus_transform_disagreement_px(
    left: &Matrix3<f64>,
    right: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> Option<f64> {
    let (width, height) = dimensions;
    let samples = [
        Point2::new(width as f64 * 0.2, height as f64 * 0.2),
        Point2::new(width as f64 * 0.8, height as f64 * 0.2),
        Point2::new(width as f64 * 0.5, height as f64 * 0.5),
        Point2::new(width as f64 * 0.2, height as f64 * 0.8),
        Point2::new(width as f64 * 0.8, height as f64 * 0.8),
    ];
    let mut errors = samples
        .into_iter()
        .filter_map(|point| {
            let left = transformed_point(left, point)?;
            let right = transformed_point(right, point)?;
            let error = (left - right).norm();
            error.is_finite().then_some(error)
        })
        .collect::<Vec<_>>();
    median_value(&mut errors)
}

/// Median Sharpness_Score over the valid pixels of one Source_RAW (需求 2.1).
///
/// Measured on the already prepared analysis image, resampled to the
/// Intra_Station analysis budget of 需求 2.2 so the number is a property of the
/// photograph rather than of whichever alignment budget the run happened to
/// pick.  The score is still the unbounded acutance ratio; task 7.6 normalises
/// it monotonically, which cannot change which frame is sharpest.
pub(super) fn median_value(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

pub(super) struct Dsu {
    parent: Vec<usize>,
}

impl Dsu {
    pub(super) fn new(n: usize) -> Self {
        Dsu {
            parent: (0..n).collect(),
        }
    }

    pub(super) fn find(&mut self, i: usize) -> usize {
        if self.parent[i] == i {
            i
        } else {
            self.parent[i] = self.find(self.parent[i]);
            self.parent[i]
        }
    }

    pub(super) fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            self.parent[root_i] = root_j;
        }
    }
}
