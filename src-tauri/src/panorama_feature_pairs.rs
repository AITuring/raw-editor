//! Candidate pair generation and bounded feature retrieval.
//!
//! This module owns deterministic candidate enumeration. Full feature
//! extraction and geometric verification remain in their existing stages.

use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::Path;

use crate::panorama_utils::processing;

use super::{
    BlendMode, FOCUS_AUTO_ORDER_EXHAUSTIVE_MAX_SOURCES, Feature, ImageInfo,
    LARGE_STACK_NEIGHBOR_WINDOW, LARGE_STACK_RETRIEVAL_FEATURES, LARGE_STACK_RETRIEVAL_MIN_MATCHES,
    LARGE_STACK_RETRIEVAL_NEIGHBORS, SCALABLE_MATCH_RATIO_THRESHOLD, SCALABLE_STACK_THRESHOLD,
    SCALE_ROBUST_EXHAUSTIVE_MAX_SOURCES, SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES,
};

pub(super) fn pairs_to_match(image_count: usize) -> Vec<(usize, usize)> {
    let neighbor_window = if image_count > SCALABLE_STACK_THRESHOLD {
        LARGE_STACK_NEIGHBOR_WINDOW
    } else {
        image_count.saturating_sub(1)
    };
    (0..image_count)
        .flat_map(|first| {
            let end = image_count.min(first.saturating_add(neighbor_window + 1));
            (first + 1..end).map(move |second| (first, second))
        })
        .collect()
}

pub(super) fn all_image_pairs(image_count: usize) -> Vec<(usize, usize)> {
    (0..image_count)
        .flat_map(|first| (first + 1..image_count).map(move |second| (first, second)))
        .collect()
}

pub(super) fn natural_path_cmp(left: &str, right: &str) -> Ordering {
    let left = Path::new(left)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(left)
        .as_bytes();
    let right = Path::new(right)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(right)
        .as_bytes();
    let mut left_index = 0usize;
    let mut right_index = 0usize;

    while left_index < left.len() && right_index < right.len() {
        if left[left_index].is_ascii_digit() && right[right_index].is_ascii_digit() {
            let left_start = left_index;
            let right_start = right_index;
            while left_index < left.len() && left[left_index].is_ascii_digit() {
                left_index += 1;
            }
            while right_index < right.len() && right[right_index].is_ascii_digit() {
                right_index += 1;
            }
            let left_digits = &left[left_start..left_index];
            let right_digits = &right[right_start..right_index];
            let left_trimmed = left_digits
                .iter()
                .position(|digit| *digit != b'0')
                .map_or(&left_digits[left_digits.len()..], |index| {
                    &left_digits[index..]
                });
            let right_trimmed = right_digits
                .iter()
                .position(|digit| *digit != b'0')
                .map_or(&right_digits[right_digits.len()..], |index| {
                    &right_digits[index..]
                });
            let number_order = left_trimmed
                .len()
                .cmp(&right_trimmed.len())
                .then_with(|| left_trimmed.cmp(right_trimmed))
                .then_with(|| left_digits.len().cmp(&right_digits.len()));
            if number_order != Ordering::Equal {
                return number_order;
            }
        } else {
            let byte_order = left[left_index]
                .to_ascii_lowercase()
                .cmp(&right[right_index].to_ascii_lowercase());
            if byte_order != Ordering::Equal {
                return byte_order;
            }
            left_index += 1;
            right_index += 1;
        }
    }

    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}

fn add_neighbor_pairs(
    order: &[usize],
    neighbor_window: usize,
    unique_pairs: &mut HashSet<(usize, usize)>,
) {
    for (position, &first) in order.iter().enumerate() {
        let end = order
            .len()
            .min(position.saturating_add(neighbor_window + 1));
        for &second in &order[position + 1..end] {
            unique_pairs.insert((first.min(second), first.max(second)));
        }
    }
}

pub(super) fn pairs_to_match_for_images(
    images: &[ImageInfo],
    blend_mode: BlendMode,
) -> Vec<(usize, usize)> {
    if images.len() <= SCALABLE_STACK_THRESHOLD {
        return pairs_to_match(images.len());
    }
    // The 65–96 image range is common for high-resolution artwork scans and is
    // still small enough for an exhaustive bounded-resolution pass.  A lens
    // switch can move the only useful bridge dozens of filenames away, so a
    // neighbour-only graph is too brittle here.  Larger selections retain the
    // bounded visual-retrieval path below.
    if (SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES..=SCALE_ROBUST_EXHAUSTIVE_MAX_SOURCES)
        .contains(&images.len())
    {
        return all_image_pairs(images.len());
    }
    if blend_mode == BlendMode::FocusStack
        && images.len() <= FOCUS_AUTO_ORDER_EXHAUSTIVE_MAX_SOURCES
    {
        return all_image_pairs(images.len());
    }

    let input_order = (0..images.len()).collect::<Vec<_>>();
    let mut filename_order = input_order.clone();
    filename_order.sort_by(|&left, &right| {
        natural_path_cmp(&images[left].filename, &images[right].filename)
            .then_with(|| left.cmp(&right))
    });

    let mut unique_pairs = HashSet::new();
    add_neighbor_pairs(&input_order, LARGE_STACK_NEIGHBOR_WINDOW, &mut unique_pairs);
    add_neighbor_pairs(
        &filename_order,
        LARGE_STACK_NEIGHBOR_WINDOW,
        &mut unique_pairs,
    );
    let mut pairs = unique_pairs.into_iter().collect::<Vec<_>>();
    pairs.sort_unstable();
    augment_large_stack_candidate_pairs(images, &mut pairs);
    pairs
}

fn retrieval_feature_subset(features: &[Feature], limit: usize) -> Vec<Feature> {
    if features.len() <= limit {
        return features.to_vec();
    }
    // Feature vectors are ordered by detector strength.  Evenly sampling the
    // vector retains strong corners throughout the image instead of taking only
    // the first (often high-contrast) region.
    (0..limit)
        .map(|slot| {
            let index = slot
                .saturating_mul(features.len())
                .checked_div(limit)
                .unwrap_or(0)
                .min(features.len().saturating_sub(1));
            features[index].clone()
        })
        .collect()
}

/// Add a bounded set of visually likely pairs to the normal input/filename
/// neighbour graph.  File-picker order is often random, and a focal-length
/// switch can put the first useful bridge well outside the neighbour window.
/// The retrieval pass only counts compact descriptor matches; full-resolution
/// geometric verification still decides whether a pair is accepted.
fn augment_large_stack_candidate_pairs(images: &[ImageInfo], pairs: &mut Vec<(usize, usize)>) {
    if images.len() <= SCALABLE_STACK_THRESHOLD || images.len() < 2 {
        return;
    }
    let compact_features = images
        .iter()
        .map(|image| retrieval_feature_subset(&image.features, LARGE_STACK_RETRIEVAL_FEATURES))
        .collect::<Vec<_>>();
    if compact_features.iter().all(Vec::is_empty) {
        return;
    }

    let existing = pairs.iter().copied().collect::<HashSet<_>>();
    let scores = all_image_pairs(images.len())
        .into_par_iter()
        .filter_map(|(left, right)| {
            let score = processing::count_mutual_descriptor_matches_with_ratio(
                &compact_features[left],
                &compact_features[right],
                SCALABLE_MATCH_RATIO_THRESHOLD,
            );
            (score >= LARGE_STACK_RETRIEVAL_MIN_MATCHES).then_some((score, left, right))
        })
        .collect::<Vec<_>>();
    if scores.is_empty() {
        return;
    }

    let mut by_image = vec![Vec::<(usize, usize)>::new(); images.len()];
    for (score, left, right) in scores {
        by_image[left].push((score, right));
        by_image[right].push((score, left));
    }
    let mut additions = HashSet::new();
    for (image_index, neighbours) in by_image.iter_mut().enumerate() {
        neighbours.sort_by(|(left_score, left_index), (right_score, right_index)| {
            right_score
                .cmp(left_score)
                .then_with(|| {
                    natural_path_cmp(
                        &images[*left_index].filename,
                        &images[*right_index].filename,
                    )
                })
                .then_with(|| left_index.cmp(right_index))
        });
        for &(_, neighbour) in neighbours.iter().take(LARGE_STACK_RETRIEVAL_NEIGHBORS) {
            let pair = (image_index.min(neighbour), image_index.max(neighbour));
            if !existing.contains(&pair) {
                additions.insert(pair);
            }
        }
    }
    pairs.extend(additions);
    pairs.sort_unstable();
    pairs.dedup();
}
