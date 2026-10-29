//! The focus-stack stitching order and its report.
//!
//! Builds the focus-stack order and reference homographies from the match graph
//! and records the decisions in the stack report.

use crate::panorama_utils::stack_pipeline::report as stack_report;
use nalgebra::Matrix3;
use std::collections::HashMap;
use std::collections::HashSet;

#[cfg(test)]
use super::add_capture_sequence_bridges;
#[cfg(test)]
use super::build_focus_sequence_homographies;
use super::{
    ImageInfo, MatchInfo, SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES,
    attach_focus_bracket_orphans_to_pose_graph, build_graph_stitching_order,
    canonical_focus_order_direction, focus_auto_order_edge_weight, focus_auto_order_greedy_path,
    focus_auto_order_motion_scale, focus_auto_order_path_max_motion,
    focus_auto_order_path_motion_cost, focus_auto_order_path_score,
    focus_capture_geometry_diagnostics, focus_capture_geometry_passes,
    focus_capture_group_geometry_diagnostics, focus_capture_group_geometry_passes,
    focus_capture_groups, focus_cycle_edge_factor, focus_filename_order_with_overview_gaps,
    focus_geometry_capture_order, focus_graph_capture_sequence_boost,
    focus_match_graph_is_dense_continuous_scan, focus_match_is_local_bracket,
    focus_stack_spine_matches, lock_focus_local_bracket_poses,
    optimize_focus_stack_global_homographies_with_reference,
    solve_focus_capture_group_poses_with_report,
};

pub(super) const FOCUS_BRACKET_GRAPH_WEIGHT_BOOST: f64 = 24.0;

#[cfg(test)]
pub(super) fn build_focus_stack_stitching_order(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> (Vec<usize>, HashMap<usize, Matrix3<f64>>) {
    let mut closure_report = stack_report::ClosureReport::default();
    let mut station_relations_report = stack_report::StationRelationsReport::default();
    build_focus_stack_stitching_order_with_report(
        images,
        matches,
        &mut closure_report,
        &mut station_relations_report,
        true,
    )
}

pub(super) fn build_focus_stack_stitching_order_with_report(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
    solve_source_station_geometry: bool,
) -> (Vec<usize>, HashMap<usize, Matrix3<f64>>) {
    if images.len() < 2 {
        let mut homographies = HashMap::new();
        if let Some(image) = images.first() {
            homographies.insert(image.id, Matrix3::identity());
        }
        return ((0..images.len()).collect(), homographies);
    }

    // Focus stacking still has an order-sensitive ownership pass, but that
    // order must come from the image evidence rather than from upload order.
    // Build a maximum-confidence overlap graph, preferring strong, precise,
    // spatially-near matches. Compare several deterministic paths against the
    // graph: upload order, natural filename order, a graph traversal, and a
    // greedy visual path. This lets a camera sequence encoded in filenames win
    // when it is genuinely supported by image evidence, while still working
    // when filenames carry no useful order.
    let motion_scale = focus_auto_order_motion_scale(images, matches);
    let spine_matches = focus_stack_spine_matches(images, matches);
    if spine_matches.len() < matches.len() {
        println!(
            "  - Focus stack pose spine uses {}/{} pairwise edges (station representatives + brackets)",
            spine_matches.len(),
            matches.len()
        );
    }
    let (mut graph_order, mut graph_homographies) =
        build_graph_stitching_order(images, &spine_matches, |source, target, match_info| {
            let bracket_boost = if focus_match_is_local_bracket(source, target, match_info) {
                FOCUS_BRACKET_GRAPH_WEIGHT_BOOST
            } else {
                1.0
            };
            focus_auto_order_edge_weight(source, target, match_info, motion_scale)
                * focus_cycle_edge_factor(images, matches, source.id, target.id)
                * focus_graph_capture_sequence_boost(source, target, match_info)
                * bracket_boost
        });
    if graph_order.len() != images.len() {
        let attached = attach_focus_bracket_orphans_to_pose_graph(
            images,
            matches,
            &mut graph_order,
            &mut graph_homographies,
        );
        if attached > 0 {
            println!("  - Attached {attached} same-pose focus layer(s) to the verified pose graph");
        }
    }
    let filename_order = focus_filename_order_with_overview_gaps(images, matches);
    let input_order =
        canonical_focus_order_direction(&(0..images.len()).collect::<Vec<_>>(), images);
    let filename_order = canonical_focus_order_direction(&filename_order, images);
    let graph_order = canonical_focus_order_direction(&graph_order, images);
    let greedy_order = canonical_focus_order_direction(
        &focus_auto_order_greedy_path(images, matches, motion_scale),
        images,
    );
    let candidate_orders = vec![
        ("input", input_order, 0u8),
        ("filename", filename_order.clone(), 3u8),
        ("graph", graph_order.clone(), 1u8),
        ("visual", greedy_order, 2u8),
    ];

    #[cfg(test)]
    if std::env::var_os("RAW_EDITOR_STACK_FORCE_SEQUENCE_GEOMETRY").is_some() {
        let mut sequence_matches = matches.clone();
        let added = add_capture_sequence_bridges(images, &mut sequence_matches);
        if let Some(sequence_homographies) = build_focus_sequence_homographies(
            &filename_order,
            images,
            &sequence_matches,
            motion_scale,
        ) {
            let (valid_edges, total_edges, median_overlap, median_scale) =
                focus_capture_geometry_diagnostics(images, &filename_order, &sequence_homographies);
            println!(
                "  - Capture-sequence geometry experiment: {valid_edges}/{total_edges} adjacent edges, {added} synthetic gaps, median overlap {:.1}%, median scale {:.3}",
                median_overlap * 100.0,
                median_scale,
            );
            return (filename_order, sequence_homographies);
        }
        println!("  - Capture-sequence geometry experiment could not link every source");
    }
    let mut candidate_summaries = Vec::new();
    let mut selected = candidate_orders
        .into_iter()
        .filter(|(_, order, _)| order.len() == images.len())
        .map(|(name, order, priority)| {
            let (matched_edges, score) =
                focus_auto_order_path_score(&order, images, matches, motion_scale);
            let maximum_motion = focus_auto_order_path_max_motion(&order, images, matches);
            let motion_cost = focus_auto_order_path_motion_cost(&order, images, matches);
            candidate_summaries.push(format!(
                "{name}={matched_edges}/{}:{score:.2}/motion{motion_cost:.3}/max{maximum_motion:.3}",
                images.len().saturating_sub(1),
            ));
            (
                name,
                order,
                matched_edges,
                score,
                motion_cost,
                maximum_motion,
                priority,
            )
        })
        .max_by(|left, right| {
            // Synthetic filename bridges are only a fallback for an actual
            // gap. They must never outrank a visual path merely because they
            // make the edge count look complete: that creates a plausible
            // filename order with incorrect canvas poses and rectangular
            // duplicate tiles. Prefer the path's evidence score first, then
            // its real-edge coverage and motion stability.
            left.3
                .total_cmp(&right.3)
                .then_with(|| left.2
                .cmp(&right.2)
                // A direct match across several captured positions can have
                // many inliers, but letting it become one edge of the path
                // makes the ownership pass jump over the intermediate focus
                // layers. Prefer the path with the smallest single jump
                // before comparing its total travel distance.
                .then_with(|| right.5.total_cmp(&left.5))
                .then_with(|| right.4.total_cmp(&left.4))
                .then_with(|| left.6.cmp(&right.6)))
        });
    let Some((selected_name, selected_order, matched_edges, selected_score, _, _, _)) =
        selected.take()
    else {
        return (graph_order, graph_homographies);
    };

    // Rendering ownership may stay in the deterministic capture order, but
    // geometry must come from the verified maximum-confidence overlap graph.
    // Chaining filename neighbours accumulates camera-pass resets into the
    // diagonal snake/collage failure seen on large focus-stack mosaics.
    // If a demonstrably weak frame is absent from the graph, filter it out of
    // the capture order instead of falling back to graph traversal; otherwise
    // one skipped frame makes hundreds of valid layers render in tree order.
    let graph_sources = graph_order.iter().copied().collect::<HashSet<_>>();
    let connected_filename_order = filename_order
        .iter()
        .copied()
        .filter(|index| graph_sources.contains(index))
        .collect::<Vec<_>>();
    // Focus ownership is order-sensitive. Keep captures that were made next
    // to each other adjacent in the renderer even when a greedy visual path
    // can collect more graph edges by jumping between distant parts of the
    // scroll. Geometry still comes exclusively from the verified graph.
    let large_scan = images.len() >= SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES;
    let dense_continuous_scan = focus_match_graph_is_dense_continuous_scan(images, matches);
    if graph_order.len() != images.len() {
        let render_order = if large_scan || dense_continuous_scan {
            connected_filename_order
        } else {
            graph_order
        };
        return (render_order, graph_homographies);
    }
    let render_order = if large_scan || dense_continuous_scan {
        filename_order
    } else {
        selected_order
    };
    let reference_index = graph_order[0];
    let optimized_homographies = optimize_focus_stack_global_homographies_with_reference(
        images,
        matches,
        &graph_homographies,
        reference_index,
    );
    // Bracket locking belongs to the station model, not to the capture-order
    // heuristic. Infer Capture_Stations from the locked intra-station poses on
    // every default focus run; raw image count must never bypass the strict
    // Station_Pose_Solver gates (requirements 6, 7 and 15.1).
    let locked_homographies =
        lock_focus_local_bracket_poses(images, matches, &optimized_homographies);
    let inferred_station_count = focus_capture_groups(images, matches, &locked_homographies)
        .map(|(groups, _)| groups.len())
        .unwrap_or_default();
    let solve_station_geometry = solve_source_station_geometry && inferred_station_count >= 2;
    let optimized_or_locked_homographies = if solve_station_geometry {
        let mut translation_geometry_verified = false;
        solve_focus_capture_group_poses_with_report(
            images,
            matches,
            &locked_homographies,
            reference_index,
            &mut translation_geometry_verified,
            closure_report,
            station_relations_report,
        )
    } else if inferred_station_count >= 2 {
        // Default two-stage Virtual_Tile flow: preserve the image-level solve
        // only as local station geometry.  Bracket locking keeps every source
        // in its station anchor coordinates; inter-station placement is
        // replaced by the authoritative fused-tile solve after rendering.
        locked_homographies.clone()
    } else {
        // A single Capture_Station has no inter-station relation or closure to
        // solve. Keep compact single-station rendering on its established
        // image-level geometry rather than applying bracket-lock side effects.
        optimized_homographies.clone()
    };
    if station_relations_report.connectivity.components > 1 {
        return (Vec::new(), HashMap::new());
    }
    let geometry_render_order = if large_scan {
        focus_geometry_capture_order(images, matches, &optimized_or_locked_homographies)
            .filter(|order| order.len() == images.len())
            .unwrap_or_else(|| render_order.clone())
    } else {
        render_order.clone()
    };
    let (valid_geometry_edges, total_geometry_edges, median_overlap, median_scale) =
        focus_capture_geometry_diagnostics(
            images,
            &geometry_render_order,
            &optimized_or_locked_homographies,
        );
    println!(
        "  - Focus geometry validation: {valid_geometry_edges}/{total_geometry_edges} adjacent edges, median overlap {:.1}%, median scale {:.3}",
        median_overlap * 100.0,
        median_scale,
    );
    let (
        connected_group_count,
        total_group_count,
        valid_group_relations,
        candidate_group_relations,
        median_group_overlap,
    ) = focus_capture_groups(images, matches, &optimized_or_locked_homographies)
        .map(|(groups, _)| {
            focus_capture_group_geometry_diagnostics(
                images,
                matches,
                &groups,
                &optimized_or_locked_homographies,
            )
        })
        .unwrap_or((0, 0, 0, 0, 0.0));
    if solve_station_geometry {
        println!(
            "  - Focus station validation: {connected_group_count}/{total_group_count} connected groups, {valid_group_relations}/{candidate_group_relations} verified relations, median overlap {:.1}%",
            median_group_overlap * 100.0,
        );
    }
    let global_homographies = if !solve_source_station_geometry
        || !large_scan
        || (focus_capture_geometry_passes(
            images,
            valid_geometry_edges,
            total_geometry_edges,
            median_overlap,
            &optimized_or_locked_homographies,
        ) && (!solve_source_station_geometry
            || focus_capture_group_geometry_passes(
                connected_group_count,
                total_group_count,
                valid_group_relations,
                median_group_overlap,
            ))) {
        optimized_or_locked_homographies
    } else {
        // The long-scan refinement can be pulled by a repeated character or
        // seal. Retry the same verified graph without the capture-group
        // translation solve before rejecting the stack outright.
        println!(
            "  - Focus geometry rejected ({valid_geometry_edges}/{total_geometry_edges} continuous edges); retrying the ungrouped global pose"
        );
        let (retry_valid_edges, retry_total_edges, retry_overlap, retry_scale) =
            focus_capture_geometry_diagnostics(
                images,
                &geometry_render_order,
                &optimized_homographies,
            );
        println!(
            "  - Focus geometry retry: {retry_valid_edges}/{retry_total_edges} adjacent edges, median overlap {:.1}%, median scale {:.3}",
            retry_overlap * 100.0,
            retry_scale,
        );
        let (
            retry_connected_groups,
            retry_total_groups,
            retry_valid_relations,
            retry_candidate_relations,
            retry_group_overlap,
        ) = focus_capture_groups(images, matches, &optimized_homographies)
            .map(|(groups, _)| {
                focus_capture_group_geometry_diagnostics(
                    images,
                    matches,
                    &groups,
                    &optimized_homographies,
                )
            })
            .unwrap_or((0, 0, 0, 0, 0.0));
        println!(
            "  - Focus station retry: {retry_connected_groups}/{retry_total_groups} connected groups, {retry_valid_relations}/{retry_candidate_relations} verified relations, median overlap {:.1}%",
            retry_group_overlap * 100.0,
        );
        if focus_capture_geometry_passes(
            images,
            retry_valid_edges,
            retry_total_edges,
            retry_overlap,
            &optimized_homographies,
        ) && (!solve_source_station_geometry
            || focus_capture_group_geometry_passes(
                retry_connected_groups,
                retry_total_groups,
                retry_valid_relations,
                retry_group_overlap,
            ))
        {
            optimized_homographies
        } else {
            println!(
                "  - Focus geometry rejected after retry; refusing to render a discontinuous focus stack"
            );
            return (Vec::new(), HashMap::new());
        }
    };
    println!(
        "Focus auto-order diagnostic selected {selected_name} path with {matched_edges}/{} adjacent matches and score {selected_score:.2}; rendering uses stable capture order (candidates: {}; local motion scale {:.3}%)",
        images.len().saturating_sub(1),
        candidate_summaries.join(", "),
        motion_scale * 100.0,
    );
    if dense_continuous_scan {
        println!(
            "  - Dense continuous focus scan: preserving filename capture order for detail ownership"
        );
    }
    (geometry_render_order, global_homographies)
}
