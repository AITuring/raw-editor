use super::*;

/// The four 需求 1.1 measurements of one pairwise relation plus the verdict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EvidenceView {
    pub(crate) inliers: usize,
    pub(crate) overlap_ncc: f64,
    pub(crate) overlap_ncc_measured: bool,
    pub(crate) spatial_support: f64,
    pub(crate) scale_ratio: f64,
    pub(crate) score: f64,
    /// [`StationEvidence::accepts`] — the membership decision itself.
    pub(crate) accepts: bool,
}

/// One recorded Capture_Station split (需求 1.4 / 1.10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SplitView {
    pub(crate) component: usize,
    pub(crate) position: usize,
    pub(crate) motion_ratio: f64,
    pub(crate) reason: &'static str,
}

impl From<&StationSplit> for SplitView {
    fn from(split: &StationSplit) -> Self {
        Self {
            component: split.component,
            position: split.position,
            motion_ratio: split.motion_ratio,
            reason: split.reason,
        }
    }
}

/// [`StationGroupingOutcome`] in a shape a sibling module can hold.
#[derive(Clone, Debug, Default)]
pub(crate) struct GroupingView {
    pub(crate) components: Vec<Vec<usize>>,
    pub(crate) isolated: Vec<usize>,
    pub(crate) splits: Vec<SplitView>,
}

/// The 需求 1.1 / 1.4 / 1.5 thresholds, so a property test can state the
/// requirement's own numbers *and* assert the production constants are them.
pub(crate) const MIN_INLIER_MATCHES: usize = STATION_MIN_INLIER_MATCHES;
pub(crate) const MIN_OVERLAP_NCC: f64 = STATION_MIN_OVERLAP_NCC;
pub(crate) const MIN_INLIER_SPATIAL_SUPPORT: f64 = STATION_MIN_INLIER_SPATIAL_SUPPORT;
pub(crate) const SCALE_RATIO_MIN: f64 = STATION_SCALE_RATIO_MIN;
pub(crate) const SCALE_RATIO_MAX: f64 = STATION_SCALE_RATIO_MAX;
pub(crate) const MAX_ACCUMULATED_CENTER_MOTION_RATIO: f64 =
    STATION_MAX_ACCUMULATED_CENTER_MOTION_RATIO;
pub(crate) const MAX_STATION_MEMBERS: usize = FOCUS_BRACKET_MAX_COMPONENT_SOURCES;

/// [`focus_station_link_evidence`] — the measured 需求 1.1 evidence of one
/// pair, `None` when the pair carries no overlap evidence at all.
pub(crate) fn station_link_evidence(
    images: &[ImageInfo],
    first: usize,
    second: usize,
    match_info: &MatchInfo,
) -> Option<EvidenceView> {
    focus_station_link_evidence(images, first, second, match_info).map(|evidence| EvidenceView {
        inliers: evidence.inliers,
        overlap_ncc: evidence.overlap_ncc,
        overlap_ncc_measured: evidence.overlap_ncc_measured,
        spatial_support: evidence.spatial_support,
        scale_ratio: evidence.scale_ratio,
        score: evidence.score,
        accepts: evidence.accepts(),
    })
}

pub(crate) fn has_verified_scan_overlap(
    images: &[ImageInfo],
    first: usize,
    second: usize,
    match_info: &MatchInfo,
) -> bool {
    focus_match_has_verified_scan_overlap(images, first, second, match_info)
}

/// [`focus_match_is_capture_station_link`] — the membership decision the
/// grouper's union-find is driven by.
pub(crate) fn is_capture_station_link(
    images: &[ImageInfo],
    first: usize,
    second: usize,
    match_info: &MatchInfo,
) -> bool {
    focus_match_is_capture_station_link(images, first, second, match_info)
}

/// [`focus_station_grouping`] — membership, isolated sources and splits.
pub(crate) fn station_grouping(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> GroupingView {
    let outcome = focus_station_grouping(images, matches);
    GroupingView {
        components: outcome.components,
        isolated: outcome.isolated,
        splits: outcome.splits.iter().map(SplitView::from).collect(),
    }
}

/// [`split_focus_local_component_by_motion`] — the accumulated displacement
/// split of one verified-link component (需求 1.4 / 1.5 / 1.10).
pub(crate) fn split_component_by_motion(
    component: Vec<usize>,
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> (Vec<Vec<usize>>, Vec<SplitView>) {
    let mut splits = Vec::new();
    let groups = split_focus_local_component_by_motion(component, 0, images, matches, &mut splits);
    (groups, splits.iter().map(SplitView::from).collect())
}

/// [`focus_frame_median_sharpness`] — the 需求 2.1 anchor measurement.
pub(crate) fn frame_median_sharpness(image: &ImageInfo) -> f64 {
    focus_frame_median_sharpness(image)
}

/// [`excluded_source_records`] — the 需求 1.6 / 1.9 report records.
pub(crate) fn excluded_sources(
    images: &[ImageInfo],
    indices: &[usize],
    reason: &'static str,
) -> Vec<stack_report::ExcludedSourceRecord> {
    excluded_source_records(images, indices, reason)
}

/// [`undecodable_source_paths`] — the 需求 1.9 exclusion itself.
pub(crate) fn undecodable_sources(selection: &[String], decoded: &[ImageInfo]) -> Vec<String> {
    undecodable_source_paths(selection, decoded)
}
