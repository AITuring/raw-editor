//! Degradation_Manager, station level: the two degraded fusion paths of
//! 需求 12.1 / 12.2 and the "do not join any Capture_Station" rejection of
//! 需求 12.5.
//!
//! [`super::degradation`] owns the identifier set and the ledger; this module
//! owns the *decisions* those identifiers describe.  Every rule here is a small
//! deterministic function over injected measurements, so all three paths are
//! reachable from a unit test without decoding a RAW file: the production call
//! sites (`mosaic.rs`) only pass the numbers they already measured for
//! 需求 2.8 and hand the resulting plan to the ledger.
//!
//! Nothing in here loosens a threshold: the 20% inlier spatial support and the
//! 0.01 × long side symmetric error limit are read from
//! [`super::intra_station`], which is also what the per frame gates of 需求 2.6
//! and 需求 2.8 use.

use serde_json::{Value, json};

use super::degradation::{
    DegradationLedger, FUSION_SINGLE_FRAME_DEGRADED, INTRA_STATION_REGISTRATION_FAILED,
    INTRA_STATION_REJECTED_FROM_GROUP, record_run_degradation,
};
use super::intra_station::{INTRA_STATION_MIN_INLIER_AREA_COVERAGE, symmetric_error_limit_px};
use super::report::{ExcludedSourceRecord, IntraStationFrameStatus};

// ---------------------------------------------------------------------------
// 需求 12.1 / 12.2: which frames of a Capture_Station are fused
// ---------------------------------------------------------------------------

/// One member of a Capture_Station as the Degradation_Manager sees it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StationMember {
    /// Absolute path, which is also the tie-break key of 需求 12.1.
    pub path: String,
    /// Median Sharpness_Score over the member's valid pixels (需求 12.1).
    pub median_sharpness: f64,
    /// Verdict of the Intra_Station_Registrar (需求 2.7 / 2.8).
    pub status: IntraStationFrameStatus,
}

impl StationMember {
    /// `true` for a frame the registrar accepted, at either refinement level.
    pub(crate) fn registered(&self) -> bool {
        self.status != IntraStationFrameStatus::Failed
    }
}

/// The single Source_RAW a station degrades to (需求 12.1).
///
/// The highest median Sharpness_Score wins.  Unlike the anchor selection of
/// 需求 2.1 there is no tolerance window here: 需求 12.1 breaks a tie by
/// ascending absolute path only when the two scores are *exactly* equal, so a
/// measurably sharper frame always wins even by a hair.  An unmeasurable score
/// never wins over a measurable one, and a station whose every score is
/// unmeasurable still resolves to its lowest absolute path.
// The unrestricted rule is the literal text of 需求 12.1 and what the tests
// drive; production always has a candidate pool, so it calls the `_among` form.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn single_frame_source(members: &[StationMember]) -> Option<usize> {
    single_frame_source_among(members, &(0..members.len()).collect::<Vec<_>>())
}

/// [`single_frame_source`] restricted to the member indices in `pool`.
///
/// `pool` is walked in the order given; the comparison is a total order over
/// (score, path, index), so the winner does not depend on that order.
pub(crate) fn single_frame_source_among(
    members: &[StationMember],
    pool: &[usize],
) -> Option<usize> {
    pool.iter()
        .filter(|&&index| index < members.len())
        .copied()
        .min_by(|&left_index, &right_index| {
            let left = &members[left_index];
            let right = &members[right_index];
            let left_score = comparable_sharpness(left.median_sharpness);
            let right_score = comparable_sharpness(right.median_sharpness);
            // Descending score, then ascending path, then ascending index: a
            // total order, so the winner cannot depend on member order.
            right_score
                .total_cmp(&left_score)
                .then_with(|| left.path.cmp(&right.path))
                .then_with(|| left_index.cmp(&right_index))
        })
}

/// A non-finite score is treated as the lowest possible one so it can never
/// out-rank a real measurement.
fn comparable_sharpness(score: f64) -> f64 {
    if score.is_finite() {
        score
    } else {
        f64::NEG_INFINITY
    }
}

/// Which of the fusion paths of 需求 12.1 / 12.2 a Capture_Station takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StationFusionMode {
    /// The station has no member at all; nothing to fuse.
    Empty,
    /// 需求 12.1: every non-anchor frame failed, so one Source_RAW owns the
    /// whole Virtual_Tile.
    SingleFrameDegraded,
    /// 需求 12.2: at least one frame failed and at least two survived, so the
    /// fusion runs on the surviving frames only.
    SuccessfulFramesOnly,
    /// No frame failed; the full bracket is fused.
    AllFrames,
}

impl StationFusionMode {
    /// The failure reason identifier of this path, or `None` for a clean
    /// station (需求 12.7).
    pub(crate) fn reason(self) -> Option<&'static str> {
        match self {
            Self::SingleFrameDegraded => Some(FUSION_SINGLE_FRAME_DEGRADED),
            Self::SuccessfulFramesOnly => Some(INTRA_STATION_REGISTRATION_FAILED),
            Self::Empty | Self::AllFrames => None,
        }
    }
}

/// The fusion plan of one Capture_Station (需求 12.1 / 12.2).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StationFusionPlan {
    pub mode: StationFusionMode,
    /// Member indices that take part in the fusion, in member order.
    pub fused: Vec<usize>,
    /// Every excluded Source_RAW with the reason it was excluded (需求 12.2).
    pub excluded: Vec<ExcludedSourceRecord>,
}

impl StationFusionPlan {
    /// Absolute path of the single Source_RAW a degraded station owns, or
    /// `None` on any other path.
    pub(crate) fn single_frame_path(&self, members: &[StationMember]) -> Option<String> {
        (self.mode == StationFusionMode::SingleFrameDegraded)
            .then(|| self.fused.first().and_then(|&index| members.get(index)))
            .flatten()
            .map(|member| member.path.clone())
    }
}

/// Decide how a Capture_Station is fused from its members' registration
/// verdicts (需求 12.1 / 12.2).
///
/// Fewer than two surviving frames is the degraded path: a single frame has
/// nothing to select between, so the whole Ownership_Map points at the one
/// Source_RAW [`single_frame_source`] picks.  Two or more survivors keep the
/// fusion, with the failed frames listed as excluded — 需求 12.2 requires the
/// exclusion of *each* failed Source_RAW to be reported individually, not just
/// counted.
pub(crate) fn plan_station_fusion(members: &[StationMember]) -> StationFusionPlan {
    if members.is_empty() {
        return StationFusionPlan {
            mode: StationFusionMode::Empty,
            fused: Vec::new(),
            excluded: Vec::new(),
        };
    }
    let registered = members
        .iter()
        .enumerate()
        .filter(|(_, member)| member.registered())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let failed = members.len() - registered.len();
    if registered.len() >= 2 {
        let mode = if failed == 0 {
            StationFusionMode::AllFrames
        } else {
            StationFusionMode::SuccessfulFramesOnly
        };
        return StationFusionPlan {
            mode,
            fused: registered,
            excluded: members
                .iter()
                .filter(|member| !member.registered())
                .map(|member| ExcludedSourceRecord {
                    path: member.path.clone(),
                    reason: INTRA_STATION_REGISTRATION_FAILED.to_string(),
                })
                .collect(),
        };
    }
    // 需求 12.1.  The candidates are the frames the registrar accepted: the
    // Virtual_Tile lives in the anchor frame's coordinate system, so a frame
    // with no verified transform into it cannot be the frame the tile keeps —
    // reporting it as the owner would describe pixels the station never wrote.
    // The distinction is at most 需求 2.1's 0.01 anchor tie window wide: 需求 2.1
    // already selected the anchor by the same "highest median Sharpness_Score"
    // rule, so the surviving frame *is* the sharpest member up to that window.
    // When nothing registered at all, every member is equally unverified and
    // the rule of 需求 12.1 runs over the whole station.
    let pool = if registered.is_empty() {
        (0..members.len()).collect::<Vec<_>>()
    } else {
        registered
    };
    let selected =
        single_frame_source_among(members, &pool).expect("a non-empty station has a member");
    StationFusionPlan {
        mode: StationFusionMode::SingleFrameDegraded,
        fused: vec![selected],
        // Every other member of a degraded station is a failed frame: the pool
        // above only skips a registered frame when a second registered frame
        // exists, which is the 需求 12.2 branch.  需求 12.2's per-source listing
        // is kept here too, so a degraded station reports the same detail.
        excluded: members
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != selected)
            .map(|(_, member)| ExcludedSourceRecord {
                path: member.path.clone(),
                reason: INTRA_STATION_REGISTRATION_FAILED.to_string(),
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// 需求 12.5: rejected from every Capture_Station
// ---------------------------------------------------------------------------

/// What 需求 12.5 measures about one Source_RAW against a candidate station's
/// anchor frame.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GroupJoinEvidence {
    pub path: String,
    /// Homography inlier spatial support as a fraction of the overlap area
    /// (需求 2.8's definition, which 需求 12.5 reuses).
    pub inlier_area_coverage: f64,
    /// Median symmetric reprojection error of the inliers, native pixels.
    pub median_symmetric_error_px: f64,
    /// Native long side of the anchor frame, which scales the error limit.
    pub anchor_long_side_px: u32,
}

/// A Source_RAW that joins no Capture_Station, with the measurements that
/// decided it (需求 12.5).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GroupJoinRejection {
    pub path: String,
    pub inlier_area_coverage: f64,
    pub minimum_inlier_area_coverage: f64,
    pub median_symmetric_error_px: f64,
    pub symmetric_error_limit_px: f64,
    /// `true` when the spatial support was the failing criterion.
    pub support_below_minimum: bool,
    /// `true` when the symmetric error was the failing criterion.
    pub error_above_limit: bool,
}

impl GroupJoinRejection {
    pub(crate) fn detail(&self, station_index: usize) -> Value {
        json!({
            "stage": "intra_station_group_join",
            "station_index": station_index,
            "path": self.path,
            "inlier_area_coverage": self.inlier_area_coverage,
            "minimum_inlier_area_coverage": self.minimum_inlier_area_coverage,
            "median_symmetric_error_px": self.median_symmetric_error_px,
            "symmetric_error_limit_px": self.symmetric_error_limit_px,
            "support_below_minimum": self.support_below_minimum,
            "error_above_limit": self.error_above_limit,
        })
    }
}

/// 需求 12.5: inlier spatial support below 20% of the overlap area, or an
/// inlier median symmetric reprojection error above 0.01 × the long side, keeps
/// the Source_RAW out of every Capture_Station.
///
/// Both limits are the ones the per frame gates already use
/// ([`INTRA_STATION_MIN_INLIER_AREA_COVERAGE`], [`symmetric_error_limit_px`]),
/// so this decision cannot disagree with 需求 2.6 / 2.8.  A measurement that is
/// not a finite number is not evidence of a good fit and rejects as well.
pub(crate) fn group_join_rejection(evidence: &GroupJoinEvidence) -> Option<GroupJoinRejection> {
    let limit = symmetric_error_limit_px(evidence.anchor_long_side_px);
    let support_below_minimum = !(evidence.inlier_area_coverage.is_finite()
        && evidence.inlier_area_coverage >= INTRA_STATION_MIN_INLIER_AREA_COVERAGE);
    let error_above_limit = !(evidence.median_symmetric_error_px.is_finite()
        && evidence.median_symmetric_error_px <= limit);
    (support_below_minimum || error_above_limit).then(|| GroupJoinRejection {
        path: evidence.path.clone(),
        inlier_area_coverage: evidence.inlier_area_coverage,
        minimum_inlier_area_coverage: INTRA_STATION_MIN_INLIER_AREA_COVERAGE,
        median_symmetric_error_px: evidence.median_symmetric_error_px,
        symmetric_error_limit_px: limit,
        support_below_minimum,
        error_above_limit,
    })
}

// ---------------------------------------------------------------------------
// Ledger entries
// ---------------------------------------------------------------------------

/// The ledger entries a station's plan raises (需求 12.7 / 12.9).
///
/// Built as data so the same rule serves a unit test (folded into a local
/// [`DegradationLedger`]) and the production call site (folded into the run
/// scoped ledger).
pub(crate) fn station_plan_entries(
    station_index: usize,
    members: &[StationMember],
    plan: &StationFusionPlan,
) -> Vec<(&'static str, Value)> {
    let Some(reason) = plan.mode.reason() else {
        return Vec::new();
    };
    let excluded = plan
        .excluded
        .iter()
        .map(|record| {
            json!({
                "path": record.path,
                "reason": record.reason,
            })
        })
        .collect::<Vec<_>>();
    vec![(
        reason,
        json!({
            "stage": "intra_station_fusion",
            "station_index": station_index,
            "member_count": members.len(),
            "fused_frames": plan.fused.len(),
            "single_frame_path": plan.single_frame_path(members),
            "excluded": excluded,
        }),
    )]
}

/// Fold `entries` into a caller-owned ledger.  Production records into the run
/// scoped ledger; an injected ledger is how a test observes the same rule.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn record_entries(ledger: &mut DegradationLedger, entries: &[(&'static str, Value)]) {
    for (reason, detail) in entries {
        ledger.record(reason, detail.clone());
    }
}

/// Fold `entries` into the ledger of the run currently in flight.
pub(crate) fn record_run_entries(entries: &[(&'static str, Value)]) {
    for (reason, detail) in entries {
        record_run_degradation(reason, detail.clone());
    }
}

/// Record one 需求 12.5 rejection in the run scoped ledger.
pub(crate) fn record_run_group_join_rejection(
    station_index: usize,
    rejection: &GroupJoinRejection,
) {
    record_run_degradation(
        INTRA_STATION_REJECTED_FROM_GROUP,
        rejection.detail(station_index),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panorama_utils::stack_pipeline::degradation::RunResult;

    fn member(path: &str, sharpness: f64, status: IntraStationFrameStatus) -> StationMember {
        StationMember {
            path: path.to_string(),
            median_sharpness: sharpness,
            status,
        }
    }

    fn anchor(path: &str, sharpness: f64) -> StationMember {
        member(path, sharpness, IntraStationFrameStatus::Anchor)
    }

    fn local(path: &str, sharpness: f64) -> StationMember {
        member(path, sharpness, IntraStationFrameStatus::Local)
    }

    fn failed(path: &str, sharpness: f64) -> StationMember {
        member(path, sharpness, IntraStationFrameStatus::Failed)
    }

    // -- 需求 12.1 ---------------------------------------------------------

    #[test]
    fn every_non_anchor_frame_failing_degrades_to_one_source() {
        let members = vec![
            anchor("/station/0001.nef", 0.61),
            failed("/station/0002.nef", 0.42),
            failed("/station/0003.nef", 0.55),
        ];
        let plan = plan_station_fusion(&members);
        assert_eq!(plan.mode, StationFusionMode::SingleFrameDegraded);
        assert_eq!(plan.fused, vec![0]);
        assert_eq!(
            plan.single_frame_path(&members).as_deref(),
            Some("/station/0001.nef"),
            "the whole Ownership_Map points at the sharpest source"
        );
        // 需求 12.2's per-source listing still applies to the excluded frames.
        assert_eq!(
            plan.excluded,
            vec![
                ExcludedSourceRecord {
                    path: "/station/0002.nef".to_string(),
                    reason: INTRA_STATION_REGISTRATION_FAILED.to_string(),
                },
                ExcludedSourceRecord {
                    path: "/station/0003.nef".to_string(),
                    reason: INTRA_STATION_REGISTRATION_FAILED.to_string(),
                },
            ]
        );

        let mut ledger = DegradationLedger::new();
        record_entries(&mut ledger, &station_plan_entries(3, &members, &plan));
        assert_eq!(ledger.count_of(FUSION_SINGLE_FRAME_DEGRADED), 1);
        assert_eq!(ledger.result(), RunResult::Degraded);
        let detail = &ledger.entries()[0].detail;
        assert_eq!(detail["station_index"], json!(3));
        assert_eq!(detail["fused_frames"], json!(1));
        assert_eq!(detail["single_frame_path"], json!("/station/0001.nef"));
        assert_eq!(detail["excluded"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn the_degraded_source_is_the_sharpest_frame_then_the_lowest_path() {
        // A measurably sharper frame wins even by less than the 需求 2.1 anchor
        // tie window: 需求 12.1 only breaks *exact* ties by path.
        let members = vec![failed("/a/0001.nef", 0.4995), anchor("/z/0009.nef", 0.5000)];
        assert_eq!(single_frame_source(&members), Some(1));

        // Exactly equal scores: the lowest absolute path wins, whatever the
        // member order is.
        let mut members = vec![
            failed("/z/0009.nef", 0.5),
            anchor("/a/0001.nef", 0.5),
            failed("/m/0005.nef", 0.5),
        ];
        assert_eq!(
            members[single_frame_source(&members).unwrap()].path,
            "/a/0001.nef"
        );
        members.reverse();
        assert_eq!(
            members[single_frame_source(&members).unwrap()].path,
            "/a/0001.nef"
        );

        // Unmeasurable scores never out-rank a measured one, and a station of
        // only unmeasurable scores still resolves deterministically.
        let members = vec![anchor("/b.nef", f64::NAN), failed("/a.nef", 0.01)];
        assert_eq!(single_frame_source(&members), Some(1));
        let members = vec![anchor("/b.nef", f64::NAN), failed("/a.nef", f64::NAN)];
        assert_eq!(single_frame_source(&members), Some(1));
        assert_eq!(single_frame_source(&[]), None);
    }

    // -- 需求 12.2 ---------------------------------------------------------

    #[test]
    fn a_failed_frame_beside_two_survivors_is_excluded_one_by_one() {
        let members = vec![
            anchor("/station/0001.nef", 0.60),
            failed("/station/0002.nef", 0.50),
            local("/station/0003.nef", 0.55),
            failed("/station/0004.nef", 0.10),
        ];
        let plan = plan_station_fusion(&members);
        assert_eq!(plan.mode, StationFusionMode::SuccessfulFramesOnly);
        assert_eq!(
            plan.fused,
            vec![0, 2],
            "only the registration-successful frames are fused"
        );
        assert_eq!(plan.single_frame_path(&members), None);
        assert_eq!(
            plan.excluded
                .iter()
                .map(|record| (record.path.as_str(), record.reason.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("/station/0002.nef", INTRA_STATION_REGISTRATION_FAILED),
                ("/station/0004.nef", INTRA_STATION_REGISTRATION_FAILED),
            ],
            "需求 12.2 lists every excluded Source_RAW with its reason"
        );

        let mut ledger = DegradationLedger::new();
        record_entries(&mut ledger, &station_plan_entries(0, &members, &plan));
        assert_eq!(ledger.count_of(INTRA_STATION_REGISTRATION_FAILED), 1);
        assert_eq!(ledger.entries()[0].detail["fused_frames"], json!(2));
        assert_eq!(
            ledger.entries()[0].detail["excluded"][1]["path"],
            json!("/station/0004.nef")
        );
    }

    #[test]
    fn a_station_without_failures_stays_on_the_full_bracket() {
        let members = vec![
            anchor("/station/0001.nef", 0.60),
            local("/station/0002.nef", 0.55),
            member(
                "/station/0003.nef",
                0.50,
                IntraStationFrameStatus::GlobalFallback,
            ),
        ];
        let plan = plan_station_fusion(&members);
        assert_eq!(plan.mode, StationFusionMode::AllFrames);
        assert_eq!(plan.fused, vec![0, 1, 2]);
        assert!(plan.excluded.is_empty());
        assert!(station_plan_entries(0, &members, &plan).is_empty());

        // A station with no member at all raises nothing either.
        let plan = plan_station_fusion(&[]);
        assert_eq!(plan.mode, StationFusionMode::Empty);
        assert!(plan.fused.is_empty());
        assert!(station_plan_entries(0, &[], &plan).is_empty());
    }

    #[test]
    fn a_two_member_station_losing_its_partner_takes_the_single_frame_path() {
        let members = vec![anchor("/a.nef", 0.6), failed("/b.nef", 0.9)];
        let plan = plan_station_fusion(&members);
        assert_eq!(plan.mode, StationFusionMode::SingleFrameDegraded);
        // The failed frame has no verified transform into the tile's coordinate
        // system, so the registered frame is the one the tile keeps even though
        // this synthetic pair makes the failed frame the sharper one.  需求 2.1
        // cannot produce that pair: it picks the anchor by the same maximum.
        assert_eq!(plan.single_frame_path(&members).as_deref(), Some("/a.nef"));
        assert_eq!(
            plan.excluded,
            vec![ExcludedSourceRecord {
                path: "/b.nef".to_string(),
                reason: INTRA_STATION_REGISTRATION_FAILED.to_string(),
            }]
        );

        // Nothing registered at all: the rule of 需求 12.1 runs over the whole
        // station, so the sharpest member wins and an exact tie falls to the
        // lowest absolute path.
        let members = vec![failed("/a.nef", 0.6), failed("/b.nef", 0.9)];
        let plan = plan_station_fusion(&members);
        assert_eq!(plan.single_frame_path(&members).as_deref(), Some("/b.nef"));
        assert_eq!(
            plan.excluded,
            vec![ExcludedSourceRecord {
                path: "/a.nef".to_string(),
                reason: INTRA_STATION_REGISTRATION_FAILED.to_string(),
            }]
        );
    }

    // -- 需求 12.5 ---------------------------------------------------------

    #[test]
    fn low_spatial_support_is_rejected_from_every_station() {
        let evidence = GroupJoinEvidence {
            path: "/station/0007.nef".to_string(),
            inlier_area_coverage: 0.1999,
            median_symmetric_error_px: 0.8,
            anchor_long_side_px: 6_000,
        };
        let rejection = group_join_rejection(&evidence).expect("below 20% support is rejected");
        assert!(rejection.support_below_minimum);
        assert!(!rejection.error_above_limit);
        assert_eq!(rejection.minimum_inlier_area_coverage, 0.20);
        assert!((rejection.symmetric_error_limit_px - 60.0).abs() < 1e-12);

        let mut ledger = DegradationLedger::new();
        ledger.record(INTRA_STATION_REJECTED_FROM_GROUP, rejection.detail(2));
        let detail = &ledger.entries()[0].detail;
        assert_eq!(detail["path"], json!("/station/0007.nef"));
        assert_eq!(detail["inlier_area_coverage"], json!(0.1999));
        assert_eq!(detail["median_symmetric_error_px"], json!(0.8));
        assert_eq!(detail["symmetric_error_limit_px"], json!(60.0));
        assert_eq!(ledger.result(), RunResult::Degraded);
    }

    #[test]
    fn a_high_median_symmetric_error_is_rejected_from_every_station() {
        // 0.01 x 6000 = 60 native pixels.
        let evidence = |error_px: f64| GroupJoinEvidence {
            path: "/station/0008.nef".to_string(),
            inlier_area_coverage: 0.85,
            median_symmetric_error_px: error_px,
            anchor_long_side_px: 6_000,
        };
        assert!(
            group_join_rejection(&evidence(60.0)).is_none(),
            "at the limit"
        );
        let rejection = group_join_rejection(&evidence(60.1)).expect("above the limit is rejected");
        assert!(rejection.error_above_limit);
        assert!(!rejection.support_below_minimum);

        // Both criteria failing is reported as both.
        let rejection = group_join_rejection(&GroupJoinEvidence {
            inlier_area_coverage: 0.05,
            ..evidence(120.0)
        })
        .expect("both criteria fail");
        assert!(rejection.support_below_minimum && rejection.error_above_limit);

        // An unmeasurable number is not evidence of a good fit.
        assert!(group_join_rejection(&evidence(f64::NAN)).is_some());
        assert!(
            group_join_rejection(&GroupJoinEvidence {
                inlier_area_coverage: f64::NAN,
                ..evidence(1.0)
            })
            .is_some()
        );
    }

    #[test]
    fn an_accepted_frame_raises_no_rejection() {
        assert!(
            group_join_rejection(&GroupJoinEvidence {
                path: "/station/0002.nef".to_string(),
                inlier_area_coverage: 0.20,
                median_symmetric_error_px: 0.35,
                anchor_long_side_px: 6_000,
            })
            .is_none(),
            "exactly 20% support with a small error still joins the station"
        );
    }
}
