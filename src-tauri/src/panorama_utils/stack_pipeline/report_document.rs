//! Top-level Stack_Report document and serialization contract.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::report_schema_core::*;
use super::report_schema_metrics::*;
use super::report_schema_solver::*;

// Stack_Report
// ---------------------------------------------------------------------------

/// The full Stack_Report document.
///
/// Every top-level field is always serialized, including defaults, so the
/// structure can be asserted without a RAW decode (requirement 15.9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StackReport {
    pub schema: u32,
    pub pipeline_version: String,
    pub run_id: String,
    pub started_at_epoch_ms: u64,
    pub finished_at_epoch_ms: u64,
    pub selected_path: SelectedPath,

    pub input: InputReport,
    pub grouping: GroupingReport,
    pub intra_station: Vec<IntraStationReport>,
    pub fusion: Vec<FusionReport>,
    pub virtual_tiles: Vec<VirtualTileReport>,
    pub virtual_tile_cache: VirtualTileCacheSummary,
    pub topology: TopologyReport,
    pub station_relations: StationRelationsReport,
    pub closure: ClosureReport,
    pub residual_warp: ResidualWarpReport,
    pub tone: ToneReport,
    pub composition: CompositionReport,
    pub quality_gate: QualityGateReport,
    pub output: OutputReport,
    pub resources: ResourcesReport,
    pub degradation: DegradationReport,
}

impl Default for StackReport {
    fn default() -> Self {
        Self {
            schema: STACK_REPORT_SCHEMA,
            pipeline_version: String::new(),
            run_id: String::new(),
            started_at_epoch_ms: 0,
            finished_at_epoch_ms: 0,
            selected_path: SelectedPath::LayeredVirtualTile,

            input: InputReport::default(),
            grouping: GroupingReport::default(),
            intra_station: Vec::new(),
            fusion: Vec::new(),
            virtual_tiles: Vec::new(),
            virtual_tile_cache: VirtualTileCacheSummary::default(),
            topology: TopologyReport::default(),
            station_relations: StationRelationsReport::default(),
            closure: ClosureReport::default(),
            residual_warp: ResidualWarpReport::default(),
            tone: ToneReport::default(),
            composition: CompositionReport::default(),
            quality_gate: QualityGateReport::default(),
            output: OutputReport::default(),
            resources: ResourcesReport::default(),
            degradation: DegradationReport::default(),
        }
    }
}

/// Every top-level key the schema guarantees.  Used by the unit test below and
/// mirrored by `tests/focus-stack-quality-contract.mjs`.
#[allow(dead_code)]
pub(crate) const STACK_REPORT_TOP_LEVEL_FIELDS: &[&str] = &[
    "schema",
    "pipeline_version",
    "run_id",
    "started_at_epoch_ms",
    "finished_at_epoch_ms",
    "selected_path",
    "input",
    "grouping",
    "intra_station",
    "fusion",
    "virtual_tiles",
    "virtual_tile_cache",
    "topology",
    "station_relations",
    "closure",
    "residual_warp",
    "tone",
    "composition",
    "quality_gate",
    "output",
    "resources",
    "degradation",
];

impl StackReport {
    pub(crate) fn new(pipeline_version: &str) -> Self {
        Self {
            pipeline_version: pipeline_version.to_string(),
            run_id: Uuid::new_v4().to_string(),
            started_at_epoch_ms: epoch_millis(),
            ..Self::default()
        }
    }

    pub(crate) fn to_json_string(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("Failed to serialize the stack report: {error}"))
    }
}

pub(super) fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}
