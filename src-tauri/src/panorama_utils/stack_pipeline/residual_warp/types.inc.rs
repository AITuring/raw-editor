// Residual_Warp module fragment; included at parent module scope.

/// A globally aligned overlap that may receive a local residual correction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct OverlapResidual {
    pub left_station: usize,
    pub right_station: usize,
    pub world: WorldRect,
    pub p95_px: f64,
}

impl OverlapResidual {
    pub(crate) const fn new(
        left_station: usize,
        right_station: usize,
        world: WorldRect,
        p95_px: f64,
    ) -> Self {
        Self {
            left_station,
            right_station,
            world,
            p95_px,
        }
    }
}

impl Default for OverlapResidual {
    fn default() -> Self {
        Self {
            left_station: 0,
            right_station: 0,
            world: WorldRect::default(),
            p95_px: 0.0,
        }
    }
}

/// A match measured in the overlap's world frame.  `global_error_px` and
/// `residual_error_px` are optional evidence used by [`WarpRegion::revert`]
/// when deciding whether an individual cell improved the global model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WarpObservation {
    pub world: WorldPoint,
    pub displacement: [f64; 2],
    pub round_trip_error_px: f64,
    pub global_error_px: f64,
    pub residual_error_px: f64,
}

impl WarpObservation {
    pub(crate) const fn new(
        world: WorldPoint,
        displacement: [f64; 2],
        round_trip_error_px: f64,
    ) -> Self {
        Self {
            world,
            displacement,
            round_trip_error_px,
            global_error_px: f64::NAN,
            residual_error_px: f64::NAN,
        }
    }

    pub(crate) const fn with_errors(
        world: WorldPoint,
        displacement: [f64; 2],
        round_trip_error_px: f64,
        global_error_px: f64,
        residual_error_px: f64,
    ) -> Self {
        Self {
            world,
            displacement,
            round_trip_error_px,
            global_error_px,
            residual_error_px,
        }
    }
}

/// One verified control node.  Invalid nodes are retained as records so a
/// diagnostic can distinguish no evidence from an explicitly zero correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WarpNode {
    pub world: WorldPoint,
    pub displacement: [f64; 2],
    pub valid: bool,
    pub round_trip_error_px: f64,
}

impl WarpNode {
    fn identity(world: WorldPoint) -> Self {
        Self {
            world,
            displacement: [0.0, 0.0],
            valid: false,
            round_trip_error_px: f64::INFINITY,
        }
    }
}

/// The result of a bidirectional patch check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RoundTripMatch {
    pub forward: Point2<f64>,
    pub backward: Point2<f64>,
    pub round_trip_error_px: f64,
}

/// A bounded residual field for one overlap.
#[derive(Debug, Clone)]
pub(crate) struct WarpRegion {
    pub left_station: usize,
    pub right_station: usize,
    pub world: WorldRect,
    pub p95_before_px: f64,
    pub node_step_px: u32,
    pub columns: u32,
    pub rows: u32,
    field: Field<2>,
    nodes: Vec<WarpNode>,
    pub reverted_cells: u64,
    pub insufficient_evidence: bool,
    pub max_node_displacement_px: f64,
    pub max_neighbour_delta_px: f64,
}

