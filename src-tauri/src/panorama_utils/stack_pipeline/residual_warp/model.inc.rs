// Residual_Warp module fragment; included at parent module scope.

/// The complete residual model for a run.  `Default` intentionally starts
/// with an empty region list so old callers preserve their exact behaviour.
#[derive(Debug, Clone, Default)]
pub(crate) struct ResidualWarp {
    pub regions: Vec<WarpRegion>,
}

impl ResidualWarp {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Add an overlap using identity evidence.  P95 at or below the enable
    /// threshold does not allocate a region; a high-P95 overlap allocates an
    /// identity region and records insufficient evidence until observations
    /// are supplied with [`Self::add_observations`].
    pub(crate) fn add_overlap(&mut self, overlap: OverlapResidual) -> Option<usize> {
        if !overlap.p95_px.is_finite() || overlap.p95_px <= RESIDUAL_WARP_ENABLE_P95_PX {
            return None;
        }
        let region = WarpRegion::identity(overlap);
        self.regions.push(region);
        Some(self.regions.len() - 1)
    }

    /// Add an overlap and immediately populate it from round-trip observations.
    pub(crate) fn add_observations(
        &mut self,
        overlap: OverlapResidual,
        observations: &[WarpObservation],
    ) -> Option<usize> {
        if !overlap.p95_px.is_finite() || overlap.p95_px <= RESIDUAL_WARP_ENABLE_P95_PX {
            return None;
        }
        self.regions
            .push(WarpRegion::from_observations(overlap, observations));
        Some(self.regions.len() - 1)
    }

    pub(crate) fn identity(&self) -> bool {
        self.regions.iter().all(|region| {
            region
                .field
                .values
                .iter()
                .all(|value| value[0] == 0.0 && value[1] == 0.0)
        })
    }

    /// Apply the inverse residual map for one station in world coordinates.
    /// A residual is defined as `warped = world + displacement(world)`, so a
    /// few fixed-point steps recover the source world point without replacing
    /// the non-rigid field by a matrix.  Regions are keyed by their right-hand
    /// station; absent regions are an exact identity.
    pub(crate) fn inverse_world_for_station(
        &self,
        world: Point2<f64>,
        station_id: usize,
    ) -> Point2<f64> {
        let Some(region) = self.regions.iter().find(|region| {
            region.right_station == station_id && contains(&region.world, world.x, world.y)
        }) else {
            return world;
        };
        if self.identity() {
            return world;
        }
        let mut source = world;
        for _ in 0..4 {
            let displacement = region.displacement_at(source.x, source.y);
            let next = Point2::new(world.x - displacement[0], world.y - displacement[1]);
            if (next - source).norm() < 1e-6 {
                return next;
            }
            source = next;
        }
        source
    }

    /// Short compositor-facing alias for the world-space inverse query.
    pub(crate) fn warp_inverse(&self, world: Point2<f64>, station_id: usize) -> Point2<f64> {
        self.inverse_world_for_station(world, station_id)
    }

    pub(crate) fn report_records(&self) -> Vec<WarpRegionRecord> {
        self.regions.iter().map(WarpRegion::report_record).collect()
    }
}

