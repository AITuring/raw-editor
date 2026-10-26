// Residual_Warp module fragment; included at parent module scope.

/// Run-scoped report sink, matching the Intra_Station and Focus_Fuser pattern.
static RUN_RESIDUAL_WARP: Mutex<Vec<WarpRegionRecord>> = Mutex::new(Vec::new());
static RUN_RESIDUAL_WARP_IDENTITY: Mutex<bool> = Mutex::new(true);
static RUN_RESIDUAL_WARP_MODEL: Mutex<Option<ResidualWarp>> = Mutex::new(None);

pub(crate) fn reset_run_records() {
    RUN_RESIDUAL_WARP
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    *RUN_RESIDUAL_WARP_IDENTITY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
    *RUN_RESIDUAL_WARP_MODEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

pub(crate) fn record_run_model(model: &ResidualWarp) {
    *RUN_RESIDUAL_WARP_MODEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(model.clone());
    let mut records = RUN_RESIDUAL_WARP
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    records.extend(model.report_records());
    let mut identity = RUN_RESIDUAL_WARP_IDENTITY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *identity &= model.identity();
}

/// Snapshot the run model for the compositor.  An empty snapshot is the
/// compatibility path used until the panorama entry point supplies measured
/// station P95s and observations.
pub(crate) fn run_model_snapshot() -> ResidualWarp {
    RUN_RESIDUAL_WARP_MODEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
        .unwrap_or_default()
}

pub(crate) fn run_records_snapshot() -> Vec<WarpRegionRecord> {
    RUN_RESIDUAL_WARP
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

pub(crate) fn run_report_snapshot() -> ResidualWarpReport {
    ResidualWarpReport {
        regions: run_records_snapshot(),
        identity: *RUN_RESIDUAL_WARP_IDENTITY
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    }
}

