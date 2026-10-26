// Degradation_Manager fragment; included at parent module scope.

/// Severity of a degradation entry. `Rejected` means the run must not write a
/// final result file; `Degraded` means the run continues with reduced evidence;
/// `Informational` means the entry is housekeeping that left the output
/// untouched.
///
/// `Informational` exists because not every recorded path costs the result
/// anything. Evicting an old cache entry to stay inside the 64 GiB budget, for
/// example, only decides what stays on disk: the tile it makes room for is
/// synthesised from the same sources with the same code, so the pixels, the
/// thresholds and the fallback decisions of the run are all identical with and
/// without the eviction. Reporting such a run as `degraded` would tell the user
/// the output is worse than it is.
///
/// Design note: the degradation matrix in `design.md` still lists
/// `cache_evicted_for_capacity` as a degraded path and needs the same edit; the
/// table below is the authority in the meantime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Informational,
    Degraded,
    Rejected,
}

impl Severity {
    /// Stable machine readable form, identical to the serialized value.
    pub fn as_identifier(self) -> &'static str {
        match self {
            Self::Informational => "informational",
            Self::Degraded => "degraded",
            Self::Rejected => "rejected",
        }
    }
}

/// Aggregated result of a run as decided by the ledger (requirement 12.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunResult {
    Success,
    Degraded,
    Rejected,
}

/// Authoritative final decision made from one complete run ledger.
///
/// Cancellation is a rejected-severity path, but stays distinct so the report
/// and caller preserve the user's explicit terminal action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutcomeDecision {
    Success,
    Degraded,
    Rejected,
    Cancelled,
}

impl OutcomeDecision {
    pub(crate) fn permits_final_output(self) -> bool {
        matches!(self, Self::Success | Self::Degraded)
    }

    pub(crate) fn as_identifier(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Degraded => "degraded",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Filesystem boundary used by final-output publication. Tests inject an
/// in-memory implementation so they can prove exactly which paths are touched.
pub(crate) trait FinalOutputFileSystem {
    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()>;
    fn remove_file(&mut self, path: &Path) -> io::Result<()>;
    fn sync_directory(&mut self, path: &Path) -> io::Result<()>;
}

/// Production filesystem for the final output transaction.
pub(crate) struct StandardFinalOutputFileSystem;

impl FinalOutputFileSystem for StandardFinalOutputFileSystem {
    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::rename(from, to)
    }

    fn remove_file(&mut self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }

    fn sync_directory(&mut self, path: &Path) -> io::Result<()> {
        #[cfg(unix)]
        {
            std::fs::File::open(path)?.sync_all()
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            Ok(())
        }
    }
}

/// Result of resolving a staged final output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputPublication {
    Published(OutcomeDecision),
    Rejected,
    Cancelled,
}

/// Makes the sole final-output decision for a run.
pub(crate) struct DegradationManager<'a> {
    ledger: &'a DegradationLedger,
}

impl<'a> DegradationManager<'a> {
    pub(crate) const fn new(ledger: &'a DegradationLedger) -> Self {
        Self { ledger }
    }

    /// Rejection outranks success and every degradation. Cancellation is tested
    /// first only to retain its existing, more-specific reporting precedence.
    pub(crate) fn decision(&self) -> OutcomeDecision {
        if self.ledger.count_of(RUN_CANCELLED_BY_USER) > 0 {
            return OutcomeDecision::Cancelled;
        }
        match self.ledger.result() {
            RunResult::Success => OutcomeDecision::Success,
            RunResult::Degraded => OutcomeDecision::Degraded,
            RunResult::Rejected => OutcomeDecision::Rejected,
        }
    }

    /// Publish `temporary_path` only after the accepted decision is known.
    ///
    /// Rejected and cancelled runs remove only their run-owned temporary path;
    /// they never remove or truncate `final_path`, a report, a preview, or any
    /// other pre-existing user file. Accepted runs use one atomic rename and
    /// then sync the containing directory.
    pub(crate) fn publish_staged_output<F: FinalOutputFileSystem>(
        &self,
        file_system: &mut F,
        temporary_path: &Path,
        final_path: &Path,
    ) -> Result<OutputPublication, String> {
        let decision = self.decision();
        if !decision.permits_final_output() {
            match file_system.remove_file(temporary_path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(format!(
                        "Failed to remove rejected image-stack temporary output '{}': {error}",
                        temporary_path.display()
                    ));
                }
            }
            return Ok(match decision {
                OutcomeDecision::Cancelled => OutputPublication::Cancelled,
                OutcomeDecision::Rejected => OutputPublication::Rejected,
                OutcomeDecision::Success | OutcomeDecision::Degraded => unreachable!(),
            });
        }

        file_system
            .rename(temporary_path, final_path)
            .map_err(|error| {
                let _ = file_system.remove_file(temporary_path);
                format!(
                    "Failed to atomically publish image-stack output '{}': {error}",
                    final_path.display()
                )
            })?;
        let parent = final_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        file_system.sync_directory(parent).map_err(|error| {
            format!(
                "Failed to sync the image-stack output folder '{}': {error}",
                parent.display()
            )
        })?;
        Ok(OutputPublication::Published(decision))
    }
}

