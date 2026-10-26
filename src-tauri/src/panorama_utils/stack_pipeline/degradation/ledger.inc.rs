// Degradation_Manager fragment; included at parent module scope.

/// One active degraded or rejected path of a run.
#[derive(Debug, Clone, Serialize)]
pub struct DegradationEntry {
    /// ASCII lowercase, digits and underscore, at most 64 bytes.
    pub reason: &'static str,
    pub severity: Severity,
    /// Free form measured context for the Stack_Report. `Value::Null` when the
    /// call site has no numbers to report yet.
    pub detail: Value,
}

impl DegradationEntry {
    pub fn new(reason: &'static str, severity: Severity, detail: Value) -> Self {
        Self {
            reason,
            severity,
            detail,
        }
    }
}

/// Collects every degraded or rejected path of one run (requirement 12.9).
#[derive(Debug, Clone, Default, Serialize)]
pub struct DegradationLedger {
    pub entries: Vec<DegradationEntry>,
}

impl DegradationLedger {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Records `reason` with the severity declared by the design matrix.
    /// Unknown identifiers are ignored so an observation point can never
    /// introduce a reason outside the frozen set.
    pub fn record(&mut self, reason: &'static str, detail: Value) {
        if let Some(severity) = severity_of(reason) {
            self.entries
                .push(DegradationEntry::new(reason, severity, detail));
        } else {
            debug_assert!(false, "unknown degradation reason identifier: {reason}");
        }
    }

    /// Records `reason` with an explicit severity. Only used where a call site
    /// must override the table (currently nothing does).
    pub fn record_with_severity(
        &mut self,
        reason: &'static str,
        severity: Severity,
        detail: Value,
    ) {
        self.entries
            .push(DegradationEntry::new(reason, severity, detail));
    }

    pub fn entries(&self) -> &[DegradationEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn has_rejection(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.severity == Severity::Rejected)
    }

    /// `true` when at least one entry actually degraded the output. Entries of
    /// [`Severity::Informational`] are housekeeping and do not count.
    pub fn has_degradation(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.severity == Severity::Degraded)
    }

    /// Number of entries carrying `reason`.
    pub fn count_of(&self, reason: &str) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.reason == reason)
            .count()
    }

    /// Decision rule of requirement 12.9: any `Rejected` entry makes the run a
    /// rejected output, otherwise any `Degraded` entry makes it a degraded
    /// output. [`Severity::Informational`] entries are reported in full but
    /// never move the verdict, so an otherwise clean run that only did some
    /// cache housekeeping stays a success.
    pub fn result(&self) -> RunResult {
        if self.has_rejection() {
            RunResult::Rejected
        } else if self.has_degradation() {
            RunResult::Degraded
        } else {
            RunResult::Success
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

