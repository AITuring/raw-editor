// Degradation_Manager fragment; included at parent module scope.

/// Ledger of the run currently in flight.
///
/// The existing fallback sites are free functions deep inside the stitching
/// code and have no ledger parameter. A run scoped global keeps the new
/// observation additive: no signature changes, no behaviour changes, and a
/// record is a push onto a `Vec` behind an uncontended lock.
static RUN_LEDGER: Mutex<DegradationLedger> = Mutex::new(DegradationLedger::new());

/// Serialises operations that use the process-wide run observation sinks.  The
/// sinks predate concurrent runs and are intentionally kept signature-free;
/// one guard now makes reset -> render -> snapshot an atomic run transaction.
static RUN_SCOPE: Mutex<()> = Mutex::new(());
#[cfg(test)]
static RUN_SCOPE_ACTIVE: AtomicBool = AtomicBool::new(false);
/// The image-stack command increments this generation for every new request.
/// A run that already owns `RUN_SCOPE` observes a newer generation at its
/// cooperative checkpoints and exits instead of holding the process-wide sinks
/// while a replacement request waits for a completed render.
static ACTIVE_RUN_GENERATION: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static OWNED_RUN_GENERATION: Cell<usize> = const { Cell::new(0) };
}

pub(crate) struct RunScope {
    _guard: MutexGuard<'static, ()>,
}

#[cfg(test)]
impl Drop for RunScope {
    fn drop(&mut self) {
        RUN_SCOPE_ACTIVE.store(false, Ordering::Release);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RunScopeCancelled;

/// Acquire the process-wide observation scope while remaining responsive to a
/// newer UI generation. A queued replacement run never takes ownership of a
/// scope after its generation has already been superseded.
pub(crate) fn begin_run_scope() -> RunScope {
    begin_run_scope_for_generation(0).expect("a direct run must acquire its scope")
}

pub(crate) fn begin_run_scope_for_generation(
    generation: usize,
) -> Result<RunScope, RunScopeCancelled> {
    loop {
        if generation_changed_for(generation, ACTIVE_RUN_GENERATION.load(Ordering::Acquire)) {
            return Err(RunScopeCancelled);
        }
        match RUN_SCOPE.try_lock() {
            Ok(guard) => {
                if generation_changed_for(generation, ACTIVE_RUN_GENERATION.load(Ordering::Acquire))
                {
                    drop(guard);
                    return Err(RunScopeCancelled);
                }
                OWNED_RUN_GENERATION.with(|owned| owned.set(generation));
                #[cfg(test)]
                RUN_SCOPE_ACTIVE.store(true, Ordering::Release);
                return Ok(RunScope { _guard: guard });
            }
            Err(TryLockError::Poisoned(poisoned)) => {
                let guard = poisoned.into_inner();
                if generation_changed_for(generation, ACTIVE_RUN_GENERATION.load(Ordering::Acquire))
                {
                    drop(guard);
                    return Err(RunScopeCancelled);
                }
                OWNED_RUN_GENERATION.with(|owned| owned.set(generation));
                #[cfg(test)]
                RUN_SCOPE_ACTIVE.store(true, Ordering::Release);
                return Ok(RunScope { _guard: guard });
            }
            Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(5)),
        }
    }
}

pub(crate) fn active_run_generation() -> usize {
    ACTIVE_RUN_GENERATION.load(Ordering::Acquire)
}

/// Publish the generation of the newest UI request.  The next run still waits
/// for the scope to become available, but the previous owner sees the change at
/// its next checkpoint and releases the scope promptly.
pub(crate) fn set_active_run_generation(generation: usize) {
    ACTIVE_RUN_GENERATION.store(generation, Ordering::Release);
}

/// Whether the run holding the process-wide scope has been superseded by a
/// newer image-stack request.  A generation of zero is the direct/library test
/// path and deliberately has no external cancellation source.
pub(crate) fn run_generation_changed() -> bool {
    OWNED_RUN_GENERATION.with(|owned| {
        let generation = owned.get();
        run_generation_changed_for(generation)
    })
}

pub(crate) fn run_generation_changed_for(owned_generation: usize) -> bool {
    generation_changed_for(
        owned_generation,
        ACTIVE_RUN_GENERATION.load(Ordering::Acquire),
    )
}

fn generation_changed_for(owned_generation: usize, active_generation: usize) -> bool {
    owned_generation != 0 && active_generation != owned_generation
}

#[cfg(test)]
#[track_caller]
fn assert_run_scope() {
    assert!(
        RUN_SCOPE_ACTIVE.load(Ordering::Acquire),
        "run ledger write requires degradation::begin_run_scope()"
    );
}

fn with_run_ledger<T>(body: impl FnOnce(&mut DegradationLedger) -> T) -> T {
    let mut guard = RUN_LEDGER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    body(&mut guard)
}

/// Starts a fresh observation window. Called once per stitching run before any
/// diagnostic point can fire.
#[cfg_attr(test, track_caller)]
pub fn reset_run_ledger() {
    #[cfg(test)]
    assert_run_scope();
    with_run_ledger(DegradationLedger::clear);
}

/// Records an active degraded or rejected path of the current run. Parallel to
/// the existing `println!` diagnostics; it never affects control flow.
#[cfg_attr(test, track_caller)]
pub fn record_run_degradation(reason: &'static str, detail: Value) {
    #[cfg(test)]
    assert_run_scope();
    with_run_ledger(|ledger| ledger.record(reason, detail));
}

/// Copy of the ledger for report serialisation. The window stays open so late
/// terminal paths can still record.
pub fn run_ledger_snapshot() -> DegradationLedger {
    with_run_ledger(|ledger| ledger.clone())
}

