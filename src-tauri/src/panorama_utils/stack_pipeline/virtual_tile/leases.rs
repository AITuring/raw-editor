//! Virtual_Tile residency leases (需求 14.4).
//!
//! At most MAX_RESIDENT_VIRTUAL_TILES decoded tiles are resident; a lease
//! returns its slot by RAII and the registry records the residency peak.

use crate::panorama_utils::stitching::VirtualTile;
use std::sync::Arc;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Residency leases (需求 14.4)
// ---------------------------------------------------------------------------

/// How many full size Virtual_Tiles may be resident in memory at once
/// (需求 14.4).  Two is the minimum a pairwise tile-to-tile step needs.
pub(crate) const MAX_RESIDENT_VIRTUAL_TILES: usize = 2;

/// Why a lease could not be granted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LeaseError {
    /// Every resident slot is held by a live [`TileLease`], so admitting one
    /// more tile would break the ceiling of 需求 14.4.  The caller must drop a
    /// lease it already holds and ask again.
    ResidencyExhausted { limit: usize, held: usize },
    /// The tile itself could not be produced: neither the cache nor the caller's
    /// fusion closure returned one.
    Unavailable(String),
}

impl std::fmt::Display for LeaseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResidencyExhausted { limit, held } => write!(
                formatter,
                "cannot hold more than {limit} full size virtual tiles at once ({held} leased)"
            ),
            Self::Unavailable(detail) => write!(formatter, "{detail}"),
        }
    }
}

/// One tile currently held in memory by the store.
pub(super) struct Resident {
    pub(super) station_index: usize,
    pub(super) tile: Arc<VirtualTile>,
    /// Number of live [`TileLease`] handles on this tile.  A resident with zero
    /// handles is kept for reuse but may be evicted to make room.
    pub(super) held: usize,
    /// Value of [`LeaseState::clock`] at the last lease.  Lowest loses.
    pub(super) last_access: u64,
}

#[derive(Default)]
pub(super) struct LeaseState {
    pub(super) residents: Vec<Resident>,
    /// Monotonic access counter.  A logical clock rather than a wall clock, so
    /// the eviction order cannot depend on timer resolution (需求 14.6).
    pub(super) clock: u64,
    /// Highest `residents.len()` this registry ever reached.
    pub(super) max_resident: usize,
}

/// Enforces the residency ceiling of 需求 14.4 and observes the peak.
///
/// Held behind an `Arc` so a [`TileLease`] can return its slot after the store
/// reference the caller leased through has gone out of scope.
pub(crate) struct LeaseRegistry {
    pub(super) limit: usize,
    pub(super) state: Mutex<LeaseState>,
}

impl LeaseRegistry {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            limit: limit.max(1),
            state: Mutex::new(LeaseState::default()),
        }
    }

    pub(super) fn lock(&self) -> std::sync::MutexGuard<'_, LeaseState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Grant a lease on `station_index`, calling `provide` only when the tile is
    /// not resident already.
    pub(super) fn acquire(
        &self,
        station_index: usize,
        provide: impl FnOnce() -> Result<VirtualTile, String>,
    ) -> Result<Arc<VirtualTile>, LeaseError> {
        let mut state = self.lock();
        state.clock = state.clock.saturating_add(1);
        let now = state.clock;
        if let Some(resident) = state
            .residents
            .iter_mut()
            .find(|resident| resident.station_index == station_index)
        {
            // Already in memory: a second lease on the same station costs no
            // additional memory, so it never has to evict anything.
            resident.held += 1;
            resident.last_access = now;
            return Ok(Arc::clone(&resident.tile));
        }
        // Make room: least recently leased first, and only among tiles nobody
        // holds.  A tile with a live lease is still being read, so evicting it
        // would hand the caller a dangling view of the mosaic.
        while state.residents.len() >= self.limit {
            let victim = state
                .residents
                .iter()
                .enumerate()
                .filter(|(_, resident)| resident.held == 0)
                .min_by(|left, right| {
                    left.1
                        .last_access
                        .cmp(&right.1.last_access)
                        .then_with(|| left.1.station_index.cmp(&right.1.station_index))
                })
                .map(|(index, _)| index);
            let Some(victim) = victim else {
                let held = state
                    .residents
                    .iter()
                    .filter(|resident| resident.held > 0)
                    .count();
                return Err(LeaseError::ResidencyExhausted {
                    limit: self.limit,
                    held,
                });
            };
            state.residents.remove(victim);
        }
        // `provide` may decode and fuse, which is slow, but it runs under the
        // lock on purpose: the slot it was admitted into must not be handed to a
        // second caller in the meantime.
        let tile = Arc::new(provide().map_err(LeaseError::Unavailable)?);
        state.residents.push(Resident {
            station_index,
            tile: Arc::clone(&tile),
            held: 1,
            last_access: now,
        });
        state.max_resident = state.max_resident.max(state.residents.len());
        Ok(tile)
    }

    /// Return one handle.  The tile stays resident for reuse; it becomes
    /// evictable once no handle is left.
    pub(super) fn release(&self, station_index: usize) {
        let mut state = self.lock();
        if let Some(resident) = state
            .residents
            .iter_mut()
            .find(|resident| resident.station_index == station_index)
        {
            resident.held = resident.held.saturating_sub(1);
        }
    }

    pub(super) fn resident_count(&self) -> usize {
        self.lock().residents.len()
    }

    pub(super) fn held_count(&self) -> usize {
        self.lock()
            .residents
            .iter()
            .map(|resident| resident.held)
            .sum()
    }

    pub(super) fn max_resident(&self) -> usize {
        self.lock().max_resident
    }
}

/// RAII lease on a resident Virtual_Tile.  Dropping it returns the slot; the
/// tile itself may stay in memory for reuse until the store needs the slot.
pub(crate) struct TileLease {
    pub(super) station_index: usize,
    pub(super) tile: Arc<VirtualTile>,
    pub(super) registry: Arc<LeaseRegistry>,
}

impl TileLease {
    pub(crate) fn station_index(&self) -> usize {
        self.station_index
    }

    pub(crate) fn tile(&self) -> &VirtualTile {
        &self.tile
    }
}

impl std::ops::Deref for TileLease {
    type Target = VirtualTile;

    fn deref(&self) -> &Self::Target {
        &self.tile
    }
}

impl Drop for TileLease {
    fn drop(&mut self) {
        self.registry.release(self.station_index);
    }
}
