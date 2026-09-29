//! Run-scoped memory threshold resolution and peak RSS sampling.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub const MEMORY_THRESHOLD_DEFAULT_BYTES: u64 = 24 * 1024 * 1024 * 1024;
pub const MEMORY_THRESHOLD_MIN_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const MEMORY_THRESHOLD_PHYSICAL_RATIO: f64 = 0.75;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryThresholdSource {
    Default,
    UserConfigured,
    AutoCalibrated,
}

impl MemoryThresholdSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::UserConfigured => "user_configured",
            Self::AutoCalibrated => "auto_calibrated",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryThreshold {
    pub bytes: u64,
    pub source: MemoryThresholdSource,
}

pub fn resolve_memory_threshold(
    physical_memory_bytes: u64,
    user_configured_bytes: Option<u64>,
    auto_calibrated: bool,
) -> MemoryThreshold {
    let upper = ((physical_memory_bytes as f64) * MEMORY_THRESHOLD_PHYSICAL_RATIO)
        .floor()
        .min(u64::MAX as f64) as u64;

    // A machine with less than 16/3 GiB of physical memory cannot satisfy the
    // nominal 4 GiB lower bound.  In that case the physical-memory ceiling is
    // the only safe threshold, and the report must say that it was
    // auto-calibrated even when a stale user setting was supplied.
    if upper < MEMORY_THRESHOLD_MIN_BYTES {
        return MemoryThreshold {
            bytes: upper,
            source: MemoryThresholdSource::AutoCalibrated,
        };
    }

    let lower = MEMORY_THRESHOLD_MIN_BYTES.min(upper);
    let clamp = |value: u64| value.max(lower).min(upper);
    if let Some(value) = user_configured_bytes {
        return MemoryThreshold {
            bytes: clamp(value),
            source: MemoryThresholdSource::UserConfigured,
        };
    }
    if auto_calibrated {
        return MemoryThreshold {
            bytes: upper,
            source: MemoryThresholdSource::AutoCalibrated,
        };
    }
    MemoryThreshold {
        bytes: clamp(MEMORY_THRESHOLD_DEFAULT_BYTES),
        source: MemoryThresholdSource::Default,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RssSample {
    pub peak_rss_bytes: u64,
    pub sample_count: u64,
    pub threshold_exceeded: bool,
}

struct SamplerState {
    peak_rss_bytes: AtomicU64,
    sample_count: AtomicU64,
    cancel: Arc<AtomicBool>,
}

pub struct RssSampler {
    state: Arc<SamplerState>,
    stop: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl RssSampler {
    pub fn start<F>(threshold_bytes: u64, read_rss: F) -> Self
    where
        F: Fn() -> u64 + Send + Sync + 'static,
    {
        Self::start_with_interval(threshold_bytes, Duration::from_millis(500), read_rss)
    }

    pub fn start_with_interval<F>(threshold_bytes: u64, interval: Duration, read_rss: F) -> Self
    where
        F: Fn() -> u64 + Send + Sync + 'static,
    {
        let state = Arc::new(SamplerState {
            peak_rss_bytes: AtomicU64::new(0),
            sample_count: AtomicU64::new(0),
            cancel: Arc::new(AtomicBool::new(false)),
        });
        let (stop, receiver) = mpsc::channel();
        let thread_state = Arc::clone(&state);
        let reader = Arc::new(read_rss);
        let thread_reader = Arc::clone(&reader);
        let thread = thread::spawn(move || {
            loop {
                let rss = thread_reader();
                thread_state.sample_count.fetch_add(1, Ordering::Relaxed);
                thread_state
                    .peak_rss_bytes
                    .fetch_max(rss, Ordering::Relaxed);
                if rss > threshold_bytes {
                    thread_state.cancel.store(true, Ordering::Release);
                }
                match receiver.recv_timeout(interval) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
        });
        Self {
            state,
            stop: Some(stop),
            thread: Some(thread),
        }
    }

    pub fn snapshot(&self, threshold_bytes: u64) -> RssSample {
        let peak = self.state.peak_rss_bytes.load(Ordering::Acquire);
        RssSample {
            peak_rss_bytes: peak,
            sample_count: self.state.sample_count.load(Ordering::Acquire),
            threshold_exceeded: peak > threshold_bytes || self.state.cancel.load(Ordering::Acquire),
        }
    }

    /// Whether the background sampler has observed RSS above the run limit.
    /// Callers use this at stage boundaries so cancellation is cooperative.
    pub fn cancelled(&self) -> bool {
        self.state.cancel.load(Ordering::Acquire)
    }

    pub fn stop(mut self, threshold_bytes: u64) -> RssSample {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        self.snapshot(threshold_bytes)
    }
}

impl Drop for RssSampler {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn threshold_resolution_clamps_and_records_source() {
        let physical = 32 * 1024 * 1024 * 1024;
        assert_eq!(
            resolve_memory_threshold(physical, None, false),
            MemoryThreshold {
                bytes: MEMORY_THRESHOLD_DEFAULT_BYTES,
                source: MemoryThresholdSource::Default
            }
        );
        assert_eq!(
            resolve_memory_threshold(physical, Some(u64::MAX), false).source,
            MemoryThresholdSource::UserConfigured
        );
        assert_eq!(
            resolve_memory_threshold(physical, None, true).source,
            MemoryThresholdSource::AutoCalibrated
        );
        assert_eq!(
            resolve_memory_threshold(1, None, false),
            MemoryThreshold {
                bytes: 0,
                source: MemoryThresholdSource::AutoCalibrated,
            }
        );
        assert_eq!(
            resolve_memory_threshold(3 * 1024 * 1024 * 1024, Some(u64::MAX), false),
            MemoryThreshold {
                bytes: 3 * 1024 * 1024 * 1024 * 3 / 4,
                source: MemoryThresholdSource::AutoCalibrated,
            }
        );
    }

    #[test]
    fn sampler_uses_injected_rss_and_sets_cancel() {
        let values = Arc::new(AtomicU64::new(7));
        let next = Arc::clone(&values);
        let sampler = RssSampler::start_with_interval(10, Duration::from_millis(1), move || {
            next.fetch_add(5, Ordering::Relaxed)
        });
        thread::sleep(Duration::from_millis(5));
        let sample = sampler.stop(10);
        assert!(sample.sample_count >= 1);
        assert!(sample.peak_rss_bytes > 10);
        assert!(sample.threshold_exceeded);
    }
}
