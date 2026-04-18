//! Unconditional per-run profiling — CPU spans + GPU pass timings on
//! a single `.pftrace` you drop into ui.perfetto.dev.
//!
//! All the hard work lives in [`meganeura::profiler`]: it maintains
//! the global event buffer, the `tracing` subscriber, and the
//! Perfetto binary writer. This module is the mega-plays-specific
//! glue:
//!
//! - [`init_or_default`] sets everything up on startup and returns a
//!   drop-guard that writes the trace at process exit.
//! - [`record_submit`] wraps a command-encoder submit: it captures
//!   the pre-submit timestamp, runs the caller-supplied closure
//!   (which should call `gpu.submit(&mut encoder)`), and — using the
//!   `Timings` blade returns on the *next* submit — feeds the GPU
//!   track.
//!
//! `MEGA_TRACE` env var controls destination:
//! - unset → `./mega-plays.pftrace` at shutdown
//! - `off` / `0` / empty → disabled (no subscriber, no file)
//! - any other value → use as path

use std::path::PathBuf;
use std::time::Duration;

/// Drop-guard that writes the collected trace when it falls out of
/// scope. Dropping a disabled guard is a no-op.
#[must_use = "profiler writes its trace when the guard drops"]
pub struct TraceGuard {
    target: Option<PathBuf>,
}

impl TraceGuard {
    fn from_env(default_path: &str) -> Self {
        let target = match std::env::var("MEGA_TRACE") {
            Ok(v) if v.eq_ignore_ascii_case("off") || v == "0" || v.is_empty() => None,
            Ok(v) => Some(PathBuf::from(v)),
            Err(_) => Some(PathBuf::from(default_path)),
        };
        Self { target }
    }

    /// `true` if the trace will be written on drop.
    pub fn enabled(&self) -> bool {
        self.target.is_some()
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        if let Some(p) = self.target.take() {
            match meganeura::profiler::save(&p) {
                Ok(()) => {
                    log::info!(
                        "mega-plays profiler: wrote {} events to {}",
                        meganeura::profiler::event_count(),
                        p.display()
                    );
                }
                Err(e) => {
                    log::warn!(
                        "mega-plays profiler: failed to write {}: {}",
                        p.display(),
                        e
                    );
                }
            }
        }
    }
}

/// Install the tracing subscriber (if enabled) and return a guard
/// that saves the trace on drop. Call from `main()` before anything
/// else emits spans.
pub fn init_or_default() -> TraceGuard {
    let guard = TraceGuard::from_env("./mega-plays.pftrace");
    if guard.enabled() {
        meganeura::profiler::init();
    }
    guard
}

/// Forward blade's captured per-pass timings to the GPU track.
///
/// `submit_offset_ns` should come from
/// [`meganeura::profiler::now_ns`] captured *immediately before* the
/// submit call those timings are from. Pass each blade
/// `(name, duration)` through unchanged.
pub fn record_gpu_passes(submit_offset_ns: u64, timings: &[(String, Duration)]) {
    if timings.is_empty() {
        return;
    }
    meganeura::profiler::record_gpu_passes(submit_offset_ns, timings);
}

/// Nanosecond offset relative to the profiler's epoch. Safe to call
/// even if profiling is disabled (returns 0).
pub fn now_ns() -> u64 {
    meganeura::profiler::now_ns()
}
