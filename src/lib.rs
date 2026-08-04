#![allow(
    clippy::match_like_matches_macro,
    clippy::redundant_pattern_matching,
    clippy::needless_lifetimes,
    clippy::new_without_default,
    clippy::single_match,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::collapsible_match
)]
#![warn(trivial_numeric_casts, unused_extern_crates)]

//! Shared plumbing for live-learning game demos.
//!
//! The crate wires three independent pieces into one binary:
//!
//! 1. A Blade-graphics renderer with an egui overlay. Egui handles all
//!    on-screen drawing — paddles, balls, stats, sparklines. There is no
//!    custom shader, no MSAA juggling, and no text-shaping dependency;
//!    egui already ships fonts and primitive shapes.
//! 2. A meganeura training session that runs on the *same*
//!    `blade_graphics::Context` as the renderer. The context is created
//!    once, wrapped in `Arc`, and cloned into [`meganeura::Session::with_context`].
//! 3. A replay buffer and DQN trainer glue that a concrete [`Game`]
//!    implementation plugs into.
//!
//! A binary (see `src/bin/pong.rs`) supplies a [`Game`] and calls [`run`].

pub mod agent;
pub mod app;
pub mod catch;
pub mod env_loop;
pub mod game;
pub mod lander;
pub mod pong;
pub mod profiling;
pub mod stats;

pub use agent::{Action, Agent, AgentConfig, Observation, Transition};
pub use app::{AppConfig, run};
pub use game::{Game, tint};
pub use stats::{RollingStats, SparkLine};

/// A random-number stream for anything whose noise should be
/// reproducible: the agent's parameter init and ε-greedy draws, and
/// each game's own randomness (ball angles, spawn positions).
///
/// With `MEGAPLAYS_SEED=<u64>` set, streams are handed out
/// deterministically as `seed, seed+1, seed+2, …` in construction
/// order — so two runs of the same build compare like for like, while
/// parallel environments still get *different* streams and don't play
/// the same episode 32 times over. Unset, every stream comes from OS
/// entropy as before.
pub fn seeded_rng() -> rand::rngs::StdRng {
    use rand::SeedableRng;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    match std::env::var("MEGAPLAYS_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        Some(base) => {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            if n == 0 {
                log::info!("seeded from MEGAPLAYS_SEED={base}");
            }
            rand::rngs::StdRng::seed_from_u64(base.wrapping_add(n))
        }
        // rand 0.10 dropped `from_os_rng`; seed from a transient
        // ThreadRng (OS-seeded under the hood).
        None => rand::rngs::StdRng::from_rng(&mut rand::rng()),
    }
}
