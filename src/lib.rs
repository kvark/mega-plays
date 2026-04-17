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
pub mod game;
pub mod lander;
pub mod pong;
pub mod stats;

pub use agent::{Action, Agent, AgentConfig, Observation, Transition};
pub use app::{AppConfig, run};
pub use game::Game;
pub use stats::{RollingStats, SparkLine};
