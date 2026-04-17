//! The [`Game`] trait.
//!
//! Implementors describe a self-contained environment: physics, rendering,
//! observation, reward. The [`run`](crate::run) driver takes care of
//! windowing, GPU setup, the training loop, and stat overlays.
//!
//! Observations and actions are intentionally type-erased as flat vectors
//! (`Vec<f32>` / `u32`). The concrete shapes are declared once via
//! [`Game::spec`] and propagated to the DQN network builder. Keeping the
//! trait object-safe lets the driver swap games at runtime without
//! monomorphising the whole app harness per game.

use crate::agent::{Action, Observation};

/// Static description of the environment the driver needs up front —
/// before opening a window, before building the neural network graph.
#[derive(Clone, Debug)]
pub struct GameSpec {
    /// Window title.
    pub title: &'static str,
    /// Observation length in f32s.
    pub obs_dim: usize,
    /// Number of discrete actions available to an agent.
    pub num_actions: u32,
    /// Fixed physics step in seconds. Decoupled from render rate.
    pub physics_dt: f32,
    /// Logical play-area size in arbitrary units; the renderer maps this
    /// into the window. Aspect ratio is preserved; the short axis is
    /// padded.
    pub play_area: [f32; 2],
}

/// A reward and episode-boundary signal.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepOutcome {
    /// Total reward this step — terminal + any dense shaping.
    pub reward: f32,
    /// True when the episode ends on this step.
    pub done: bool,
    /// The sparse terminal-reward component (±1 for win/loss, 0 while
    /// in play). The harness uses this — not `reward` — to count wins
    /// so that shaping can't skew the scoreboard.
    pub terminal_reward: f32,
}

pub trait Game {
    fn spec(&self) -> GameSpec;

    /// Reset to a fresh episode. Called at construction and after every
    /// `done: true` step returned from [`Game::step`].
    fn reset(&mut self);

    /// Advance physics by one fixed step using `action`. For multi-agent
    /// games the concrete implementation handles scripted / mirrored
    /// opponents internally — the driver only sees one observation /
    /// action stream per frame.
    fn step(&mut self, action: Action) -> StepOutcome;

    /// Current observation for the agent under the driver's control.
    fn observation(&self) -> Observation;

    /// Draw the scene into `painter` within `rect`.
    ///
    /// Coordinates in `rect` are egui screen-space pixels; the
    /// implementation is responsible for mapping its own play-area units
    /// into that rect.
    fn paint(&self, painter: &egui::Painter, rect: egui::Rect);

    /// Game-specific panel contents (scores, mode toggles, etc.).
    /// Called once per frame inside the stats window.
    fn ui(&mut self, _ui: &mut egui::Ui) {}
}
