//! The games.
//!
//! Each one is a single [`Game`](crate::game::Game) implementation —
//! physics, observation, reward and rendering — and a thin
//! `src/bin/<name>.rs` that hands it to [`run`](crate::run). Nothing
//! else in the crate knows which game it is driving.
//!
//! They are deliberately small and fast-resolving. An episode is one
//! rally, one descent or one falling ball, so 25 parallel environments
//! produce enough labelled outcomes per second for the win-rate curve
//! to move while somebody is watching it.

pub mod catch;
pub mod lander;
pub mod pong;
