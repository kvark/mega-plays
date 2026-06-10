//! One physics substep across every parallel environment.
//!
//! All four call sites (`tick`, `train_epoch_chunk`, `run_headless`, and the
//! headless smoke test) used to inline the same gather-obs / select-actions /
//! step-games / record-transition / reset-on-done sequence. Three of them
//! also reproduced the same off-by-one bug: storing `last_action[i]` (the
//! action chosen on the *previous* substep) alongside `prev` (the observation
//! seen on the *current* substep). The replay buffer ended up with shifted
//! (obs, action) pairs and the first transition of every episode was silently
//! dropped because the `last_obs[i].replace(...)` short-circuited on the
//! first step.
//!
//! [`run_substep`] is the single correct implementation: it records the
//! observation that produced each action together with the same action, plus
//! the reward and post-step observation it caused. Per-env stats updates
//! happen in the caller-supplied `on_step` closure.
//!
//! Done envs are reset *after* `on_step` so the closure observes the
//! terminal reward.

use crate::{
    agent::{Action, Agent, Transition},
    game::{Game, StepOutcome},
};

/// Run one physics substep across `games`, recording one transition per env.
///
/// `obs_buf` must already be sized `num_envs * obs_dim`; the helper reuses it
/// across calls so the caller only allocates once per frame.
pub fn run_substep<G, F>(
    agent: &mut Agent,
    games: &mut [G],
    obs_buf: &mut [f32],
    obs_dim: usize,
    mut on_step: F,
) where
    G: Game,
    F: FnMut(usize, Action, StepOutcome),
{
    debug_assert_eq!(obs_buf.len(), games.len() * obs_dim);
    for (i, g) in games.iter().enumerate() {
        let o = g.observation();
        obs_buf[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&o);
    }
    let actions = {
        let _s = tracing::info_span!("select_actions").entered();
        agent.select_actions(obs_buf)
    };
    let _physics = tracing::info_span!("physics").entered();
    for (i, g) in games.iter_mut().enumerate() {
        let action = actions[i];
        let outcome = g.step(action);
        let next_obs = g.observation();
        agent.record(Transition {
            obs: obs_buf[i * obs_dim..(i + 1) * obs_dim].to_vec(),
            action,
            reward: outcome.reward,
            next_obs,
            done: outcome.done,
        });
        on_step(i, action, outcome);
        if outcome.done {
            g.reset();
        }
    }
}
