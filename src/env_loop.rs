//! One decision burst across every parallel environment.
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
//! [`run_burst`] is the single correct implementation. One decision is held
//! for up to `repeat` physics substeps and yields one transition per env
//! whose reward is the burst-summed reward and whose `next_obs` is the
//! post-burst observation. Per-substep stats updates (action histogram,
//! reward sparkline, episode return/length) happen in the caller-supplied
//! `on_step` closure.
//!
//! Two practical reasons to hold actions across substeps:
//!
//! - **Inference round trips dominate** when the policy network is tiny. At
//!   `repeat = 4` and a 120 Hz physics step the agent does one batched
//!   forward pass every 33 ms instead of every 8 ms — a 4× cut on the only
//!   item that touched the GPU each substep.
//! - **120 Hz ε-greedy is no exploration.** Three uniform random ±torques
//!   sampled per perception-budget average to zero net force; lander could
//!   never commit to a maneuver. A held random action over 33 ms is a real
//!   directed perturbation.

use crate::{
    agent::{Action, Agent, Transition},
    game::{Game, StepOutcome},
};

/// Run a burst of `repeat` physics substeps under one held action per env.
///
/// `obs_buf` must already be sized `num_envs * obs_dim`; the helper reuses it
/// across calls so the caller only allocates once per frame. `on_step` fires
/// once per (env, substep), so action-histogram / reward-sparkline / episode
/// counters update at the substep cadence even though only one (obs, action,
/// reward, next_obs, done) transition is recorded per env per burst.
///
/// If an env terminates mid-burst, that env stops stepping for the remainder
/// (its reward continues to sum into the burst total at 0, its `next_obs`
/// is the terminal-state observation), and is reset after `on_step` for the
/// final substep so the closure observes the terminal outcome.
pub fn run_burst<G, F>(
    agent: &mut Agent,
    games: &mut [G],
    obs_buf: &mut [f32],
    obs_dim: usize,
    repeat: u32,
    mut on_step: F,
) where
    G: Game,
    F: FnMut(usize, Action, StepOutcome),
{
    debug_assert_eq!(obs_buf.len(), games.len() * obs_dim);
    debug_assert!(repeat >= 1);
    for (i, g) in games.iter().enumerate() {
        let o = g.observation();
        obs_buf[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&o);
    }
    let actions = {
        let _s = tracing::info_span!("select_actions").entered();
        agent.select_actions(obs_buf)
    };

    let mut burst_reward = vec![0.0_f32; games.len()];
    let mut env_done = vec![false; games.len()];

    let _physics = tracing::info_span!("physics").entered();
    for _ in 0..repeat {
        for (i, g) in games.iter_mut().enumerate() {
            if env_done[i] {
                continue;
            }
            let action = actions[i];
            let outcome = g.step(action);
            burst_reward[i] += outcome.reward;
            on_step(i, action, outcome);
            if outcome.done {
                env_done[i] = true;
            }
        }
    }

    for (i, g) in games.iter_mut().enumerate() {
        let next_obs = g.observation();
        agent.record(Transition {
            obs: obs_buf[i * obs_dim..(i + 1) * obs_dim].to_vec(),
            action: actions[i],
            reward: burst_reward[i],
            next_obs,
            done: env_done[i],
        });
        if env_done[i] {
            g.reset();
        }
    }
}
