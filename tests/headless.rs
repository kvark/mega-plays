//! Headless training smoke test.
//!
//! Runs the DQN training loop for a fixed number of iterations
//! without opening a window. Works on lavapipe (software Vulkan)
//! for CI.

use std::sync::Arc;

use mega_plays::{
    agent::{Agent, AgentConfig, Transition},
    env_loop::run_burst,
    game::Game,
    pong::PongGame,
};

/// Headless training loop: run `frames` iterations, each with
/// `substeps` physics substeps across `num_envs` environments,
/// followed by `train_steps` gradient updates. Mimics the windowed
/// app's tick() cadence.
/// Returns (mean_return, win_rate, gradient_steps).
fn train_headless<G: Game>(
    games: &mut [G],
    agent: &mut Agent,
    frames: u32,
    substeps: u32,
    train_steps: u32,
) -> (f32, f32, u64) {
    let num_envs = games.len();
    let obs_dim = games[0].spec().obs_dim;
    let mut obs_buf = vec![0.0_f32; num_envs * obs_dim];
    let mut episode_return = vec![0.0_f32; num_envs];
    let mut total_return = 0.0_f32;
    let mut total_episodes = 0u64;
    let mut wins = 0u64;
    let mut losses = 0u64;
    let mut dones_no_terminal = 0u64;

    let action_repeat = agent.action_repeat();
    let bursts = (substeps / action_repeat.max(1)).max(1);
    for _ in 0..frames {
        for _ in 0..bursts {
            run_burst(
                agent,
                games,
                &mut obs_buf,
                obs_dim,
                action_repeat,
                |i, _action, outcome| {
                    episode_return[i] += outcome.reward;
                    if outcome.done || outcome.truncated {
                        if outcome.done {
                            if outcome.terminal_reward > 0.0 {
                                wins += 1;
                            } else if outcome.terminal_reward < 0.0 {
                                losses += 1;
                            } else {
                                dones_no_terminal += 1;
                            }
                        }
                        total_return += episode_return[i];
                        total_episodes += 1;
                        episode_return[i] = 0.0;
                    }
                },
            );
        }

        for _ in 0..train_steps {
            if agent.train_step().is_none() {
                break;
            }
        }
    }

    let mean_return = if total_episodes > 0 {
        total_return / total_episodes as f32
    } else {
        0.0
    };
    let win_rate = if total_episodes > 0 {
        wins as f32 / total_episodes as f32
    } else {
        0.0
    };
    eprintln!(
        "  episodes={total_episodes} wins={wins} losses={losses} \
         no_terminal={dones_no_terminal}",
    );
    (mean_return, win_rate, agent.gradient_steps)
}

/// Sanity check: can meganeura learn a fixed mapping?
/// Feed a constant obs, train toward a known target, verify Q-values move.
#[test]
fn training_moves_q_values() {
    let gpu = unsafe {
        blade_graphics::Context::init(blade_graphics::ContextDesc {
            presentation: false,
            validation: false,
            timing: false,
            ..Default::default()
        })
    }
    .expect("init Blade context");
    let gpu = Arc::new(gpu);

    let num_envs = 1;
    let cfg = AgentConfig {
        warmup: 256,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(gpu, 6, 3, num_envs, cfg);

    // Fill replay with identical transitions: obs=zeros, action=1, reward=5.0, done=true.
    for _ in 0..2000 {
        agent.record(
            0,
            Transition::step(vec![0.0; 6], 1, 5.0, vec![0.0; 6], true),
            true,
        );
    }

    // Q-values before training.
    let obs = vec![0.0_f32; 6];
    let q_before = agent.select_actions(&obs);
    eprintln!("action before training: {}", q_before[0]);

    // Train 2000 steps — plenty to learn a fixed mapping.
    for _ in 0..2000 {
        agent.train_step();
    }

    // Force greedy: epsilon is step-based but we want to check the
    // raw Q-values, so set MEGAPLAYS_FORCE_EPSILON=0.
    unsafe { std::env::set_var("MEGAPLAYS_FORCE_EPSILON", "0") };
    let q_after = agent.select_actions(&obs);
    unsafe { std::env::remove_var("MEGAPLAYS_FORCE_EPSILON") };
    eprintln!(
        "action after 2000 grad steps: {} (loss={:.4})",
        q_after[0], agent.last_loss,
    );

    assert_eq!(
        q_after[0], 1,
        "expected greedy action=1 (trained on reward=5 for action 1)"
    );
}

#[test]
fn pong_learns_to_beat_slow_opponent() {
    let gpu = unsafe {
        blade_graphics::Context::init(blade_graphics::ContextDesc {
            presentation: false,
            validation: false,
            timing: false,
            ..Default::default()
        })
    }
    .expect("init Blade context (needs Vulkan — lavapipe is fine)");
    let gpu = Arc::new(gpu);

    let num_envs = 16;
    let cfg = AgentConfig::default();

    let mut games: Vec<PongGame> = (0..num_envs).map(|_| PongGame::new()).collect();
    let spec = games[0].spec();
    let mut agent = Agent::new(gpu, spec.obs_dim, spec.num_actions, num_envs, cfg);

    // Warm up: fill the replay buffer (1000 frames × 4 substeps).
    train_headless(&mut games, &mut agent, 1000, 4, 0);

    // Train: long enough for epsilon to fully decay (20 k grad steps at
    // 8 grad/frame = 2500 frames) plus a few thousand more to learn the
    // ball-tracking policy. The earlier 28 800 frames pre-dated the
    // stale-action fix; with corrected transition recording + Double-DQN
    // + action repeat, pong reaches the 40 % win-rate gate in well
    // under 6 000 frames.
    let (mean_return, win_rate, grad_steps) = train_headless(&mut games, &mut agent, 6000, 4, 8);

    eprintln!(
        "pong headless: mean_return={mean_return:.2} win_rate={:.1}% \
         grad_steps={grad_steps} replay={} last_loss={:.4} eps={:.3} \
         game0={}:{}",
        win_rate * 100.0,
        agent.replay_len(),
        agent.last_loss,
        agent.current_epsilon(),
        games[0].score_agent(),
        games[0].score_opponent(),
    );

    assert!(
        grad_steps > 1000,
        "expected meaningful training, got {grad_steps} gradient steps"
    );
    assert!(
        agent.last_loss.is_finite(),
        "loss diverged: {}",
        agent.last_loss,
    );
    assert!(
        win_rate > 0.40,
        "expected win_rate > 40% (random baseline ~15%), got {:.1}%",
        win_rate * 100.0,
    );
}

/// Train for longer and save a checkpoint to `mega-pong.weights`.
/// Ignored by default — run explicitly with:
///   cargo test --release --test headless -- --ignored save_checkpoint
#[test]
#[ignore]
fn save_checkpoint() {
    let gpu = unsafe {
        blade_graphics::Context::init(blade_graphics::ContextDesc {
            presentation: false,
            validation: false,
            timing: false,
            ..Default::default()
        })
    }
    .expect("init Blade context");
    let gpu = Arc::new(gpu);

    let num_envs = 16;
    let cfg = AgentConfig::default();

    let mut games: Vec<PongGame> = (0..num_envs).map(|_| PongGame::new()).collect();
    let spec = games[0].spec();
    let mut agent = Agent::new(gpu, spec.obs_dim, spec.num_actions, num_envs, cfg);

    // Warm up.
    eprintln!("warming up...");
    train_headless(&mut games, &mut agent, 1000, 4, 0);

    // Train ~3x longer than the normal test.
    eprintln!("training...");
    let (mean_return, win_rate, grad_steps) = train_headless(&mut games, &mut agent, 80_000, 4, 8);

    let path = std::path::Path::new("mega-pong.weights");
    agent.save_weights(path).expect("failed to save weights");

    eprintln!(
        "saved checkpoint: mean_return={mean_return:.2} win_rate={:.1}% \
         grad_steps={grad_steps} eps={:.3}",
        win_rate * 100.0,
        agent.current_epsilon(),
    );
}

/// Measure per-batch timing to detect progressive slowdown.
/// Also tests scaling with higher num_envs (64).
#[test]
#[ignore]
fn timing_stability() {
    use std::time::Instant;

    let gpu = unsafe {
        blade_graphics::Context::init(blade_graphics::ContextDesc {
            presentation: false,
            validation: false,
            timing: false,
            ..Default::default()
        })
    }
    .expect("init Blade context");
    let gpu = Arc::new(gpu);

    let num_envs = 64;
    let cfg = AgentConfig::default();
    let mut games: Vec<PongGame> = (0..num_envs).map(|_| PongGame::new()).collect();
    let spec = games[0].spec();
    let mut agent = Agent::new(gpu, spec.obs_dim, spec.num_actions, num_envs, cfg);

    // 10 batches of 2000 frames each.
    let mut times = Vec::new();
    for batch in 0..10 {
        let t0 = Instant::now();
        train_headless(&mut games, &mut agent, 2000, 4, 8);
        let dt = t0.elapsed().as_secs_f32();
        eprintln!(
            "batch {batch}: {dt:.2}s ({:.0} frames/s) grad={} replay={}",
            2000.0 / dt,
            agent.gradient_steps,
            agent.replay_len(),
        );
        times.push(dt);
    }

    // Check that the last batch isn't >50% slower than the first.
    let first = times[0];
    let last = times[times.len() - 1];
    eprintln!(
        "first batch: {first:.2}s, last batch: {last:.2}s, ratio: {:.2}",
        last / first
    );
    assert!(
        last < first * 1.5,
        "progressive slowdown detected: first={first:.2}s last={last:.2}s"
    );
}
