//! Learning-curve and throughput measurements. Both tests are
//! `#[ignore]`d — they are experiments, not gates:
//!
//! ```text
//! cargo test --release --test curves -- --ignored --nocapture
//! cargo test --release --test curves -- --ignored --nocapture learning_curves
//! ```
//!
//! `learning_curves` fixes a **wall-clock** budget rather than a frame
//! count, because "does it visibly learn while you watch" is a wall-clock
//! question. It mimics the windowed driver's cadence (one decision burst
//! per frame, then `train_steps_per_frame` gradient steps) minus the
//! render, so absolute numbers are an upper bound on the windowed app;
//! relative comparisons between variants are what it's for.

use std::{sync::Arc, time::Instant};

use mega_plays::{
    agent::{Agent, AgentConfig, Transition},
    catch::CatchGame,
    env_loop::run_burst,
    game::Game,
    lander::LanderGame,
    pong::{self, PongGame},
};

fn ctx() -> Arc<blade_graphics::Context> {
    Arc::new(
        unsafe {
            blade_graphics::Context::init(blade_graphics::ContextDesc {
                presentation: false,
                validation: false,
                timing: false,
                ..Default::default()
            })
        }
        .expect("init Blade context"),
    )
}

/// Rolling window over the last `cap` episode outcomes.
struct Window {
    wins: std::collections::VecDeque<f32>,
    cap: usize,
}

impl Window {
    fn new(cap: usize) -> Self {
        Self {
            wins: std::collections::VecDeque::with_capacity(cap),
            cap,
        }
    }
    fn push(&mut self, v: f32) {
        if self.wins.len() == self.cap {
            self.wins.pop_front();
        }
        self.wins.push_back(v);
    }
    fn rate(&self) -> f32 {
        if self.wins.is_empty() {
            return 0.0;
        }
        self.wins.iter().sum::<f32>() / self.wins.len() as f32
    }
}

#[derive(Default)]
struct Outcomes {
    wins: u64,
    losses: u64,
    neutral: u64,
    truncated: u64,
}

/// One measurement: a config, a wall-clock budget, and the win rate it
/// is trying to reach.
struct Run<'a> {
    label: &'a str,
    envs: usize,
    train_steps: u32,
    cfg: AgentConfig,
    seconds: f32,
    milestone: f32,
}

/// Run one variant for `run.seconds` of wall clock, printing a sample
/// line every second. Returns the seconds elapsed when the rolling win
/// rate first crossed `run.milestone` (or `None`).
fn run_curve<G: Game, F: FnMut() -> G>(
    gpu: &Arc<blade_graphics::Context>,
    mut make_game: F,
    run: Run<'_>,
) -> Option<f32> {
    let Run {
        label,
        envs: num_envs,
        train_steps: train_steps_per_frame,
        cfg,
        seconds,
        milestone,
    } = run;
    let mut games: Vec<G> = (0..num_envs).map(|_| make_game()).collect();
    let spec = games[0].spec();
    let mut agent = Agent::new(gpu.clone(), spec.obs_dim, spec.num_actions, num_envs, cfg);
    let repeat = agent.action_repeat();
    let mut obs_buf = vec![0.0_f32; num_envs * spec.obs_dim];

    // 50 outcomes ≈ 5 s of play across 9+ envs: responsive enough to
    // show a policy improving while you watch, long enough not to be
    // pure noise. A 200-deep window is indistinguishable from the
    // lifetime average over a 30 s run.
    let mut win_window = Window::new(50);
    let mut outcomes = Outcomes::default();
    let mut prev = Outcomes::default();
    let mut hit: Option<f32> = None;
    let start = Instant::now();
    let mut next_sample = 1.0_f32;

    println!("--- {label} ---");
    println!(
        "    t(s)  grad/s   eps   win%(50)  win%(interval)   won  neutral  lost  trunc   loss"
    );
    while start.elapsed().as_secs_f32() < seconds {
        run_burst(
            &mut agent,
            &mut games,
            &mut obs_buf,
            spec.obs_dim,
            repeat,
            |_, _, o| {
                if o.done {
                    if o.terminal_reward > 0.0 {
                        outcomes.wins += 1;
                        win_window.push(1.0);
                    } else if o.terminal_reward < 0.0 {
                        outcomes.losses += 1;
                        win_window.push(0.0);
                    } else {
                        outcomes.neutral += 1;
                        win_window.push(0.0);
                    }
                } else if o.truncated {
                    outcomes.truncated += 1;
                    win_window.push(0.0);
                }
            },
        );
        for _ in 0..train_steps_per_frame {
            if agent.train_step().is_none() {
                break;
            }
        }

        let t = start.elapsed().as_secs_f32();
        if hit.is_none() && win_window.wins.len() >= 50 && win_window.rate() >= milestone {
            hit = Some(t);
        }
        if t >= next_sample {
            next_sample = t.ceil() + 1.0;
            let d_win = outcomes.wins - prev.wins;
            let d_end = (outcomes.wins + outcomes.losses + outcomes.neutral + outcomes.truncated)
                - (prev.wins + prev.losses + prev.neutral + prev.truncated);
            let interval_wr = if d_end > 0 {
                100.0 * d_win as f32 / d_end as f32
            } else {
                0.0
            };
            println!(
                "  {:6.1}  {:>6.0}  {:.2}     {:5.1}          {:5.1}      {:>5}  {:>7}  {:>4}  {:>5}  {:.4}",
                t,
                agent.gradient_steps as f32 / t,
                agent.current_epsilon(),
                win_window.rate() * 100.0,
                interval_wr,
                outcomes.wins,
                outcomes.neutral,
                outcomes.losses,
                outcomes.truncated,
                agent.last_loss,
            );
            prev = Outcomes {
                wins: outcomes.wins,
                losses: outcomes.losses,
                neutral: outcomes.neutral,
                truncated: outcomes.truncated,
            };
        }
    }
    match hit {
        Some(t) => println!("  => reached {:.0}% at {t:.1}s", milestone * 100.0),
        None => println!("  => never reached {:.0}%", milestone * 100.0),
    }
    agent.destroy();
    hit
}

struct Variant {
    name: &'static str,
    envs: usize,
    train_steps: u32,
    cfg: AgentConfig,
}

/// `CURVE_GAME=pong|lander CURVE_SECS=30 CURVE_ONLY=<substring>`
#[test]
#[ignore]
fn learning_curves() {
    let gpu = ctx();
    let secs: f32 = std::env::var("CURVE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let game = std::env::var("CURVE_GAME").unwrap_or_else(|_| "pong".into());
    let only = std::env::var("CURVE_ONLY").unwrap_or_default();

    let base = if game == "lander" {
        AgentConfig {
            td_target_clamp: mega_plays::lander::TERMINAL_REWARD * 1.05,
            ..AgentConfig::default()
        }
    } else {
        AgentConfig::default()
    };
    // The pre-2026-08 defaults, for reference: slow schedules, no
    // n-step, 9 environments.
    let old = AgentConfig {
        warmup: 5_000,
        epsilon_decay_steps: 20_000,
        n_step: 1,
        ..base.clone()
    };

    let shipped = mega_plays::AppConfig::default();
    let variants = vec![
        Variant {
            // Exactly what `cargo run --bin <game>` gives you.
            name: "shipped defaults",
            envs: shipped.num_envs,
            train_steps: shipped.train_steps_per_frame,
            cfg: base.clone(),
        },
        Variant {
            name: "pre-tuning baseline",
            envs: 9,
            train_steps: 8,
            cfg: old,
        },
        Variant {
            name: "envs=9",
            envs: 9,
            train_steps: 8,
            cfg: base.clone(),
        },
        Variant {
            name: "envs=16",
            envs: 16,
            train_steps: 8,
            cfg: base.clone(),
        },
        Variant {
            name: "envs=32",
            envs: 32,
            train_steps: 8,
            cfg: base.clone(),
        },
        Variant {
            name: "n5, 16 envs",
            envs: 16,
            train_steps: 8,
            cfg: AgentConfig {
                n_step: 5,
                ..base.clone()
            },
        },
        Variant {
            name: "train4, 16 envs",
            envs: 16,
            train_steps: 4,
            cfg: base.clone(),
        },
    ];

    // Lander's milestone is the pad-rate, which starts from nothing;
    // pong and catch both have a ~15 % random baseline.
    let milestone = if game == "lander" { 0.2 } else { 0.5 };
    for v in variants {
        if !only.is_empty() && !v.name.contains(&only) {
            continue;
        }
        let label = format!(
            "{game} / {} ({} envs, {} train)",
            v.name, v.envs, v.train_steps
        );
        let run = Run {
            label: &label,
            envs: v.envs,
            train_steps: v.train_steps,
            cfg: v.cfg,
            seconds: secs,
            milestone,
        };
        match game.as_str() {
            "lander" => run_curve(&gpu, LanderGame::new, run),
            "catch" => run_curve(&gpu, CatchGame::new, run),
            _ => run_curve(&gpu, PongGame::new, run),
        };
    }
}

/// What a *random* policy reaches, per game. The first question to ask
/// of any new game or reward tweak: if uniform-random play never
/// stumbles into a win, no amount of DQN tuning will find one, and the
/// best policy the agent can learn is to stall.
///
/// The lander was exactly that case — zero soft landings in a thousand
/// random episodes against the design tolerances — which is why it
/// hovered until the time limit instead of landing.
///
/// `CURVE_GAME=pong|lander|catch`
#[test]
#[ignore]
fn random_baseline() {
    use rand::RngExt;
    let game = std::env::var("CURVE_GAME").unwrap_or_else(|_| "pong".into());
    let mut rng = rand::rng();

    fn measure<G: Game>(g: &mut G, rng: &mut impl RngExt, name: &str) {
        let spec = g.spec();
        let (mut win, mut neutral, mut loss, mut timeout) = (0, 0, 0, 0);
        let (mut action, mut held) = (0u32, 0);
        for _ in 0..400_000 {
            if held == 0 {
                action = rng.random_range(0..spec.num_actions);
                held = 4; // action_repeat
            }
            held -= 1;
            let out = g.step(action);
            if out.done || out.truncated {
                if out.truncated {
                    timeout += 1;
                } else if out.terminal_reward > 0.0 {
                    win += 1;
                } else if out.terminal_reward < 0.0 {
                    loss += 1;
                } else {
                    neutral += 1;
                }
                g.reset();
            }
        }
        let total = win + neutral + loss + timeout;
        println!(
            "{name}: {total} episodes — win={win} neutral={neutral} loss={loss} \
             timeout={timeout}  ({:.1}% win)",
            100.0 * win as f32 / total.max(1) as f32,
        );
    }

    match game.as_str() {
        "lander" => measure(&mut LanderGame::new(), &mut rng, "lander"),
        "catch" => measure(&mut CatchGame::new(), &mut rng, "catch"),
        _ => measure(&mut PongGame::new(), &mut rng, "pong"),
    }
}

fn fill(agent: &mut Agent, n: usize) {
    use rand::RngExt;
    let mut rng = rand::rng();
    for _ in 0..n {
        let obs: Vec<f32> = (0..pong::OBS_DIM)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let next_obs: Vec<f32> = (0..pong::OBS_DIM)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let done = rng.random_range(0.0..1.0) < 0.02;
        agent.record(
            0,
            Transition::step(
                obs,
                rng.random_range(0..pong::NUM_ACTIONS),
                rng.random_range(-1.0..1.0),
                next_obs,
                done,
            ),
            done,
        );
    }
}

/// Where a gradient step's wall clock goes, as a function of batch size.
#[test]
#[ignore]
fn throughput_breakdown() {
    let gpu = ctx();
    for &batch in &[128usize, 256, 512, 1024] {
        let t0 = Instant::now();
        let cfg = AgentConfig {
            batch_size: batch,
            warmup: 1024,
            ..AgentConfig::default()
        };
        let mut agent = Agent::new(gpu.clone(), pong::OBS_DIM, pong::NUM_ACTIONS, 9, cfg);
        let build_ms = t0.elapsed().as_secs_f32() * 1e3;
        fill(&mut agent, 4096);

        for _ in 0..20 {
            agent.train_step();
        }
        let n = 200;
        let t1 = Instant::now();
        for _ in 0..n {
            agent.train_step();
        }
        let step_ms = t1.elapsed().as_secs_f32() * 1e3 / n as f32;

        let obs = vec![0.0_f32; 9 * pong::OBS_DIM];
        for _ in 0..20 {
            agent.select_actions(&obs);
        }
        let t2 = Instant::now();
        for _ in 0..n {
            agent.select_actions(&obs);
        }
        let inf_ms = t2.elapsed().as_secs_f32() * 1e3 / n as f32;

        println!(
            "batch {batch:>5}: build {build_ms:>6.0} ms | train_step {step_ms:>6.3} ms \
             ({:>7.0} samples/s) | select_actions {inf_ms:.3} ms",
            batch as f32 * 1e3 / step_ms,
        );
        agent.destroy();
    }
}
