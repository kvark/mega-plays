//! Double-DQN agent: MLP policy, experience replay, target network,
//! vectorised inference.
//!
//! Design notes:
//!
//! - The policy is a small MLP built at runtime via meganeura. Two
//!   sessions share one Blade context: an *inference* session with a
//!   fixed batch of `num_envs` (one forward pass per simulation
//!   substep for every parallel environment), and a *training* session
//!   that owns autodiff + Adam state. After each gradient step the
//!   updated parameters are read back to host memory and re-uploaded
//!   into the inference session. That round trip is negligible for
//!   the ~few-thousand-parameter MLPs used here; when networks grow,
//!   a GPU-side weight copy becomes worth the effort.
//!
//! - The target network is a Polyak-averaged CPU-side snapshot of the
//!   training weights. *Double*-DQN: the online network picks the
//!   action at the next state, the target network evaluates its
//!   Q-value. The decoupling kills the positive bias of vanilla DQN —
//!   `max_a Q'(s', a)` over a noisy Q' systematically overestimates
//!   the true max — and is the difference between a run that holds
//!   its gains and one that regresses late in training.
//!
//! - Epsilon decays by gradient steps, not wall clock, so training
//!   progress is deterministic regardless of frame rate.
//!
//! - Transitions live in a plain `VecDeque<Transition>`. No lock-free
//!   queue — the driver is single-threaded for the initial iteration.

use std::{collections::VecDeque, sync::Arc};

use meganeura::{Graph, Session, nn};
use rand::{RngExt, seq::IteratorRandom};

/// Magic bytes at the head of a `.weights` file — `"MEGA"` in ASCII,
/// little-endian. Lets `load_weights` distinguish a real file from
/// garbage before guessing at sizes.
const MEGA_WEIGHTS_MAGIC: u32 = u32::from_le_bytes(*b"MEGA");
/// Bump when the on-disk weights layout changes.
const MEGA_WEIGHTS_VERSION: u32 = 2;

/// Observation vector. Flat f32s, caller-defined layout, normalised to
/// roughly `[-1, 1]`.
pub type Observation = Vec<f32>;

/// Discrete action index. Valid range is `0..num_actions` from the
/// game's [`GameSpec`](crate::game::GameSpec).
pub type Action = u32;

/// One step from the environment's perspective.
#[derive(Clone, Debug)]
pub struct Transition {
    pub obs: Observation,
    pub action: Action,
    pub reward: f32,
    pub next_obs: Observation,
    pub done: bool,
}

/// DQN hyperparameters tuned for small, fast-converging tasks like
/// Pong — not a general-purpose RL configuration.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// Width of each hidden layer: `obs → hidden → hidden → actions`.
    pub hidden: usize,
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub discount: f32,
    pub learning_rate: f32,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    /// Gradient steps over which epsilon linearly decays.
    pub epsilon_decay_steps: u64,
    /// Polyak averaging factor for soft target-network updates.
    /// `target ← (1-τ) * target + τ * online` after each gradient step.
    pub target_tau: f32,
    /// Minimum transitions in the buffer before training starts.
    pub warmup: usize,
    /// Hold each selected action for this many physics substeps before
    /// the next decision. Cuts inference round trips proportionally and
    /// turns ε-greedy into a real exploration signal (sub-perception
    /// dithering at 120 Hz averages to zero net torque/velocity). The
    /// recorded transition's reward is the sum across the burst and
    /// `next_obs` is the post-burst observation.
    pub action_repeat: u32,
    /// Symmetric clamp on the Bellman target `r + γ·max_a' Q'(s', a')`.
    /// Stops Q-value runaway when bootstrap mistakes feed back into the
    /// next step's targets. Must be ≥ the largest expected |terminal
    /// reward| or the sparse signal gets compressed and learning stalls
    /// (lander's ±10 ate a hard ±5 clamp before this knob existed).
    pub td_target_clamp: f32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            hidden: 128,
            replay_capacity: 50_000,
            batch_size: 256,
            discount: 0.99,
            learning_rate: 1e-3,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay_steps: 20_000,
            target_tau: 0.005,
            warmup: 5_000,
            action_repeat: 4,
            td_target_clamp: 5.0,
        }
    }
}

struct ParamShape {
    name: String,
    shape: Vec<usize>,
}

impl ParamShape {
    fn len(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A Double-DQN agent backed by two meganeura sessions.
pub struct Agent {
    cfg: AgentConfig,
    obs_dim: usize,
    num_actions: u32,
    num_envs: usize,

    inference: Session,
    training: Session,

    params: Vec<ParamShape>,
    /// Frozen copy of the online network that produces bootstrap Q targets.
    target_snapshot: Vec<Vec<f32>>,

    replay: VecDeque<Transition>,
    rng: rand::rngs::ThreadRng,

    // (soft update replaces the hard-copy counter)
    pub gradient_steps: u64,
    pub inferences: u64,
    pub last_loss: f32,
    /// Latest Q-values from `select_actions`, row-major
    /// `[num_envs, num_actions]`. Zeros during ε≥1 warmup.
    last_q: Vec<f32>,
}

impl Agent {
    pub fn new(
        gpu: Arc<blade_graphics::Context>,
        obs_dim: usize,
        num_actions: u32,
        num_envs: usize,
        cfg: AgentConfig,
    ) -> Self {
        assert!(num_envs >= 1, "num_envs must be ≥ 1");

        // Inference graph: obs[num_envs, obs_dim] -> q[num_envs, num_actions]
        let na = num_actions as usize;
        let mut g_inf = Graph::new();
        let obs = g_inf.input("obs", &[num_envs, obs_dim]);
        let fc1 = nn::Linear::new(&mut g_inf, "fc1", obs_dim, cfg.hidden);
        let fc2 = nn::Linear::new(&mut g_inf, "fc2", cfg.hidden, na);
        let h = fc1.forward(&mut g_inf, obs);
        let h = g_inf.relu(h);
        let q = fc2.forward(&mut g_inf, h);
        g_inf.set_outputs(vec![q]);

        // Training graph: masked MSE (see loss definition in the crate README).
        let mut g_train = Graph::new();
        let batch = cfg.batch_size;
        let obs_b = g_train.input("obs", &[batch, obs_dim]);
        let act_mask = g_train.input("act_mask", &[batch, na]);
        let target = g_train.input("target", &[batch, na]);
        let t_fc1 = nn::Linear::new(&mut g_train, "fc1", obs_dim, cfg.hidden);
        let t_fc2 = nn::Linear::new(&mut g_train, "fc2", cfg.hidden, na);
        let th = t_fc1.forward(&mut g_train, obs_b);
        let th = g_train.relu(th);
        let q_all = t_fc2.forward(&mut g_train, th);
        let masked_q = g_train.mul(q_all, act_mask);
        let masked_t = g_train.mul(target, act_mask);
        let loss = g_train.mse_loss(masked_q, masked_t);
        g_train.set_outputs(vec![loss]);

        let inference = meganeura::build(
            &g_inf,
            meganeura::SessionConfig {
                mode: meganeura::Mode::Inference,
                gpu: Some(gpu.clone()),
                ..meganeura::SessionConfig::default()
            },
        )
        .0;
        let training = meganeura::build(
            &g_train,
            meganeura::SessionConfig {
                gpu: Some(gpu),
                ..meganeura::SessionConfig::default()
            },
        )
        .0;

        let params = vec![
            ParamShape {
                name: "fc1.weight".into(),
                shape: vec![obs_dim, cfg.hidden],
            },
            ParamShape {
                name: "fc1.bias".into(),
                shape: vec![cfg.hidden],
            },
            ParamShape {
                name: "fc2.weight".into(),
                shape: vec![cfg.hidden, na],
            },
            ParamShape {
                name: "fc2.bias".into(),
                shape: vec![na],
            },
        ];

        let mut agent = Self {
            inference,
            training,
            params,
            target_snapshot: Vec::new(),
            replay: VecDeque::with_capacity(cfg.replay_capacity),
            rng: rand::rng(),
            gradient_steps: 0,
            inferences: 0,
            last_loss: 0.0,
            last_q: vec![0.0; num_envs * num_actions as usize],
            cfg,
            obs_dim,
            num_actions,
            num_envs,
        };

        agent.init_parameters();
        agent.target_snapshot = agent.snapshot_training();
        agent
    }

    /// Wait for all in-flight GPU work on every session. Called
    /// before the render resources are torn down so that the later
    /// `Session::drop` pipeline cleanup doesn't race with the GPU.
    pub fn destroy(&mut self) {
        self.inference.wait();
        self.training.wait();
    }

    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    pub fn action_repeat(&self) -> u32 {
        self.cfg.action_repeat.max(1)
    }

    fn init_parameters(&mut self) {
        for p in &self.params {
            let data = if p.name.ends_with(".weight") {
                let fan_in = p.shape[0] as f32;
                let bound = (6.0 / fan_in).sqrt();
                (0..p.len())
                    .map(|_| self.rng.random_range(-bound..bound))
                    .collect::<Vec<_>>()
            } else {
                vec![0.0; p.len()]
            };
            self.training.set_parameter(&p.name, &data);
            self.inference.set_parameter(&p.name, &data);
        }
    }

    fn snapshot_training(&self) -> Vec<Vec<f32>> {
        self.params
            .iter()
            .map(|p| {
                let mut buf = vec![0.0_f32; p.len()];
                self.training.read_param(&p.name, &mut buf);
                buf
            })
            .collect()
    }

    /// Epsilon-greedy actions for a batch of `num_envs` observations.
    ///
    /// `obs_batch` is the flat row-major layout: `[env0.obs, env1.obs,
    /// ..., envN-1.obs]`, length `num_envs * obs_dim`.
    pub fn select_actions(&mut self, obs_batch: &[f32]) -> Vec<Action> {
        assert_eq!(obs_batch.len(), self.num_envs * self.obs_dim);
        let eps = self.current_epsilon();

        // During warmup epsilon is pinned at 1.0 — every action is
        // uniform random, no Q-value is consulted, and the GPU inference
        // round trip would be discarded. Skip it entirely.
        if eps >= 1.0 {
            self.inferences += self.num_envs as u64;
            // Keep last_q at zeros so the Q-bar overlay reads "no
            // signal yet" rather than stale post-bootstrap noise.
            for v in self.last_q.iter_mut() {
                *v = 0.0;
            }
            return (0..self.num_envs)
                .map(|_| self.rng.random_range(0..self.num_actions))
                .collect();
        }

        self.inference.set_input("obs", obs_batch);
        self.inference.step();
        self.inference.wait();
        self.inferences += self.num_envs as u64;

        let na = self.num_actions as usize;
        self.inference.read_output_by_index(0, &mut self.last_q);

        let mut out = Vec::with_capacity(self.num_envs);
        for i in 0..self.num_envs {
            let a = if self.rng.random::<f32>() < eps {
                self.rng.random_range(0..self.num_actions)
            } else {
                argmax(&self.last_q[i * na..(i + 1) * na]) as Action
            };
            out.push(a);
        }
        out
    }

    pub fn current_epsilon(&self) -> f32 {
        if let Ok(v) = std::env::var("MEGAPLAYS_FORCE_EPSILON") {
            if let Ok(f) = v.parse::<f32>() {
                return f.clamp(0.0, 1.0);
            }
        }
        let t = (self.gradient_steps as f32 / self.cfg.epsilon_decay_steps as f32).clamp(0.0, 1.0);
        self.cfg.epsilon_start + (self.cfg.epsilon_end - self.cfg.epsilon_start) * t
    }

    pub fn record(&mut self, t: Transition) {
        if self.replay.len() >= self.cfg.replay_capacity {
            self.replay.pop_front();
        }
        self.replay.push_back(t);
    }

    /// Run one minibatch gradient step if enough transitions have been
    /// collected. Returns the loss, or `None` when skipped.
    ///
    /// Implements *Double-DQN* (van Hasselt et al., 2015): the online
    /// network picks the action at the next state, the target network
    /// evaluates its Q-value. This decouples action selection from
    /// value estimation, killing the positive bias of vanilla DQN
    /// (`max_a Q'(s', a)` over a noisy Q' systematically overestimates
    /// the true max), which in practice is the difference between a
    /// run that holds its gains and one that regresses late in training.
    pub fn train_step(&mut self) -> Option<f32> {
        if self.replay.len() < self.cfg.warmup.max(self.cfg.batch_size) {
            return None;
        }

        // Pre-step online snapshot — used for the Double-DQN argmax
        // *and* as the source for the post-step Polyak target update.
        // The Polyak τ is so small (default 0.005) that a one-step lag
        // between pre and post in the target update is negligible.
        let online = self.snapshot_training();

        let batch = self.cfg.batch_size;
        let obs_dim = self.obs_dim;
        let na = self.num_actions as usize;

        let mut obs_flat = vec![0.0_f32; batch * obs_dim];
        let mut mask = vec![0.0_f32; batch * na];
        let mut target = vec![0.0_f32; batch * na];

        let indices: Vec<usize> = (0..self.replay.len()).sample(&mut self.rng, batch);

        for (i, &ri) in indices.iter().enumerate() {
            let t = &self.replay[ri];
            obs_flat[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&t.obs);
            mask[i * na + t.action as usize] = 1.0;

            let next_q_max = if t.done {
                0.0
            } else {
                let online_q = cpu_forward(&online, &t.next_obs, self.obs_dim, self.cfg.hidden, na);
                let best_a = argmax(&online_q);
                let target_q = cpu_forward(
                    &self.target_snapshot,
                    &t.next_obs,
                    self.obs_dim,
                    self.cfg.hidden,
                    na,
                );
                target_q[best_a]
            };
            // Clamp the Bellman target to prevent Q-value divergence.
            // The clamp must be ≥ the largest |terminal reward| or the
            // sparse signal gets compressed (lander's ±10 ate the old
            // hard ±5 before this knob existed).
            let clip = self.cfg.td_target_clamp;
            let td = (t.reward + self.cfg.discount * next_q_max).clamp(-clip, clip);
            target[i * na + t.action as usize] = td;
        }

        self.training.set_input("obs", &obs_flat);
        self.training.set_input("act_mask", &mask);
        self.training.set_input("target", &target);

        self.training
            .set_adam(self.cfg.learning_rate, 0.9, 0.999, 1e-8);
        self.training.step();
        self.training.wait();
        self.gradient_steps += 1;

        let loss = self.training.read_output(1).first().copied().unwrap_or(0.0);
        self.last_loss = loss;

        // Soft (Polyak) target update: target ← (1-τ)*target + τ*online.
        let tau = self.cfg.target_tau;
        for (tgt, src) in self.target_snapshot.iter_mut().zip(online.iter()) {
            for (t, &s) in tgt.iter_mut().zip(src.iter()) {
                *t = *t * (1.0 - tau) + s * tau;
            }
        }

        // Sync inference from the post-step weights — re-snapshot so
        // inference sees the most recent gradient update (online[] is
        // pre-step). One extra read_param + wait per train_step; for a
        // ~4k-param MLP this is microseconds.
        let online_post = self.snapshot_training();
        for (p, buf) in self.params.iter().zip(online_post.iter()) {
            self.inference.set_parameter(&p.name, buf);
        }
        Some(loss)
    }

    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }

    /// Total trainable f32 element count across all parameters
    /// (weights and biases of fc1, fc2). Reported in the overlay so
    /// the user can see this is a *small* live-trained net, not the
    /// gigantic checkpoint they might assume.
    pub fn param_count(&self) -> usize {
        self.params.iter().map(|p| p.len()).sum()
    }

    /// Most-recent batch of Q-values from the inference session, in
    /// row-major `[num_envs, num_actions]` layout. Refreshed every
    /// `select_actions` call (set to zero during ε ≥ 1 warmup, since
    /// the network isn't consulted). Used to draw Q-bars over the
    /// hero env in the overlay.
    pub fn last_q(&self) -> &[f32] {
        &self.last_q
    }

    /// Save online weights and training state (Adam moments + step
    /// count) to a binary file.
    ///
    /// Resuming with only the weights resets Adam's first / second
    /// moment estimates to zero, which causes a transient loss spike
    /// and one or two thousand grad steps of recovery noise. Saving
    /// the moments alongside means a loaded checkpoint continues
    /// where the run left off.
    ///
    /// Format (`v2`, little-endian):
    ///
    /// ```text
    /// u32 magic = 0x4147454D ("MEGA")
    /// u32 version = 2
    /// u64 gradient_steps
    /// u32 adam_step_count
    /// for each param:
    ///     u32 len
    ///     f32 * len   weights
    ///     f32 * len   adam m (first moment estimate)
    ///     f32 * len   adam v (second moment estimate)
    /// ```
    pub fn save_weights(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let weights = self.snapshot_training();
        let mut f = std::fs::File::create(path)?;
        f.write_all(&MEGA_WEIGHTS_MAGIC.to_le_bytes())?;
        f.write_all(&MEGA_WEIGHTS_VERSION.to_le_bytes())?;
        f.write_all(&self.gradient_steps.to_le_bytes())?;
        f.write_all(&self.training.adam_step_count().to_le_bytes())?;
        for (p, w) in self.params.iter().zip(weights.iter()) {
            let len = w.len() as u32;
            f.write_all(&len.to_le_bytes())?;
            f.write_all(bytemuck::cast_slice(&w[..]))?;
            let mut m = vec![0.0_f32; w.len()];
            let mut v = vec![0.0_f32; w.len()];
            self.training.read_adam_m(&p.name, &mut m);
            self.training.read_adam_v(&p.name, &mut v);
            f.write_all(bytemuck::cast_slice(&m[..]))?;
            f.write_all(bytemuck::cast_slice(&v[..]))?;
        }
        Ok(())
    }

    /// Load weights and Adam state from a binary file produced by
    /// [`save_weights`]. Overwrites training, inference, target-network
    /// weights and Adam moment estimates, and restores gradient_steps
    /// (so epsilon resumes correctly) and the Adam bias-correction
    /// step count.
    pub fn load_weights(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;

        let mut hdr32 = [0u8; 4];
        f.read_exact(&mut hdr32)?;
        let magic = u32::from_le_bytes(hdr32);
        if magic != MEGA_WEIGHTS_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("not a mega-plays weights file (magic 0x{magic:08x})"),
            ));
        }
        f.read_exact(&mut hdr32)?;
        let version = u32::from_le_bytes(hdr32);
        if version != MEGA_WEIGHTS_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "weights file version {version}, expected {MEGA_WEIGHTS_VERSION}; \
                     re-train"
                ),
            ));
        }

        let mut u64_buf = [0u8; 8];
        f.read_exact(&mut u64_buf)?;
        self.gradient_steps = u64::from_le_bytes(u64_buf);
        f.read_exact(&mut hdr32)?;
        let adam_step = u32::from_le_bytes(hdr32);

        let mut len_buf = [0u8; 4];
        for p in &self.params {
            f.read_exact(&mut len_buf)?;
            let len = u32::from_le_bytes(len_buf) as usize;
            if len != p.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "param {}: expected {} elements, file has {}",
                        p.name,
                        p.len(),
                        len,
                    ),
                ));
            }
            let mut w = vec![0.0_f32; len];
            let mut m = vec![0.0_f32; len];
            let mut v = vec![0.0_f32; len];
            f.read_exact(bytemuck::cast_slice_mut(&mut w[..]))?;
            f.read_exact(bytemuck::cast_slice_mut(&mut m[..]))?;
            f.read_exact(bytemuck::cast_slice_mut(&mut v[..]))?;
            self.training.set_parameter(&p.name, &w);
            self.inference.set_parameter(&p.name, &w);
            self.training.write_adam_m(&p.name, &m);
            self.training.write_adam_v(&p.name, &v);
        }
        self.training.set_adam_step_count(adam_step);
        self.target_snapshot = self.snapshot_training();
        log::info!(
            "loaded weights from {} (grad_steps={}, adam_step={}, eps={:.3})",
            path.display(),
            self.gradient_steps,
            adam_step,
            self.current_epsilon(),
        );
        Ok(())
    }
}

/// CPU forward pass through a 2-layer MLP snapshot. Returns all
/// Q-values for the given observation.
fn cpu_forward(
    weights: &[Vec<f32>],
    obs: &[f32],
    obs_dim: usize,
    hidden: usize,
    na: usize,
) -> Vec<f32> {
    let w1 = &weights[0];
    let b1 = &weights[1];
    let w2 = &weights[2];
    let b2 = &weights[3];

    let mut h = vec![0.0_f32; hidden];
    for j in 0..hidden {
        let mut acc = b1[j];
        for i in 0..obs_dim {
            acc += obs[i] * w1[i * hidden + j];
        }
        h[j] = acc.max(0.0); // ReLU
    }

    let mut q = vec![0.0_f32; na];
    for a in 0..na {
        let mut val = b2[a];
        for j in 0..hidden {
            val += h[j] * w2[j * na + a];
        }
        q[a] = val;
    }
    q
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}
