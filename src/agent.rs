//! DQN agent: MLP policy, experience replay, target network, vectorised
//! inference.
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
//! - Transitions live in a plain `VecDeque<Transition>`. No lock-free
//!   queue — the driver is single-threaded for the initial iteration.

use std::{collections::VecDeque, sync::Arc};

use meganeura::{Graph, Session, nn};
use rand::{Rng, seq::IteratorRandom};

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
    pub hidden: usize,
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub discount: f32,
    pub learning_rate: f32,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    /// Wall-clock seconds over which epsilon linearly decays.
    pub epsilon_decay_secs: f32,
    /// Gradient steps between target-network hard copies.
    pub target_sync_interval: u32,
    /// Minimum transitions in the buffer before training starts.
    pub warmup: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            hidden: 128,
            replay_capacity: 50_000,
            batch_size: 128,
            discount: 0.97,
            learning_rate: 3e-4,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_secs: 45.0,
            target_sync_interval: 1000,
            warmup: 1024,
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

/// A DQN agent backed by two meganeura sessions.
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

    pub steps_since_target_sync: u32,
    pub gradient_steps: u64,
    pub inferences: u64,
    pub last_loss: f32,
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
        let mut g_inf = Graph::new();
        let obs = g_inf.input("obs", &[num_envs, obs_dim]);
        let fc1 = nn::Linear::new(&mut g_inf, "fc1", obs_dim, cfg.hidden);
        let fc2 = nn::Linear::new(&mut g_inf, "fc2", cfg.hidden, num_actions as usize);
        let h = fc1.forward(&mut g_inf, obs);
        let h = g_inf.relu(h);
        let q = fc2.forward(&mut g_inf, h);
        g_inf.set_outputs(vec![q]);

        // Training graph: masked MSE (see loss definition in the crate README).
        let mut g_train = Graph::new();
        let batch = cfg.batch_size;
        let obs_b = g_train.input("obs", &[batch, obs_dim]);
        let act_mask = g_train.input("act_mask", &[batch, num_actions as usize]);
        let target = g_train.input("target", &[batch, num_actions as usize]);
        let t_fc1 = nn::Linear::new(&mut g_train, "fc1", obs_dim, cfg.hidden);
        let t_fc2 = nn::Linear::new(&mut g_train, "fc2", cfg.hidden, num_actions as usize);
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
            ParamShape { name: "fc1.weight".into(), shape: vec![obs_dim, cfg.hidden] },
            ParamShape { name: "fc1.bias".into(), shape: vec![cfg.hidden] },
            ParamShape { name: "fc2.weight".into(), shape: vec![cfg.hidden, num_actions as usize] },
            ParamShape { name: "fc2.bias".into(), shape: vec![num_actions as usize] },
        ];

        let mut agent = Self {
            inference,
            training,
            params,
            target_snapshot: Vec::new(),
            replay: VecDeque::with_capacity(cfg.replay_capacity),
            rng: rand::rng(),
            steps_since_target_sync: 0,
            gradient_steps: 0,
            inferences: 0,
            last_loss: 0.0,
            cfg,
            obs_dim,
            num_actions,
            num_envs,
        };

        agent.init_parameters();
        agent.target_snapshot = agent.snapshot_training();
        agent
    }

    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    pub fn obs_dim(&self) -> usize {
        self.obs_dim
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

    fn sync_inference_from_training(&mut self) {
        for p in &self.params {
            let mut buf = vec![0.0_f32; p.len()];
            self.training.read_param(&p.name, &mut buf);
            self.inference.set_parameter(&p.name, &buf);
        }
    }

    /// Epsilon-greedy actions for a batch of `num_envs` observations.
    ///
    /// `obs_batch` is the flat row-major layout: `[env0.obs, env1.obs,
    /// ..., envN-1.obs]`, length `num_envs * obs_dim`.
    pub fn select_actions(&mut self, obs_batch: &[f32], wall_secs: f32) -> Vec<Action> {
        assert_eq!(obs_batch.len(), self.num_envs * self.obs_dim);
        self.inference.set_input("obs", obs_batch);
        self.inference.step();
        // step() submits async; read_output maps CPU-side memory whose
        // contents are undefined until the GPU has finished. Skipping
        // wait() here silently returns stale data and breaks training.
        self.inference.wait();
        self.inferences += self.num_envs as u64;

        let na = self.num_actions as usize;
        let mut q = vec![0.0_f32; self.num_envs * na];
        self.inference.read_output_by_index(0, &mut q);

        let eps = self.current_epsilon(wall_secs);
        let mut out = Vec::with_capacity(self.num_envs);
        for i in 0..self.num_envs {
            let a = if self.rng.random::<f32>() < eps {
                self.rng.random_range(0..self.num_actions)
            } else {
                argmax(&q[i * na..(i + 1) * na]) as Action
            };
            out.push(a);
        }
        out
    }

    pub fn current_epsilon(&self, wall_secs: f32) -> f32 {
        // Debug knob: pin epsilon to a fixed value for sanity checks.
        if let Ok(v) = std::env::var("MEGAPLAYS_FORCE_EPSILON") {
            if let Ok(f) = v.parse::<f32>() {
                return f.clamp(0.0, 1.0);
            }
        }
        let t = (wall_secs / self.cfg.epsilon_decay_secs).clamp(0.0, 1.0);
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
    pub fn train_step(&mut self) -> Option<f32> {
        if self.replay.len() < self.cfg.warmup.max(self.cfg.batch_size) {
            return None;
        }

        let batch = self.cfg.batch_size;
        let obs_dim = self.obs_dim;
        let na = self.num_actions as usize;

        let mut obs_flat = vec![0.0_f32; batch * obs_dim];
        let mut mask = vec![0.0_f32; batch * na];
        let mut target = vec![0.0_f32; batch * na];

        let indices: Vec<usize> = (0..self.replay.len()).choose_multiple(&mut self.rng, batch);

        for (i, &ri) in indices.iter().enumerate() {
            let t = &self.replay[ri];
            obs_flat[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&t.obs);
            mask[i * na + t.action as usize] = 1.0;

            let next_q_max = if t.done {
                0.0
            } else {
                self.target_forward_max(&t.next_obs)
            };
            target[i * na + t.action as usize] = t.reward + self.cfg.discount * next_q_max;
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

        self.steps_since_target_sync += 1;
        if self.steps_since_target_sync >= self.cfg.target_sync_interval {
            self.target_snapshot = self.snapshot_training();
            self.steps_since_target_sync = 0;
        }

        self.sync_inference_from_training();

        Some(loss)
    }

    /// Forward pass through the target network on host (CPU). The MLP
    /// is tiny, so this beats a third GPU session for target-Q
    /// evaluation: no extra compile, no extra pipeline, no extra
    /// parameter-upload traffic each target-sync interval.
    fn target_forward_max(&self, obs: &[f32]) -> f32 {
        let w1 = &self.target_snapshot[0];
        let b1 = &self.target_snapshot[1];
        let w2 = &self.target_snapshot[2];
        let b2 = &self.target_snapshot[3];

        let hidden = self.cfg.hidden;
        let na = self.num_actions as usize;

        let mut h = vec![0.0_f32; hidden];
        for j in 0..hidden {
            let mut acc = b1[j];
            for i in 0..self.obs_dim {
                acc += obs[i] * w1[i * hidden + j];
            }
            h[j] = acc.max(0.0);
        }

        let mut best = f32::NEG_INFINITY;
        for a in 0..na {
            let mut q = b2[a];
            for j in 0..hidden {
                q += h[j] * w2[j * na + a];
            }
            if q > best {
                best = q;
            }
        }
        best
    }

    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }
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
