//! Winit driver: window, Blade context, egui overlay, training loop.
//!
//! The driver is generic over any [`Game`] and runs `num_envs` of them
//! in parallel. All rendering flows through egui — no custom Blade
//! pipeline — which means no WGSL shaders to ship, no MSAA resolve
//! pass, and no text-shaping dependency.
//!
//! Each physics substep gathers observations from every game, runs one
//! batched inference pass on the shared policy to produce N actions,
//! then advances every game by one step and records their transitions
//! into the shared replay buffer. The games are independent — no
//! communication between them — so vectorisation is pure throughput.

use std::{sync::Arc, time::Instant};

use blade_graphics as gpu;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};

use crate::{
    agent::{Agent, AgentConfig},
    env_loop::run_burst,
    game::{Game, GameSpec, HumanInput},
    stats::{RollingStats, SparkLine},
};

/// Top-level knobs for [`run`]. The [`AgentConfig`] comes from the game
/// wiring; everything else governs the host harness.
pub struct AppConfig {
    pub agent: AgentConfig,
    /// Number of parallel environments. Batched inference does one
    /// forward pass over all of them per substep.
    pub num_envs: usize,
    /// Physics substeps per rendered frame at speed = 1. The in-UI
    /// speed slider multiplies this at runtime.
    pub base_substeps_per_frame: u32,
    /// Minibatch gradient steps per rendered frame.
    pub train_steps_per_frame: u32,
    /// Per-frame time budget for the physics + training loop (ms). The
    /// slider can ask for many substeps; we stop early if we blow this
    /// budget so the UI stays interactive. The displayed sim-rate
    /// reflects what we actually achieved.
    pub frame_budget_ms: u32,
    /// Clear colour behind the game grid.
    pub clear_color: Color32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            agent: AgentConfig::default(),
            num_envs: 9,
            base_substeps_per_frame: 4,
            train_steps_per_frame: 8,
            frame_budget_ms: 12,
            clear_color: Color32::from_rgb(8, 10, 14),
        }
    }
}

/// Run the driver until the window is closed.
///
/// `build_game` is called `num_envs` times, each with a clone of the
/// shared `Arc<Context>`. Games are independent; each gets its own
/// [`Game`] instance.
pub fn run<G, F>(config: AppConfig, build_game: F)
where
    G: Game + 'static,
    F: FnMut(Arc<gpu::Context>) -> G + 'static,
{
    env_logger::init();
    // Drop-guard that saves a Perfetto-compatible `.pftrace` on exit
    // (path controlled by MEGA_TRACE, default `./mega-plays.pftrace`).
    // Kept alive for the whole event loop — dropped when `run` returns.
    let _profile_guard = crate::profiling::init_or_default();

    // Headless mode — MEGA_HEADLESS=1 or MEGA_HEADLESS=<frames>. Skips
    // window/surface/render entirely (no presentation, no DRI3) and
    // runs the tick loop for `frames` iterations. Intended for trace
    // generation on machines where blade can't acquire a display
    // surface (e.g. xvfb without DRI3, CI runners). The compute path
    // — context init, games, agent training, meganeura dispatches —
    // is exercised fully; only the egui render pass is omitted.
    if let Ok(v) = std::env::var("MEGA_HEADLESS") {
        let frames: u32 = match v.as_str() {
            "1" | "" => 2000,
            s => s.parse().unwrap_or(2000),
        };
        run_headless(config, build_game, frames);
        return;
    }

    let event_loop = winit::event_loop::EventLoop::new().expect("event loop");
    let mut app = App::<G> {
        state: AppState::Uninit {
            build_game: Some(Box::new(build_game)),
            config,
        },
    };
    event_loop.run_app(&mut app).expect("run app");
    if let AppState::Running(mut running) = app.state {
        running.destroy();
    }
}

type Builder<G> = Box<dyn FnMut(Arc<gpu::Context>) -> G>;

/// MEGA_HEADLESS code path: builds the compute context + games + agent
/// and drives `frames` physics substeps (one per iteration), without
/// any window, surface, or render calls. Useful for trace generation
/// on hosts that can't acquire a display (xvfb without DRI3, CI
/// runners, headless Linux servers) — everything except rendering
/// still happens, so the resulting `.pftrace` has CPU spans and
/// meganeura GPU dispatches on the same tracks as a windowed run.
fn run_headless<G, F>(mut config: AppConfig, mut build_game: F, frames: u32)
where
    G: Game + 'static,
    F: FnMut(Arc<gpu::Context>) -> G + 'static,
{
    // Compute-only context — no presentation, no surface extensions.
    let gpu = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            presentation: false,
            validation: cfg!(debug_assertions),
            timing: true,
            ..Default::default()
        })
    }
    .expect("init Blade context (headless)");
    let gpu = Arc::new(gpu);

    if let Some(n) = std::env::var("MEGAPLAYS_NUM_ENVS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        config.num_envs = n.max(1);
    }

    let mut games: Vec<G> = (0..config.num_envs)
        .map(|_| build_game(gpu.clone()))
        .collect();
    let spec = games[0].spec();

    let mut agent = Agent::new(
        gpu.clone(),
        spec.obs_dim,
        spec.num_actions,
        config.num_envs,
        config.agent.clone(),
    );

    let num_envs = games.len();
    let obs_dim = spec.obs_dim;
    let mut obs_buf = vec![0.0_f32; num_envs * obs_dim];

    let start = Instant::now();
    log::info!(
        "mega-plays headless: {} frames × {} substeps × {} envs",
        frames,
        config.base_substeps_per_frame,
        num_envs
    );

    let action_repeat = agent.action_repeat();
    let bursts_per_frame = (config.base_substeps_per_frame / action_repeat).max(1);
    // Four-way outcome counters: a 0% win-rate by itself can't tell
    // "crashes constantly" from "hovers until truncation". The
    // breakdown distinguishes them.
    let mut wins = 0u64; // terminal_reward > 0
    let mut losses = 0u64; // terminal_reward < 0
    let mut partials = 0u64; // done && terminal_reward == 0 (soft off-pad etc)
    let mut timeouts = 0u64; // truncated (time-limit / OOB-without-terminal)
    let mut last = [0u64; 4];
    let mut loss_ema = 0.0f32;
    for frame in 0..frames {
        let _tick = tracing::info_span!("tick").entered();
        for _ in 0..bursts_per_frame {
            let _sub = tracing::info_span!("substep").entered();
            run_burst(
                &mut agent,
                &mut games,
                &mut obs_buf,
                obs_dim,
                action_repeat,
                |_, _, outcome| {
                    if outcome.done {
                        if outcome.terminal_reward > 0.0 {
                            wins += 1;
                        } else if outcome.terminal_reward < 0.0 {
                            losses += 1;
                        } else {
                            partials += 1;
                        }
                    } else if outcome.truncated {
                        timeouts += 1;
                    }
                },
            );
        }
        let _train = tracing::info_span!("train").entered();
        for _ in 0..config.train_steps_per_frame {
            let _s = tracing::info_span!("train_step").entered();
            if let Some(loss) = agent.train_step() {
                loss_ema = if loss_ema == 0.0 {
                    loss
                } else {
                    loss_ema * 0.99 + loss * 0.01
                };
            } else {
                break;
            }
        }
        if frame.is_multiple_of(500) {
            let dwin = wins - last[0];
            let dloss = losses - last[1];
            let dpart = partials - last[2];
            let dtime = timeouts - last[3];
            let dtotal = dwin + dloss + dpart + dtime;
            let interval_wr = if dtotal > 0 {
                100.0 * dwin as f32 / dtotal as f32
            } else {
                0.0
            };
            log::info!(
                "headless frame {} / {}  {:.1}s  replay={} grad={} eps={:.3} \
                 wr_last_window={:.1}%  win/loss/partial/timeout={}/{}/{}/{}  \
                 loss_ema={:.4}",
                frame,
                frames,
                start.elapsed().as_secs_f32(),
                agent.replay_len(),
                agent.gradient_steps,
                agent.current_epsilon(),
                interval_wr,
                dwin,
                dloss,
                dpart,
                dtime,
                loss_ema,
            );
            last = [wins, losses, partials, timeouts];
        }
    }
    log::info!(
        "headless done: {} frames in {:.2}s",
        frames,
        start.elapsed().as_secs_f32()
    );
}

enum AppState<G: Game> {
    Uninit {
        build_game: Option<Builder<G>>,
        config: AppConfig,
    },
    Running(Box<Running<G>>),
    Dead,
}

struct App<G: Game> {
    state: AppState<G>,
}

impl<G: Game + 'static> winit::application::ApplicationHandler for App<G> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let (mut build_game, mut config) = match std::mem::replace(&mut self.state, AppState::Dead)
        {
            AppState::Uninit { build_game, config } => (
                build_game.expect("resumed called twice without a game builder"),
                config,
            ),
            other => {
                self.state = other;
                return;
            }
        };

        let gpu = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                timing: true,
                ..Default::default()
            })
        }
        .expect("init Blade context");
        let gpu = Arc::new(gpu);

        // Allow overriding the number of parallel environments at
        // launch via MEGAPLAYS_NUM_ENVS=<n>. Useful for quick
        // experiments without recompiling.
        if let Some(n) = std::env::var("MEGAPLAYS_NUM_ENVS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            config.num_envs = n.max(1);
        }

        let games: Vec<G> = (0..config.num_envs)
            .map(|_| build_game(gpu.clone()))
            .collect();
        let spec = games[0].spec();

        let window_attributes = winit::window::Window::default_attributes()
            .with_title(spec.title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
        let window = event_loop
            .create_window(window_attributes)
            .expect("create window");

        let window_size = window.inner_size();
        let surface = gpu
            .create_surface_configured(&window, make_surface_config(window_size))
            .expect("create surface");
        let surface_info = surface.info();
        let gui_painter = blade_egui::GuiPainter::new(surface_info, &gpu);

        let command_encoder = gpu.create_command_encoder(gpu::CommandEncoderDesc {
            name: "mega-plays",
            buffer_count: 2,
            manual_barriers: false,
        });

        let egui_ctx = egui::Context::default();
        let viewport_id = egui_ctx.viewport_id();
        let egui_winit = egui_winit::State::new(egui_ctx, viewport_id, &window, None, None, None);

        let agent = Agent::new(
            gpu.clone(),
            spec.obs_dim,
            spec.num_actions,
            config.num_envs,
            config.agent.clone(),
        );

        let exit_deadline = std::env::var("MEGAPLAYS_EXIT_AFTER_SECS")
            .ok()
            .and_then(|s| s.parse::<f32>().ok());
        let view_mode = match std::env::var("MEGAPLAYS_VIEW").ok().as_deref() {
            Some("overlay") => ViewMode::Overlay,
            _ => ViewMode::Grid,
        };

        let now = Instant::now();
        let initial_num_envs = config.num_envs as u32;
        let device_label = device_label_for(&gpu);
        let param_count = agent.param_count();
        self.state = AppState::Running(Box::new(Running {
            gpu,
            surface,
            command_encoder,
            prev_sync_point: None,
            prev_submit_offset_ns: None,
            gui_painter,
            window,
            egui_winit,
            games,
            spec,
            agent,
            build_game: Some(build_game),
            config,
            loss_hist: RollingStats::new(400),
            reward_hist: RollingStats::new(400),
            episode_return: Vec::new(),
            episode_return_hist: RollingStats::new(200),
            episode_len: Vec::new(),
            episode_len_hist: RollingStats::new(200),
            win_rate_hist: RollingStats::new(400),
            difficulty_hist: RollingStats::new(400),
            last_sample_time: now,
            scores_agent: 0,
            scores_opp: 0,
            last_return: Vec::new(),
            action_hist: [0; 8],
            loss_ema: 0.0,
            return_ema: 0.0,
            pending_num_envs: initial_num_envs,
            auto_difficulty: true,
            target_win_ratio: 1.5,
            win_rate_ema: 0.6, // = 1.5 / 2.5, matches the default target
            start_time: now,
            last_frame: now,
            physics_accum: 0.0,
            paused: false,
            show_overlay: true,
            view_mode,
            human_play: false,
            human_up: false,
            human_down: false,
            speed_mul: 1,
            achieved_subs: 0,
            requested_subs: 0,
            last_dt: 1.0 / 60.0,
            tick_ms: 0.0,
            exit_deadline,
            frame_counter: 0,
            last_heartbeat: 0.0,
            last_heartbeat_frame: 0,
            epoch_remaining: 0,
            epoch_total: 0,
            render_prims: 0,
            render_verts: 0,
            device_label,
            param_count,
            grad_rate: 0.0,
            prev_grad_steps: 0,
            prev_grad_time: now,
            status_msg: String::new(),
            status_time: now,
        }));
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let AppState::Running(r) = &self.state {
            r.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        use winit::event::{ElementState, KeyEvent, WindowEvent};
        use winit::keyboard::{KeyCode, PhysicalKey};

        let AppState::Running(r) = &mut self.state else {
            return;
        };

        let response = r.egui_winit.on_window_event(&r.window, &event);
        if response.repaint {
            r.window.request_redraw();
        }
        if response.consumed {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => r.resize(size),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = matches!(state, ElementState::Pressed);
                match key {
                    KeyCode::ArrowUp => r.human_up = pressed,
                    KeyCode::ArrowDown => r.human_down = pressed,
                    _ if pressed => match key {
                        KeyCode::Escape => event_loop.exit(),
                        KeyCode::Space => r.paused = !r.paused,
                        KeyCode::KeyG => r.show_overlay = !r.show_overlay,
                        KeyCode::KeyV => r.view_mode = r.view_mode.toggled(),
                        KeyCode::KeyR => r.reset_learning(),
                        KeyCode::KeyT => r.train_epoch(),
                        KeyCode::KeyS => r.save_weights(),
                        KeyCode::KeyL => r.load_weights(),
                        KeyCode::KeyP => {
                            r.human_play = !r.human_play;
                            r.set_status(if r.human_play {
                                "human play: ON (arrow keys control env 0's opponent)".into()
                            } else {
                                "human play: OFF".into()
                            });
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => r.redraw(event_loop),
            _ => {}
        }
    }
}

struct Running<G: Game> {
    gpu: Arc<gpu::Context>,
    surface: gpu::Surface,
    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,
    /// Nanosecond profile-epoch offset captured just before the most
    /// recent submit. The timestamp-query results from that submit are
    /// readable once the fence signals, which happens before the next
    /// frame's `render()` waits; we forward them to the GPU track
    /// tagged with this offset so they align with the submit point.
    prev_submit_offset_ns: Option<u64>,
    gui_painter: blade_egui::GuiPainter,

    window: winit::window::Window,
    egui_winit: egui_winit::State,

    games: Vec<G>,
    spec: GameSpec,
    agent: Agent,
    build_game: Option<Builder<G>>,
    config: AppConfig,

    loss_hist: RollingStats,
    reward_hist: RollingStats,
    episode_return: Vec<f32>,
    episode_return_hist: RollingStats,
    episode_len: Vec<u32>,
    episode_len_hist: RollingStats,
    /// Sampled `win_rate_ema` and `difficulty` over wall time. Each
    /// sample is one tick of `sample_interval` (default 0.5 s), so the
    /// plot's x-axis is naturally time-scoped. The win-rate trace is
    /// what the auto-curriculum reads — when the curve hugs the dashed
    /// `target_win_ratio` line the controller is doing its job and
    /// `win_rate` will look stationary by design (this is the trace to
    /// watch instead).
    win_rate_hist: RollingStats,
    difficulty_hist: RollingStats,
    /// Wall-clock time of the last (win_rate, difficulty) sample push.
    last_sample_time: Instant,
    scores_agent: u64,
    scores_opp: u64,
    /// Per-env total reward from the most recently completed episode.
    /// Drives the overlay's hero pick — "best recent run" instead of
    /// "longest live episode", so the camera highlights *good* play
    /// rather than a policy that's just stalling for time.
    last_return: Vec<f32>,
    /// Rolling count of each action taken since last heartbeat. Reset
    /// at every log print. Useful for catching policy collapse.
    action_hist: [u64; 8],

    start_time: Instant,
    last_frame: Instant,
    physics_accum: f32,

    /// Exponential moving average of training loss (α = 0.01).
    loss_ema: f32,
    /// Exponential moving average of episode return (α = 0.02).
    return_ema: f32,

    /// Editable in the UI; applied on the next reset.
    pending_num_envs: u32,

    paused: bool,
    show_overlay: bool,
    view_mode: ViewMode,
    /// True when the user has taken over env 0 with the keyboard. The
    /// scripted opponent path keeps driving the other envs so the
    /// agent still trains against a stationary distribution; env 0
    /// becomes the killer-demo "you vs the live-learning agent".
    human_play: bool,
    /// Arrow-key state for the human-play env. Polled each frame, fed
    /// to env 0 as `HumanInput { axis_y: up - down }`.
    human_up: bool,
    human_down: bool,
    /// Speed multiplier from the UI slider. Total substeps per frame
    /// is `base_substeps_per_frame * speed_mul`.
    speed_mul: i32,
    /// Substeps actually advanced on the last frame, before the budget
    /// fired. Used to display the effective sim multiplier so the user
    /// can see when the slider is pinned by the time budget rather
    /// than by their setting.
    achieved_subs: u32,
    /// Substeps the slider asked for on the last frame.
    requested_subs: u32,
    /// Wall-clock dt of the last frame, in seconds. Used to derive the
    /// "1× = real time" sim multiplier display.
    last_dt: f32,
    /// Wall-clock time spent inside `tick()` last frame (ms).
    tick_ms: f32,

    /// Auto-curriculum: adjust game difficulty to maintain a target
    /// win/loss ratio. When enabled, the controller nudges
    /// `Game::set_difficulty` after every episode.
    auto_difficulty: bool,
    /// Target win/loss ratio for auto-curriculum (e.g. 1.5 = agent
    /// wins 60% of episodes).
    target_win_ratio: f32,
    /// EMA of the agent's win rate (0.0–1.0), α = 0.02.
    win_rate_ema: f32,

    /// If set (via `MEGAPLAYS_EXIT_AFTER_SECS`), self-exit once wall
    /// time exceeds this many seconds. Used for headless smoke tests.
    exit_deadline: Option<f32>,
    frame_counter: u64,
    last_heartbeat: f32,
    last_heartbeat_frame: u64,

    /// Remaining frames in the current training epoch. When > 0 each
    /// redraw processes a chunk and shows a progress bar.
    epoch_remaining: u32,
    epoch_total: u32,

    /// Per-frame render timing (updated every frame, logged at heartbeat).
    render_prims: usize,
    render_verts: usize,

    /// Cached device name (e.g. "llvmpipe (LLVM 20.1.2, 256 bits)" on
    /// lavapipe, or the real GPU name otherwise) and the count of
    /// trainable parameters. Sampled once at startup; shown in the
    /// overlay as a one-line proof that this *is* a live-trained net
    /// on real GPU hardware.
    device_label: String,
    param_count: usize,
    /// Sliding measurement of gradient steps per wall-clock second.
    /// `prev_grad_steps` is the value at `prev_grad_time`; the rate is
    /// `(grad_steps - prev_grad_steps) / (now - prev_grad_time)`.
    /// Refreshed at each ≥1 s interval so the display is stable.
    grad_rate: f32,
    prev_grad_steps: u64,
    prev_grad_time: Instant,

    /// Brief status message shown in the UI, fades after a few seconds.
    status_msg: String,
    status_time: Instant,
}

fn make_surface_config(size: winit::dpi::PhysicalSize<u32>) -> gpu::SurfaceConfig {
    gpu::SurfaceConfig {
        size: gpu::Extent {
            width: size.width.max(1),
            height: size.height.max(1),
            depth: 1,
        },
        usage: gpu::TextureUsage::TARGET,
        display_sync: gpu::DisplaySync::Recent,
        ..Default::default()
    }
}

impl<G: Game> Running<G> {
    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if let Some(sp) = self.prev_sync_point.take() {
            let _ = self.gpu.wait_for(&sp, !0);
        }
        self.gpu
            .reconfigure_surface(&mut self.surface, make_surface_config(size));
    }

    fn reset_learning(&mut self) {
        // Apply pending num_envs change.
        let new_num_envs = (self.pending_num_envs as usize).max(1);
        self.config.num_envs = new_num_envs;

        let cfg = self.config.agent.clone();
        self.agent = Agent::new(
            self.gpu.clone(),
            self.spec.obs_dim,
            self.spec.num_actions,
            new_num_envs,
            cfg,
        );
        self.loss_hist = RollingStats::new(400);
        self.reward_hist = RollingStats::new(400);
        self.episode_return_hist = RollingStats::new(200);
        self.win_rate_hist = RollingStats::new(400);
        self.difficulty_hist = RollingStats::new(400);
        self.last_sample_time = Instant::now();
        self.loss_ema = 0.0;
        self.return_ema = 0.0;
        self.scores_agent = 0;
        self.scores_opp = 0;
        self.last_return = vec![0.0; new_num_envs];
        self.win_rate_ema = self.target_win_ratio / (1.0 + self.target_win_ratio);

        // Rebuild the game vector to match the new env count.
        if let Some(build) = self.build_game.as_mut() {
            self.games = (0..new_num_envs).map(|_| build(self.gpu.clone())).collect();
        } else {
            self.games.resize_with(new_num_envs, || unreachable!());
            for g in &mut self.games {
                g.reset();
            }
        }
    }

    /// Index of the env to paint on top in overlay mode. We promote the
    /// env with the highest *most-recently-completed* return, plus the
    /// current in-flight episode return as a tiebreaker / catch-all
    /// when nothing has completed yet. The previous "longest live
    /// episode" heuristic celebrated a stalling policy that never
    /// terminated (especially the lander hoverer before truncation
    /// landed). Ties break toward the lowest-index env.
    fn hero_env(&self) -> usize {
        // Human-play forces env 0 to be the hero — the player must see
        // the env they're piloting on top of the cloud.
        if self.human_play {
            return 0;
        }
        let n = self.games.len().max(1);
        let last_return = if self.last_return.len() == n {
            &self.last_return[..]
        } else {
            &[]
        };
        let episode_return = if self.episode_return.len() == n {
            &self.episode_return[..]
        } else {
            &[]
        };
        let key = |i: usize| -> f32 {
            let lr = last_return.get(i).copied().unwrap_or(0.0);
            let ip = episode_return.get(i).copied().unwrap_or(0.0);
            // Last completed dominates; in-flight breaks ties.
            lr + 0.1 * ip
        };
        let mut best = 0;
        let mut best_k = key(0);
        for i in 1..n {
            let k = key(i);
            if k > best_k {
                best_k = k;
                best = i;
            }
        }
        best
    }

    /// Start a training epoch: 10 000 headless frames processed in
    /// chunks across multiple redraws so the UI stays responsive.
    fn train_epoch(&mut self) {
        const EPOCH_FRAMES: u32 = 10_000;
        if self.epoch_remaining > 0 {
            return; // already in progress
        }
        self.epoch_remaining = EPOCH_FRAMES;
        self.epoch_total = EPOCH_FRAMES;
    }

    /// Process a chunk of the current training epoch. Called each
    /// redraw while `epoch_remaining > 0`. Returns `true` while work
    /// remains.
    fn train_epoch_chunk(&mut self) -> bool {
        if self.epoch_remaining == 0 {
            return false;
        }

        let obs_dim = self.spec.obs_dim;
        let num_envs = self.games.len();
        let mut obs_buf = vec![0.0_f32; num_envs * obs_dim];

        if self.episode_return.len() != num_envs {
            self.episode_return = vec![0.0; num_envs];
            self.episode_len = vec![0; num_envs];
        }

        // Process a small chunk per redraw to keep the UI responsive.
        // Each frame does `substeps` worth of physics in bursts of
        // `action_repeat` substeps per inference + `train_steps`
        // gradient steps, so even 10 frames can take tens of ms.
        let action_repeat = self.agent.action_repeat();
        let bursts_per_frame = (self.config.base_substeps_per_frame / action_repeat.max(1)).max(1);
        let train_steps = self.config.train_steps_per_frame;
        let chunk = self.epoch_remaining.min(10);
        for _ in 0..chunk {
            for _ in 0..bursts_per_frame {
                let agent = &mut self.agent;
                let games = &mut self.games;
                let episode_return = &mut self.episode_return;
                let episode_len = &mut self.episode_len;
                let scores_agent = &mut self.scores_agent;
                let scores_opp = &mut self.scores_opp;
                let win_rate_ema = &mut self.win_rate_ema;
                let episode_return_hist = &mut self.episode_return_hist;
                let return_ema = &mut self.return_ema;
                run_burst(
                    agent,
                    games,
                    &mut obs_buf,
                    obs_dim,
                    action_repeat,
                    |i, _action, outcome| {
                        episode_return[i] += outcome.reward;
                        episode_len[i] += 1;
                        if outcome.done || outcome.truncated {
                            if outcome.done {
                                let agent_won = outcome.terminal_reward > 0.0;
                                if agent_won {
                                    *scores_agent += 1;
                                } else if outcome.terminal_reward < 0.0 {
                                    *scores_opp += 1;
                                }
                                *win_rate_ema =
                                    ema(*win_rate_ema, if agent_won { 1.0 } else { 0.0 }, 0.02);
                            }
                            episode_return_hist.push(episode_return[i]);
                            *return_ema = ema(*return_ema, episode_return[i], 0.02);
                            episode_return[i] = 0.0;
                            episode_len[i] = 0;
                        }
                    },
                );
            }
            for _ in 0..train_steps {
                if let Some(loss) = self.agent.train_step() {
                    self.loss_hist.push(loss);
                    self.loss_ema = ema(self.loss_ema, loss, 0.01);
                } else {
                    break;
                }
            }
        }
        self.epoch_remaining -= chunk;

        if self.epoch_remaining == 0 {
            log::info!(
                "train epoch done: grad={} eps={:.3} replay={}",
                self.agent.gradient_steps,
                self.agent.current_epsilon(),
                self.agent.replay_len(),
            );
        }
        self.epoch_remaining > 0
    }

    fn weights_path(&self) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("{}.weights", self.spec.title))
    }

    fn set_status(&mut self, msg: String) {
        self.status_msg = msg;
        self.status_time = Instant::now();
    }

    fn save_weights(&mut self) {
        let path = self.weights_path();
        match self.agent.save_weights(&path) {
            Ok(()) => {
                let msg = format!("saved to {}", path.display());
                log::info!("{msg}");
                self.set_status(msg);
            }
            Err(e) => {
                let msg = format!("save failed: {e}");
                log::error!("{msg}");
                self.set_status(msg);
            }
        }
    }

    fn load_weights(&mut self) {
        let path = self.weights_path();
        match self.agent.load_weights(&path) {
            Ok(()) => {
                let msg = format!(
                    "loaded from {} (eps={:.3})",
                    path.display(),
                    self.agent.current_epsilon(),
                );
                log::info!("{msg}");
                self.set_status(msg);
            }
            Err(e) => {
                let msg = format!("load failed: {e}");
                log::error!("{msg}");
                self.set_status(msg);
            }
        }
    }

    fn tick(&mut self) {
        let _tick = tracing::info_span!("tick").entered();
        let tick_start = Instant::now();
        if self.episode_return.len() != self.games.len() {
            self.episode_return = vec![0.0; self.games.len()];
            self.episode_len = vec![0; self.games.len()];
            self.last_return = vec![0.0; self.games.len()];
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.physics_accum += dt;
        self.last_dt = dt;

        // Human-play wiring: feed env 0 the arrow-key axis, hand every
        // other env a zero so the scripted opponent path stays
        // stationary for the policy. When the toggle is off, every env
        // including 0 sees zero — pong's `update_opponent` then takes
        // the scripted branch.
        let human = if self.human_play {
            HumanInput {
                axis_y: (self.human_up as i32 - self.human_down as i32) as f32,
                ..Default::default()
            }
        } else {
            HumanInput::default()
        };
        for (i, g) in self.games.iter_mut().enumerate() {
            g.set_human_input(if i == 0 { human } else { HumanInput::default() });
        }

        // Wall-clock-anchored sim speed. `speed_mul = 1` means "physics
        // runs at real time" — the substep count this frame is the
        // number of physics_dt intervals that fit into `dt × speed_mul`
        // of wall time. The slider can ask for more bursts than fit a
        // per-frame time budget; stop early and surface the achieved
        // rate so the user can tell when they're pinned by compute
        // rather than by their setting.
        let total_subs = if self.paused {
            0
        } else {
            let secs = dt * self.speed_mul.max(1) as f32;
            (secs / self.spec.physics_dt).round().max(1.0) as u32
        };
        let action_repeat = self.agent.action_repeat().max(1);
        let bursts_requested = total_subs.div_ceil(action_repeat);
        let frame_budget =
            std::time::Duration::from_millis(self.config.frame_budget_ms.max(1) as u64);
        let obs_dim = self.spec.obs_dim;
        let num_envs = self.games.len();
        let mut obs_buf = vec![0.0_f32; num_envs * obs_dim];

        let mut bursts_done = 0u32;
        for _ in 0..bursts_requested {
            if tick_start.elapsed() >= frame_budget {
                break;
            }
            let _sub = tracing::info_span!("substep").entered();
            // Split borrows so the closure can mutate every stat field
            // independently of `agent` and `games`.
            let agent = &mut self.agent;
            let games = &mut self.games;
            let episode_return = &mut self.episode_return;
            let episode_len = &mut self.episode_len;
            let last_return = &mut self.last_return;
            let action_hist = &mut self.action_hist;
            let reward_hist = &mut self.reward_hist;
            let scores_agent = &mut self.scores_agent;
            let scores_opp = &mut self.scores_opp;
            let win_rate_ema = &mut self.win_rate_ema;
            let episode_return_hist = &mut self.episode_return_hist;
            let episode_len_hist = &mut self.episode_len_hist;
            let return_ema = &mut self.return_ema;
            let hist_slots = action_hist.len();
            run_burst(
                agent,
                games,
                &mut obs_buf,
                obs_dim,
                action_repeat,
                |i, action, outcome| {
                    episode_return[i] += outcome.reward;
                    episode_len[i] += 1;
                    if (action as usize) < hist_slots {
                        action_hist[action as usize] += 1;
                    }
                    reward_hist.push(outcome.reward);
                    if outcome.done || outcome.truncated {
                        if outcome.done {
                            let agent_won = outcome.terminal_reward > 0.0;
                            if agent_won {
                                *scores_agent += 1;
                            } else if outcome.terminal_reward < 0.0 {
                                *scores_opp += 1;
                            }
                            *win_rate_ema =
                                ema(*win_rate_ema, if agent_won { 1.0 } else { 0.0 }, 0.02);
                        }
                        let ret = episode_return[i];
                        episode_return_hist.push(ret);
                        *return_ema = ema(*return_ema, ret, 0.02);
                        episode_len_hist.push(episode_len[i] as f32);
                        last_return[i] = ret;
                        episode_return[i] = 0.0;
                        episode_len[i] = 0;
                    }
                },
            );
            bursts_done += 1;
            self.physics_accum =
                (self.physics_accum - self.spec.physics_dt * action_repeat as f32).max(0.0);
        }
        let achieved_subs = bursts_done * action_repeat;
        self.achieved_subs = achieved_subs;
        self.requested_subs = bursts_requested * action_repeat;

        // Auto-curriculum: keep `win_rate_ema` close to `target_wr`.
        // Bidirectional — when the agent is winning we make the game
        // harder; when it's losing (e.g. after a policy regression) we
        // walk the difficulty back down. The asymmetric rates favour
        // climbing because exploration noise alone can push wr below
        // target for a few episodes. EMA-driven, not lifetime-averaged
        // (that one only updates ~1% per episode after the first 100,
        // i.e. essentially never). The hard ceiling/floor live in each
        // game's `set_difficulty` clamp — `set_difficulty` itself is
        // the only place that knows how much harder the game can get.
        let total_episodes = self.scores_agent + self.scores_opp;
        if !self.paused && self.auto_difficulty && total_episodes >= 30 {
            let target_wr = self.target_win_ratio / (1.0 + self.target_win_ratio);
            let err = self.win_rate_ema - target_wr;
            const DEADBAND: f32 = 0.04;
            let d = self.games[0].difficulty();
            let new_d = if err > DEADBAND {
                let adj = (1.0 + err * 0.20 * dt).clamp(1.0, 1.010);
                d * adj
            } else if err < -DEADBAND {
                let adj = (1.0 + err * 0.10 * dt).clamp(0.992, 1.0);
                d * adj
            } else {
                d
            };
            if (new_d - d).abs() > f32::EPSILON {
                for g in &mut self.games {
                    g.set_difficulty(new_d);
                }
                // Read back what the game accepted — clamps may differ
                // per game (pong caps at 3, lander up to 5).
            }
        }

        // Sample win-rate-EMA and difficulty into time-windowed plots
        // every `sample_interval` of wall time. 0.5 s × 400 capacity =
        // ~3 minutes of training visible at a glance.
        const SAMPLE_INTERVAL_SECS: f32 = 0.5;
        if tick_start
            .duration_since(self.last_sample_time)
            .as_secs_f32()
            >= SAMPLE_INTERVAL_SECS
        {
            self.win_rate_hist.push(self.win_rate_ema);
            let cur_difficulty = self.games.first().map_or(0.0, |g| g.difficulty());
            self.difficulty_hist.push(cur_difficulty);
            self.last_sample_time = tick_start;
        }

        // Training continues even when physics is paused — that's what
        // makes pause genuinely useful for "stop the action, let the
        // network finish chewing on what it just saw". Scale by the
        // *achieved* sim rate so a budget-capped frame doesn't also
        // fire spurious training that ran on stale data, and stop
        // early if the frame budget is exhausted.
        let train_steps = if self.paused {
            self.config.train_steps_per_frame
        } else {
            self.config.train_steps_per_frame * achieved_subs / self.config.base_substeps_per_frame
        };
        let _train = tracing::info_span!("train").entered();
        for _ in 0..train_steps {
            if tick_start.elapsed() >= frame_budget {
                break;
            }
            if let Some(loss) = {
                let _s = tracing::info_span!("train_step").entered();
                self.agent.train_step()
            } {
                self.loss_hist.push(loss);
                self.loss_ema = ema(self.loss_ema, loss, 0.01);
            } else {
                break;
            }
        }
        self.tick_ms = tick_start.elapsed().as_secs_f32() * 1000.0;

        // Refresh grad-rate at ≥1 s intervals so the display reads
        // cleanly. Anything finer-grained twitches with every train
        // step and conveys nothing extra.
        let since = tick_start
            .saturating_duration_since(self.prev_grad_time)
            .as_secs_f32();
        if since >= 1.0 {
            let dgrad = self
                .agent
                .gradient_steps
                .saturating_sub(self.prev_grad_steps);
            self.grad_rate = dgrad as f32 / since;
            self.prev_grad_steps = self.agent.gradient_steps;
            self.prev_grad_time = tick_start;
        }
    }

    fn redraw(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.tick();
        self.train_epoch_chunk();
        self.frame_counter += 1;

        let wall = self.start_time.elapsed().as_secs_f32();
        if wall - self.last_heartbeat > 2.0 {
            let interval_frames = self.frame_counter - self.last_heartbeat_frame;
            let interval_secs = wall - self.last_heartbeat;
            let interval_fps = interval_frames as f32 / interval_secs;
            let frame_ms = if interval_frames > 0 {
                interval_secs * 1000.0 / interval_frames as f32
            } else {
                0.0
            };
            self.last_heartbeat = wall;
            self.last_heartbeat_frame = self.frame_counter;
            let total = self.scores_agent + self.scores_opp;
            let win_rate = if total > 0 {
                self.scores_agent as f32 / total as f32
            } else {
                0.0
            };
            let na = self.spec.num_actions as usize;
            use std::fmt::Write;
            let act_sum: u64 = self.action_hist[..na].iter().sum();
            let mut action_str = String::with_capacity(4 * na);
            action_str.push('[');
            for a in 0..na {
                if a > 0 {
                    action_str.push('/');
                }
                let frac = if act_sum > 0 {
                    self.action_hist[a] as f32 / act_sum as f32
                } else {
                    0.0
                };
                let _ = write!(action_str, "{frac:.2}");
            }
            action_str.push(']');
            let galleys = self
                .egui_winit
                .egui_ctx()
                .fonts(|f| f.num_galleys_in_cache());
            let atlas_fill = self
                .egui_winit
                .egui_ctx()
                .fonts(|f| f.font_atlas_fill_ratio());
            log::info!(
                "t={:5.1}s fps={:5.1} ({:.1}ms) prims={} verts={} galleys={} atlas={:.0}% \
                 eps={:.3} grad={:>5} loss={:.4} \
                 ret={:6.2} len={:6.1} wins={}/{} ({:.1}%) actions={}",
                wall,
                interval_fps,
                frame_ms,
                self.render_prims,
                self.render_verts,
                galleys,
                atlas_fill * 100.0,
                self.agent.current_epsilon(),
                self.agent.gradient_steps,
                self.agent.last_loss,
                self.episode_return_hist.mean(),
                self.episode_len_hist.mean(),
                self.scores_agent,
                total,
                win_rate * 100.0,
                action_str,
            );
            for h in self.action_hist.iter_mut() {
                *h = 0;
            }
        }

        if let Some(deadline) = self.exit_deadline {
            if self.start_time.elapsed().as_secs_f32() > deadline {
                log::info!(
                    "MEGAPLAYS_EXIT_AFTER_SECS={deadline:.1} hit after {} frames; \
                     replay={} grad_steps={} loss={:.4}",
                    self.frame_counter,
                    self.agent.replay_len(),
                    self.agent.gradient_steps,
                    self.agent.last_loss,
                );
                event_loop.exit();
                return;
            }
        }

        let raw_input = self.egui_winit.take_egui_input(&self.window);
        let window_size = self.window.inner_size();
        let pixels_per_point =
            egui_winit::pixels_per_point(self.egui_winit.egui_ctx(), &self.window);

        let egui_output = self
            .egui_winit
            .egui_ctx()
            .clone()
            .run_ui(raw_input, |ctx| self.build_ui(ctx, event_loop));

        self.egui_winit
            .handle_platform_output(&self.window, egui_output.platform_output);

        let primitives = self
            .egui_winit
            .egui_ctx()
            .tessellate(egui_output.shapes, pixels_per_point);
        let screen_desc = blade_egui::ScreenDescriptor {
            physical_size: (window_size.width, window_size.height),
            scale_factor: pixels_per_point,
        };

        self.render_prims = primitives.len();
        self.render_verts = primitives
            .iter()
            .map(|p| match &p.primitive {
                egui::epaint::Primitive::Mesh(m) => m.vertices.len(),
                _ => 0,
            })
            .sum();

        let t_render = Instant::now();
        self.render(&primitives, &egui_output.textures_delta, &screen_desc);
        let render_ms = t_render.elapsed().as_secs_f32() * 1000.0;
        if render_ms > 20.0 {
            log::warn!(
                "slow render: {render_ms:.1}ms prims={} verts={} tex_set={} tex_free={} galleys={}",
                self.render_prims,
                self.render_verts,
                egui_output.textures_delta.set.len(),
                egui_output.textures_delta.free.len(),
                self.egui_winit
                    .egui_ctx()
                    .fonts(|f| f.num_galleys_in_cache()),
            );
        }
    }

    fn build_ui(&mut self, ctx: &egui::Context, event_loop: &winit::event_loop::ActiveEventLoop) {
        let clear = self.config.clear_color;
        let play_aspect = self.spec.play_area[0] / self.spec.play_area[1];
        let view_mode = self.view_mode;
        #[allow(deprecated)]
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(Color32::from_rgb(clear.r(), clear.g(), clear.b()))
                    .inner_margin(0.0),
            )
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                let painter = ui.painter();
                match view_mode {
                    ViewMode::Grid => {
                        let (cols, rows) = grid_dims(self.games.len());
                        let cell_w = rect.width() / cols as f32;
                        let cell_h = rect.height() / rows as f32;
                        let pad = 2.0;
                        let highlight_human = self.human_play;
                        for (i, g) in self.games.iter().enumerate() {
                            let col = i % cols;
                            let row = i / cols;
                            let cell = Rect::from_min_size(
                                Pos2::new(
                                    rect.min.x + col as f32 * cell_w + pad,
                                    rect.min.y + row as f32 * cell_h + pad,
                                ),
                                Vec2::new(cell_w - 2.0 * pad, cell_h - 2.0 * pad),
                            );
                            let play = fit_rect(cell, play_aspect);
                            g.paint(painter, play, 255);
                            let (stroke_w, stroke_c) = if highlight_human && i == 0 {
                                (2.0, Color32::from_rgb(230, 170, 60))
                            } else {
                                (1.0, Color32::from_gray(28))
                            };
                            painter.rect_stroke(
                                cell,
                                0.0,
                                Stroke::new(stroke_w, stroke_c),
                                egui::StrokeKind::Inside,
                            );
                        }
                    }
                    ViewMode::Overlay => {
                        // Super-Meat-Boy-style: every env in the same
                        // rect. The hero (longest current episode) draws
                        // first so its opaque backdrop goes down, then
                        // the alpha-blended ghosts layer on top, and
                        // finally the hero's entities re-paint at full
                        // opacity so the current best attempt clearly
                        // stands out against the ghost cloud.
                        let play = fit_rect(rect, play_aspect);
                        let hero = self.hero_env();
                        let ghost_alpha = ghost_alpha_for(self.games.len());
                        // Hero backdrop first (opaque, sets the scene).
                        self.games[hero].paint(painter, play, 255);
                        // Ghosts on top of the backdrop.
                        for (i, g) in self.games.iter().enumerate() {
                            if i == hero {
                                continue;
                            }
                            g.paint(painter, play, ghost_alpha);
                        }
                        // Re-paint the hero's entities so they sit on
                        // top of the ghost cloud, fully opaque.
                        self.games[hero].paint(painter, play, 254);
                        painter.rect_stroke(
                            play,
                            0.0,
                            Stroke::new(1.5, Color32::from_rgb(230, 210, 90)),
                            egui::StrokeKind::Inside,
                        );
                        // Q-bars for the hero env: one bar per action,
                        // height ∝ Q-value (centred at zero so negative
                        // bars hang below). The argmax bar is bright
                        // green — that's the action ε-greedy would pick.
                        draw_q_bars(
                            painter,
                            play,
                            self.agent.last_q(),
                            hero,
                            self.spec.num_actions as usize,
                        );
                    }
                }
            });

        if !self.show_overlay {
            return;
        }

        egui::Window::new("training")
            .default_pos([16.0, 16.0])
            .resizable(false)
            .collapsible(true)
            .show(ctx, |ui| {
                let wall = self.start_time.elapsed().as_secs_f32();

                // Editable env count + reset on one line.
                ui.horizontal(|ui| {
                    ui.label("envs");
                    ui.add(
                        egui::DragValue::new(&mut self.pending_num_envs)
                            .range(1..=256)
                            .speed(0.2),
                    );
                    if ui.button("reset [R]").clicked() {
                        self.reset_learning();
                    }
                });

                // Showcase line — proves at a glance what this is:
                // a tiny MLP being trained live on the user's GPU.
                ui.label(
                    egui::RichText::new(format!(
                        "training {} params  →  {} @ {:.0} grad/s",
                        format_count(self.param_count),
                        self.device_label,
                        self.grad_rate,
                    ))
                    .small()
                    .color(Color32::from_rgb(140, 200, 230)),
                );

                ui.label(format!("wall time       {:7.1} s", wall));
                ui.label(format!(
                    "epsilon         {:7.3}",
                    self.agent.current_epsilon()
                ));
                ui.label(format!(
                    "replay / cap    {:>6} / {}",
                    self.agent.replay_len(),
                    self.config.agent.replay_capacity
                ));
                ui.label(format!("grad steps      {:>7}", self.agent.gradient_steps));
                ui.label(format!("inferences      {:>7}", self.agent.inferences));

                ui.separator();
                ui.add(
                    egui::Slider::new(&mut self.speed_mul, 1..=32)
                        .text("speed")
                        .integer(),
                );
                // Show the achieved-vs-requested sim rate so the user
                // can see when the slider is pinned by the frame budget
                // rather than by their setting. "tick" is how long the
                // physics + training loop spent inside this frame.
                // Sim multiplier is wall-clock-anchored: 1× = physics
                // runs at real time. Less than the slider value = pinned.
                let achieved_x = if self.last_dt > 0.0 && self.requested_subs > 0 {
                    self.achieved_subs as f32 * self.spec.physics_dt / self.last_dt
                } else {
                    0.0
                };
                let pinned = self.requested_subs > 0
                    && self.achieved_subs < self.requested_subs
                    && !self.paused;
                let color = if pinned {
                    Color32::from_rgb(220, 170, 60)
                } else {
                    Color32::from_gray(180)
                };
                ui.label(
                    egui::RichText::new(format!(
                        "  effective    {:>5.1}×   tick {:>4.1} ms{}",
                        achieved_x,
                        self.tick_ms,
                        if pinned { "  (budget)" } else { "" },
                    ))
                    .small()
                    .color(color),
                );
                ui.horizontal(|ui| {
                    ui.label("view [V]");
                    ui.selectable_value(&mut self.view_mode, ViewMode::Grid, "grid");
                    ui.selectable_value(&mut self.view_mode, ViewMode::Overlay, "overlay");
                });
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(self.human_play, "human play [P]")
                        .on_hover_text(
                            "Take over env 0's opponent with ↑/↓ — you vs the live-learning agent",
                        )
                        .clicked()
                    {
                        self.human_play = !self.human_play;
                    }
                    if self.human_play {
                        ui.colored_label(
                            Color32::from_rgb(220, 170, 60),
                            "  env 0: ↑/↓ control orange paddle",
                        );
                    }
                });
                if ui
                    .button(if self.paused {
                        "resume [Space]"
                    } else {
                        "pause [Space]"
                    })
                    .clicked()
                {
                    self.paused = !self.paused;
                }
                if self.epoch_remaining > 0 {
                    let done = self.epoch_total - self.epoch_remaining;
                    let frac = done as f32 / self.epoch_total as f32;
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .text(format!("training {}/{}", done, self.epoch_total,))
                            .desired_width(260.0),
                    );
                } else if ui
                    .button("train epoch [T]")
                    .on_hover_text("Run 10 000 headless train steps (skips rendering)")
                    .clicked()
                {
                    self.train_epoch();
                }
                ui.horizontal(|ui| {
                    if ui.button("save [S]").clicked() {
                        self.save_weights();
                    }
                    if ui.button("load [L]").clicked() {
                        self.load_weights();
                    }
                });
                // Show status message for 3 seconds after save/load.
                if !self.status_msg.is_empty() && self.status_time.elapsed().as_secs_f32() < 3.0 {
                    ui.label(
                        egui::RichText::new(&self.status_msg)
                            .small()
                            .color(Color32::from_rgb(180, 200, 140)),
                    );
                }

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("training loss");
                    ui.label(
                        egui::RichText::new(format!("{:.4}", self.loss_ema))
                            .strong()
                            .color(Color32::from_rgb(220, 90, 80)),
                    );
                });
                let (loss_rect, _) =
                    ui.allocate_exact_size(Vec2::new(260.0, 80.0), egui::Sense::hover());
                draw_plot(
                    ui.painter(),
                    loss_rect,
                    &self.loss_hist,
                    Color32::from_rgb(220, 90, 80),
                    true,
                );

                ui.horizontal(|ui| {
                    ui.label("episode return");
                    ui.label(
                        egui::RichText::new(format!("{:.2}", self.return_ema))
                            .strong()
                            .color(Color32::from_rgb(80, 220, 120)),
                    );
                });
                let (ret_rect, _) =
                    ui.allocate_exact_size(Vec2::new(260.0, 60.0), egui::Sense::hover());
                draw_plot(
                    ui.painter(),
                    ret_rect,
                    &self.episode_return_hist,
                    Color32::from_rgb(80, 220, 120),
                    false,
                );

                ui.label("instantaneous reward");
                let (rew_rect, _) =
                    ui.allocate_exact_size(Vec2::new(260.0, 30.0), egui::Sense::hover());
                SparkLine::draw(
                    ui.painter(),
                    rew_rect,
                    &self.reward_hist,
                    Color32::from_rgb(120, 180, 255),
                );

                ui.separator();
                self.games[0].ui(ui);
                // Propagate slider changes from game[0] to all others.
                for i in 1..self.games.len() {
                    let (head, tail) = self.games.split_at_mut(i);
                    tail[0].sync_settings(&head[0]);
                }

                ui.separator();
                ui.checkbox(&mut self.auto_difficulty, "auto difficulty");
                if self.auto_difficulty {
                    ui.horizontal(|ui| {
                        ui.label("target W/L");
                        ui.add(
                            egui::DragValue::new(&mut self.target_win_ratio)
                                .range(1.0..=5.0)
                                .speed(0.05)
                                .fixed_decimals(1),
                        );
                    });
                }
                let total = self.scores_agent + self.scores_opp;
                let wr = if total > 0 {
                    self.scores_agent as f32 / total as f32
                } else {
                    0.0
                };
                ui.label(format!(
                    "win rate        {:>5.1}%  (EMA {:.1}%)  difficulty {:.2}",
                    wr * 100.0,
                    self.win_rate_ema * 100.0,
                    self.games[0].difficulty(),
                ));

                // Win-rate EMA + difficulty over time. When
                // auto-curriculum is on, win-rate is the *controlled*
                // variable — by design it stays near `target_wr`, so
                // the live signal is whether the controller can keep it
                // there *and* whether difficulty is climbing. Both
                // tracks share the same x-axis (wall time) and the
                // dashed target line shows where the controller is
                // pulling.
                ui.horizontal(|ui| {
                    ui.label("win-rate EMA (target dashed)");
                });
                let (wr_rect, _) =
                    ui.allocate_exact_size(Vec2::new(260.0, 50.0), egui::Sense::hover());
                let target_wr = self.target_win_ratio / (1.0 + self.target_win_ratio);
                draw_plot_with_marker(
                    ui.painter(),
                    wr_rect,
                    &self.win_rate_hist,
                    Color32::from_rgb(120, 220, 200),
                    Some(target_wr),
                    (0.0, 1.0),
                );
                ui.label("difficulty over wall time");
                let (df_rect, _) =
                    ui.allocate_exact_size(Vec2::new(260.0, 40.0), egui::Sense::hover());
                draw_plot(
                    ui.painter(),
                    df_rect,
                    &self.difficulty_hist,
                    Color32::from_rgb(230, 180, 120),
                    false,
                );

                ui.separator();
                if ui.button("quit [Esc]").clicked() {
                    event_loop.exit();
                }
            });
    }

    fn render(
        &mut self,
        gui_primitives: &[egui::ClippedPrimitive],
        gui_textures: &egui::TexturesDelta,
        screen_desc: &blade_egui::ScreenDescriptor,
    ) {
        let _render = tracing::info_span!("render").entered();
        let frame = {
            let _s = tracing::info_span!("acquire_frame").entered();
            self.surface.acquire_frame()
        };
        let frame_view = frame.texture_view();
        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        self.gui_painter
            .update_textures(&mut self.command_encoder, gui_textures, &self.gpu);

        {
            let _draw = tracing::info_span!("encode_ui").entered();
            let mut pass = self.command_encoder.render(
                "draw ui",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: frame_view,
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            );
            self.gui_painter
                .paint(&mut pass, gui_primitives, screen_desc, &self.gpu);
        }

        self.command_encoder.present(frame);
        // Capture the shared-epoch timestamp immediately before submit so
        // blade's per-pass timings from *this* encoder land at the right
        // offset on the Perfetto GPU track. The timings themselves become
        // readable after the corresponding fence signals, which blade
        // takes care of internally on the next submit — we just re-read
        // them off the encoder once the wait below has returned, then
        // forward to the profiler.
        let submit_offset_ns = crate::profiling::now_ns();
        let sync_point = {
            let _s = tracing::info_span!("submit").entered();
            self.gpu.submit(&mut self.command_encoder)
        };
        self.gui_painter.after_submit(&sync_point);

        if let Some(sp) = self.prev_sync_point.take() {
            let _s = tracing::info_span!("wait_prev").entered();
            let _ = self.gpu.wait_for(&sp, !0);
        }
        // Now that the previous frame's fence has been waited on (or
        // there was no previous frame), blade has resolved the *previous*
        // submit's timestamp queries. Drain them onto the GPU track, using
        // the submit-offset we captured for that previous frame.
        if let Some(prev_offset) = self.prev_submit_offset_ns.take() {
            let timings = self.command_encoder.timings().clone();
            crate::profiling::record_gpu_passes(prev_offset, &timings);
        }
        self.prev_sync_point = Some(sync_point);
        self.prev_submit_offset_ns = Some(submit_offset_ns);
    }

    fn destroy(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            let _ = self.gpu.wait_for(&sp, !0);
        }
        // Drop the agent (and its meganeura sessions) first — each
        // session's Drop waits for its in-flight GPU work and then
        // destroys its pipelines and buffers. This must happen before
        // we tear down the render resources below, otherwise Vulkan
        // validation complains about objects destroyed out of order.
        self.agent.destroy();
        self.gpu.destroy_command_encoder(&mut self.command_encoder);
        self.gui_painter.destroy(&self.gpu);
        self.gpu.destroy_surface(&mut self.surface);
    }
}

/// Q-bars for the hero env: one short vertical bar per action along
/// the bottom edge of `play`, height proportional to the Q-value with
/// the argmax in bright green. The whole strip occupies ~10% of the
/// play rect height. Zeros (warmup, no inference yet) render as a
/// dimmed baseline.
fn draw_q_bars(painter: &egui::Painter, play: Rect, last_q: &[f32], hero: usize, na: usize) {
    if na == 0 || last_q.len() < (hero + 1) * na {
        return;
    }
    let q = &last_q[hero * na..(hero + 1) * na];
    let span = q.iter().fold(0.0_f32, |a, &v| a.max(v.abs())).max(1e-3);
    let strip_h = play.height() * 0.10;
    let strip_y = play.max.y - strip_h - 2.0;
    let bar_w = (play.width() * 0.7) / na as f32;
    let strip_w = bar_w * na as f32;
    let strip_x0 = play.center().x - strip_w * 0.5;
    let mid_y = strip_y + strip_h * 0.5;
    painter.line_segment(
        [
            Pos2::new(strip_x0, mid_y),
            Pos2::new(strip_x0 + strip_w, mid_y),
        ],
        Stroke::new(0.5, Color32::from_gray(80)),
    );
    let mut argmax = 0;
    for i in 1..na {
        if q[i] > q[argmax] {
            argmax = i;
        }
    }
    for (i, &v) in q.iter().enumerate() {
        let h = (v / span).clamp(-1.0, 1.0) * (strip_h * 0.5);
        let x0 = strip_x0 + i as f32 * bar_w + 1.0;
        let x1 = x0 + bar_w - 2.0;
        let (y0, y1) = if h >= 0.0 {
            (mid_y - h, mid_y)
        } else {
            (mid_y, mid_y - h)
        };
        let color = if i == argmax {
            Color32::from_rgb(120, 220, 120)
        } else {
            Color32::from_rgb(100, 130, 170)
        };
        painter.rect_filled(
            Rect::from_min_max(Pos2::new(x0, y0), Pos2::new(x1, y1)),
            0.0,
            color,
        );
    }
}

/// Compact integer for the overlay: 1_234 → "1.2 k", 4_500_000 → "4.5 M".
/// Below 1000 it's just the count. Slightly different from
/// `humantime`-style for the trainable-params context (where "4.5 M"
/// is unambiguous and reads cleanly at a glance).
fn format_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1} M", n as f32 / 1.0e6)
    } else if n >= 1_000 {
        format!("{:.1} k", n as f32 / 1.0e3)
    } else {
        format!("{n}")
    }
}

/// Build the device label shown in the overlay, e.g.
/// `"llvmpipe (LLVM 20.1.2, 256 bits)  [software]"` or
/// `"NVIDIA GeForce RTX 4090"`. Surfaces the software-emulation flag
/// because performance numbers on lavapipe are a stress floor, not a
/// reflection of the real hardware path.
fn device_label_for(gpu: &gpu::Context) -> String {
    let info = gpu.device_information();
    let mut s = info.device_name.clone();
    if info.is_software_emulated {
        s.push_str("  [software]");
    }
    s
}

fn grid_dims(n: usize) -> (usize, usize) {
    // Pick a near-square layout: cols = ceil(sqrt(n)), rows = ceil(n / cols).
    let cols = (n as f32).sqrt().ceil().max(1.0) as usize;
    let rows = n.div_ceil(cols);
    (cols, rows)
}

/// Layout of the central panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    /// One tile per environment — the default "see all agents at once".
    Grid,
    /// Every environment paints into the same rect; the hero env
    /// (currently the one with the longest live episode) paints last
    /// at full opacity over the alpha-blended ghosts. Super Meat Boy
    /// replay-style.
    Overlay,
}

impl ViewMode {
    fn toggled(self) -> Self {
        match self {
            Self::Grid => Self::Overlay,
            Self::Overlay => Self::Grid,
        }
    }
}

/// Alpha used for ghost envs in overlay mode. Tuned so that 16
/// overlapping ghosts read as a softly-saturated cloud without
/// drowning out the hero.
fn ghost_alpha_for(n_envs: usize) -> u8 {
    // Inverse proportional to sqrt(N) with a generous floor so
    // ghosts stay visible even with many envs — 4 envs ≈ 120,
    // 16 envs ≈ 60, 64 envs ≈ 40.
    let n = (n_envs as f32).max(1.0);
    ((240.0 / n.sqrt()).clamp(40.0, 200.0)) as u8
}

/// Exponential moving average: `current ← (1-α) * current + α * sample`.
fn ema(current: f32, sample: f32, alpha: f32) -> f32 {
    current * (1.0 - alpha) + sample * alpha
}

fn fit_rect(avail: Rect, aspect: f32) -> Rect {
    let avail_aspect = avail.width() / avail.height();
    if avail_aspect > aspect {
        let w = avail.height() * aspect;
        let x0 = avail.center().x - w * 0.5;
        Rect::from_min_size(Pos2::new(x0, avail.min.y), Vec2::new(w, avail.height()))
    } else {
        let h = avail.width() / aspect;
        let y0 = avail.center().y - h * 0.5;
        Rect::from_min_size(Pos2::new(avail.min.x, y0), Vec2::new(avail.width(), h))
    }
}

/// Line plot with a subtle grid and min/max labels. `log_y = true`
/// uses `log(1 + v)` so the loss curve's early spike doesn't flatten
/// the rest of the trace.
fn draw_plot(
    painter: &egui::Painter,
    rect: Rect,
    stats: &RollingStats,
    color: Color32,
    log_y: bool,
) {
    painter.rect_filled(rect, 0.0, Color32::from_black_alpha(120));
    let grid = Stroke::new(0.5, Color32::from_gray(40));
    for i in 1..4 {
        let y = rect.min.y + rect.height() * i as f32 / 4.0;
        painter.line_segment([Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)], grid);
    }

    if stats.len() < 2 {
        return;
    }
    let transform = |v: f32| if log_y { v.max(0.0).ln_1p() } else { v };
    let (lo, hi) = stats.min_max();
    let (lo_t, hi_t) = (transform(lo), transform(hi));
    let span = (hi_t - lo_t).max(1e-6);
    let n = stats.len().max(1) - 1;
    let mut prev: Option<Pos2> = None;
    for (i, v) in stats.iter().enumerate() {
        let t = i as f32 / n as f32;
        let y = 1.0 - (transform(v) - lo_t) / span;
        let p = rect.min + Vec2::new(t * rect.width(), y * rect.height());
        if let Some(a) = prev {
            painter.line_segment([a, p], Stroke::new(1.2, color));
        }
        prev = Some(p);
    }

    let font = egui::FontId::monospace(10.0);
    let label_color = Color32::from_gray(170);
    painter.text(
        Pos2::new(rect.min.x + 2.0, rect.min.y + 2.0),
        egui::Align2::LEFT_TOP,
        format!("max {hi:.3}"),
        font.clone(),
        label_color,
    );
    painter.text(
        Pos2::new(rect.min.x + 2.0, rect.max.y - 2.0),
        egui::Align2::LEFT_BOTTOM,
        format!("min {lo:.3}"),
        font,
        label_color,
    );
}

/// Plot a [0,1]-bounded trace (e.g. win-rate EMA) with a fixed y range
/// and a dashed reference marker (the target the controller is pulling
/// the trace toward). Avoids `draw_plot`'s auto-scaling, which would
/// crush the visible variation when the trace hugs the target.
fn draw_plot_with_marker(
    painter: &egui::Painter,
    rect: Rect,
    stats: &RollingStats,
    color: Color32,
    marker: Option<f32>,
    (lo, hi): (f32, f32),
) {
    painter.rect_filled(rect, 0.0, Color32::from_black_alpha(120));
    let grid = Stroke::new(0.5, Color32::from_gray(40));
    for i in 1..4 {
        let y = rect.min.y + rect.height() * i as f32 / 4.0;
        painter.line_segment([Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)], grid);
    }
    let span = (hi - lo).max(1e-6);
    if let Some(m) = marker {
        let y = rect.min.y + (1.0 - (m - lo) / span) * rect.height();
        let dash_w = 5.0;
        let gap = 4.0;
        let mut x = rect.min.x;
        while x < rect.max.x {
            let x2 = (x + dash_w).min(rect.max.x);
            painter.line_segment(
                [Pos2::new(x, y), Pos2::new(x2, y)],
                Stroke::new(1.0, Color32::from_rgb(200, 200, 90)),
            );
            x += dash_w + gap;
        }
    }
    if stats.len() < 2 {
        return;
    }
    let n = stats.len().max(1) - 1;
    let mut prev: Option<Pos2> = None;
    for (i, v) in stats.iter().enumerate() {
        let t = i as f32 / n as f32;
        let y = 1.0 - (v.clamp(lo, hi) - lo) / span;
        let p = rect.min + Vec2::new(t * rect.width(), y * rect.height());
        if let Some(a) = prev {
            painter.line_segment([a, p], Stroke::new(1.2, color));
        }
        prev = Some(p);
    }
}
