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
    agent::{Agent, AgentConfig, Transition},
    game::{Game, GameSpec},
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
    /// Clear colour behind the game grid.
    pub clear_color: Color32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            agent: AgentConfig::default(),
            num_envs: 16,
            base_substeps_per_frame: 4,
            train_steps_per_frame: 4,
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
        let (mut build_game, config) = match std::mem::replace(&mut self.state, AppState::Dead) {
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
        self.state = AppState::Running(Box::new(Running {
            gpu,
            surface,
            command_encoder,
            prev_sync_point: None,
            gui_painter,
            window,
            egui_winit,
            games,
            spec,
            agent,
            build_game: Some(build_game),
            config,
            last_obs: Vec::new(),
            last_action: Vec::new(),
            loss_hist: RollingStats::new(400),
            reward_hist: RollingStats::new(400),
            episode_return: Vec::new(),
            episode_return_hist: RollingStats::new(200),
            episode_len: Vec::new(),
            episode_len_hist: RollingStats::new(200),
            scores_agent: 0,
            scores_opp: 0,
            action_hist: [0; 8],
            start_time: now,
            last_frame: now,
            physics_accum: 0.0,
            paused: false,
            show_overlay: true,
            view_mode,
            speed_mul: 1,
            exit_deadline,
            frame_counter: 0,
            last_heartbeat: 0.0,
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
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape => event_loop.exit(),
                KeyCode::Space => r.paused = !r.paused,
                KeyCode::KeyG => r.show_overlay = !r.show_overlay,
                KeyCode::KeyV => r.view_mode = r.view_mode.toggled(),
                KeyCode::KeyR => r.reset_learning(),
                _ => {}
            },
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
    gui_painter: blade_egui::GuiPainter,

    window: winit::window::Window,
    egui_winit: egui_winit::State,

    games: Vec<G>,
    spec: GameSpec,
    agent: Agent,
    build_game: Option<Builder<G>>,
    config: AppConfig,

    last_obs: Vec<Option<Vec<f32>>>,
    last_action: Vec<u32>,

    loss_hist: RollingStats,
    reward_hist: RollingStats,
    episode_return: Vec<f32>,
    episode_return_hist: RollingStats,
    episode_len: Vec<u32>,
    episode_len_hist: RollingStats,
    scores_agent: u64,
    scores_opp: u64,
    /// Rolling count of each action taken since last heartbeat. Reset
    /// at every log print. Useful for catching policy collapse.
    action_hist: [u64; 8],

    start_time: Instant,
    last_frame: Instant,
    physics_accum: f32,

    paused: bool,
    show_overlay: bool,
    view_mode: ViewMode,
    /// Speed multiplier from the UI slider. Total substeps per frame
    /// is `base_substeps_per_frame * speed_mul`.
    speed_mul: u32,

    /// If set (via `MEGAPLAYS_EXIT_AFTER_SECS`), self-exit once wall
    /// time exceeds this many seconds. Used for headless smoke tests.
    exit_deadline: Option<f32>,
    frame_counter: u64,
    last_heartbeat: f32,
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
        let cfg = self.config.agent.clone();
        let num_envs = self.config.num_envs;
        self.agent = Agent::new(
            self.gpu.clone(),
            self.spec.obs_dim,
            self.spec.num_actions,
            num_envs,
            cfg,
        );
        self.loss_hist = RollingStats::new(400);
        self.reward_hist = RollingStats::new(400);
        self.episode_return_hist = RollingStats::new(200);
        self.episode_return.iter_mut().for_each(|r| *r = 0.0);
        if let Some(build) = self.build_game.as_mut() {
            for g in &mut self.games {
                *g = build(self.gpu.clone());
            }
        } else {
            for g in &mut self.games {
                g.reset();
            }
        }
        self.last_obs.clear();
        self.last_action.clear();
    }

    /// Index of the env to paint on top in overlay mode. We use the
    /// longest-running *current* episode as a proxy for "doing best
    /// right now": for pong that's the longest rally, for lander it's
    /// the agent that has survived the most substeps without
    /// crashing. Ties break toward the lowest-index env.
    fn hero_env(&self) -> usize {
        self.episode_len
            .iter()
            .enumerate()
            .max_by_key(|&(_, len)| *len)
            .map_or(0, |(i, _)| i)
    }

    fn tick(&mut self) {
        if self.last_obs.len() != self.games.len() {
            self.last_obs = vec![None; self.games.len()];
            self.last_action = vec![0; self.games.len()];
            self.episode_return = vec![0.0; self.games.len()];
            self.episode_len = vec![0; self.games.len()];
        }

        if self.paused {
            return;
        }

        let wall = self.start_time.elapsed().as_secs_f32();
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.physics_accum += dt;

        // Number of substeps this frame: base * slider, but capped by
        // how much wall-clock dt has actually accumulated so the
        // display doesn't lie about the simulation rate.
        let total_subs = (self.config.base_substeps_per_frame * self.speed_mul).max(1);
        let obs_dim = self.spec.obs_dim;
        let num_envs = self.games.len();
        let mut obs_buf = vec![0.0_f32; num_envs * obs_dim];

        for _ in 0..total_subs {
            for (i, g) in self.games.iter().enumerate() {
                let o = g.observation();
                obs_buf[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&o);
            }
            let actions = self.agent.select_actions(&obs_buf, wall);

            for (i, g) in self.games.iter_mut().enumerate() {
                let outcome = g.step(actions[i]);
                let next = g.observation();
                self.episode_return[i] += outcome.reward;
                self.episode_len[i] += 1;
                if let Some(prev) = self.last_obs[i].replace(next.clone()) {
                    self.agent.record(Transition {
                        obs: prev,
                        action: self.last_action[i],
                        reward: outcome.reward,
                        next_obs: next,
                        done: outcome.done,
                    });
                }
                self.last_action[i] = actions[i];
                if (actions[i] as usize) < self.action_hist.len() {
                    self.action_hist[actions[i] as usize] += 1;
                }
                self.reward_hist.push(outcome.reward);

                if outcome.done {
                    if outcome.terminal_reward > 0.0 {
                        self.scores_agent += 1;
                    } else if outcome.terminal_reward < 0.0 {
                        self.scores_opp += 1;
                    }
                    self.episode_return_hist.push(self.episode_return[i]);
                    self.episode_len_hist.push(self.episode_len[i] as f32);
                    self.episode_return[i] = 0.0;
                    self.episode_len[i] = 0;
                    g.reset();
                    self.last_obs[i] = None;
                }
            }
            self.physics_accum = (self.physics_accum - self.spec.physics_dt).max(0.0);
        }

        for _ in 0..self.config.train_steps_per_frame {
            if let Some(loss) = self.agent.train_step() {
                self.loss_hist.push(loss);
            } else {
                break;
            }
        }
    }

    fn redraw(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.tick();
        self.frame_counter += 1;

        let wall = self.start_time.elapsed().as_secs_f32();
        if wall - self.last_heartbeat > 2.0 {
            self.last_heartbeat = wall;
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
            log::info!(
                "t={:5.1}s fps={:5.1} eps={:.3} grad={:>5} loss={:.4} \
                 ret={:6.2} len={:6.1} wins={}/{} ({:.1}%) actions={}",
                wall,
                self.frame_counter as f32 / wall,
                self.agent.current_epsilon(wall),
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
            .run(raw_input, |ctx| self.build_ui(ctx, event_loop));

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

        self.render(&primitives, &egui_output.textures_delta, &screen_desc);
    }

    fn build_ui(&mut self, ctx: &egui::Context, event_loop: &winit::event_loop::ActiveEventLoop) {
        let clear = self.config.clear_color;
        let play_aspect = self.spec.play_area[0] / self.spec.play_area[1];
        let view_mode = self.view_mode;
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
                            painter.rect_stroke(
                                cell,
                                0.0,
                                Stroke::new(1.0, Color32::from_gray(28)),
                                egui::StrokeKind::Inside,
                            );
                        }
                    }
                    ViewMode::Overlay => {
                        // Super-Meat-Boy-style: every env in the same
                        // rect. The "hero" env (longest current episode
                        // — proxy for best live performer) paints last
                        // at full opacity; the rest are alpha-blended
                        // ghosts, so the active attempt stands out
                        // against a cloud of parallel runs.
                        let play = fit_rect(rect, play_aspect);
                        let hero = self.hero_env();
                        let ghost_alpha = ghost_alpha_for(self.games.len());
                        for (i, g) in self.games.iter().enumerate() {
                            if i == hero {
                                continue;
                            }
                            g.paint(painter, play, ghost_alpha);
                        }
                        self.games[hero].paint(painter, play, 255);
                        painter.rect_stroke(
                            play,
                            0.0,
                            Stroke::new(1.5, Color32::from_rgb(230, 210, 90)),
                            egui::StrokeKind::Inside,
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
                ui.label(format!("envs            {}", self.games.len()));
                ui.label(format!("wall time       {:7.1} s", wall));
                ui.label(format!(
                    "epsilon         {:7.3}",
                    self.agent.current_epsilon(wall)
                ));
                ui.label(format!(
                    "replay / cap    {:>6} / {}",
                    self.agent.replay_len(),
                    self.config.agent.replay_capacity
                ));
                ui.label(format!("grad steps      {:>7}", self.agent.gradient_steps));
                ui.label(format!("inferences      {:>7}", self.agent.inferences));
                ui.label(format!("last loss       {:>7.4}", self.agent.last_loss));
                if !self.episode_return_hist.is_empty() {
                    ui.label(format!(
                        "mean return     {:>7.2}",
                        self.episode_return_hist.mean()
                    ));
                }

                ui.separator();
                ui.add(
                    egui::Slider::new(&mut self.speed_mul, 1..=32)
                        .text("speed")
                        .integer(),
                );
                ui.horizontal(|ui| {
                    ui.label("view [V]");
                    ui.selectable_value(&mut self.view_mode, ViewMode::Grid, "grid");
                    ui.selectable_value(&mut self.view_mode, ViewMode::Overlay, "overlay");
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

                ui.separator();
                ui.label("training loss");
                let (loss_rect, _) =
                    ui.allocate_exact_size(Vec2::new(260.0, 80.0), egui::Sense::hover());
                draw_plot(
                    ui.painter(),
                    loss_rect,
                    &self.loss_hist,
                    Color32::from_rgb(220, 90, 80),
                    true,
                );

                ui.label("episode return (200)");
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

                ui.separator();
                if ui.button("reset [R]").clicked() {
                    self.reset_learning();
                }
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
        let frame = self.surface.acquire_frame();
        let frame_view = frame.texture_view();
        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        self.gui_painter
            .update_textures(&mut self.command_encoder, gui_textures, &self.gpu);

        {
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
        let sync_point = self.gpu.submit(&mut self.command_encoder);
        self.gui_painter.after_submit(&sync_point);

        if let Some(sp) = self.prev_sync_point.take() {
            let _ = self.gpu.wait_for(&sp, !0);
        }
        self.prev_sync_point = Some(sync_point);
    }

    fn destroy(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            let _ = self.gpu.wait_for(&sp, !0);
        }
        self.gpu.destroy_command_encoder(&mut self.command_encoder);
        self.gui_painter.destroy(&self.gpu);
        self.gpu.destroy_surface(&mut self.surface);
    }
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
    // Inverse proportional to sqrt(N) with a reasonable floor — 4
    // envs read at ~80, 16 envs read at ~40, 64 envs read at ~20.
    let n = (n_envs as f32).max(1.0);
    ((160.0 / n.sqrt()).clamp(16.0, 200.0)) as u8
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
