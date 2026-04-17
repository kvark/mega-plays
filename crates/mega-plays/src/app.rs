//! Winit driver: window, Blade context, egui overlay, training loop.
//!
//! The driver is generic over any [`Game`]. Rendering is done entirely
//! through egui — no custom Blade pipeline. That means:
//!
//! - no WGSL shaders to ship,
//! - no MSAA resolve pass (egui tessellates with AA fringe),
//! - no text-shaping dependency; egui's bundled fonts render our stats.
//!
//! Trade-off: we pay for every on-screen primitive through egui's
//! tessellator. For tens of primitives (Pong, simple arenas) this is
//! nothing. A game that needs thousands of sprites per frame should add
//! a sibling crate with a direct Blade pipeline.

use std::sync::Arc;
use std::time::Instant;

use blade_graphics as gpu;
use egui::{Color32, Rect};

use crate::agent::{Agent, AgentConfig, Transition};
use crate::game::{Game, GameSpec};
use crate::stats::{RollingStats, SparkLine};

/// Top-level knobs for [`run`]. The [`AgentConfig`] comes from the game
/// wiring; everything else governs the host harness.
pub struct AppConfig {
    pub agent: AgentConfig,
    /// How many minibatch gradient steps to run per rendered frame. At
    /// 60 Hz render, `train_steps_per_frame = 2` gives ~120 Hz training.
    pub train_steps_per_frame: u32,
    /// Clear color for the game surface.
    pub clear_color: Color32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            agent: AgentConfig::default(),
            train_steps_per_frame: 2,
            clear_color: Color32::from_rgb(8, 10, 14),
        }
    }
}

/// Run the driver until the window is closed.
///
/// The game is constructed in place via `build_game` once the Blade
/// context exists — this lets games pre-allocate GPU resources on the
/// shared context if they need to. For the initial pong build, the
/// callback ignores the context and just constructs its physics state.
pub fn run<G, F>(config: AppConfig, build_game: F)
where
    G: Game + 'static,
    F: FnOnce(Arc<gpu::Context>) -> G + 'static,
{
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().expect("event loop");
    let mut app = App::<G, F> {
        state: AppState::Uninit {
            build_game: Some(build_game),
            config,
        },
    };
    event_loop.run_app(&mut app).expect("run app");
    if let AppState::Running(mut running) = app.state {
        running.destroy();
    }
}

enum AppState<G, F>
where
    G: Game + 'static,
    F: FnOnce(Arc<gpu::Context>) -> G,
{
    Uninit {
        build_game: Option<F>,
        config: AppConfig,
    },
    Running(Running<G>),
}

struct App<G, F>
where
    G: Game + 'static,
    F: FnOnce(Arc<gpu::Context>) -> G,
{
    state: AppState<G, F>,
}

struct Running<G: Game> {
    gpu: Arc<gpu::Context>,
    surface: gpu::Surface,
    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,
    gui_painter: blade_egui::GuiPainter,

    window: winit::window::Window,
    egui_winit: egui_winit::State,

    game: G,
    spec: GameSpec,
    agent: Agent,
    config: AppConfig,

    last_obs: Option<Vec<f32>>,
    last_action: u32,

    loss_hist: RollingStats,
    reward_hist: RollingStats,
    start_time: Instant,
    last_frame: Instant,
    physics_accum: f32,

    paused: bool,
    show_overlay: bool,
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

impl<G, F> winit::application::ApplicationHandler for App<G, F>
where
    G: Game + 'static,
    F: FnOnce(Arc<gpu::Context>) -> G + 'static,
{
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let (build_game, config) = match std::mem::replace(
            &mut self.state,
            AppState::Uninit {
                build_game: None,
                config: AppConfig::default(),
            },
        ) {
            AppState::Uninit { build_game, config } => (build_game, config),
            s @ AppState::Running(_) => {
                self.state = s;
                return;
            }
        };
        let build_game = build_game.expect("resumed called twice without a game builder");

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

        let game = build_game(gpu.clone());
        let spec = game.spec();

        let window_attributes = winit::window::Window::default_attributes()
            .with_title(spec.title)
            .with_inner_size(winit::dpi::LogicalSize::new(960.0, 640.0));
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
        let egui_winit =
            egui_winit::State::new(egui_ctx, viewport_id, &window, None, None, None);
        let _ = viewport_id;

        let agent = Agent::new(
            gpu.clone(),
            spec.obs_dim,
            spec.num_actions,
            config.agent.clone(),
        );

        let now = Instant::now();
        self.state = AppState::Running(Running {
            gpu,
            surface,
            command_encoder,
            prev_sync_point: None,
            gui_painter,
            window,
            egui_winit,
            game,
            spec,
            agent,
            config,
            last_obs: None,
            last_action: 0,
            loss_hist: RollingStats::new(240),
            reward_hist: RollingStats::new(240),
            start_time: now,
            last_frame: now,
            physics_accum: 0.0,
            paused: false,
            show_overlay: true,
        });
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
        let r = match &mut self.state {
            AppState::Running(r) => r,
            _ => return,
        };

        let response = r.egui_winit.on_window_event(&r.window, &event);
        if response.repaint {
            r.window.request_redraw();
        }
        if response.consumed {
            return;
        }

        use winit::event::{ElementState, KeyEvent, WindowEvent};
        use winit::keyboard::{KeyCode, PhysicalKey};

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
                KeyCode::KeyR => r.reset_learning(),
                _ => {}
            },
            WindowEvent::RedrawRequested => r.redraw(event_loop),
            _ => {}
        }
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
        // Rebuild the agent from scratch. The existing sessions' GPU
        // buffers will be released when they drop; the Blade context
        // stays alive via our Arc.
        let cfg = self.config.agent.clone();
        self.agent = Agent::new(self.gpu.clone(), self.spec.obs_dim, self.spec.num_actions, cfg);
        self.loss_hist = RollingStats::new(240);
        self.reward_hist = RollingStats::new(240);
        self.game.reset();
        self.last_obs = None;
    }

    fn tick(&mut self) {
        if self.paused {
            return;
        }
        let wall = self.start_time.elapsed().as_secs_f32();
        let now = Instant::now();
        let mut dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.physics_accum += dt;

        while self.physics_accum >= self.spec.physics_dt {
            self.physics_accum -= self.spec.physics_dt;
            let obs = self.game.observation();
            let action = self.agent.select_action(&obs, wall);
            let outcome = self.game.step(action);
            let next_obs = self.game.observation();

            if let Some(prev) = self.last_obs.replace(next_obs.clone()) {
                self.agent.record(Transition {
                    obs: prev,
                    action: self.last_action,
                    reward: outcome.reward,
                    next_obs: next_obs.clone(),
                    done: outcome.done,
                });
            }
            self.last_action = action;
            self.reward_hist.push(outcome.reward);

            if outcome.done {
                self.game.reset();
                self.last_obs = None;
            }

            dt = 0.0; // silence the unused-tail warning in optimised builds
            let _ = dt;
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

    fn build_ui(
        &mut self,
        ctx: &egui::Context,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) {
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(
                        self.config.clear_color.r(),
                        self.config.clear_color.g(),
                        self.config.clear_color.b(),
                    ))
                    .inner_margin(0.0),
            )
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                let play = self.fit_play_rect(rect);
                self.game.paint(ui.painter(), play);
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

                ui.separator();
                ui.label("loss (rolling 240)");
                let (loss_rect, _) = ui
                    .allocate_exact_size(egui::vec2(220.0, 40.0), egui::Sense::hover());
                SparkLine::draw(
                    ui.painter(),
                    loss_rect,
                    &self.loss_hist,
                    egui::Color32::from_rgb(220, 80, 80),
                );

                ui.label("reward (rolling 240)");
                let (rew_rect, _) = ui
                    .allocate_exact_size(egui::vec2(220.0, 40.0), egui::Sense::hover());
                SparkLine::draw(
                    ui.painter(),
                    rew_rect,
                    &self.reward_hist,
                    egui::Color32::from_rgb(80, 220, 120),
                );

                ui.separator();
                self.game.ui(ui);

                ui.separator();
                if ui.button("reset [R]").clicked() {
                    self.reset_learning();
                }
                if ui.button("quit [Esc]").clicked() {
                    event_loop.exit();
                }
            });
    }

    fn fit_play_rect(&self, avail: Rect) -> Rect {
        let aspect = self.spec.play_area[0] / self.spec.play_area[1];
        let avail_aspect = avail.width() / avail.height();
        if avail_aspect > aspect {
            let w = avail.height() * aspect;
            let x0 = avail.center().x - w * 0.5;
            Rect::from_min_size(
                egui::pos2(x0, avail.min.y),
                egui::vec2(w, avail.height()),
            )
        } else {
            let h = avail.width() / aspect;
            let y0 = avail.center().y - h * 0.5;
            Rect::from_min_size(
                egui::pos2(avail.min.x, y0),
                egui::vec2(avail.width(), h),
            )
        }
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
