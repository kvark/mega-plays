//! Catch for mega-plays: one paddle, one falling ball, one second per
//! episode.
//!
//! This is the fastest-feedback game in the crate and exists for that
//! reason. Pong needs a rally before anything is decided and the lander
//! needs a whole descent; catch resolves +1 / -1 about once a second per
//! environment, so a 16-env grid produces ~15 labelled episodes every
//! second and the win-rate curve moves while you are still reading the
//! overlay.
//!
//! All coordinates are in play-area units: width 1.6, height 1.0, origin
//! at the centre. The paddle slides along the bottom; the ball spawns at
//! the top with a random horizontal position and drift, bounces off the
//! side walls, and the episode ends the moment it reaches paddle height.
//!
//! Observation is five floats, normalised to roughly `[-1, 1]`:
//!
//! - paddle x, ball x, ball y,
//! - ball vx, ball vy.
//!
//! Actions are three discrete values: 0 = stay, 1 = left, 2 = right.
//!
//! Reward is `+1` for a catch, `-1` for a miss, plus small potential-based
//! shaping for closing the horizontal gap — the same trick pong uses, and
//! the reason the policy starts tracking the ball before it has ever
//! caught one.

use egui::{Color32, CornerRadius, Painter, Pos2, Rect, Stroke, Vec2};

use crate::{
    agent::{Action, Observation},
    game::{Game, GameSpec, StepOutcome},
};

pub const OBS_DIM: usize = 5;
pub const NUM_ACTIONS: u32 = 3;
pub const PHYSICS_DT: f32 = 1.0 / 120.0;

pub const PLAY_WIDTH: f32 = 1.6;
pub const PLAY_HEIGHT: f32 = 1.0;

pub const PADDLE_HALF_W: f32 = 0.12;
pub const PADDLE_HEIGHT: f32 = 0.04;
pub const PADDLE_Y: f32 = -0.42;
pub const PADDLE_SPEED: f32 = 1.3;
pub const BALL_RADIUS: f32 = 0.025;

/// Vertical fall speed at difficulty 1. One episode is the ball
/// covering ~0.9 units, i.e. about 1.2 s.
pub const FALL_SPEED: f32 = 0.75;
/// Largest horizontal drift the ball spawns with, at difficulty 1.
pub const DRIFT_SPEED: f32 = 0.35;

pub const TERMINAL_REWARD: f32 = 1.0;
/// Weight of the potential-based shaping on the horizontal gap. Small
/// next to the ±1 terminal, but it is what gives the policy a gradient
/// during the first seconds, before any catch has landed in replay.
pub const SHAPING_WEIGHT: f32 = 0.1;

pub struct CatchGame {
    paddle_x: f32,
    ball: Pos2,
    ball_vel: Vec2,
    catches: u32,
    misses: u32,
    /// Curriculum knob: scales fall speed and drift together. 1.0 is
    /// the design difficulty; the auto-curriculum drives it.
    difficulty: f32,
    /// Previous normalised horizontal gap, for potential-based shaping.
    prev_gap: f32,
    rng: rand::rngs::StdRng,
}

impl CatchGame {
    pub fn new() -> Self {
        let mut g = Self {
            paddle_x: 0.0,
            ball: Pos2::ZERO,
            ball_vel: Vec2::ZERO,
            catches: 0,
            misses: 0,
            difficulty: 1.0,
            prev_gap: 0.0,
            rng: crate::seeded_rng(),
        };
        g.spawn_ball();
        g
    }

    pub fn catches(&self) -> u32 {
        self.catches
    }

    pub fn misses(&self) -> u32 {
        self.misses
    }

    fn spawn_ball(&mut self) {
        use rand::RngExt;
        let half_w = PLAY_WIDTH * 0.5;
        self.ball = Pos2::new(
            self.rng.random_range(-half_w * 0.75..half_w * 0.75),
            PLAY_HEIGHT * 0.5 - BALL_RADIUS,
        );
        let drift = DRIFT_SPEED * self.difficulty;
        self.ball_vel = Vec2::new(
            self.rng.random_range(-drift..drift),
            -FALL_SPEED * self.difficulty,
        );
        self.prev_gap = self.gap();
    }

    /// Horizontal paddle-ball gap, normalised so 1.0 is the width of
    /// the play area.
    fn gap(&self) -> f32 {
        (self.ball.x - self.paddle_x).abs() / PLAY_WIDTH
    }
}

impl Default for CatchGame {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for CatchGame {
    fn spec(&self) -> GameSpec {
        GameSpec {
            title: "mega-catch",
            obs_dim: OBS_DIM,
            num_actions: NUM_ACTIONS,
            physics_dt: PHYSICS_DT,
            play_area: [PLAY_WIDTH, PLAY_HEIGHT],
        }
    }

    fn reset(&mut self) {
        self.paddle_x = 0.0;
        self.spawn_ball();
    }

    fn step(&mut self, action: Action) -> StepOutcome {
        let dt = PHYSICS_DT;

        let dx = match action {
            1 => -PADDLE_SPEED,
            2 => PADDLE_SPEED,
            _ => 0.0,
        };
        let limit = PLAY_WIDTH * 0.5 - PADDLE_HALF_W;
        self.paddle_x = (self.paddle_x + dx * dt).clamp(-limit, limit);

        self.ball += self.ball_vel * dt;
        // Side walls: reflect, so the ball always stays reachable and
        // the drift stays interesting instead of leaving the field.
        let wall = PLAY_WIDTH * 0.5 - BALL_RADIUS;
        if self.ball.x > wall {
            self.ball.x = wall;
            self.ball_vel.x = -self.ball_vel.x.abs();
        } else if self.ball.x < -wall {
            self.ball.x = -wall;
            self.ball_vel.x = self.ball_vel.x.abs();
        }

        // The episode is decided the moment the ball reaches paddle
        // height: caught if the paddle covers it, missed otherwise.
        let paddle_top = PADDLE_Y + PADDLE_HEIGHT * 0.5;
        let mut terminal_r = 0.0;
        let mut done = false;
        if self.ball.y - BALL_RADIUS <= paddle_top {
            let caught = (self.ball.x - self.paddle_x).abs() <= PADDLE_HALF_W + BALL_RADIUS;
            if caught {
                self.catches += 1;
                terminal_r = TERMINAL_REWARD;
            } else {
                self.misses += 1;
                terminal_r = -TERMINAL_REWARD;
            }
            done = true;
        }

        // Potential-based shaping on the horizontal gap: closing on the
        // ball earns, drifting off costs, and standing still is free.
        let gap = self.gap();
        let shaping = SHAPING_WEIGHT * (self.prev_gap - gap);
        self.prev_gap = gap;

        StepOutcome {
            reward: terminal_r + shaping,
            done,
            truncated: false,
            terminal_reward: terminal_r,
        }
    }

    fn observation(&self) -> Observation {
        let half_w = PLAY_WIDTH * 0.5;
        let half_h = PLAY_HEIGHT * 0.5;
        vec![
            self.paddle_x / half_w,
            self.ball.x / half_w,
            self.ball.y / half_h,
            self.ball_vel.x / DRIFT_SPEED,
            self.ball_vel.y / FALL_SPEED,
        ]
    }

    fn paint(&self, painter: &Painter, rect: Rect, alpha: u8) {
        let (sx, sy) = (rect.width() / PLAY_WIDTH, rect.height() / PLAY_HEIGHT);
        let cx = rect.center().x;
        let cy = rect.center().y;
        let to_screen = |p: Pos2| Pos2::new(cx + p.x * sx, cy - p.y * sy);

        if alpha == 255 {
            painter.rect_filled(rect, 0.0, Color32::from_rgb(14, 18, 24));
            // Floor line the ball is racing toward.
            let y = to_screen(Pos2::new(0.0, PADDLE_Y - PADDLE_HEIGHT)).y;
            painter.line_segment(
                [Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)],
                Stroke::new(1.0_f32, Color32::from_rgb(60, 70, 88)),
            );
        }

        let paddle_tl = to_screen(Pos2::new(
            self.paddle_x - PADDLE_HALF_W,
            PADDLE_Y + PADDLE_HEIGHT * 0.5,
        ));
        let paddle_br = to_screen(Pos2::new(
            self.paddle_x + PADDLE_HALF_W,
            PADDLE_Y - PADDLE_HEIGHT * 0.5,
        ));
        painter.rect_filled(
            Rect::from_two_pos(paddle_tl, paddle_br),
            CornerRadius::ZERO,
            crate::tint(Color32::from_rgb(160, 200, 255), alpha),
        );

        painter.circle_filled(
            to_screen(self.ball),
            BALL_RADIUS * sx.min(sy),
            crate::tint(Color32::from_rgb(240, 210, 120), alpha),
        );

        if alpha == 255 {
            painter.text(
                Pos2::new(cx, rect.min.y + 6.0),
                egui::Align2::CENTER_TOP,
                format!("{} / {}", self.catches, self.catches + self.misses),
                egui::FontId::monospace(16.0),
                Color32::from_rgb(200, 210, 225),
            );
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        let total = self.catches + self.misses;
        ui.label(format!(
            "caught          {} / {} ({:.0}%)",
            self.catches,
            total,
            if total > 0 {
                100.0 * self.catches as f32 / total as f32
            } else {
                0.0
            },
        ));
        ui.add(egui::Slider::new(&mut self.difficulty, 0.3..=3.0).text("ball speed"));
    }

    fn sync_settings(&mut self, source: &Self) {
        self.difficulty = source.difficulty;
    }

    fn difficulty(&self) -> f32 {
        self.difficulty
    }

    fn set_difficulty(&mut self, level: f32) {
        self.difficulty = level.clamp(0.3, 3.0);
    }
}
