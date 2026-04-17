//! Pong for mega-plays.
//!
//! All coordinates are in play-area units: width = 2.0, height = 1.0,
//! origin at the centre. The paddle along `x = -1` is the *agent* paddle
//! (the one the DQN controls); the paddle at `x = +1` is the opponent,
//! driven by a scripted tracker with a small delay and noise. A mirrored
//! self-play setup is a straightforward future extension.
//!
//! The observation is six floats, normalised to roughly `[-1, 1]`:
//!
//! - agent paddle y,
//! - opponent paddle y,
//! - ball x, ball y,
//! - ball vx, ball vy.
//!
//! Actions are three discrete values: 0 = stay, 1 = up, 2 = down.
//!
//! Reward is +1 when the opponent misses (agent scores), -1 when the
//! agent misses, and 0 elsewhere. The episode ends on either score.

use egui::{Color32, CornerRadius, Painter, Pos2, Rect, Stroke, Vec2};

use crate::{
    agent::{Action, Observation},
    game::{Game, GameSpec, StepOutcome},
};

pub const OBS_DIM: usize = 6;
pub const NUM_ACTIONS: u32 = 3;
pub const PADDLE_HEIGHT: f32 = 0.18;
pub const PADDLE_WIDTH: f32 = 0.02;
pub const BALL_RADIUS: f32 = 0.02;
pub const PADDLE_SPEED: f32 = 1.6;
pub const INITIAL_BALL_SPEED: f32 = 1.2;
pub const PHYSICS_DT: f32 = 1.0 / 120.0;
pub const PLAY_WIDTH: f32 = 2.0;
pub const PLAY_HEIGHT: f32 = 1.0;

/// Scripted opponent speed, relative to the agent. Below 1.0 makes it
/// trackable. Default 0.25 lets a random policy score ~1 in 4 episodes,
/// which is enough density of +1 rewards for DQN to learn quickly.
/// Crank it up via the in-UI slider once the agent is competent.
pub const OPPONENT_SPEED_FRACTION: f32 = 0.25;

/// Terminal reward magnitude. Scoring is `±TERMINAL_REWARD`. Scaling
/// above ±1 makes the sparse terminal signal dominate the mean-squared
/// Bellman loss against an otherwise-zero batch of non-terminal
/// transitions. DQN learning speed on small networks is very
/// sensitive to this.
pub const TERMINAL_REWARD: f32 = 10.0;

/// Small dense reward for aligning the agent paddle with the ball.
/// Provides a non-terminal gradient signal so the policy has something
/// to optimise before the first terminal reward lands in replay.
pub const SHAPING_WEIGHT: f32 = 0.05;

pub struct PongGame {
    agent_y: f32,
    opponent_y: f32,
    ball: Pos2,
    ball_vel: Vec2,
    score_agent: u32,
    score_opponent: u32,
    opponent_noise: f32,
    opponent_speed_frac: f32,
    shaping_weight: f32,
    rng: rand::rngs::ThreadRng,
}

impl PongGame {
    pub fn new() -> Self {
        let mut g = Self {
            agent_y: 0.0,
            opponent_y: 0.0,
            ball: Pos2::ZERO,
            ball_vel: Vec2::ZERO,
            score_agent: 0,
            score_opponent: 0,
            opponent_noise: 0.08,
            opponent_speed_frac: OPPONENT_SPEED_FRACTION,
            shaping_weight: SHAPING_WEIGHT,
            rng: rand::rng(),
        };
        g.reset_ball(true);
        g
    }

    fn reset_ball(&mut self, toward_agent: bool) {
        use rand::Rng;
        self.ball = Pos2::ZERO;
        let angle: f32 = self.rng.random_range(-0.3..0.3);
        let direction: f32 = if toward_agent { -1.0 } else { 1.0 };
        let speed = INITIAL_BALL_SPEED;
        self.ball_vel = Vec2::new(direction * speed * angle.cos(), speed * angle.sin());
    }

    fn move_paddle(y: f32, action: Action, dt: f32) -> f32 {
        let d = match action {
            1 => PADDLE_SPEED,
            2 => -PADDLE_SPEED,
            _ => 0.0,
        };
        let half = PLAY_HEIGHT * 0.5 - PADDLE_HEIGHT * 0.5;
        (y + d * dt).clamp(-half, half)
    }

    fn update_opponent(&mut self, dt: f32) {
        use rand::Rng;
        let target = self.ball.y + self.rng.random_range(-self.opponent_noise..self.opponent_noise);
        let diff = target - self.opponent_y;
        let max_step = PADDLE_SPEED * self.opponent_speed_frac * dt;
        let step = diff.clamp(-max_step, max_step);
        let half = PLAY_HEIGHT * 0.5 - PADDLE_HEIGHT * 0.5;
        self.opponent_y = (self.opponent_y + step).clamp(-half, half);
    }

    /// Reflect the ball off walls and paddles. Returns the reward for
    /// this substep (non-zero only when the ball goes past a paddle).
    fn collide(&mut self, dt: f32) -> (f32, bool) {
        self.ball += self.ball_vel * dt;

        // Top / bottom walls.
        let half_h = PLAY_HEIGHT * 0.5 - BALL_RADIUS;
        if self.ball.y > half_h {
            self.ball.y = half_h;
            self.ball_vel.y = -self.ball_vel.y.abs();
        } else if self.ball.y < -half_h {
            self.ball.y = -half_h;
            self.ball_vel.y = self.ball_vel.y.abs();
        }

        let agent_x = -PLAY_WIDTH * 0.5 + PADDLE_WIDTH;
        let opp_x = PLAY_WIDTH * 0.5 - PADDLE_WIDTH;

        // Paddle collisions. Spin based on contact offset.
        if self.ball.x < agent_x + BALL_RADIUS && self.ball_vel.x < 0.0 {
            if (self.ball.y - self.agent_y).abs() < PADDLE_HEIGHT * 0.5 + BALL_RADIUS {
                self.ball.x = agent_x + BALL_RADIUS;
                self.ball_vel.x = self.ball_vel.x.abs();
                let offset = (self.ball.y - self.agent_y) / (PADDLE_HEIGHT * 0.5);
                self.ball_vel.y += offset * 0.4;
                self.ball_vel *= 1.03;
            }
        }
        if self.ball.x > opp_x - BALL_RADIUS && self.ball_vel.x > 0.0 {
            if (self.ball.y - self.opponent_y).abs() < PADDLE_HEIGHT * 0.5 + BALL_RADIUS {
                self.ball.x = opp_x - BALL_RADIUS;
                self.ball_vel.x = -self.ball_vel.x.abs();
                let offset = (self.ball.y - self.opponent_y) / (PADDLE_HEIGHT * 0.5);
                self.ball_vel.y += offset * 0.4;
                self.ball_vel *= 1.03;
            }
        }

        // Scoring.
        if self.ball.x < -PLAY_WIDTH * 0.5 {
            self.score_opponent += 1;
            self.reset_ball(false);
            return (-TERMINAL_REWARD, true);
        }
        if self.ball.x > PLAY_WIDTH * 0.5 {
            self.score_agent += 1;
            self.reset_ball(true);
            return (TERMINAL_REWARD, true);
        }
        (0.0, false)
    }
}

impl Default for PongGame {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for PongGame {
    fn spec(&self) -> GameSpec {
        GameSpec {
            title: "mega-pong",
            obs_dim: OBS_DIM,
            num_actions: NUM_ACTIONS,
            physics_dt: PHYSICS_DT,
            play_area: [PLAY_WIDTH, PLAY_HEIGHT],
        }
    }

    fn reset(&mut self) {
        self.agent_y = 0.0;
        self.opponent_y = 0.0;
        self.reset_ball(true);
    }

    fn step(&mut self, action: Action) -> StepOutcome {
        let dt = PHYSICS_DT;
        self.agent_y = Self::move_paddle(self.agent_y, action, dt);
        self.update_opponent(dt);
        let (terminal_r, done) = self.collide(dt);
        // Dense shaping: small negative reward proportional to paddle /
        // ball y-misalignment. Pushes the agent to track the ball even
        // while epsilon-greedy random actions dominate the replay
        // buffer. Zero weight disables it.
        let alignment = (self.ball.y - self.agent_y).abs() / (PLAY_HEIGHT * 0.5);
        let shaping = -self.shaping_weight * alignment;
        StepOutcome {
            reward: terminal_r + shaping,
            done,
            terminal_reward: terminal_r,
        }
    }

    fn observation(&self) -> Observation {
        vec![
            self.agent_y / (PLAY_HEIGHT * 0.5),
            self.opponent_y / (PLAY_HEIGHT * 0.5),
            self.ball.x / (PLAY_WIDTH * 0.5),
            self.ball.y / (PLAY_HEIGHT * 0.5),
            self.ball_vel.x / INITIAL_BALL_SPEED,
            self.ball_vel.y / INITIAL_BALL_SPEED,
        ]
    }

    fn paint(&self, painter: &Painter, rect: Rect) {
        let (sx, sy) = (rect.width() / PLAY_WIDTH, rect.height() / PLAY_HEIGHT);
        let cx = rect.center().x;
        let cy = rect.center().y;
        let to_screen = |p: Pos2| Pos2::new(cx + p.x * sx, cy - p.y * sy);

        // Court backdrop + centre line.
        painter.rect_filled(rect, 0.0, Color32::from_rgb(14, 16, 22));
        let dash_len = 0.04 * sy;
        let gap = 0.03 * sy;
        let mut y = rect.min.y;
        let white = Color32::from_rgb(90, 96, 110);
        while y < rect.max.y {
            painter.line_segment(
                [Pos2::new(cx, y), Pos2::new(cx, (y + dash_len).min(rect.max.y))],
                Stroke::new(1.0, white),
            );
            y += dash_len + gap;
        }

        let paddle_color = Color32::from_rgb(220, 225, 235);
        let paddle = |painter: &Painter, x: f32, paddle_y: f32| {
            let tl = to_screen(Pos2::new(x - PADDLE_WIDTH, paddle_y + PADDLE_HEIGHT * 0.5));
            let br = to_screen(Pos2::new(x + PADDLE_WIDTH, paddle_y - PADDLE_HEIGHT * 0.5));
            let r = Rect::from_two_pos(tl, br);
            painter.rect_filled(r, CornerRadius::ZERO, paddle_color);
        };
        paddle(painter, -PLAY_WIDTH * 0.5 + PADDLE_WIDTH, self.agent_y);
        paddle(painter, PLAY_WIDTH * 0.5 - PADDLE_WIDTH, self.opponent_y);

        // Ball.
        let ball_screen = to_screen(self.ball);
        painter.rect_filled(
            Rect::from_center_size(
                ball_screen,
                Vec2::splat(BALL_RADIUS * 2.0 * sx.min(sy)),
            ),
            CornerRadius::ZERO,
            paddle_color,
        );

        // Score.
        let score_color = Color32::from_rgb(200, 210, 225);
        painter.text(
            Pos2::new(cx - rect.width() * 0.15, rect.min.y + 24.0),
            egui::Align2::CENTER_TOP,
            format!("{}", self.score_agent),
            egui::FontId::monospace(28.0),
            score_color,
        );
        painter.text(
            Pos2::new(cx + rect.width() * 0.15, rect.min.y + 24.0),
            egui::Align2::CENTER_TOP,
            format!("{}", self.score_opponent),
            egui::FontId::monospace(28.0),
            score_color,
        );
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label(format!(
            "score            {} - {}",
            self.score_agent, self.score_opponent
        ));
        ui.add(
            egui::Slider::new(&mut self.opponent_noise, 0.0..=0.3)
                .text("opponent noise"),
        );
        ui.add(
            egui::Slider::new(&mut self.opponent_speed_frac, 0.2..=1.2)
                .text("opponent speed"),
        );
        ui.add(
            egui::Slider::new(&mut self.shaping_weight, 0.0..=0.1)
                .text("shaping reward"),
        );
    }
}
