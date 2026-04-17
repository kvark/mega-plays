//! Lunar lander for mega-plays.
//!
//! One rigid body in a constant-gravity field with three discrete
//! thrusters (main, left RCS, right RCS) plus idle. Physics is pure
//! semi-implicit Euler — no physics engine, no contacts beyond a
//! horizontal ground plane. That keeps the crate dependency-free
//! beyond what pong already pulls in.
//!
//! World coordinates: `x ∈ [-1, 1]` horizontal, `y ∈ [-1, 1]` vertical
//! with `y = -1` the ground plane. A small landing pad sits at the
//! centre of the ground. The lander spawns at a random horizontal
//! position near the top with a small sideways nudge.
//!
//! Observation (7 floats, roughly `[-1, 1]`):
//!
//! - position x, position y,
//! - velocity x, velocity y,
//! - `sin(angle)`, `cos(angle)` (avoids the `±π` discontinuity),
//! - angular velocity / ANG_VEL_SCALE.
//!
//! Actions (4 discrete):
//!
//! - 0 — idle
//! - 1 — main engine (thrust along the lander's "up" axis)
//! - 2 — left RCS   (torque CCW: rotates the lander counter-clockwise)
//! - 3 — right RCS  (torque CW)
//!
//! Terminal reward:
//!
//! - `+TERMINAL_REWARD` for a soft landing on the pad (upright,
//!   velocity within thresholds, horizontal position inside the pad);
//! - `+PARTIAL_LANDING_REWARD` for a soft landing anywhere else on
//!   the ground;
//! - `-TERMINAL_REWARD` for a crash (touching ground with too much
//!   velocity or tilt) or going out of the horizontal bounds.
//!
//! Plus dense shaping per step: penalise distance to the pad, tilt,
//! and fuel usage. The shaping is kept small so the sparse ±10 signal
//! still dominates the Bellman target — same reasoning as pong.

use egui::{Color32, CornerRadius, Painter, Pos2, Rect, Stroke, Vec2};

use crate::{
    agent::{Action, Observation},
    game::{Game, GameSpec, StepOutcome},
};

pub const OBS_DIM: usize = 7;
pub const NUM_ACTIONS: u32 = 4;
pub const PHYSICS_DT: f32 = 1.0 / 120.0;

pub const PLAY_WIDTH: f32 = 2.0;
pub const PLAY_HEIGHT: f32 = 2.0;

pub const GRAVITY: f32 = 0.8;
/// Main engine acceleration when firing (world units / s²). Above `GRAVITY`
/// so the lander can hover or climb.
pub const MAIN_THRUST: f32 = 1.6;
/// Rotational acceleration from one RCS thruster (rad / s²).
pub const RCS_TORQUE: f32 = 4.0;
/// Angular velocity scale used to normalise the observation.
pub const ANG_VEL_SCALE: f32 = 2.0;

pub const BODY_HALF_W: f32 = 0.05;
pub const BODY_HALF_H: f32 = 0.06;
pub const PAD_HALF_W: f32 = 0.15;
pub const GROUND_Y: f32 = -1.0;

pub const SOFT_VEL_X: f32 = 0.30;
pub const SOFT_VEL_Y: f32 = 0.40;
pub const SOFT_TILT_COS: f32 = 0.95; // cos(angle) >= 0.95 ~ |angle| < ~18°

pub const TERMINAL_REWARD: f32 = 10.0;
pub const PARTIAL_LANDING_REWARD: f32 = 2.0;

/// Per-step shaping weights. Kept small so the sparse terminal signal
/// still dominates the Bellman target.
pub const SHAPE_DIST: f32 = 0.01;
pub const SHAPE_TILT: f32 = 0.01;
pub const SHAPE_FUEL: f32 = 0.003;

pub struct LanderGame {
    pos: Vec2,
    vel: Vec2,
    angle: f32,
    ang_vel: f32,
    last_thrusting: bool,
    rng: rand::rngs::ThreadRng,

    landings: u32,
    crashes: u32,
    partials: u32,
}

impl LanderGame {
    pub fn new() -> Self {
        let mut g = Self {
            pos: Vec2::ZERO,
            vel: Vec2::ZERO,
            angle: 0.0,
            ang_vel: 0.0,
            last_thrusting: false,
            rng: rand::rng(),
            landings: 0,
            crashes: 0,
            partials: 0,
        };
        g.spawn();
        g
    }

    fn spawn(&mut self) {
        use rand::Rng;
        self.pos = Vec2::new(self.rng.random_range(-0.7..0.7), 0.85);
        self.vel = Vec2::new(self.rng.random_range(-0.2..0.2), 0.0);
        self.angle = self.rng.random_range(-0.2..0.2);
        self.ang_vel = 0.0;
        self.last_thrusting = false;
    }

    fn thrust_vec(&self) -> Vec2 {
        // Craft "up" axis: (0, 1) rotated by `angle` CCW.
        // `angle = 0` means the lander points straight up.
        Vec2::new(-self.angle.sin(), self.angle.cos()) * MAIN_THRUST
    }
}

impl Default for LanderGame {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
enum Landing {
    None,
    SoftOnPad,
    SoftElsewhere,
    Crash,
    OutOfBounds,
}

impl LanderGame {
    fn classify_ground_contact(&self) -> Landing {
        if self.pos.x.abs() > PLAY_WIDTH * 0.5 {
            return Landing::OutOfBounds;
        }
        let touching = self.pos.y - BODY_HALF_H <= GROUND_Y;
        if !touching {
            return Landing::None;
        }
        let upright = self.angle.cos() >= SOFT_TILT_COS;
        let slow = self.vel.x.abs() <= SOFT_VEL_X && self.vel.y.abs() <= SOFT_VEL_Y;
        if upright && slow {
            if self.pos.x.abs() <= PAD_HALF_W {
                Landing::SoftOnPad
            } else {
                Landing::SoftElsewhere
            }
        } else {
            Landing::Crash
        }
    }
}

impl Game for LanderGame {
    fn spec(&self) -> GameSpec {
        GameSpec {
            title: "mega-lander",
            obs_dim: OBS_DIM,
            num_actions: NUM_ACTIONS,
            physics_dt: PHYSICS_DT,
            play_area: [PLAY_WIDTH, PLAY_HEIGHT],
        }
    }

    fn reset(&mut self) {
        self.spawn();
    }

    fn step(&mut self, action: Action) -> StepOutcome {
        let dt = PHYSICS_DT;

        let thrusting = action == 1;
        let torque = match action {
            2 => RCS_TORQUE,
            3 => -RCS_TORQUE,
            _ => 0.0,
        };
        self.last_thrusting = thrusting;

        // Semi-implicit Euler: update velocity first with the accelerations
        // from this step's inputs, then advance position with the new
        // velocity. More energy-stable than explicit Euler for the same
        // cost.
        let mut accel = Vec2::new(0.0, -GRAVITY);
        if thrusting {
            accel += self.thrust_vec();
        }
        self.vel += accel * dt;
        self.ang_vel += torque * dt;
        self.pos += self.vel * dt;
        self.angle += self.ang_vel * dt;

        let landing = self.classify_ground_contact();
        let (terminal_r, done) = match landing {
            Landing::SoftOnPad => {
                self.landings += 1;
                (TERMINAL_REWARD, true)
            }
            Landing::SoftElsewhere => {
                self.partials += 1;
                (PARTIAL_LANDING_REWARD, true)
            }
            Landing::Crash | Landing::OutOfBounds => {
                self.crashes += 1;
                (-TERMINAL_REWARD, true)
            }
            Landing::None => (0.0, false),
        };

        // Dense shaping: distance to pad, tilt, fuel. Small compared
        // with the ±10 terminal so the sparse signal still drives
        // learning.
        let dist = (self.pos - Vec2::new(0.0, GROUND_Y)).length();
        let tilt = 1.0 - self.angle.cos().max(0.0);
        let fuel = if thrusting { 1.0 } else { 0.0 };
        let shaping = -SHAPE_DIST * dist - SHAPE_TILT * tilt - SHAPE_FUEL * fuel;

        if done {
            self.spawn();
        }

        StepOutcome {
            reward: terminal_r + shaping,
            done,
            terminal_reward: terminal_r,
        }
    }

    fn observation(&self) -> Observation {
        vec![
            self.pos.x,
            self.pos.y,
            self.vel.x,
            self.vel.y,
            self.angle.sin(),
            self.angle.cos(),
            self.ang_vel / ANG_VEL_SCALE,
        ]
    }

    fn paint(&self, painter: &Painter, rect: Rect, alpha: u8) {
        let (sx, sy) = (rect.width() / PLAY_WIDTH, rect.height() / PLAY_HEIGHT);
        let cx = rect.center().x;
        let cy = rect.center().y;
        let to_screen = |p: Vec2| Pos2::new(cx + p.x * sx, cy - p.y * sy);
        let tint = |c: Color32| crate::tint(c, alpha);

        // Backdrop + static scenery are only drawn on the opaque
        // pass. In the alpha-blended overlay view, 16 ghost passes
        // would otherwise re-stamp the sky and drown out the ghosts.
        if alpha == 255 {
            painter.rect_filled(rect, 0.0, Color32::from_rgb(12, 16, 28));
            for i in 0..24_u32 {
                let h = ((i.wrapping_mul(2_654_435_761)) ^ 0xdead_beef) as f32;
                let sx_ = ((h * 0.000_000_12).fract() - 0.5) * PLAY_WIDTH;
                let sy_ = (((h * 0.000_000_73).fract()) - 0.2) * 0.9;
                painter.circle_filled(
                    to_screen(Vec2::new(sx_, sy_.max(-0.3))),
                    1.0,
                    Color32::from_gray(80 + ((i as u8) & 0x3f)),
                );
            }
            let ground_y_screen = to_screen(Vec2::new(0.0, GROUND_Y)).y;
            painter.line_segment(
                [
                    Pos2::new(rect.min.x, ground_y_screen),
                    Pos2::new(rect.max.x, ground_y_screen),
                ],
                Stroke::new(1.5, Color32::from_gray(110)),
            );
            let pad_top_left = to_screen(Vec2::new(-PAD_HALF_W, GROUND_Y + 0.02));
            let pad_bot_right = to_screen(Vec2::new(PAD_HALF_W, GROUND_Y));
            painter.rect_filled(
                Rect::from_two_pos(pad_top_left, pad_bot_right),
                CornerRadius::ZERO,
                Color32::from_rgb(210, 200, 90),
            );
        }

        // Lander: body triangle + landing legs, rotated by `angle`.
        let rot = |p: Vec2| {
            let (s, c) = self.angle.sin_cos();
            Vec2::new(p.x * c - p.y * s, p.x * s + p.y * c)
        };
        let local_verts = [
            Vec2::new(0.0, BODY_HALF_H),
            Vec2::new(-BODY_HALF_W, -BODY_HALF_H * 0.6),
            Vec2::new(BODY_HALF_W, -BODY_HALF_H * 0.6),
        ];
        let verts: Vec<Pos2> = local_verts
            .iter()
            .map(|&v| to_screen(self.pos + rot(v)))
            .collect();
        painter.add(egui::Shape::convex_polygon(
            verts,
            tint(Color32::from_rgb(200, 210, 220)),
            Stroke::new(1.0, tint(Color32::from_gray(40))),
        ));
        let leg_l = to_screen(self.pos + rot(Vec2::new(-BODY_HALF_W * 1.2, -BODY_HALF_H)));
        let leg_r = to_screen(self.pos + rot(Vec2::new(BODY_HALF_W * 1.2, -BODY_HALF_H)));
        let leg_anchor_l = to_screen(self.pos + rot(Vec2::new(-BODY_HALF_W, -BODY_HALF_H * 0.6)));
        let leg_anchor_r = to_screen(self.pos + rot(Vec2::new(BODY_HALF_W, -BODY_HALF_H * 0.6)));
        let leg_stroke = Stroke::new(1.5, tint(Color32::from_gray(180)));
        painter.line_segment([leg_anchor_l, leg_l], leg_stroke);
        painter.line_segment([leg_anchor_r, leg_r], leg_stroke);

        if self.last_thrusting {
            let flame_local = [
                Vec2::new(-BODY_HALF_W * 0.5, -BODY_HALF_H * 0.6),
                Vec2::new(BODY_HALF_W * 0.5, -BODY_HALF_H * 0.6),
                Vec2::new(0.0, -BODY_HALF_H * 1.8),
            ];
            let flame_verts: Vec<Pos2> = flame_local
                .iter()
                .map(|&v| to_screen(self.pos + rot(v)))
                .collect();
            painter.add(egui::Shape::convex_polygon(
                flame_verts,
                tint(Color32::from_rgb(255, 160, 40)),
                Stroke::NONE,
            ));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label(format!(
            "landings        {:>5}  (pad={}, off={}, crash={})",
            self.landings + self.partials,
            self.landings,
            self.partials,
            self.crashes,
        ));
        let total = self.landings + self.partials + self.crashes;
        if total > 0 {
            ui.label(format!(
                "pad-rate        {:>5.1}%",
                100.0 * self.landings as f32 / total as f32,
            ));
        }
    }
}
