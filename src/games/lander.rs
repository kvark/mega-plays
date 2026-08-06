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
//! Rewards split into a *sparse* component that goes into the agent's
//! reward stream and a *terminal sentinel* that the harness counts for
//! the scoreboard:
//!
//! - On-pad soft landing:  reward `+TERMINAL_REWARD`,   sentinel `+TERMINAL_REWARD`  → win.
//! - Off-pad soft landing: reward `+PARTIAL_LANDING_REWARD`, sentinel `0`        → not a win.
//! - Crash / out-of-bounds: reward `-TERMINAL_REWARD`,  sentinel `-TERMINAL_REWARD`  → loss.
//!
//! The off-pad soft landing case is the intermediate skill —
//! "touch down softly *anywhere*" — between "don't crash" and "hit the
//! ±0.15 pad". We pay the agent for it so the policy has a stepping
//! stone, but pad-rate (the headline metric) only counts the on-pad
//! case.
//!
//! Plus dense potential-based shaping per step: the agent is paid the
//! *change* in a potential Φ that rewards being close to the pad, slow
//! and upright, so approaching earns and drifting away costs, while
//! merely existing costs nothing. A full descent is worth a few points
//! against the ±10 terminal — enough to steer, not enough to drown it.

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
/// Attitude-hold damping on angular velocity (1 / s): the craft bleeds
/// off spin with a ~0.5 s time constant, as if reaction wheels were
/// holding it steady between thruster taps.
///
/// Without it a single RCS tap spins the lander forever — nothing in
/// the world removes angular momentum — and a random policy tumbles
/// within a second. Measured on the undamped build: 82 % of touchdowns
/// failed *both* the tilt and the speed check, and not one of a
/// thousand random episodes landed softly. Damping makes attitude a
/// controllable thing rather than a coin flip at spawn.
pub const ANG_DAMPING: f32 = 2.0;
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

/// Weights of the shaping *potential* Φ — see [`LanderGame::potential`].
/// They set how strongly the descent is pulled toward the pad, toward
/// zero speed, and toward upright. Because only the change in Φ is paid
/// out, a full descent is worth about `SHAPE_DIST × spawn_height` ≈ 3.7
/// against the ±10 terminal: enough to steer, not enough to drown it.
pub const SHAPE_DIST: f32 = 2.0;
pub const SHAPE_SPEED: f32 = 1.0;
pub const SHAPE_TILT: f32 = 1.0;
/// Per-substep cost of running the main engine. A genuine cost, not
/// shaping — kept tiny (a full 3 s burn is ~0.4) so it discourages
/// pointless hovering without discouraging the engine itself.
pub const SHAPE_FUEL: f32 = 0.001;

/// Force a truncated reset after this many substeps. A hovering policy
/// that never touches the ground used to live forever (since the only
/// termination paths were ground contact and horizontal out-of-bounds),
/// and the overlay's hero pick — longest live episode — was literally
/// celebrating the hoverer. Free fall from the spawn height takes ~2 s
/// and a controlled descent ~4 s, so 8 s is generous; every second past
/// that is demo time spent watching a stalled episode.
pub const MAX_EPISODE_STEPS: u32 = 8 * 120;
/// World ceiling: the lander is considered out of bounds (truncated, not
/// crashed) if it climbs higher than this. Above the spawn altitude with
/// some slack so a tossing-up-on-spawn maneuver isn't punished.
pub const CEILING_Y: f32 = 1.0;

pub struct LanderGame {
    pos: Vec2,
    vel: Vec2,
    angle: f32,
    ang_vel: f32,
    last_thrusting: bool,
    rng: rand::rngs::StdRng,
    step_count: u32,
    truncations: u32,
    /// Curriculum knob driven by the harness. `1.0` is the nominal
    /// design difficulty: pad half-width [`PAD_HALF_W`] and the
    /// [`SOFT_VEL_X`] / [`SOFT_VEL_Y`] / [`SOFT_TILT_COS`] touchdown
    /// limits. Lower values widen the pad *and* open the tolerance;
    /// higher values tighten both. Lander boots at `0.4` — a fat target
    /// and a forgiving touchdown, so a fresh DQN can actually stumble
    /// into its first landing — and the auto-curriculum walks it back
    /// toward 1.0 as the pad-rate climbs.
    difficulty: f32,

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
            rng: crate::seeded_rng(),
            step_count: 0,
            truncations: 0,
            difficulty: 0.3,
            landings: 0,
            crashes: 0,
            partials: 0,
        };
        g.spawn();
        g
    }

    fn spawn(&mut self) {
        use rand::RngExt;
        self.pos = Vec2::new(self.rng.random_range(-0.7..0.7), 0.85);
        self.vel = Vec2::new(self.rng.random_range(-0.2..0.2), 0.0);
        self.angle = self.rng.random_range(-0.2..0.2);
        self.ang_vel = 0.0;
        self.last_thrusting = false;
        self.step_count = 0;
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
    /// Shaping potential Φ: how good this state looks on its way to a
    /// landing, in reward units. Near the pad, slow and upright is 0;
    /// everything else is negative. Only differences of Φ are ever paid
    /// out — see the shaping block in [`Game::step`].
    fn potential(&self) -> f32 {
        let dist = (self.pos - Vec2::new(0.0, GROUND_Y)).length();
        let speed = self.vel.length();
        let tilt = 1.0 - self.angle.cos().clamp(-1.0, 1.0);
        -(SHAPE_DIST * dist + SHAPE_SPEED * speed + SHAPE_TILT * tilt)
    }

    /// Curriculum-modulated pad half-width. `difficulty = 1` gives the
    /// design [`PAD_HALF_W`]; the clamp keeps the pad from collapsing to
    /// nothing or growing past the play area.
    fn effective_pad_half_w(&self) -> f32 {
        (PAD_HALF_W / self.difficulty.clamp(0.3, 5.0)).clamp(0.03, PLAY_WIDTH * 0.5 - 0.05)
    }

    /// Curriculum-modulated touchdown tolerance: `(max |vx|, max |vy|,
    /// min cos angle)`. `difficulty = 1` is the design spec.
    ///
    /// Scaling this — not just the pad width — is what gets the lander
    /// off the ground at all. A random policy touches down at a mean
    /// |vy| of 1.37 with the craft tumbling; against the design limits
    /// (0.4 and 18°) that is *zero* soft landings in a thousand
    /// episodes, so the +2 / +10 rewards are unreachable by exploration
    /// and the best policy the agent can find is to hover until the
    /// time limit. Opening the tolerance early gives it a first
    /// success to imitate; the auto-curriculum closes it back down as
    /// the pad-rate climbs.
    fn tolerance(&self) -> (f32, f32, f32) {
        let d = self.difficulty.clamp(0.3, 5.0);
        (
            SOFT_VEL_X / d,
            SOFT_VEL_Y / d,
            1.0 - (1.0 - SOFT_TILT_COS) / d,
        )
    }

    fn classify_state(&self) -> Landing {
        if self.pos.x.abs() > PLAY_WIDTH * 0.5 || self.pos.y > CEILING_Y {
            return Landing::OutOfBounds;
        }
        let touching = self.pos.y - BODY_HALF_H <= GROUND_Y;
        if !touching {
            return Landing::None;
        }
        let (max_vx, max_vy, min_cos) = self.tolerance();
        let upright = self.angle.cos() >= min_cos;
        let slow = self.vel.x.abs() <= max_vx && self.vel.y.abs() <= max_vy;
        if upright && slow {
            if self.pos.x.abs() <= self.effective_pad_half_w() {
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
        let potential_before = self.potential();

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
        self.ang_vel *= 1.0 - (ANG_DAMPING * dt).min(1.0);
        self.pos += self.vel * dt;
        self.angle += self.ang_vel * dt;
        self.step_count += 1;

        let landing = self.classify_state();
        // Returns (sparse_reward, terminal_sentinel, done). `sparse_reward`
        // folds into the learning signal; `terminal_sentinel` is what the
        // harness counts as a win/loss for the scoreboard. They differ for
        // partial landings: a soft touchdown off-pad is the natural
        // stepping-stone skill between "don't crash" and "hit the ±0.15
        // pad", so we still pay the agent for it — but it's not a "win"
        // for pad-rate accounting.
        let (sparse_reward, terminal_r, done) = match landing {
            Landing::SoftOnPad => {
                self.landings += 1;
                (TERMINAL_REWARD, TERMINAL_REWARD, true)
            }
            Landing::SoftElsewhere => {
                self.partials += 1;
                (PARTIAL_LANDING_REWARD, 0.0, true)
            }
            Landing::Crash => {
                self.crashes += 1;
                (-TERMINAL_REWARD, -TERMINAL_REWARD, true)
            }
            Landing::OutOfBounds => {
                self.crashes += 1;
                (-TERMINAL_REWARD, -TERMINAL_REWARD, true)
            }
            Landing::None => (0.0, 0.0, false),
        };

        // Truncation: hard time limit forces a reset so a hovering
        // policy can't camp forever. The bootstrap target *survives*
        // truncation (helper records done=false), so the value
        // estimate doesn't get falsely cut at an arbitrary substep.
        let truncated = !done && self.step_count >= MAX_EPISODE_STEPS;
        if truncated {
            self.truncations += 1;
        }

        // Potential-based shaping: pay out the *change* in Φ, never Φ
        // itself. The previous form charged `-SHAPE_DIST × dist` every
        // substep, which made time itself expensive: a 15 s hover ran up
        // ~-27 against a -10 crash, so the cheapest way to stop losing
        // points was to hit the ground hard. A difference of potentials
        // can't reorder the optimal policy (Ng et al., 1999) — it only
        // says which way is downhill.
        //
        // Φ is evaluated on the real final state at a terminal rather
        // than forced to 0, so a crash keeps its speed penalty instead
        // of collecting a parting bonus for arriving at the ground.
        let fuel = if thrusting { 1.0 } else { 0.0 };
        let shaping = self.potential() - potential_before - SHAPE_FUEL * fuel;

        // The helper resets on done or truncated; we used to self-reset
        // here, which made the post-step `observation()` already point
        // at the freshly-spawned pose. Let the helper handle it.
        StepOutcome {
            reward: sparse_reward + shaping,
            done,
            truncated,
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
                Stroke::new(1.5_f32, Color32::from_gray(110)),
            );
            let pad_w = self.effective_pad_half_w();
            let pad_top_left = to_screen(Vec2::new(-pad_w, GROUND_Y + 0.02));
            let pad_bot_right = to_screen(Vec2::new(pad_w, GROUND_Y));
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
            Stroke::new(1.0_f32, tint(Color32::from_gray(40))),
        ));
        let leg_l = to_screen(self.pos + rot(Vec2::new(-BODY_HALF_W * 1.2, -BODY_HALF_H)));
        let leg_r = to_screen(self.pos + rot(Vec2::new(BODY_HALF_W * 1.2, -BODY_HALF_H)));
        let leg_anchor_l = to_screen(self.pos + rot(Vec2::new(-BODY_HALF_W, -BODY_HALF_H * 0.6)));
        let leg_anchor_r = to_screen(self.pos + rot(Vec2::new(BODY_HALF_W, -BODY_HALF_H * 0.6)));
        let leg_stroke = Stroke::new(1.5_f32, tint(Color32::from_gray(180)));
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
            "outcomes        pad={} off={} crash={} timeout={}",
            self.landings, self.partials, self.crashes, self.truncations,
        ));
        let total = self.landings + self.partials + self.crashes + self.truncations;
        if total > 0 {
            ui.label(format!(
                "pad-rate        {:>5.1}%",
                100.0 * self.landings as f32 / total as f32,
            ));
        }
        ui.label(format!(
            "pad half-width  {:.3}   (design {:.3})",
            self.effective_pad_half_w(),
            PAD_HALF_W,
        ));
    }

    fn difficulty(&self) -> f32 {
        self.difficulty
    }

    fn set_difficulty(&mut self, level: f32) {
        self.difficulty = level.clamp(0.3, 5.0);
    }
}
