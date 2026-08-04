# mega-plays

Games where the opponents learn in real time, on your laptop's GPU, in Rust —
no CUDA, no Python.

Showcase for [meganeura](https://github.com/kvark/meganeura) (GPU neural
network inference and training) and [blade-graphics](https://github.com/kvark/blade).
Every agent you see playing is training live: fresh-start on launch,
visibly improving inside a minute, converged in two.

## Status

Three games shipping. Each starts from random weights when you launch
it and is visibly better before you have finished reading the overlay.

Measured on an AMD 780M integrated GPU at the shipped defaults (25
parallel environments, one decision burst plus 8 gradient steps per
frame), as the wall-clock time until the win rate over the last 50
episodes first crosses the mark — see `tests/curves.rs`:

| game | milestone | reached |
| --- | --- | --- |
| **catch** — paddle under a falling ball | 50 % caught | **2.4 s** |
| **pong** — against a scripted tracker | 50 % of points | **6.2 s** |
| **lander** — soft touchdown on the pad | 20 % on-pad | **~13 s** ± |

± the lander's first success is still a lottery: 3 s, 12 s, 14 s and
15 s over four seeds, where pong and catch repeat to within a second or
two. Once it lands once it climbs to ~90 % on-pad and stays there.

For scale, the same pong milestone took ~18 s before this round of
tuning, and the lander never reached its milestone at all — it learned
to hover until the time limit and stayed there. What changed is
described under [learning fast enough to
watch](#learning-fast-enough-to-watch).

All three then keep climbing — pong and catch to 90-100 %, the lander
to ~90 % on-pad — at which point the auto-curriculum starts making the
game harder instead, and the **difficulty plot** becomes the real
progress curve.

Each is a separate binary (`cargo run --release --bin pong` /
`--bin lander` / `--bin catch`). They share the `mega-plays` library:
same driver, same DQN agent, same overlay — only the `Game` impl
differs.

Each "decision" the agent makes is held for `action_repeat` physics
substeps (default 4) — DQN's standard frame-skip trick. Cuts inference
round-trip count 4× and turns ε-greedy into a real exploration signal
(120 Hz sub-perception dithering averages to zero net torque).

## Layout

```
mega-plays/
├── Cargo.toml
├── AUDIT_PLAN.md           # what's intentionally still rough, and why
├── src/
│   ├── lib.rs              # re-exports
│   ├── agent.rs            # Double-DQN: replay buffer, target net, meganeura
│   ├── app.rs              # winit driver, Blade context, egui overlay, main loop
│   ├── env_loop.rs         # one decision burst across N parallel envs
│   ├── game.rs             # Game trait, StepOutcome { done | truncated }
│   ├── stats.rs            # rolling stats, sparkline
│   ├── pong.rs             # Pong physics + rendering
│   ├── lander.rs           # Lunar lander physics + rendering
│   ├── catch.rs            # Catch physics + rendering
│   ├── profiling.rs        # Perfetto trace glue (off by default)
│   └── bin/
│       ├── pong.rs
│       ├── lander.rs
│       └── catch.rs
└── tests/
    ├── headless.rs         # CI smoke: does pong actually learn?
    └── curves.rs           # ignored: learning curves, throughput,
                            #   random-policy baselines
```

Future games land as additional `src/<name>.rs` modules and
`src/bin/<name>.rs` binaries. The crate stays small on purpose and is
not intended as a general RL library.

## Building

```
cargo run --release --bin pong
```

Meganeura is pulled in as a normal git dependency pinned by SHA in
`Cargo.toml`. Blade is pinned by SHA the same way. Bump both in lockstep
— two crate versions of the same FFI type are two different Rust types,
so a mismatched blade rev between mega-plays and meganeura simply
doesn't link.

Release mode is strongly preferred — debug throughput on the training
loop is not representative and the overlay stats will misread.

To smoke-test headlessly (e.g. CI), install `mesa-vulkan-drivers` and
`xvfb`, then:

```
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
XDG_RUNTIME_DIR=/tmp/xdg \
MEGAPLAYS_EXIT_AFTER_SECS=60 \
xvfb-run -s "-screen 0 1280x800x24" cargo run --release --bin pong
```

`MEGAPLAYS_EXIT_AFTER_SECS` self-exits after the given wall time. The
windowed app prints a per-2-second heartbeat so you can watch the win
rate move without a display. There's also `MEGA_HEADLESS=<frames>` for
a no-window, no-render path (useful for traces on hosts that can't
acquire a display surface — the heartbeat now prints rolling win-rate
and loss-EMA there too).

## Learning fast enough to watch

The engine was never the bottleneck — a gradient step on the 33 k-param
policy costs about a millisecond, so the driver lands ~800 of them a
second while also running physics and drawing. What cost the demo its
first minute was everything around it. Four changes, all measured with
`tests/curves.rs`, which fixes a *wall-clock* budget rather than a step
count because "does it visibly learn while you watch" is a wall-clock
question:

### n-step returns

`AgentConfig::n_step` (default 3) folds three consecutive decisions into
one replay entry: the reward is the discounted sum over them and the
bootstrap term carries γ³ instead of γ. The network gets real reward
from further ahead instead of trusting its own half-trained estimate for
those steps. It is the single biggest win here — pong's 50 % milestone
went from 7.9 s to 5.0 s, and the lander's from 17.6 s to 3.8 s — because
both games pay out only at the end of an episode.

### Schedules measured in seconds, not in steps

Warmup was 5 000 transitions and ε decayed over 20 000 gradient steps.
At the driver's rate that is five seconds of uniform-random play before
the first gradient step and forty before the policy stops dithering —
most of a demo, spent on flailing. Now 1 000 and 5 000: training starts
before the window has settled and the policy is greedy inside ten
seconds. Nothing about the algorithm changed, only the clock it is
scheduled against.

### More environments

25 by default (was 9). Stepping a game is a rounding error next to a
gradient step, one batched inference covers all of them, and each extra
environment means more *fresh* experience per gradient step — the
replay ratio was over 200 samples drawn per new transition, which is
where a DQN starts memorising instead of learning. Time to pong's 50 %,
three seeds each: 9 envs 12.5 s, 16 envs 8.6 s, 32 envs 5.7 s.

### Potential-based shaping, everywhere

Every game now pays out the *change* in a potential Φ rather than a
per-step penalty (Ng et al., 1999 — a difference of potentials cannot
reorder the optimal policy). This matters more than it sounds: the
lander's old shaping charged distance-to-pad every substep, so a 15 s
hover ran up about -27 against a -10 crash, and the cheapest way to stop
losing points was to hit the ground hard.

## Design choices and departures from the original sketch

### Rendering goes through egui — no cosmic-text, no custom pipeline

Every on-screen element — paddles, ball, scores, stats, sparklines — is
an egui primitive. We do not compile our own WGSL shaders, do not run an
MSAA resolve pass, and do not depend on `cosmic-text`. Egui ships a
perfectly serviceable monospace font and tessellates rectangles with
anti-aliased edges. For tens of primitives per frame (which is every
game we reasonably care about at this stage) the tessellator is
invisible in profiles.

When a future game needs thousands of sprites per frame, add a sibling
crate with a direct Blade pipeline; the driver already owns the
`Arc<blade_graphics::Context>` needed to build one.

### One Blade context, shared between renderer and meganeura

`meganeura::Session::with_context(plan, Arc<Context>)` lets the driver
create the Blade context once and hand a clone to both the renderer's
egui painter and the training / inference sessions. Same device, same
queue, same memory allocator, no device-enumeration surprises.

### Vectorised environments, shared policy

The driver runs `num_envs` (default 25) parallel games against a single
DQN. Every burst gathers observations from all environments, does **one**
batched forward pass through the inference session, and picks 25 actions
at once. The replay buffer collects transitions from all environments
indiscriminately. Warmup fills in well under a second.

### Single-threaded driver (for now)

Each frame advances physics N times (in bursts of `action_repeat`
substeps per inference), runs minibatch gradient steps, then renders.
The per-frame physics+training loop is capped at `frame_budget_ms`
(default 12 ms) so the speed slider can ask for more than the machine
can deliver without freezing the UI; the overlay surfaces the achieved
sim multiplier so the user can tell "slider is in the way" from "machine
is in the way." A worker-thread split — moving the training session
off the event loop so render fps is independent of grad-step throughput
— is sketched in `AUDIT_PLAN.md`. Blade's Vulkan `Context` already
guards its queue with a `Mutex`, so the structural change is supported.

### `step()` is async; `wait()` before reading

Meganeura's `Session::step` submits GPU work but doesn't block. Reading
any buffer afterwards (inference outputs, training loss, parameters to
copy into a target snapshot) without an intervening `wait()` returns
whatever was in the host-visible memory *before* the submission landed.
During bring-up this produced a policy that learned a stable bad
strategy — loss dropped cleanly, but the action choices were driven by
stale uninitialised Q values. The fix is mundane: `step(); wait();
read_*(...);` everywhere meganeura's buffers are consumed on the host.

### Real Double-DQN

The online network picks the action at the next state, the target
network evaluates its Q-value. Decoupling action selection from value
estimation kills the systematic positive bias of vanilla DQN
(`max_a Q'(s', a)` over a noisy Q' is biased above the true max); in
practice that's the difference between a run that holds its gains and
one that regresses late in training. Both networks live as CPU-side
weight snapshots (the network is ~4 k parameters, host-side forward is
sub-microsecond).

### DQN training: mask-based target fitting

The training graph feeds a one-hot action mask and a target Q value
scattered into the same action slot. The loss is plain MSE:

```
masked_q = q_all * action_mask
masked_t = target   * action_mask
loss     = mean((masked_q - masked_t)^2)
```

Only the column of `fc2` corresponding to the taken action receives
gradient. This avoids needing a gather op, which meganeura does not
currently expose. If / when gather lands we can switch to the more
standard Huber loss over a single Q-value.

### `done` vs `truncated` — episode boundaries that don't lie to bootstrap

`StepOutcome` carries both: `done` means the world reached an absorbing
state (Bellman target is cut, no bootstrap past it), `truncated` means
the env reset for management reasons (time limit, escape boundary —
bootstrap target *survives*, the recorded transition's `done` flag
stays false). Lander uses `truncated` for both its 15 s episode
time-limit and its world ceiling. Without that the hover-forever
policy was a stable local optimum.

## Pong specifics

- 6-float observation: own paddle y, opponent paddle y, ball (x, y),
  ball (vx, vy).
- 3 discrete actions: stay, up, down. Held for 4 substeps each.
- Reward: ±1 on scoring, plus a small potential-based shaping for
  paddle–ball alignment (default 0.1, slider-tunable to 0).
- Opponent: scripted tracker with adjustable y-noise and speed-fraction
  sliders. Auto-curriculum nudges its tracking speed to maintain the
  target win/loss ratio (default 1.5 ≈ 60 % wins).
- Fixed-step physics at 120 Hz; render at whatever the window reports.

## Lander specifics

- 7-float observation: position (x, y), velocity (vx, vy), `sin/cos`
  of the lander's angle, angular velocity scaled to roughly [-1, 1].
- 4 discrete actions: idle, main engine, left RCS, right RCS.
- Physics: pure semi-implicit Euler, constant gravity, main thrust
  along the craft's "up" axis, RCS torque, and attitude damping. No
  physics library — the craft is one rigid body in a horizontal-ground
  world, which doesn't need one.
- Terminal reward: `+TERMINAL_REWARD` for a soft landing on the pad
  (upright, slow, on target), `+PARTIAL_LANDING_REWARD` for a soft
  landing off pad (episode ends, and it counts as a stepping stone but
  not as a "win"), `-TERMINAL_REWARD` for a crash or going out of the
  horizontal/vertical bounds.
- Truncation: 8 s of physics with no terminal → reset, bootstrap
  survives. The hoverer is no longer a stable local optimum, and
  the overlay's "hero env" doesn't celebrate it either.
- The lander binary sets `td_target_clamp = TERMINAL_REWARD * 1.05`
  so the ±10 sparse signal isn't compressed by the agent's default
  ±5 clamp.

### Why the lander needed the *game* fixed, not the agent

For a long time this game did not land, and every attempt to fix it
from the RL side (bigger net, longer warmup, n-step) failed the same
way: crash, crash, then hover until the time limit. `curves.rs`'s
`random_baseline` explains why. Under uniform-random play against the
design tolerances the craft touched down **994 times without a single
soft landing** — 82 % of those failed the tilt *and* the speed check at
once, at a mean vertical speed of 1.37 against a 0.40 limit. The
positive rewards were not merely rare, they were unreachable, so
hovering (worth 0) really was the best policy available and the agent
was right to find it.

Two changes make success reachable, and both belong to the world rather
than the learner:

- **Attitude damping** (`ANG_DAMPING`, ~0.5 s time constant). Nothing
  used to remove angular momentum, so one RCS tap span the craft
  forever and a random policy tumbled within a second. With damping,
  attitude is something a policy can hold.
- **The curriculum scales the touchdown tolerance, not just the pad
  width.** At the starting difficulty of 0.4 the pad is wider *and* the
  speed / tilt limits are looser, which takes the random-play success
  rate from 0 % to about 2 % — enough to bootstrap from. The
  auto-curriculum then tightens both back toward the design spec as the
  pad-rate climbs, and that tightening is what the difficulty plot
  shows.

## Catch specifics

- 5-float observation: paddle x, ball (x, y), ball (vx, vy).
- 3 discrete actions: stay, left, right.
- Reward: ±1 caught / missed, plus potential-based shaping on the
  horizontal gap so the policy starts tracking before it has ever
  caught anything.
- The ball spawns at the top with a random x and sideways drift and
  bounces off the side walls; the episode is decided the moment it
  reaches paddle height, about a second later.
- Difficulty scales fall speed and drift together. Catch saturates at
  ~100 % within seconds, so in practice the auto-curriculum is what you
  end up watching — the ball gets faster until the agent is winning
  about 60 % of the time.
- This is the game to launch first if you want to *see* the point of
  the project: 25 environments resolve about 20 episodes a second, so
  the win-rate curve moves in real time.

Both rendering and physics are dependency-free beyond `glam`. The
lander is drawn as a triangle body plus landing legs; the main engine
plume shows as an orange triangle under the craft while firing.

## Controls

- `Space` — pause physics. **Training continues** so the network keeps
  chewing on what's already in replay.
- `G` — toggle the stats panel.
- `V` — switch between **grid** (one tile per env) and **overlay** (all
  envs painted into the same rect with alpha blending, the current
  best-performing env — highest most-recent return — on top at full
  opacity). Super Meat Boy's replay screen is the visual reference.
  Initial mode also settable via `MEGAPLAYS_VIEW=overlay`.
- `R` — reset agent weights and replay buffer.
- `T` — run a 10 000-frame headless training epoch in chunks (progress bar).
- `S` — save current weights to `<game-title>.weights`.
- `L` — load weights from `<game-title>.weights`.
- `Esc` — quit.

The training panel exposes:

- Number of parallel envs (applied on next reset).
- Speed multiplier slider (1×–32×). Pinned by the per-frame compute
  budget shows in amber; "effective N×" tells you the achieved rate.
- View toggle (grid / overlay).
- Pause / resume.
- Save / load weights.
- Training loss curve (log-y), episode return curve, instantaneous
  reward sparkline.
- Game-specific sliders (pong opponent speed / noise / shaping).
- Auto-curriculum toggle + target W/L drag-value.
- **Win-rate EMA plot with the target as a dashed reference line** —
  the controlled variable when curriculum is on.
- **Difficulty over wall time** — the controlling variable, where to
  look for "we're learning" when win-rate is pinned to target.
- **Milestones** — a timestamped list of what the agent has managed so
  far: first win, each win-rate crossing, each difficulty step. The
  plots say something is improving; this says *it won its first point
  at 4 s and was winning half of them by 11 s*, which is the part a
  viewer remembers. Also echoed to the log, so a headless run tells the
  same story.

## Environment variables

- `MEGAPLAYS_NUM_ENVS=<n>` — override the default `num_envs` at launch
  (windowed and headless).
- `MEGAPLAYS_VIEW=overlay` — start in overlay view.
- `MEGAPLAYS_EXIT_AFTER_SECS=<n>` — self-exit after this wall time.
- `MEGAPLAYS_FORCE_EPSILON=<f>` — pin ε to this value (for tests).
- `MEGAPLAYS_SEED=<u64>` — seed every random stream in the process
  (network init, ε-greedy draws, replay sampling, and each game's own
  noise). Streams are handed out as `seed, seed+1, …` in construction
  order, so parallel environments still differ from each other while
  two runs of the same build compare like for like. Unset, everything
  comes from OS entropy.
- `MEGA_HEADLESS=<frames>` — skip window/surface/render entirely, run
  `frames` ticks of the physics+training loop, exit. Logs rolling
  win-rate and loss EMA every 500 frames.
- `MEGA_TRACE=<path|off>` — Perfetto trace destination. Off by default
  for normal runs; set to a path to capture CPU spans + GPU pass
  timings into a single `.pftrace`.

## Candidates for the next game

The harness is deliberately game-agnostic: a new game is one `impl Game`
module plus a thin `src/bin/<name>.rs` that wires it into [`run`].
We're explicitly looking for the shortest possible feedback loop —
something that converges in well under a minute on a modest laptop
GPU, keeps the observation space flat and ≤ ~16 floats, and produces
an on-screen policy the viewer can *see* getting better.

Catch — the previous list's first entry — shipped; it converges in
about three seconds and is now the reference for "the harness is
fine, the game is hard".

1. **Grid-based find-the-food.** 8×8 grid, agent + food glyph, 4
   directional actions. Trivial physics, classic RL benchmark,
   benefits directly from the vectorised-env pipeline.
2. **Flappy-pipe.** Agent with gravity + one "flap" action, pipes
   scroll in. Episodes end on hit. Famous for being almost trivial
   with the right reward shaping and catastrophic without it — a
   good stress test for the harness's stability knobs.
3. **Simple arena-dodger.** Agent dodges projectiles in an arena;
   reward is time alive. Related in shape to pong but single-agent,
   no opponent model needed. A reasonable step toward the multi-
   agent / league-sampling variants sketched in `AUDIT_PLAN.md`.

**Non-candidates for now** — Breakout (physics is only superficially
simple; tile state blows observation size), Atari-pixel games (CNN-on-
pixels is an explicit v0.4 goal, not v0.2), anything multi-agent (needs
the self-play machinery we haven't built yet).

## Platform support

- **macOS**: Metal via blade-graphics. Primary development target.
- **Linux**: Vulkan.
- **Windows**: Vulkan or DX12 via blade-graphics.

No WebGPU — blade-graphics does not ship a WebGPU backend.

## Non-goals

- Beating state-of-the-art RL benchmarks.
- Pretrained weights or transfer learning.
- Being a general-purpose RL library.
- Browser deployment.

## License

MIT.

## Credits

Built on [meganeura](https://github.com/kvark/meganeura) and
[blade-graphics](https://github.com/kvark/blade).
