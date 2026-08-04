# Audit plan — making the live-learning visualization satisfying

The engine works. The product didn't *feel* like it learned. This doc
lists each finding from two passes (the original audit + a follow-up
Fable-on-Fable second look) and tracks what shipped.

## Still missing — known live worklist

- **Training-on-worker-thread.** Still deferred. See "Deferred" below.
- **Lander's on-pad rate is the noisiest number in the repo.** Time to
  20 % on-pad ranges from ~4 s to ~18 s across seeds where pong and
  catch are repeatable to a second or two. It gets there, and it holds
  ~90 % once it does, but the first success is still a lottery. A
  proper fix is probably prioritised replay (the handful of successful
  episodes are drowned by hundreds of crashes in a uniform sample).

## Round three — learning fast enough to watch (2026-08-04)

The prior passes fixed the harness; this one went after the wall clock
and the lander's reward landscape. Everything below was measured with
the new `tests/curves.rs`, which budgets **wall-clock seconds** instead
of frames.

- **n-step returns** (`AgentConfig::n_step`, default 3). Biggest single
  win: pong's 50 %-win milestone 7.9 s → 5.0 s, lander's 20 %-on-pad
  17.6 s → 3.8 s.
- **Schedules re-sized in seconds**: warmup 5 000 → 1 000, ε decay
  20 000 → 5 000 gradient steps. Five seconds of dead random play and
  forty seconds of dithering were most of a demo.
- **25 environments** (was 9). Three seeds each, time to pong's 50 %:
  9 envs 12.5 s, 16 envs 8.6 s, 32 envs 5.7 s. 25 keeps the grid square
  and legible.
- **The lander was an unreachable-goal problem, not an RL problem.**
  `curves.rs::random_baseline` under uniform-random play: 994
  touchdowns, **zero** soft landings, 82 % failing tilt *and* speed at
  once. Fixed in the world, not the learner — attitude damping so the
  craft stops tumbling, and a curriculum that scales the touchdown
  tolerance (not only the pad width) so random play succeeds ~2 % of
  the time and there is something to bootstrap from. It now reaches
  ~90 % on-pad within a minute.
- **Potential-based shaping** in the lander (pong and catch already had
  it). The old absolute per-step distance penalty made a 15 s hover cost
  ~-27 against a -10 crash — crashing was the cheap way out.
- **Milestone log** in the overlay and the driver log: first win,
  win-rate crossings, difficulty steps, each with a timestamp.
- **`MEGAPLAYS_SEED` now seeds the games too**, not just the agent, with
  a distinct stream per environment. A/B runs are comparable.
- **catch** shipped as a third game: ~1 s episodes, 50 % caught at
  2.4 s, ~100 % by 7 s. It is the fastest visible feedback in the repo
  and the control case for "is the harness at fault, or is the game
  hard?".

## Status

All shipped this session except the worker-thread split, which is deferred
intentionally — see "Deferred" at the bottom. The single-threaded design is
fully **lock-step**: every inference and every training step waits on its own
GPU fence before returning. Predictable, debuggable.

## P0 — Stale-action recording (~2× faster learning, free)  — shipped

`Transition.action` was recorded as `last_action[i]` instead of `actions[i]`.
The obs–action pair stored in replay was shifted by one substep. A/B over 6
chunks × 8000 grad steps: fix reaches 80 % win rate at 32 k grad steps vs
48 k bugged.

Code was duplicated in four places (`tick`, `run_headless`,
`train_epoch_chunk`, `tests/headless.rs`); fixed once by extracting
`env_loop::run_burst`.

Same edit also fixed the dropped-first-transition bug
(`last_obs[i].replace(...)` unconditionally returned `None` on the first step
of an episode, so the spawn state never fed replay).

## P0 — Sim-speed/interactivity divorce  — shipped

Every interactive lever fought GPU round-trip latency or frame rate.
Shipped fixes:

- **Action repeat k=4.** Hold a single decision for k substeps and record one
  transition per burst with the summed reward + final obs. Cuts inference 4×
  and gives ε-greedy a real exploration signal (120 Hz dithering averaged to
  near-zero net torque/velocity — a likely root cause of lander never
  committing).
- **Skip inference when ε ≥ 1.0.** During warmup every action is uniform
  random; take them on host directly. Combined with action repeat, headless
  600 frames × 4 substeps runs in 2.4 s — **23× faster** than the
  pre-audit baseline.
- **Budget-paced substep count.** Each `tick()` stops as soon as the per-frame
  `frame_budget_ms` (default 12) has elapsed, surfaces the achieved substep
  count in the overlay, and scales `train_steps` by what actually ran.
  The UI shows "effective N× tick T ms (budget)" in amber whenever the
  slider is pinned by compute, so the user can tell "slider in the way"
  from "machine in the way".
- **Pause physics only.** Training continues at `train_steps_per_frame` even
  when physics is paused. Matches the README/button text, which was
  previously a lie.

## P0 — Lander cannot learn as configured  — shipped

Three independent fixes:

1. **TD-target clamp is per-agent-config** (`AgentConfig::td_target_clamp`,
   default 5). Lander's binary sets it to `TERMINAL_REWARD * 1.05 = 10.5`
   so the ±10 sparse signal isn't compressed.
2. **Episode time limit (truncation).** New `StepOutcome::truncated` field.
   Lander truncates after 15 s of physics with no terminal — bootstrap target
   survives (recorded `done=false`), the env resets. No more hover-forever.
3. **World ceiling.** Out-of-bounds at `y > 1.0`, treated as a crash.

The `StepOutcome` `done` vs `truncated` distinction is now first-class through
the env loop helper and the stat counters.

## P1 — Real Double-DQN  — shipped

Module docstring claimed Double-DQN; `train_step` implemented vanilla DQN
(`max` over the target). Replaced with: snapshot online network once per
train_step, use it to argmax over `next_obs`, evaluate that chosen action's
Q on the target snapshot. The standard remedy for the late-stage regression
that showed up in the A/B (chunk 5 dropped 91 % → 80 %).

## P1 — Make the learning visible  — shipped

- **Auto-curriculum now reads `win_rate_ema`** (was lifetime average, which
  is essentially frozen after the first 100 episodes). Bidirectional — climb
  fast, descend slow with a 4-point deadband. Lower bound 0.05 so a
  regression can recover instead of freezing.
- **Win-rate-EMA plot with the target as a dashed reference line** + a
  **difficulty plot over wall time**, both sampled every 0.5 s and shown in
  the training panel. When auto-curriculum is on the win-rate hugs the
  target by design; difficulty is the actual progress curve.
- **Hero env = best most-recent return**, plus 0.1× in-flight return as a
  tiebreaker. Was "longest live episode", which celebrated stallers.
- `run_headless` heartbeat now prints rolling win-rate and loss EMA every
  500 frames so a CI run actually tells you whether anything is learning.

## Second-pass findings  — shipped

A fresh-eyes review after the first audit landed surfaced eight more
items, all addressed in `8f4a0d6..2543340`:

- **Lander's +2 partial-landing reward** was accidentally zeroed by the
  earlier "honest pad-rate" commit and the docstring stopped matching
  the code. Restored as the reward component (sentinel still 0 so the
  scoreboard is honest), plus added a `win/loss/partial/timeout`
  breakdown to the headless heartbeat that immediately revealed
  lander dies 100 % by crash/OOB, not by timeout-after-hovering.
- **Pad-width curriculum** for lander. Difficulty 1.0 is the design
  half-width; lander boots at 0.5 (twice the design) and the
  auto-curriculum tightens it as pad-rate climbs. The hard ceiling /
  floor live in `Game::set_difficulty` clamps now, not in the
  controller.
- **Adam state in save/load.** `read_adam_m/v`, `write_adam_m/v`,
  `set_adam_step_count` wired in. Format v2 carries the moments and
  step count under a "MEGA" magic + version field.
- **Profiling default flipped off.** README said "off by default", code
  defaulted to *on* with an unbounded `Vec<TraceEvent>` upstream
  (hundreds of MB/h in a long windowed session). `MEGA_TRACE` now
  has to be set explicitly to enable.
- **Truncation handling in remaining loops.** `tick` was the only
  loop that flushed episode stats on `done || truncated`;
  `train_epoch_chunk` and `tests/headless` only checked `done`.
  Episode returns no longer accumulate across truncation resets.
- **Wall-clock-anchored sim speed.** `tick()` used to compute
  substeps as `base × speed_mul`, independent of dt — "1×" was
  ~2× real-time at 60 fps and ⅓× at 10 fps. Now `target_subs =
  (dt × speed_mul) / physics_dt`, so 1× means real-time. The frame
  budget caps both the substep *and* the training loops; a slow GPU
  no longer overruns by 100+ ms.
- **Human play.** `P` toggles human control of env 0's opponent
  paddle via ↑/↓ — the "you vs the live-learning agent" demo the
  README always promised. `Game::set_human_input` is the channel;
  defaults to no-op, pong implements it, lander ignores it.
- **Showcase line + Q-bars.** Training panel shows `param-count →
  device-name @ grad-rate`; hero env in overlay mode gets a row of
  Q-value bars with the argmax in green, so the user sees not just
  *that* the policy improves but *what* it thinks each substep.
- **Misc:** `AgentConfig::hidden` docstring corrected (one hidden
  layer, not two); `MEGAPLAYS_SEED=<u64>` seeds the Agent RNG for
  reproducible A/Bs; meganeura plan cache wired at
  `target/meganeura-cache/`; CI release-builds both binaries; egglog
  graph-naming WARN spam silenced by default `RUST_LOG`.

## P2 — Smaller findings (first audit) — shipped (interleaved)

- meganeura `Session::drop` leaked `grad_clip_acc` → fixed on
  `claude/game-learning-agents-nSKEt` of meganeura; SHA bumped here. The bump
  also picks up `e0dffee` (persistent optimizer config) and `3f4dfd3` (Adam
  state read/write — `save_weights`/`load_weights` no longer drop optimizer
  momentum).
- `pong.rs::reset` now clears `prev_dist` (was leaking a spurious shaping
  reward into the first substep of every new episode).
- `mega-pong.weights` deleted (contradicted the README's non-goal).
- `load_weights` / `save_weights` use `bytemuck::cast_slice_mut` /
  `cast_slice` instead of a hand-rolled unsafe byte cast.
- `MEGAPLAYS_NUM_ENVS` honored in `run_headless` as well as windowed.
- Pad-rate (not landing-rate) is the lander's honest win metric; partial
  off-pad landings no longer flip the `terminal_reward > 0` win branch.
- CI smoke test `pong_learns_to_beat_slow_opponent` 28 800 → 6 000 frames,
  still gates at 40 % win-rate (now reaches ~50 % in ~4 min on lavapipe).
- README rewritten end-to-end to match current behaviour.

## Deferred — Training worker thread

The original plan listed an `Agent` refactor splitting the training session
onto a worker thread so render fps becomes independent of grad-step
throughput. Intentionally skipped for now:

1. The single-threaded design is **lock-step**: every inference and every
   training step waits on its own GPU fence before returning. Predictable,
   debuggable. After the action-repeat + ε=1-skip wins above, the
   interactive demo is no longer bottlenecked on round trips — pong shows
   visible learning inside ~60 s on a CPU-Vulkan stress floor.
2. The thread split would require meganeura's `Session` to be `Send`
   (unverified — Session contains a blade `CommandEncoder` and a long list of
   internal handles), and lavapipe is the worst surface to debug concurrent
   GPU submit issues on.

Sketch for the future if we do want to revisit:

- Replay → `Arc<Mutex<VecDeque<Transition>>>`. Critical section per `record`
  is one push, per `sample` is `batch × obs_dim` copies (≈ tens of µs).
- Worker owns `training: Session`, `target_snapshot`. Loops:
  acquire latest online via `snapshot_training` → sample batch → train_step
  → wait → polyak update → publish post-step online snapshot to an
  `Arc<Mutex<Option<Vec<Vec<f32>>>>>`.
- Main thread: each substep, drain the published snapshot (if any) into
  `inference.set_parameter`, then run `select_actions` as today.
- Shutdown: stop flag + join before the blade Context drops.

Blade's Vulkan `Context` already guards its queue with a `Mutex`, so the
concurrent-submit path is supported on the rendering side.
