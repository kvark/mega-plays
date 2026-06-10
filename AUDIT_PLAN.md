# Audit plan — making the live-learning visualization satisfying

The engine works. The product didn't *feel* like it learned. Four root causes,
ordered by impact; this doc lists each fix and tracks what shipped.

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

## P2 — Smaller findings  — shipped (interleaved)

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
