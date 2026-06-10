# Audit plan — making the live-learning visualization satisfying

The engine works. The product doesn't yet *feel* like it learns. Four root causes,
ordered by impact; everything below is a step toward fixing one of them.

## P0 — Stale-action recording (~2× faster learning, free)

`Transition.action` is recorded as `last_action[i]` instead of `actions[i]`. The
obs–action pair stored in replay is shifted by one substep. A/B over 6 chunks
× 8000 grad steps: fix reaches 80% win rate at 32k grad steps vs 48k bugged.

Code is duplicated in four places (`tick`, `run_headless`, `train_epoch_chunk`,
`tests/headless.rs`); fix once by extracting a helper.

Same edit also fixes the dropped-first-transition bug (`last_obs[i].replace`
unconditionally returns `None` on the first step of an episode, so the spawn
state never feeds replay).

## P0 — Sim-speed/interactivity divorce

Every interactive lever currently fights GPU round-trip latency or frame rate:

- **`tick()` is fully serialized on GPU fences.** 12 inference/training round
  trips per frame; on lavapipe that's ~120 ms of `wait()` for ~1.2 k params of
  actual math. The speed slider multiplies those round trips linearly, so
  speed=32 stalls the UI instead of accelerating learning.
- **Physics is frame-rate dependent.** `physics_accum` is never consulted to gate
  substeps. The visualization runs at ⅓ real-time at 10 fps and 2× at 60 fps
  on the same slider setting.
- **Pause stops training** despite the README/button promising otherwise.
- **Warmup wastes inference.** ε=1 means every action is uniform random; the
  GPU forward pass is computed and discarded for ~5 000 substeps on startup.

Fix order:

1. **Action repeat k=4.** Select one action, hold it for k substeps, store the
   k-step sum reward + final obs as one transition. Cuts inference 4× and gives
   the policy temporally-extended exploration (120 Hz ε-dithering averages to
   near-zero net torque/velocity — likely a big part of why lander never commits
   to a maneuver).
2. **Skip inference when ε ≥ 1.0.** Take random actions on host.
3. **Budget-paced substep count.** Slider sets a *target sim multiplier*, driver
   runs as many physics substeps as a ~10 ms per-frame budget allows (or
   `dt * sim_mul / physics_dt`, whichever is smaller). Display achieved rate.
4. **Pause physics only.** Training continues to drain replay even when paused.
5. **Training on a worker thread.** Blade's Vulkan `Context` already guards its
   queue with a `Mutex`, so the shared `Arc<Context>` can submit from two
   threads. The training session moves off the event loop; render fps is then
   independent of grad-step throughput. Replay buffer wrapped in `Arc<Mutex<_>>`,
   parameter sync via a small `Arc<Mutex<Snapshot>>` consumed by the inference
   loop.

## P0 — Lander cannot learn as configured

Three independent issues:

1. **Bellman target clamp** (`agent.rs:340` clamp ±5) silently halves lander's
   ±10 terminals and erases the +10 vs +2 distinction. Make the clamp
   per-`AgentConfig` (or per-game, defaulting to the max terminal magnitude).
2. **No time limit / no vertical bound.** A policy that hovers or flies up
   off-screen never receives a terminal. The overlay's hero pick is "longest
   live episode", so the camera literally celebrates the hoverer.
   - Add `max_episode_steps` to `GameSpec`; driver enforces it as
     **truncation** (game resets, transition records `done=false` so the
     bootstrap target survives).
   - Add a top-of-world boundary (out-of-bounds) in lander.
3. **120 Hz dithering exploration** — solved by action repeat (P0 #2 above).

## P1 — Real Double-DQN

`agent.rs:16-19` documents Double-DQN, `train_step` implements vanilla DQN
(`max` over target). The standard remedy for the late-stage regression we
observed and the README itself flags. Cheap given we already host-snapshot
weights every step: snapshot online once per train_step; use online for
`argmax(next_obs)`, target for the Q evaluation.

## P1 — The learning is partly invisible

- Plot **difficulty over time** and **win-rate EMA** — when auto-curriculum is
  on, win rate is the controlled variable and stops moving by design. The
  unread `win_rate_ema` field becomes a first-class plot.
- Replace the lifetime-average win rate the curriculum currently reads with the
  EMA, and make the controller bidirectional (slow-walk difficulty back down
  when win rate dips, so a regression recovers instead of freezing).
- Hero env in overlay = "highest recent return" instead of "longest live
  episode", so the camera highlights *good* play, not stalling.
- Time-windowed plots (loss = last 8 s, return = last episode-count). Reward
  sparkline currently shows ≈1 frame of data; widen it.
- `run_headless` tracks no stats; teach it to print win-rate / loss EMA on its
  heartbeat.

## P2 — Smaller findings (interleaved with the above)

- meganeura `Session::drop` leaks `grad_clip_acc`. One-line fix on the meganeura
  branch, then bump the SHA. Bump also picks up `e0dffee` (persistent optimizer
  config — currently we re-arm Adam every step as a workaround) and `3f4dfd3`
  (Adam state read/write — `save_weights`/`load_weights` currently discard
  optimizer momentum, so a resumed run spikes its loss).
- `pong.rs::reset` doesn't clear `prev_dist`; one spurious shaping reward per
  episode reset. One line.
- `mega-pong.weights` committed contradicts the README's "no pretrained
  weights" non-goal — delete it.
- `load_weights` hand-rolls an unsafe byte cast; `bytemuck` is already pulled in.
- `MEGAPLAYS_NUM_ENVS` works windowed but `run_headless` ignores it. One line.
- Pad-rate (not landing-rate) is the honest lander win metric; partial off-pad
  landings should not flip the `terminal_reward > 0.0` win-counting branch.
- CI is ~30 minutes, almost all in the long headless smoke. After the
  recording fix the same confidence costs roughly half; tune frame count.
- README has gone stale: 16 envs vs the actual 9, blade `=0.8.2` pin vs the
  git-rev pins, target_sync interval vs Polyak, missing T/S/L keybindings,
  the wrong claim that pause keeps training.

## Order of implementation

Grouped into commits the user can review individually:

1. **(meganeura)** Fix the `grad_clip_acc` leak in `Session::drop`. Push to
   meganeura branch.
2. **(mega-plays)** Bump meganeura SHA; drop the committed `.weights`; commit
   this plan doc.
3. **(mega-plays)** Dedupe the env loop into one helper. The four call sites
   become trivial wrappers. No behavior change. ← lets every subsequent fix
   touch one place.
4. **(mega-plays)** Fix the stale-action bug at the dedup helper. Recovers the
   first-transition-per-episode at the same time. Loosen CI smoke threshold
   accordingly (faster).
5. **(mega-plays)** Action repeat (k=4) + ε=1 inference skip. Speed slider now
   means "target sim multiplier". Sim rate is displayed.
6. **(mega-plays)** Budget-paced substeps + honest pause (training keeps
   draining replay when physics is paused).
7. **(mega-plays)** TD-target clamp → per-game config. Truncation vs
   termination in `StepOutcome` and the driver; bootstrap survives time-outs.
8. **(mega-plays)** Lander: top-of-world boundary, configurable
   `max_episode_steps`, partial-landing no longer counts as a win, pad-rate
   surfaced in UI.
9. **(mega-plays)** Real Double-DQN.
10. **(mega-plays)** Auto-curriculum reads `win_rate_ema`, bidirectional.
    Plot difficulty + win-EMA over time. Hero env = best recent return.
    Time-windowed plots.
11. **(mega-plays)** Training worker thread.
12. **(mega-plays)** Small polish: `pong::reset` clears `prev_dist`,
    `MEGAPLAYS_NUM_ENVS` honored headless, `bytemuck::cast_slice_mut` in
    `load_weights`, README rewrite to match current behavior.

The training-thread commit (#11) is structurally largest and the most likely
place I want a checkpoint with the user before merging.
