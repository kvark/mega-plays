# Profiling — unified CPU + GPU trace per run

Every `pong` / `lander` run produces one `.pftrace` that Perfetto UI
loads directly, with two tracks:

- **CPU** — kindle-style `tracing` spans around physics, agent
  forward/learn, overlay paint, and blade submit/wait boundaries.
- **GPU** — meganeura's compute passes + blade's render passes,
  time-aligned on the same epoch via meganeura's submit-offset
  calibration.

No new crates. We reuse `meganeura::profiler` (hand-rolled Perfetto
binary emitter) and pull `tracing = "0.1"` transitively from
meganeura.

## Design

### Epoch and alignment

`meganeura::profiler` stores an `Instant::now()` epoch on first
init. All events are placed at `elapsed_ns` from that epoch. CPU
spans use `Instant::now()` at begin/end. GPU pass durations come from
blade's timestamp queries (already enabled via
`ContextDesc { timing: true, .. }`), submitted as a contiguous block
starting at `meganeura::profiler::now_ns()` immediately before
`gpu.submit()`.

The two blade sources — meganeura's compute-pass encoder and
mega-plays' render-pass encoder — share the track and chain through
their respective `record_gpu_passes` calls.

### Entry point

```rust
fn main() {
    env_logger::init();
    let _trace = mega_plays::profiling::init_or_default();
    // ... run the event loop ...
    // _trace drops at end of main → save `./mega-plays.pftrace`
}
```

`MEGA_TRACE=path` overrides the destination, `MEGA_TRACE=off`
disables.

### Headless trace generation

On hosts without a usable display surface (xvfb without DRI3, CI
runners, headless servers), set `MEGA_HEADLESS=<frames>` to run the
tick loop directly — no winit, no window, no render — for that many
frames. `MEGA_HEADLESS=1` defaults to 2000 frames. The resulting
`.pftrace` has full CPU spans and meganeura GPU-pass timings on the
same tracks; blade render-pass timings are absent because `render()`
is skipped. Example:

```
MEGA_HEADLESS=500 MEGA_TRACE=/tmp/pong.pftrace cargo run --release --bin pong
```

### CPU spans (in `app.rs`)

- `tick`: one span around the whole `tick` method (physics +
  training substeps).
- `substep`: one span per substep inside `tick`.
- `select_actions`: around `agent.select_actions(...)` — this is
  where meganeura's inference session runs and emits its own GPU
  events.
- `train_step`: around `agent.train_step()` — likewise a meganeura
  backward-pass dispatch.
- `render`: the whole rendering path.
- `render_submit`: just the `gpu.submit(...)` call (the wait on
  the previous frame is a separate span too).

### GPU render-pass track

Right after `self.gpu.submit(&mut self.command_encoder)` in
`render()`:

```rust
let submit_offset_ns = meganeura::profiler::now_ns();
let sync_point = self.gpu.submit(&mut self.command_encoder);
// Timings are retrieved on the *next* submit or when the previous
// fence is waited — depends on blade internals. In practice we read
// `self.command_encoder.timings()` on the following frame after the
// wait, tagging each entry with the submit-offset we captured before
// the corresponding submit.
```

Timestamp readback is fence-gated in blade: the command encoder's
`timings()` returns the *previous submit's* durations after the
fence has signalled. We keep a small ring of pending
`(submit_offset_ns, ExpectingSyncPoint)` and flush them when the
fence waits in the next frame. See `app.rs` for the actual
plumbing.

### Size

- ~50 FPS, ~3 render passes, ~5 CPU spans per frame, ~10 events per
  meganeura dispatch × 2 dispatches per frame ≈ 30 events/frame.
- 10 minutes = 30k seconds = 30k * 30 / 1000 ≈ 900 events/s → 540k
  events = ~27 MB trace (50 B/event). Easy to handle.

For unbounded sessions, `MEGA_TRACE=off` skips the whole emitter.

## Why not add our own protobuf encoder, chrome JSON, or tracy?

- `meganeura::profiler` already works, is minimal (one file), and
  the GPU timing path is already wired through it. Reusing it
  avoids a second clock domain and guarantees the two blade
  encoders land on the same track.
- Chrome JSON would blow up past 100 MB per minute at this event
  rate, and we want UNCONDITIONAL tracing.
- Tracy requires a server-side daemon and a viewer; Perfetto UI is
  a web app with a file drop.

## Non-goals

- Per-shader or per-draw-call breakdown inside a render pass —
  blade emits one timestamp pair per pass.
- Cross-process tracing — single process is enough here.
- Flamegraph aggregation — perfetto's built-in slice hierarchy
  covers it.
