# mega-plays

Games where the opponents learn in real time, on your laptop's GPU, in Rust — no CUDA, no Python.

Showcase for [meganeura](https://github.com/kvark/meganeura) (GPU neural
network inference and training) and [blade-graphics](https://github.com/kvark/blade).
Every agent you see playing is training live: fresh-start on launch,
visibly improving within a minute, converged within a few.

## Status

Early scaffolding. The workspace builds, the pong binary opens a window,
the agent plays against a scripted tracker, and the DQN pipeline is
wired end-to-end through meganeura. Training convergence targets from
the original design have not yet been measured on hardware.

## Layout

```
mega-plays/
├── Cargo.toml
├── src/
│   ├── lib.rs                # re-exports
│   ├── agent.rs              # DQN agent: replay buffer, target net, meganeura wiring
│   ├── app.rs                # winit driver, Blade context, egui overlay, main loop
│   ├── game.rs               # Game trait
│   ├── stats.rs              # rolling stats, sparkline
│   ├── pong.rs               # Pong physics + rendering
│   └── bin/
│       └── pong.rs           # thin binary
```

Future games land as additional `src/<name>.rs` modules and
`src/bin/<name>.rs` binaries. The crate stays small on purpose and is
not intended as a general RL library.

## Building

Both `mega-plays` and `meganeura` are under active co-development. The
workspace consumes meganeura via a relative path dependency, so during
development you need the two repos checked out side by side:

```
my-work/
├── meganeura/        # on branch claude/game-learning-agents-nSKEt
└── mega-plays/       # on branch claude/game-learning-agents-nSKEt
```

Then, from `mega-plays/`:

```
cargo run --release --bin pong
```

Release mode is strongly preferred — debug throughput on the training
loop is not representative and the overlay stats will misread.

## Design choices and departures from the original sketch

The `README` version that seeded this project served as a design brief;
the following notes explain where the implementation diverged and why.

### Rendering goes through egui — no `cosmic-text`, no custom pipeline

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

`meganeura::Session::with_context(plan, Arc<Context>)` — added as part
of this project on the companion meganeura branch — lets the driver
create the Blade context once and hand a clone to both the renderer's
egui painter and the training / inference sessions. Same device, same
queue, same memory allocator, no device-enumeration surprises.

Context sharing goes through the unified `meganeura::build(graph,
SessionConfig { gpu: Some(ctx), .. })` entry point, which replaced the
`build_session` / `build_session_with` / `build_session_with_report` /
`build_session_cached` / `build_inference_session_with` family on the
companion branch.

### Lockstep blade-graphics versioning

Because two crate versions of the same FFI type are two different
types at the Rust level, meganeura and mega-plays have to agree on
*exact* `blade-graphics` version. The workspace pins
`blade-graphics = "=0.8.2"` — the same version meganeura 0.2 depends
on. Bumping either side requires bumping both.

### No cross-thread training, yet

The initial driver is single-threaded: each frame runs physics at
120 Hz against a fixed timestep, selects the agent's action via a
one-sample inference pass, records a transition, and runs a small
number of minibatch gradient steps. No lock-free MPSC queue, no
double-buffered weight handoff. The scaffolding exists exactly where
it will be swapped in — see the `Agent::train_step` and `Running::tick`
call sites — but shipping that concurrency now before any numbers are
measured would be optimising the imagined profile, not the real one.

### DQN variant: mask-based target fitting

The training graph feeds a one-hot action mask and a target Q value
scattered into the same action slot. The loss is plain MSE:

```
masked_q = q_all * action_mask
masked_t = target * action_mask
loss     = mean((masked_q - masked_t)^2)
```

Only the column of `fc2` corresponding to the taken action receives
gradient. This avoids needing a gather op, which meganeura does not
currently expose. If / when gather lands we can switch to the more
standard Huber loss over a single Q-value.

### Target network: host-side forward pass, for now

The target network is a parameter snapshot refreshed every N gradient
steps. Bootstrap targets are computed on host with a straightforward
MLP evaluation (two matrix multiplies, ReLU) — the MLP is ~4k weights,
a rounding error next to the GPU training minibatch. When the network
grows, a dedicated GPU inference session with target parameters is the
right next step; swapping parameters into the live inference session
every step would conflict with action selection on the main loop.

## Pong specifics

- 6-float observation: own paddle y, opponent paddle y, ball (x, y), ball (vx, vy).
- 3 discrete actions: stay, up, down.
- Reward: +1 on scoring, -1 on being scored on, 0 otherwise.
- Opponent: scripted tracker with adjustable y-noise (tune from the overlay).
- Fixed-step physics at 120 Hz; render at whatever the window reports.

Self-play, league sampling, scripted-bootstrap mixing, human control and
recording are all on the roadmap but intentionally not shipped in v0 —
we want empirical evidence that the learning loop works against a
stationary opponent first.

## Controls

- `Space` — pause / resume physics (training continues).
- `G` — toggle overlay.
- `R` — reset agent weights and replay buffer.
- `Esc` — quit.

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
