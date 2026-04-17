# mega-plays

Games where the opponents learn in real time, on your laptop's GPU, in Rust — no CUDA, no Python.

A showcase for [Meganeura](https://github.com/kvark/meganeura) (GPU neural network inference and training in Rust) and [blade-graphics](https://github.com/kvark/blade). Every agent you see playing is training live: fresh-start on launch, visibly improving within a minute, converged within a few.

## Status

Early development. First game: Pong, self-play.

## Games

- **Pong** *(in progress)* — classic two-paddle Pong with self-play DQN agents.
- *(more to come)*

## Why

Mainstream ML tooling assumes CUDA. Meganeura doesn't. This repo is proof that a Rust/GPU ML stack targeting Metal, Vulkan, and DX12 can handle end-to-end reinforcement learning workloads — training, inference, rendering — in a single native binary, on hardware that Python pipelines tend to treat as second-class (Apple Silicon, AMD, Intel Arc).

The goal isn't Pong. The goal is to make the training loop visible and compelling so the underlying stack is too.

---

## Pong — first implementation

### Design targets

- **Cold start to entertaining**: two untrained self-play agents begin as soon as the binary runs. Learning must be visibly progressing within ~30 seconds.
- **Convergence**: competent rallies within a few minutes on a typical laptop GPU (M1 / M2 / mid-range discrete).
- **Single binary**: no runtime dependencies, no Python, no downloads. Signed binaries for macOS, Windows, Linux.
- **Shareable**: a built-in hotkey records the last 30 seconds as GIF/MP4.

### Validate before building

Before committing to full Rust implementation, validate the learning-speed assumption. Prototype the Pong DQN in Python (PyTorch, ~100 lines) with the same state representation, network size, and self-play setup described below. Confirm visible policy improvement within 30 seconds and competent play within a few minutes on a laptop CPU. If it takes longer, the scope needs adjustment — smaller network, simpler dynamics, or a different starter game — before Rust implementation begins.

### Game

- Two paddles, one ball. Standard Pong mechanics.
- Fixed-timestep physics at 120 Hz, rendering at 60 Hz.
- First to 11 wins; auto-reset.
- **Modes**: self-play (default), human-vs-AI (press `P` to take over the left paddle).

Art direction: minimalist, monochrome, crisp. The visuals should read as "precise technical demo," not "skeuomorphic arcade." Paddles and ball are simple rectangles. The overlay type is clean monospace.

### Learning setup

- **Algorithm**: DQN with experience replay and a target network.
- **Observation**: state-based, 6–8 floats — own paddle y, opponent paddle y, ball x/y, ball vx/vy, optionally normalized time-to-contact. Do *not* use pixels. Normalize everything to roughly [-1, 1].
- **Action space**: discrete, 3 actions — up, down, stay.
- **Reward**: +1 on scoring, -1 on being scored on, small optional shaping term for paddle-to-ball y-alignment (configurable; off once agents are competent).
- **Network**: MLP, 2 hidden layers of 64–128 units, ReLU. Small on purpose.
- **Discount**: γ = 0.99.
- **Exploration**: ε-greedy, ε decaying from 1.0 to 0.05 over the first few minutes of wall time.
- **Replay buffer**: ~100k transitions, uniform sampling.
- **Target network**: hard update every ~1000 gradient steps.
- **Optimizer**: Adam, lr ≈ 1e-3.
- **Batch size**: 256.

Hyperparameters live in `config.toml` next to the binary, with sensible defaults compiled in. Do not expose them in the UI for v1.

### Self-play stability

Pure self-play is non-stationary and prone to degenerate equilibria (both agents learn to stand still; one dominates and the other never sees winning states). Mitigations, in order of importance:

1. **Symmetric policy**: both paddles share a single policy network. The right paddle's observation is mirrored before being fed in, and its action is mirrored after. Halves parameters and variance.
2. **Scripted bootstrap**: for the first ~30 seconds of wall time, one of the two paddles is a scripted tracker (follows the ball with a small delay and noise). This ensures the replay buffer fills with meaningful transitions before pure self-play begins.
3. **League sampling**: maintain a ring buffer of ~8 past policy checkpoints snapshotted at regular intervals. With probability ~0.3, one paddle's action is sampled from a historical checkpoint rather than the live policy.

### Threading and data flow

This is the most important architectural constraint. Getting it right once makes the demo feel instant; getting it wrong makes everything janky.

- **Main thread**: game simulation, rendering, input handling, overlay. 60 Hz render / 120 Hz physics.
- **Training thread**: drains a lock-free transition queue, runs minibatch DQN updates on the GPU via Meganeura, periodically publishes new policy weights.
- **Weight handoff**: double-buffered. The training thread writes fresh weights into a back buffer, then atomically swaps a pointer. The main thread reads from the front buffer without locks.
- **Inference on the main thread**: runs through Meganeura against the front-buffer weights. Must never block on training.

Transition flow: main thread pushes `(obs, action, reward, next_obs, done)` into a bounded MPSC queue after every physics step. Training thread pulls, appends to replay buffer, samples, steps.

### Visualization overlay

All overlay elements update once or twice per second; they must never touch the physics timing. Positioned around the play area edges; never obscure the ball.

- **Loss curve** — rolling TD loss over the last ~60 seconds of wall time. Corner sparkline.
- **Win-rate** — rolling win-rate of the live policy vs. the scripted baseline (while it's still in the loop) and vs. itself.
- **Policy heatmap** — 2D grid over (ball_x, ball_y) with paddle fixed at center, colored by argmax action (3 colors). The core educational visual: you watch the policy form in real time.
- **Throughput counter** — inferences/sec and training steps/sec. Use this shamelessly as a bragging number; it reinforces the performance story.
- **Episode counter, total gradient steps, wall time**.

### Controls

- `Space` — pause / resume simulation (training continues).
- `R` — reset both networks to random init, wipe replay buffer and league.
- `P` — toggle human control of the left paddle (arrow keys).
- `G` — toggle overlay visibility.
- `F9` — save the last 30 seconds to GIF + MP4 in the user's pictures/videos directory.
- `Esc` — quit.

### Logging

Structured logs (`tracing`) to stderr at info level by default, with a `--log-file` flag for persistent logs. Training loop should log episode reward, loss, ε, and throughput at a low frequency (once per second). No telemetry phoned home.

---

## Architecture

### Dependencies

- [`meganeura`](https://github.com/kvark/meganeura) — network definition, inference, training.
- [`blade-graphics`](https://github.com/kvark/blade) — rendering.
- `winit` — window and input.
- `cosmic-text` (or a minimal bitmap font) — overlay text.
- `serde` + `toml` — configuration.
- `tracing` — structured logging.
- `crossbeam` — lock-free queue for the transition channel.
- `image` + `gif` — screenshot and recording output.

No PyTorch, no ONNX, no tch-rs, no Candle, no Burn. Meganeura is the point.

### Repository layout

```
mega-plays/
  Cargo.toml              # workspace
  README.md
  crates/
    mp-core/              # replay buffer, DQN loop, visualization primitives, threading harness
    mp-pong/              # Pong physics, rendering, agent wiring, config
  apps/
    pong/                 # thin binary that wires mp-core + mp-pong
  assets/                 # fonts, shaders, icons
  tools/
    py-prototype/         # the Python validation prototype
```

Future games get their own `crates/mp-<game>/` + `apps/<game>/`. Shared RL plumbing stays in `mp-core`. `mp-core` is internal — not intended as a general RL library.

### Configuration

Runtime config in `config.toml` adjacent to the binary, with all defaults compiled in so the binary runs with no config file present.

## Building

```
cargo run --release -p pong
```

Release mode is mandatory — debug builds will miss the convergence window and misrepresent throughput.

## Platform support

- **macOS**: Metal via blade-graphics. Primary development target. Universal binary, notarized.
- **Windows**: Vulkan or DX12 via blade-graphics. MSI installer, signed.
- **Linux**: Vulkan. AppImage or plain tarball.

No WebGPU — blade-graphics has no WebGPU backend.

---

## Roadmap

- **v0.1** — Pong as specified above.
- **v0.2** — Polish, packaging, signed binaries, launch.
- **v0.3** — Second game. Candidates: catch-the-falling-object, simple top-down arena, Breakout.
- **v0.4+** — CNN-on-pixels mode, multi-agent variants, small league tournaments with Elo tracking.

## Non-goals

- Beating state-of-the-art RL benchmarks.
- Pretrained weights or transfer learning.
- Being a general-purpose RL library.
- Browser deployment.

## License

MIT.

## Credits

Built on [Meganeura](https://github.com/kvark/meganeura) and [blade-graphics](https://github.com/kvark/blade).
