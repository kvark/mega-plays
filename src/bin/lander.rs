use mega_plays::{AppConfig, lander, lander::LanderGame, run};

fn main() {
    let mut cfg = AppConfig::default();
    // Lander's terminal rewards are ±TERMINAL_REWARD; the default
    // ±5 clamp would silently halve that signal. Give the clamp a hair
    // of headroom over the largest terminal so a maxed-out bootstrap
    // target isn't pinned to the limit.
    cfg.agent.td_target_clamp = lander::TERMINAL_REWARD * 1.05;
    run(cfg, |_ctx| LanderGame::new());
}
