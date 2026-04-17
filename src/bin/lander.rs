use mega_plays::{AppConfig, lander::LanderGame, run};

fn main() {
    run(AppConfig::default(), |_ctx| LanderGame::new());
}
