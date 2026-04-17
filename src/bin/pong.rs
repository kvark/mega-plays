use mega_plays::{AppConfig, pong::PongGame, run};

fn main() {
    run(AppConfig::default(), |_ctx| PongGame::new());
}
