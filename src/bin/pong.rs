use mega_plays::{AppConfig, games::pong::PongGame, run};

fn main() {
    run(AppConfig::default(), |_ctx| PongGame::new());
}
