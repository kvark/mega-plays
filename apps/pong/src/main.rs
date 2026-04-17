use mega_plays::{AppConfig, run};
use mega_pong::PongGame;

fn main() {
    let config = AppConfig::default();
    run(config, |_ctx| PongGame::new());
}
