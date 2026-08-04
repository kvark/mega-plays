use mega_plays::{AppConfig, catch::CatchGame, run};

fn main() {
    run(AppConfig::default(), |_ctx| CatchGame::new());
}
