use mega_plays::{AppConfig, games::catch::CatchGame, run};

fn main() {
    run(AppConfig::default(), |_ctx| CatchGame::new());
}
