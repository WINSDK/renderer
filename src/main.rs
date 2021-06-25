extern crate nalgebra as na;

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
compile_error!("Renderer can only be build for windows, macos and linux.");

mod events;
mod uniforms;
pub use uniforms::*;
mod window;
pub use window::*;
mod texture;
pub use texture::*;
mod util;
pub use util::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        std::env::set_var("RUST_LOG", "info, gfx_backend_metal=warn");
    }

    env_logger::init();
    let window = window::Window::new().await;
    events::run(window).await;

    Ok(())
}
