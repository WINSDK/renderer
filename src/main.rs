extern crate nalgebra as na;

#[cfg(not(any(target_os = "windows", target_family = "unix")))]
compile_error!("Renderer can only be build for windows, macos and linux.");

mod crc;
pub use crc::*;
mod events;
mod uniforms;
pub use uniforms::*;
mod window;
pub use window::*;
mod texture;
pub use texture::*;
mod util;
pub use util::*;
mod png;


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
