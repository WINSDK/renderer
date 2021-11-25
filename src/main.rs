#![feature(asm)]

extern crate nalgebra as na;

#[cfg(not(any(target_os = "windows", target_family = "unix")))]
compile_error!("Renderer can only be build for windows, macos and linux.");

#[cfg(target_arch = "x86")]
use std::arch::x86 as intrinsics;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as intrinsics;

mod camera;
mod controls;
mod crc;
mod events;
mod math;
mod png;
mod texture;
mod uniforms;
mod util;
mod window;

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
