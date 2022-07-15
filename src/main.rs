#![feature(repr_simd)]

#[cfg(not(any(target_os = "windows", target_family = "unix")))]
compile_error!("Renderer can only be build for windows, macos and linux.");

#[cfg(target_arch = "x86")]
use std::arch::x86 as intrinsics;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as intrinsics;

#[macro_use]
mod math;
mod camera;
mod controls;
#[allow(dead_code)]
mod crc;
mod events;
mod png;
mod texture;
mod uniforms;
mod util;
mod window;

pub use util::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        std::env::set_var("RUST_LOG", "info, gfx_backend_metal=warn");
    }

    env_logger::init();
    tokio::runtime::Runtime::new()?.block_on(async {
        let window = window::Window::new().await;
        events::run(window).await;

        Ok(())
    })
}
