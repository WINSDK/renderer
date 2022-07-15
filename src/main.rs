#![feature(repr_simd)]

#[cfg(not(any(target_os = "windows", target_family = "unix")))]
compile_error!("Renderer can only be build for windows, macos and linux");

#[cfg(not(any(feature = "naga", feature = "shaderc")))]
compile_error!("Must use at least one feature from `naga` or `shaderc`");

#[cfg(all(feature = "naga", feature = "shaderc"))]
compile_error!("Must use either feature `naga` or `shaderc`");

#[cfg(target_arch = "x86")]
use std::arch::x86 as intrinsics;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as intrinsics;

#[rustfmt::skip::macros(matrix, vector)]
#[macro_use]
mod math;
#[rustfmt::skip::macros(matrix, vector)]
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

fn main() {
    let log_info = std::env::var("RUST_LOG");
    let log_info = match log_info {
        Ok(ref preset) => preset,
        _ if cfg!(debug_assertions) => "info, gfx_backend_vulkan=info",
        _ if cfg!(feature = "basic_info") => "warn, gfx_backend_vulkan=warn",
        _ => "",
    };

    std::env::set_var("RUST_LOG", log_info);
    env_logger::init();

    tokio::runtime::Runtime::new().expect("Tokio runtime failed to start").block_on(async {
        let window = window::Window::new().await;
        events::run(window).await;
    })
}
