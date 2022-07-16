#![feature(repr_simd)]

#[cfg(not(any(target_os = "windows", target_family = "unix")))]
compile_error!("Renderer can only be build for windows, macos and linux");

#[cfg(not(any(feature = "naga", feature = "shaderc")))]
compile_error!("Must use at least one feature from `naga` or `shaderc`");

#[cfg(all(feature = "naga", feature = "shaderc"))]
compile_error!("Must use either feature `naga` or `shaderc`");

mod intrinsics {
    #[cfg(target_arch = "x86")]
    pub use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    pub use std::arch::x86_64::*;

    pub const fn _mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
        ((z << 6) | (y << 4) | (x << 2) | w) as i32
    }
}

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
        let display = window::Display::new().await.unwrap();
        events::run(display).await;
    })
}
