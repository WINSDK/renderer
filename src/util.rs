use png::{BitDepth, ColorType};
use std::borrow::Cow;
use std::fmt::Debug;
use std::fs;
use std::io::{self, ErrorKind, Read, Write};
use std::mem::{align_of, size_of, size_of_val};
use std::path::{Path, PathBuf};
use wgpu::{Device, ShaderModule, ShaderSource, ShaderStages};

#[macro_export]
macro_rules! consume {
    ($reader:expr, $ty:ty) => {{
        use std::io::Read;

        let mut tmp = [0u8; std::mem::size_of::<$ty>()];
        unsafe { $reader.read_exact(&mut tmp).map(|_| std::mem::transmute::<_, $ty>(tmp)) }
    }};
}

pub struct Png {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub color_type: ColorType,
    pub bit_depth: BitDepth,
    pub line_size: usize,
}

#[inline(always)]
pub async fn read_png<P: AsRef<Path>>(path: P) -> Result<Png, io::Error> {
    let bytes = fs::read(&path)?;
    log::info!("reading png from: {:?}", path.as_ref());
    convert_to_png(bytes.as_slice()).await
}

pub async fn convert_to_png(bytes: &[u8]) -> Result<Png, io::Error> {
    let decoder = png::Decoder::new(bytes);
    let (info, mut reader) = decoder.read_info().map_err(|_| ErrorKind::InvalidData)?;
    let mut data = vec![0; info.buffer_size()];
    reader.next_frame(&mut data).map_err(|_| ErrorKind::InvalidData)?;

    debug_assert!(info.color_type as u32 > 3); // ColorType::(RGBA | GrayscaleAlpha)

    Ok(Png {
        data,
        width: info.width,
        height: info.height,
        color_type: info.color_type,
        bit_depth: info.bit_depth,
        line_size: info.line_size,
    })
}

pub async fn generate_vulkan_shader_module<P: AsRef<Path> + Debug>(
    path: P,
    stage: ShaderStages,
    device: &Device,
) -> io::Result<ShaderModule> {
    match shader_checksum(&path, device) {
        Err(_) => compile_shader(path, stage, device).await,
        Ok(shader) => Ok(shader),
    }
}

fn create_cache_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let cache_path = path.as_ref().with_extension("spv");
    let cache_path = cache_path.file_name().unwrap();
    if cfg!(target_os = "windows") {
        Path::new(&std::env::var("TMP").unwrap()).join(cache_path)
    } else {
        Path::new("/tmp").join(cache_path)
    }
}

/// checks if shader is already cached, if so returns a ShaderModule
fn shader_checksum<P: AsRef<Path> + Debug>(path: P, device: &Device) -> io::Result<ShaderModule> {
    let src_file = fs::File::open(&path)?;
    let mut cache_file = fs::File::open(create_cache_path(&path))?;

    let cache_stamp = consume!(cache_file, std::time::SystemTime)?;

    // Check if the src_file's modified date equals the modified date stored in the cache file,
    // this ensures that if the source file get's modified, the cache file must be outdated.
    if src_file.metadata()?.modified()? == cache_stamp {
        let mut shader: Vec<u8> = Vec::new();

        cache_file.read_to_end(&mut shader)?;

    // cached shader matches shader source.
    if hash == u32::from_le_bytes(buf) {
        let shader: Vec<u32> = shader[4..] // ignoring the 4 checksum bytes
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        log::info!("Reading cached shader from: {:?}", path);
        Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        let shader: Vec<u32> = {
            let ptr = shader.as_mut_ptr() as *mut u32;
            let len = shader.len() / 4;
            let capacity = shader.capacity() / 4;

            let new_vec = unsafe { Vec::from_raw_parts(ptr, len, capacity) };
            std::mem::forget(shader);
            new_vec
        };

        log::info!("Reading cached shader from: {:?}", path);
        let res =  Ok(device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::SpirV(Cow::Borrowed(shader.as_slice())),
        }));

        return res;
    }


    Err(io::ErrorKind::InvalidData.into())
}

#[cfg(not(feature = "shaderc"))]
async fn compile_shader<P: AsRef<Path> + Debug>(
    path: P,
    stage: ShaderStages,
    device: &Device,
) -> io::Result<ShaderModule> {
    use naga::{
        back::spv::{self, Options as SpvOptions},
        front::glsl::{Options as GlslOptions, Parser},
        valid::{Capabilities, ValidationFlags, Validator},
    };

    let mut src_file = fs::File::open(&path)?;
    let mut cache_file = fs::File::create(create_cache_path(&path))?;

    let stage = match stage {
        ShaderStages::COMPUTE => naga::ShaderStage::Compute,
        ShaderStages::VERTEX => naga::ShaderStage::Vertex,
        ShaderStages::FRAGMENT => naga::ShaderStage::Fragment,
        _ => unreachable!(),
    };

    let module = {
        log::info!("Reading shader from: {:?}", path);

        let mut src = String::new();
        src_file.read_to_string(&mut src)?;

        let mut parser = Parser::default();
        let module = parser.parse(&GlslOptions::from(stage), &src[..]).unwrap();

        log::info!("compiling shader: {:?}", parser.metadata());
        module
    };

    let mut validator = if cfg!(debug_assertions) {
        Validator::new(ValidationFlags::all(), Capabilities::empty())
    } else {
        Validator::new(ValidationFlags::empty(), Capabilities::empty())
    };

    let module_info = validator.validate(&module).unwrap();
    let binary = spv::write_vec(&module, &module_info, &SpvOptions::default(), None).unwrap();

    let date_modified = src_file.metadata()?.modified()?;

    cache_file.write_all(cast_bytes(&date_modified))?;
    cache_file.write_all(cast_slice(binary.as_slice()))?;

    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::SpirV(Cow::Borrowed(&binary[..])),
    }))
}

#[cfg(feature = "shaderc")]
async fn compile_shader<P: AsRef<Path> + Debug>(
    path: P,
    stage: ShaderStages,
    device: &Device,
) -> io::Result<ShaderModule> {
    let stage = match stage {
        ShaderStages::COMPUTE => shaderc::ShaderKind::Compute,
        ShaderStages::VERTEX => shaderc::ShaderKind::Vertex,
        ShaderStages::FRAGMENT => shaderc::ShaderKind::Fragment,
        _ => unreachable!(), // can't have unreachable_unchecked() in case of accidental use VERTEX_FRAGMENT
    };

    let mut compiler = shaderc::Compiler::new().unwrap();
    let src = fs::read_to_string(&path).await?;
    let binary = compiler
        .compile_into_spirv(&src, stage, path.as_ref().to_str().unwrap(), "main", None)
        .unwrap();

    let hash = crc32(binary.as_binary_u8()).await;
    let mut file = Vec::with_capacity(binary.as_binary_u8().len() + size_of_val(&hash));
    file.extend(u32::to_le_bytes(hash));
    file.extend_from_slice(binary.as_binary_u8());

    fs::write(create_cache_path(path), file).await?;

    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::SpirV(Cow::Borrowed(binary.as_binary())),
    }))
}

#[inline]
pub fn cast_bytes<T>(p: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((p as *const T) as *const u8, std::mem::size_of::<T>()) }
pub fn cast_bytes<'bytes, T>(p: &'bytes T) -> &'bytes [u8] {
    unsafe { std::slice::from_raw_parts((p as *const T) as *const u8, size_of::<T>()) }
}

#[inline]
pub fn cast_slice<A, B>(slice: &[A]) -> &[B] {
    if align_of::<B>() > align_of::<A>() && (slice.as_ptr() as usize) % align_of::<B>() != 0 {
        panic!("Unaligned cast");
    } else if size_of::<B>() == size_of::<A>() {
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const B, slice.len()) }
    } else if size_of::<A>() == 0 || size_of::<B>() == 0 {
        panic!("mismatched size");
    } else if size_of_val(slice) % size_of::<B>() == 0 {
        let new_len = size_of_val(slice) / size_of::<B>();
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const B, new_len) }
    } else {
        panic!("unknown cast fault");
    }
}
