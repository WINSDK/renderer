use crate::crc::crc32;
use png::{BitDepth, ColorType};
use std::mem::{align_of, size_of, size_of_val, ManuallyDrop};
use std::{
    borrow::Cow,
    fmt::Debug,
    fs,
    io::{self, ErrorKind, Read},
    path::{Path, PathBuf},
};
use wgpu::{Device, ShaderModule, ShaderSource, ShaderStages};

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
    log::info!("Reading shader from: {:?}", path);
    let cache_path = create_cache_path(&path);

    match shader_checksum(&cache_path, device).await {
        Err(_) => compile_shader(path, stage, device).await,
        Ok(shader) => Ok(shader),
    }
}

#[inline]
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
async fn shader_checksum<P: AsRef<Path> + Debug>(
    path: P,
    device: &Device,
) -> io::Result<ShaderModule> {
    let shader = fs::read(&path)?;
    let hash = crc32(&shader[4..]).await;

    let mut handle = fs::File::open(&path)?;
    let mut buf = [0u8; 4];

    handle.read_exact(&mut buf)?;

    // cached shader matches shader source.
    if hash == u32::from_le_bytes(buf) {
        let shader: Vec<u32> = shader[4..] // ignoring the 4 checksum bytes
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        log::info!("Reading cached shader from: {:?}", path);
        Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::SpirV(Cow::Borrowed(shader.as_slice())),
        }))
    } else {
        Err(io::ErrorKind::NotFound.into())
    }
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

    let stage = match stage {
        ShaderStages::COMPUTE => naga::ShaderStage::Compute,
        ShaderStages::VERTEX => naga::ShaderStage::Vertex,
        ShaderStages::FRAGMENT => naga::ShaderStage::Fragment,
        _ => unreachable!(),
    };

    let module = {
        let src = fs::read_to_string(&path)?;
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
    let mut binary = ManuallyDrop::new(
        spv::write_vec(&module, &module_info, &SpvOptions::default(), None).unwrap(),
    );

    // SAFETY: use this before create_shader_module as the destructor can be run on the original
    // vec resulting in the vec pointing to deallocated memory
    let binary_u8 = unsafe {
        let width = size_of::<u32>();
        let len = binary.len() * width;
        let capacity = binary.capacity() * width;
        let ptr = binary.as_mut_ptr();

        Vec::from_raw_parts(ptr as *mut u8, len, capacity)
    };

    let hash = crc32(&binary_u8[..]).await;
    let mut file = Vec::with_capacity(binary_u8.len() + size_of_val(&hash));
    file.extend(u32::to_le_bytes(hash));
    file.extend_from_slice(&binary_u8[..]);

    fs::write(create_cache_path(path), file)?;

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
