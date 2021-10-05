use crate::crc::crc32;
use png::{BitDepth, ColorType};
use std::{
    borrow::Cow,
    fmt::Debug,
    io::{self, ErrorKind},
    mem::size_of_val,
    path::{Path, PathBuf},
};
use tokio::fs;
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
    let bytes = tokio::fs::read(&path).await?;
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

#[inline(always)]
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
    use tokio::io::AsyncReadExt;

    let shader = fs::read(&path).await?;
    let hash = crc32(&shader[4..]);

    let mut handle = fs::File::open(&path).await?;
    let mut buf = [0u8; 4];

    handle.read_exact(&mut buf).await?;

    // cached shader matches shader source.
    if hash == u32::from_le_bytes(buf) {
        use std::convert::TryInto;

        let shader: Vec<u32> = shader[4..] // ignoring the 4 checksum bytes
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        log::info!("Reading cached shader from: {:?}", path);
        Ok(device.create_shader_module(&wgpu::ShaderModuleDescriptor {
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
        back::spv,
        front::glsl,
        valid::{ValidationFlags, Validator},
    };

    let stage = match stage {
        ShaderStages::COMPUTE => naga::ShaderStage::Compute,
        ShaderStages::VERTEX => naga::ShaderStage::Vertex,
        ShaderStages::FRAGMENT => naga::ShaderStage::Fragment,
        _ => unreachable!(),
    };

    let src = fs::read_to_string(&path).await?;
    let module = {
        let mut entry_points = naga::FastHashMap::default();
        entry_points.insert("main".to_string(), stage);
        glsl::parse_str(&src, &glsl::Options { entry_points, defines: Default::default() }).unwrap()
    };

    let analysis = if cfg!(debug_assertions) {
        Validator::new(ValidationFlags::all()).validate(&module).unwrap()
    } else {
        Validator::new(ValidationFlags::empty()).validate(&module).unwrap()
    };

    let binary = spv::write_vec(&module, &analysis, &spv::Options::default())
        .map_err(io::ErrorKind::InvalidData)?;

    let hash = crc32(binary.as_binary_u8());
    let mut file = Vec::with_capacity(binary.as_binary_u8().len() + size_of_val(&hash));
    file.extend(u32::to_le_bytes(hash));
    file.extend_from_slice(binary.as_binary_u8());

    fs::write(create_cache_path(path), file).await?;

    Ok(device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::SpirV(Cow::Borrowed(binary.as_binary())),
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

    let hash = crc32(binary.as_binary_u8());
    let mut file = Vec::with_capacity(binary.as_binary_u8().len() + size_of_val(&hash));
    file.extend(u32::to_le_bytes(hash));
    file.extend_from_slice(binary.as_binary_u8());

    fs::write(create_cache_path(path), file).await?;

    Ok(device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::SpirV(Cow::Borrowed(binary.as_binary())),
    }))
}
