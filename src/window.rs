use std::{
    borrow::Cow,
    fmt::Debug,
    io,
    mem::size_of,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::fs;
use wgpu::*;
use winit::{
    dpi::{PhysicalSize, Size},
    event_loop::EventLoop,
    window::{Icon, Window as WindowHandle, WindowBuilder},
};

const MIN_WIN_SIZE: Size = Size::Physical(PhysicalSize::new(350, 250));

#[cfg(not(target_os = "windows"))]
fn generate_window(title: &str, icon: Option<Icon>, event_loop: &EventLoop<()>) -> WindowHandle {
    WindowBuilder::new()
        .with_title(title)
        .with_always_on_top(true)
        .with_window_icon(icon)
        .with_min_inner_size(MIN_WIN_SIZE)
        .build(event_loop)
        .unwrap()
}

#[cfg(target_os = "windows")]
fn generate_window(title: &str, icon: Option<Icon>, event_loop: &EventLoop<()>) -> WindowHandle {
    use winit::platform::windows::WindowBuilderExtWindows;

    WindowBuilder::new()
        .with_title(title)
        .with_transparent(true)
        .with_drag_and_drop(true)
        .with_always_on_top(true)
        .with_taskbar_icon(icon.clone())
        .with_window_icon(icon)
        .with_min_inner_size(MIN_WIN_SIZE)
        .build(&event_loop)
        .unwrap()
}

fn create_cache_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let cache_path = path.as_ref().with_extension("spv");
    let cache_path = cache_path.file_name().unwrap();
    if cfg!(target_os = "unix") {
        Path::new("/tmp").join(cache_path)
    } else {
        Path::new(&std::env::var("TMP").unwrap()).join(cache_path)
    }
}

#[cfg(all(feature = "naga", not(feature = "shaderc")))]
async fn generate_vulkan_shader_module<P: AsRef<Path> + Debug>(
    path: P,
    stage: ShaderStage,
    flags: ShaderFlags,
    device: &Device,
) -> io::Result<ShaderModule> {
    use naga::{
        back::spv,
        front::glsl,
        valid::{ValidationFlags, Validator},
    };

    let stage = match stage {
        ShaderStage::COMPUTE => naga::ShaderStage::Compute,
        ShaderStage::VERTEX => naga::ShaderStage::Vertex,
        ShaderStage::FRAGMENT => naga::ShaderStage::Fragment,
        _ => unsafe { std::hint::unreachable_unchecked() },
    };

    let cache_path = create_cache_path(&path);
    log::info!("Reading shader from: {:?}", path);

    let src = fs::read_to_string(&path).await?;
    let module = {
        let mut entry_points = naga::FastHashMap::default();
        entry_points.insert("main".to_string(), stage);
        glsl::parse_str(
            &src,
            &glsl::Options {
                entry_points,
                defines: Default::default(),
            },
        )
        .unwrap()
    };

    let analysis = if cfg!(debug_assertions) {
        Validator::new(ValidationFlags::all())
            .validate(&module)
            .unwrap()
    } else {
        Validator::new(ValidationFlags::empty())
            .validate(&module)
            .unwrap()
    };

    let spv = spv::write_vec(&module, &analysis, &spv::Options::default())
        .map_err(io::ErrorKind::InvalidData)?;

    let binary = spv
        .iter()
        .fold(Vec::with_capacity(spv.len() * 4), |mut v, w| {
            v.extend_from_slice(&w.to_le_bytes());
            v
        });

    fs::write(cache_path, binary.as_slice()).await?;
    let binary: Vec<u32> = unsafe { std::mem::transmute(binary) };

    // TODO: do a checksum.
    Ok(device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::SpirV(Cow::from(binary)),
        flags,
    }))
}

#[cfg(feature = "shaderc")]
async fn generate_vulkan_shader_module<P: AsRef<Path> + Debug>(
    path: P,
    stage: ShaderStage,
    flags: ShaderFlags,
    device: &Device,
) -> io::Result<ShaderModule> {
    let stage = match stage {
        ShaderStage::COMPUTE => shaderc::ShaderKind::Compute,
        ShaderStage::VERTEX => shaderc::ShaderKind::Vertex,
        ShaderStage::FRAGMENT => shaderc::ShaderKind::Fragment,
        _ => unsafe { std::hint::unreachable_unchecked() },
    };

    let cache_path = create_cache_path(path.as_ref());
    log::info!("Reading shader from: {:?}", path);

    let mut compiler = shaderc::Compiler::new().unwrap();
    let src = fs::read_to_string(&path).await?;
    let binary = compiler
        .compile_into_spirv(&src, stage, cache_path.to_str().unwrap(), "main", None)
        .unwrap_or_else(|err| panic!("{}", err));

    fs::write(cache_path, binary.as_binary_u8()).await?;

    // TODO: do a checksum.
    Ok(device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::SpirV(Cow::from(binary.as_binary())),
        flags,
    }))
}

pub struct Display {
    window: Arc<WindowHandle>,
    event_loop: EventLoop<()>,
    size: PhysicalSize<u32>,
    pub instance: Instance,
    pub surface: Surface,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub backend: BackendBit,
}

impl Display {
    async fn new() -> Result<Self, wgpu::RequestDeviceError> {
        let event_loop = EventLoop::new();
        let window = {
            #[cfg(any(target_os = "windows", target_os = "macos"))]
            const ICON: &[u8] = include_bytes!("../res/iconx256.png");
            #[cfg(target_os = "linux")]
            const ICON: &[u8] = include_bytes!("../res/iconx64.png");

            let decoder = png::Decoder::new(ICON);
            let (info, mut reader) = decoder.read_info().unwrap();
            let mut icon = vec![0; info.buffer_size()];
            reader.next_frame(&mut icon).unwrap();
            let icon = Icon::from_rgba(icon, info.width, info.height).ok();

            Arc::new(generate_window("Remotely possible", icon, &event_loop))
        };

        log::info!("Finished creating window..");

        #[cfg(any(target_os = "windows", target_os = "linux"))]
        let backend = BackendBit::VULKAN;
        #[cfg(target_os = "macos")]
        let backend = BackendBit::METAL;

        let instance = Instance::new(backend);
        let power_preference = PowerPreference::HighPerformance;

        let size = window.inner_size();
        let surface = unsafe { instance.create_surface(window.clone().as_ref()) };
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference,
                compatible_surface: Some(&surface),
            })
            .await
            .ok_or(wgpu::RequestDeviceError)?;

        let trace_dir = std::env::var("WGPU_TRACE");
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Primary device"),
                    features: adapter.features(),
                    limits: adapter.limits(),
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
            )
            .await?;

        log::info!("Requested device..");

        Ok(Self {
            window,
            event_loop,
            instance,
            size,
            surface,
            adapter,
            device,
            queue,
            backend,
        })
    }
}

pub struct Window {
    pub display: Display,
    pub swap_chains: Vec<SwapChains>,
    pub bind_groups: Vec<BindGroups>,
    pub pipelines: Vec<Pipelines>,
    pub uniform_buffers: Vec<Buffer>,
    pub vertex_buffers: Vec<Buffer>,
}

impl Window {
    pub async fn new() -> Self {
        let display = Display::new().await.unwrap();
        let swap_chain_desc = SwapChainDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT,
            format: display
                .adapter
                .get_swap_chain_preferred_format(&display.surface)
                .unwrap(),
            width: display.size.width,
            height: display.size.height,
            present_mode: PresentMode::Mailbox,
        };

        let swap_chain = display
            .device
            .create_swap_chain(&display.surface, &swap_chain_desc);

        log::info!("Initializing vertex data..");
        let vertex_buffer = display.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera vertex buffer"),
            size: 3 * size_of::<crate::Vertex>() as u64,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            mapped_at_creation: true,
        });

        display
            .queue
            .write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(crate::VERTICES));

        log::info!("Initializing uniform buffer data..");
        let uniform_buffer = display.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera uniform buffer"),
            size: size_of::<crate::CameraUniform>() as u64,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            mapped_at_creation: true,
        });

        let uniform = crate::CameraUniform::new(display.size.width, display.size.height);
        display
            .queue
            .write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

        let bind_group_layout =
            display
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Primary bind group layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStage::FRAGMENT | ShaderStage::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(
                                size_of::<crate::CameraUniform>() as u64
                            ),
                        },
                        count: None,
                    }],
                });

        let bind_group = display.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let mut flags = ShaderFlags::VALIDATION;
        if display.backend == BackendBit::METAL | BackendBit::VULKAN {
            flags |= ShaderFlags::EXPERIMENTAL_TRANSLATION;
        }

        let module = generate_vulkan_shader_module(
            "./shaders/cam.glsl",
            ShaderStage::VERTEX,
            flags,
            &display.device,
        )
        .await
        .unwrap();

        let pipeline_layout = display
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create the render pipelines. These describe how the data will flow through the GPU, and what
        // constraints and modifiers it will have.
        let pipeline = display
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Primary pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &module,
                    entry_point: "main",
                    // TODO: add objects to the world
                    buffers: &[VertexBufferLayout {
                        array_stride: size_of::<crate::Vertex>() as BufferAddress,
                        step_mode: InputStepMode::Vertex,
                        attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                    }],
                },
                fragment: None,
                // How the triangles will be rasterized
                primitive: PrimitiveState {
                    // type of data we are passing in
                    topology: PrimitiveTopology::TriangleList,
                    front_face: FrontFace::Cw,
                    ..Default::default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: CompareFunction::Less,
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState::default(),
            });

        Self {
            display,
            swap_chains: SwapChains::new_as_vec(swap_chain, swap_chain_desc),
            bind_groups: BindGroups::new_as_vec(bind_group, bind_group_layout),
            pipelines: Pipelines::new_as_vec(pipeline, pipeline_layout),
            uniform_buffers: vec![uniform_buffer],
            vertex_buffers: vec![vertex_buffer],
        }
    }

    pub fn get_event_loop<'a>(&mut self) -> EventLoop<()> {
        std::mem::replace(&mut self.display.event_loop, EventLoop::new())
    }

    pub fn get_window_handle(&self) -> Arc<WindowHandle> {
        self.display.window.clone()
    }
}

macro_rules! gen_struct_pair {
    ($name:ident: { $($field:ident: $val:ident $(,)?)+ } ) => {
        pub struct $name {
            $(pub $field: $val),+
        }

        impl $name {
            fn new($($field: $val),+) -> Self {
                Self { $($field),+ }
            }

            fn new_as_vec($($field: $val),+) -> Vec<Self> {
                vec![Self { $($field),+ }]
            }
        }
    }

}

gen_struct_pair![SwapChains: {
    chain: SwapChain,
    desc: SwapChainDescriptor
}];

gen_struct_pair![Pipelines: {
    pipe: RenderPipeline,
    layout: PipelineLayout,
}];

gen_struct_pair![BindGroups: {
    group: BindGroup,
    layout: BindGroupLayout,
}];
