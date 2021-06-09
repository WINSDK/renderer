#![allow(dead_code)]

use std::mem::size_of;
use wgpu::*;
use winit::{
    event_loop::EventLoop,
    window::{Icon, Window as WindowHandle, WindowBuilder},
};

#[cfg(not(target_os = "windows"))]
fn generate_window(title: &str, icon: Option<Icon>, event_loop: &EventLoop<()>) -> WindowHandle {
    WindowBuilder::new()
        .with_title(title)
        .with_always_on_top(true)
        .with_window_icon(icon)
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
        .with_taskbar_icon(icon)
        .with_window_icon(icon)
        .build(&event_loop)
        .unwrap()
}

struct Display {
    window: winit::window::Window,
    size: winit::dpi::PhysicalSize<u32>,
    event_loop: EventLoop<()>,
    instance: Instance,
    surface: Surface,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    backend: BackendBit,
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

            generate_window("Remotely possible", icon, &event_loop)
        };

        log::info!("finished creating window..");

        #[cfg(any(target_os = "windows", target_os = "linux"))]
        let backend = BackendBit::VULKAN;
        #[cfg(target_os = "macos")]
        let backend = BackendBit::METAL;

        let instance = Instance::new(backend);
        let power_preference = PowerPreference::HighPerformance;

        let size = window.inner_size();
        let surface = unsafe { instance.create_surface(&window) };
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

        log::info!("requested device..");

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
    display: Display,
    swap_chains: Vec<SwapChains>,
    bind_groups: Vec<BindGroups>,
    pipelines: Vec<Pipelines>,
    uniform_buffers: Vec<Buffer>,
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

        let pipeline_layout = display
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let uniform_buffer = display.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera uniform buffer"),
            size: size_of::<crate::CameraUniform>() as u64,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            mapped_at_creation: false, // no clue what this does so i'll leave it as false
        });

        log::info!("initializing uniform buffer data..");

        let uniform = crate::CameraUniform::new(display.size.width, display.size.height);
        display
            .queue
            .write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));

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

        let module = display
            .device
            .create_shader_module(&include_spirv!("../shaders/cam.glsl"));

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
                    buffers: &[],
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
        }
    }
}

macro_rules! gen_struct_pair {
    ($name:ident: { $($field:ident: $val:ident $(,)?)+ } ) => {
        struct $name {
            $($field: $val),+
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
    swaper: SwapChain,
    descer: SwapChainDescriptor
}];

gen_struct_pair![Pipelines: {
    pipe: RenderPipeline,
    layout: PipelineLayout,
}];

gen_struct_pair![BindGroups: {
    group: BindGroup,
    layout: BindGroupLayout,
}];
