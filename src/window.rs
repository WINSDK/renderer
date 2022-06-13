use crate::{camera, uniforms};
use std::{mem::size_of, sync::Arc};
use wgpu::*;
use winit::{
    dpi::{PhysicalSize, Size},
    event_loop::EventLoop,
    window::{Icon, Window as WindowHandle, WindowBuilder},
};

pub const MIN_REAL_SIZE: PhysicalSize<u32> = PhysicalSize::new(350, 250);
pub const MIN_WIN_SIZE: Size = Size::Physical(MIN_REAL_SIZE);

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
        .build(event_loop)
        .unwrap()
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
    pub backend: Backends,
}

impl Display {
    async fn new() -> Result<Self, wgpu::RequestDeviceError> {
        let event_loop = EventLoop::new();
        let window = {
            #[cfg(target_os = "linux")]
            let icon = crate::read_png("./res/iconx64.png").await.unwrap();
            #[cfg(any(target_os = "windows", target_os = "macos"))]
            let icon = crate::read_png("./res/iconx256.png").await.unwrap();
            let icon = Icon::from_rgba(icon.data, icon.width, icon.height).ok();
            Arc::new(generate_window("Remotely possible", icon, &event_loop))
        };

        log::info!("Finished creating window..");

        #[cfg(any(target_os = "windows", target_os = "linux"))]
        let backend = Backends::VULKAN;
        #[cfg(target_os = "macos")]
        let backend = Backends::METAL;

        let instance = Instance::new(backend);

        let size = window.inner_size();
        let surface = unsafe { instance.create_surface(&*window.clone()) };
        let adapter = instance
            .enumerate_adapters(backend)
            .find(|adapter| surface.get_preferred_format(adapter).is_some())
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

        Ok(Self { window, event_loop, instance, size, surface, adapter, device, queue, backend })
    }
}

pub struct Window {
    pub display: Display,
    pub camera: camera::Camera,
    pub surfaces: Vec<Surfaces>,
    pub bind_groups: Vec<BindGroups>,
    pub pipelines: Vec<Pipelines>,
    pub vertex_buffers: Vec<Buffer>,
    pub index_buffers: Vec<Buffer>,
    pub index_count: usize,
}

impl Window {
    pub async fn new() -> Self {
        let display = Display::new().await.unwrap();

        let (vertices, indices) = uniforms::create_vertices();
        log::info!("Reading texture and writting to queue..");

        let surface = {
            let window_size = display.window.inner_size();
            let surface = unsafe { display.instance.create_surface(display.window.as_ref()) };

            let format =
                surface.get_preferred_format(&display.adapter).unwrap_or(TextureFormat::Bgra8Unorm);
            let config = SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format,
                present_mode: PresentMode::Mailbox,
                width: window_size.width,
                height: window_size.height,
            };

            surface.configure(&display.device, &config);
            Surfaces::new(surface, config, format)
        };

        let texture = crate::texture::Texture::new(
            "./test_cases/joe_biden.png",
            &display.device,
            &display.queue,
        )
        .await
        .unwrap();

        let texture_bind_group_layout =
            display.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Texture bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let texture_bind_group = display.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Texture bind group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
        });

        log::info!("Initializing vertex data..");
        let vertex_buffer = display.device.create_buffer(&BufferDescriptor {
            label: Some("Vertex buffer"),
            size: (size_of::<uniforms::Vertex>() * vertices.len()) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        display.queue.write_buffer(&vertex_buffer, 0, crate::cast_slice(vertices.as_slice()));

        log::info!("Initializing indices..");
        let index_buffer = display.device.create_buffer(&BufferDescriptor {
            label: Some("Indices buffer"),
            size: (size_of::<u16>() * indices.len()) as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        display.queue.write_buffer(&index_buffer, 0, crate::cast_slice(indices.as_slice()));

        let camera = camera::Camera::new(&display.device, &display.queue, display.size);
        let uniform_bind_group_layout =
            display.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Camera uniform bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(size_of::<uniforms::Camera>() as u64),
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = display.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Camera uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[BindGroupEntry { binding: 0, resource: camera.buffer() }],
        });

        let now = std::time::Instant::now();
        let (vert_module, frag_module) = tokio::try_join!(
            crate::generate_vulkan_shader_module(
                "./shaders/cam.glsl",
                ShaderStages::VERTEX,
                &display.device,
            ),
            crate::generate_vulkan_shader_module(
                "./shaders/cam_frag.glsl",
                ShaderStages::FRAGMENT,
                &display.device,
            ),
        )
        .unwrap();

        log::info!("took {:#?} to generate shaders", now.elapsed());

        let pipeline_layout = display.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Pipeline layout"),
            bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create the render pipelines. These describe how the data will flow through the GPU, and what
        // constraints and modifiers it will have.
        let pipeline = display.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Primary pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &vert_module,
                entry_point: "main",
                // TODO: add objects to the world
                buffers: &[VertexBufferLayout {
                    array_stride: size_of::<uniforms::Vertex>() as BufferAddress,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &frag_module,
                entry_point: "main",
                targets: &[wgpu::ColorTargetState {
                    format: surface.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            // How the triangles will be rasterized
            primitive: PrimitiveState {
                // type of data we are passing in
                topology: PrimitiveTopology::TriangleList,
                front_face: FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState {
                // The number of samples for multisampling
                count: 1,
                // a mask for what samples are active: !0 means all of them
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            display,
            camera,
            surfaces: vec![surface],
            bind_groups: vec![
                BindGroups::new(texture_bind_group, texture_bind_group_layout),
                BindGroups::new(uniform_bind_group, uniform_bind_group_layout),
            ],
            pipelines: Pipelines::new_as_vec(pipeline, pipeline_layout),
            vertex_buffers: vec![vertex_buffer],
            index_buffers: vec![index_buffer],
            index_count: indices.len(),
        }
    }

    pub fn redraw(&mut self) {
        let frame = self.surfaces[0].handle.get_current_texture().unwrap();
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .display
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some("Primary encoder") });

        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color { r: 0.1, g: 0.5, b: 0.8, a: 1.0 }),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&self.pipelines[0].pipe);
        render_pass.set_bind_group(0, &self.bind_groups[0].group, &[]);
        render_pass.set_bind_group(1, &self.bind_groups[1].group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffers[0].slice(..));
        render_pass.set_index_buffer(self.index_buffers[0].slice(..), IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.index_count as u32, 0, 0..1);

        // Required because render_pass and queue takes a &T.
        drop(render_pass);

        self.display.queue.submit(Some(encoder.finish()));

        // Schedule texture to be renderer on surface.
        frame.present();
    }

    pub fn get_event_loop(&mut self) -> EventLoop<()> {
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

        #[allow(dead_code)]
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

gen_struct_pair![Surfaces: {
    handle: Surface,
    config: SurfaceConfiguration,
    format: TextureFormat,
}];

gen_struct_pair![Pipelines: {
    pipe: RenderPipeline,
    layout: PipelineLayout,
}];

gen_struct_pair![BindGroups: {
    group: BindGroup,
    layout: BindGroupLayout,
}];
