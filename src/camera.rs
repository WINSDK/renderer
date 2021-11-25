use std::f32::consts::FRAC_PI_2;
use winit::dpi::PhysicalSize;

use crate::math::Radians;
use crate::uniforms;

#[derive(Debug)]
pub struct Camera {
    pub uniform: uniforms::Camera,
    uniform_buffer: wgpu::Buffer,
    position: na::Point3<f32>,
    yaw: Radians<f32>,
    pitch: Radians<f32>,
    roll: Radians<f32>,
    aspect_ratio: f32,
    fov: Radians<f32>,
    near: f32,
    far: f32,
}

impl Camera {
    /// Creates a new camera with sensible defaults for a player.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, size: PhysicalSize<u32>) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera uniform buffer"),
            size: std::mem::size_of::<uniforms::Camera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut this = Self {
            uniform: unsafe { std::mem::zeroed() },
            uniform_buffer,
            position: na::Point3::new(0.0, 0.0, 0.0),
            pitch: Radians(FRAC_PI_2),
            aspect_ratio: size.width as f32 / size.height as f32,
            fov: Radians(FRAC_PI_2),
            near: 10.0,
            far: 1000.0,
            roll: Radians(0.0),
            yaw: Radians(0.0),
        };

        this.uniform = uniforms::Camera { proj: this.view_projection(), position: this.position() };
        queue.write_buffer(&this.uniform_buffer, 0, crate::cast_bytes(&this.uniform));
        this
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.aspect_ratio = new_size.width as f32 / new_size.height as f32;
    }

    pub fn buffer(&self) -> wgpu::BindingResource {
        self.uniform_buffer.as_entire_binding()
    }

    pub fn update(&mut self) {
        self.uniform.position = self.position();
        self.uniform.proj = self.projection() * self.view_projection();
    }

    //pub fn handle_input(&mut self) {
    //    let (yaw_sin, yaw_cos) = self.yaw.0.sin_cos();
    //    let forward = na::Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
    //    let right = na::Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
    //    self.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
    //    self.position += right * (self.amount_right - self.amount_left) * self.speed * dt;
    //}

    fn position(&self) -> na::Vector4<f32> {
        self.position.to_homogeneous()
    }

    fn projection(&self) -> na::Matrix4<f32> {
        na::Matrix4::new_perspective(self.fov.0, self.aspect_ratio, self.near, self.far)
    }

    /// Generates view matrix, used to bring world into world/camera space.
    fn view_projection(&self) -> na::Matrix4<f32> {
        let mut target = na::Point3::new(self.yaw.0.cos(), self.pitch.0.sin(), self.yaw.0.sin());

        // Normalize the vector
        let magnitude: f32 = (target.x * target.x) + (target.y * target.y) + (target.z * target.z);
        let magnitude = magnitude.sqrt();

        target.x /= magnitude;
        target.y /= magnitude;
        target.z /= magnitude;

        na::Matrix4::look_at_rh(&self.position, &target, &na::Vector3::y())
    }
}
