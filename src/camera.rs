#[rustfmt::skip::macros(matrix, vector)]

use std::f64::consts::FRAC_PI_2;
use winit::dpi::PhysicalSize;

use crate::math::{self, Point3, Radians};
use crate::uniforms;

#[derive(Debug)]
pub struct Camera {
    pub uniform: uniforms::Camera,
    uniform_buffer: wgpu::Buffer,
    position: Point3,
    yaw: Radians,
    pitch: Radians,
    roll: Radians,
    aspect_ratio: f64,
    fov: Radians,
    near: f64,
    far: f64,
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
            position: vector![1.0, 1.5, -0.9],
            pitch: FRAC_PI_2,
            aspect_ratio: size.width as f64 / size.height as f64,
            fov: FRAC_PI_2,
            near: 10.0,
            far: 1000.0,
            roll: 0.0,
            yaw: 0.0,
        };

        this.uniform = uniforms::Camera { proj: this.view_projection(), position: this.position() };
        queue.write_buffer(&this.uniform_buffer, 0, crate::cast_bytes(&this.uniform));
        this
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.aspect_ratio = size.width as f64 / size.height as f64;
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

    fn position(&self) -> math::Vector4 {
        let math::Vector3 { x, y, z } = self.position;

        vector![x, y, z, 1.0]
    }

    fn projection(&self) -> math::Matrix4<f64> {
        let f = 1.0 / (self.fov / 2.0).tan();
        let a = f / self.aspect_ratio;
        let b = self.near / (self.far - self.near);
        let c = self.far * b;

        matrix![
            a,    0.0,  0.0, 0.0,
            0.0, -f,    0.0, 0.0,
            0.0,  0.0,  b,   c,
            0.0,  0.0, -1.0, 0.0
        ]
    }

    /// Generates view matrix, used to bring world into world/camera space.
    fn view_projection(&self) -> math::Matrix4<f64> {
        let mut target = vector![self.yaw.cos(), self.pitch.sin(), self.yaw.sin()];

        // Normalize the vector
        let magnitude = (target.x * target.x) + (target.y * target.y) + (target.z * target.z);
        let magnitude = magnitude.sqrt();

        target.x /= magnitude;
        target.y /= magnitude;
        target.z /= magnitude;

        math::look_at_rh(&self.position, &target, &vector![0.0, 1.0, 0.0])
    }
}
