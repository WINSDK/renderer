use bytemuck::{Pod, Zeroable};

#[allow(dead_code)]
#[rustfmt::skip]
pub const OPENGL_TO_WGPU: na::Matrix4<f32> = na::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub const VERTICES: &[Vertex] = &[
    Vertex {
        position: na::Vector3::new(0.0, 0.5, 0.0),
        color: na::Vector3::new(1.0, 0.0, 0.0),
    },
    Vertex {
        position: na::Vector3::new(-0.5, -0.5, 0.0),
        color: na::Vector3::new(0.0, 1.0, 0.0),
    },
    Vertex {
        position: na::Vector3::new(0.5, -0.5, 0.0),
        color: na::Vector3::new(0.0, 0.0, 1.0),
    },
];

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    position: na::Vector3<f32>,
    color: na::Vector3<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct CameraUniform {
    view: na::Matrix4<f32>,
    proj: na::Matrix4<f32>,
    model: na::Matrix4<f32>,
}

impl CameraUniform {
    pub fn new(width: u32, height: u32) -> Self {
        // camera is located at 0.0, 0.0, 1.0
        let eye = na::Point3::new(0.0, 0.0, 1.0);
        // looking at point 1.0, 0.0, 0.0
        let target = na::Point3::new(1.0, 0.0, 0.0);

        let proj = na::Perspective3::new(45.0, width as f32 / height as f32, 10.0, 400.0);
        let view = na::Isometry3::look_at_rh(&eye, &target, &na::Vector3::y());
        let model = na::Isometry3::new(na::Vector3::x(), na::zero());

        Self {
            view: view.to_homogeneous(),
            proj: proj.to_homogeneous(),
            model: model.to_homogeneous(),
        }
    }
}
