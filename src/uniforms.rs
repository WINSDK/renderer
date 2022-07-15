use crate::math;

#[allow(dead_code)]
#[rustfmt::skip]
pub const OPENGL_TO_WGPU: math::Matrix4<f64> = matrix![
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
];

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Vertex {
    pos: math::Vector3,
    tex: math::Vector2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub proj: math::Matrix4<f64>,
    pub position: math::Vector4,
}

pub fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    #[rustfmt::skip] let vertex_data = [
        // top (0, 0, 1)
        Vertex { pos: vector![-1.0, -1.0, 1.0], tex: vector![0.0, 0.0] },
        Vertex { pos: vector![1.0, -1.0, 1.0], tex: vector![1.0, 0.0] },
        Vertex { pos: vector![1.0, 1.0, 1.0], tex: vector![1.0, 1.0] },
        Vertex { pos: vector![-1.0, 1.0, 1.0], tex: vector![0.0, 1.0] },
        // bottom (0.0, 0, -1)
        Vertex { pos: vector![-1.0, 1.0, -1.0], tex: vector![1.0, 0.0] },
        Vertex { pos: vector![1.0, 1.0, -1.0], tex: vector![0.0, 0.0] },
        Vertex { pos: vector![1.0, -1.0, -1.0], tex: vector![0.0, 1.0] },
        Vertex { pos: vector![-1.0, -1.0, -1.0], tex: vector![1.0, 1.0] },
        // right (1.0, 0, 0)
        Vertex { pos: vector![1.0, -1.0, -1.0], tex: vector![0.0, 0.0] },
        Vertex { pos: vector![1.0, 1.0, -1.0], tex: vector![1.0, 0.0] },
        Vertex { pos: vector![1.0, 1.0, 1.0], tex: vector![1.0, 1.0] },
        Vertex { pos: vector![1.0, -1.0, 1.0], tex: vector![0.0, 1.0] },
        // left (-1.0, 0, 0)
        Vertex { pos: vector![-1.0, -1.0, 1.0], tex: vector![1.0, 0.0] },
        Vertex { pos: vector![-1.0, 1.0, 1.0], tex: vector![0.0, 0.0] },
        Vertex { pos: vector![-1.0, 1.0, -1.0], tex: vector![0.0, 1.0] },
        Vertex { pos: vector![-1.0, -1.0, -1.0], tex: vector![1.0, 1.0] },
        // front (0.0, 1, 0)
        Vertex { pos: vector![1.0, 1.0, -1.0], tex: vector![1.0, 0.0] },
        Vertex { pos: vector![-1.0, 1.0, -1.0], tex: vector![0.0, 0.0] },
        Vertex { pos: vector![-1.0, 1.0, 1.0], tex: vector![0.0, 1.0] },
        Vertex { pos: vector![1.0, 1.0, 1.0], tex: vector![1.0, 1.0] },
        // back (0.0, -1, 0)
        Vertex { pos: vector![1.0, -1.0, 1.0], tex: vector![0.0, 0.0] },
        Vertex { pos: vector![-1.0, -1.0, 1.0], tex: vector![1.0, 0.0] },
        Vertex { pos: vector![-1.0, -1.0, -1.0], tex: vector![1.0, 1.0] },
        Vertex { pos: vector![1.0, -1.0, -1.0], tex: vector![0.0, 1.0] },
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}
