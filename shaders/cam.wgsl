[[block]]
struct Camera {
  proj: mat4x4<f32>;
  pos: vec4<f32>;
};

[[group(1), binding(0)]] var<uniform> camera: Camera;

struct VertexInput {
    [[location(0)]] pos: vec3<f32>;
    [[location(1)]] tex: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] pos: vec4<f32>;
    [[location(1)]] tex: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;

  out.tex = in.tex;
  out.pos = camera.proj * vec4<f32>(in.pos, 1.0);

  return out;
}

[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex);
}
