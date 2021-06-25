[[block]]
struct Uniforms {
  view: mat4x4<f32>;
  proj: mat4x4<f32>;
  model: mat4x4<f32>;
};

[[group(0), binding(0)]] var<uniform> uniforms: Uniforms;

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
  out.pos = in.pos;

  let model_pos: vec4<f32> = uniforms.model * in.pos; // can't do this
  let view_pos: vec4<f32> = uniforms.view * model_pos;

  out.pos = uniforms.proj * view_pos;
  return out;
}

[[binding(0), group(0)]]
var t_diffuse: texture_2d<f32>;
[[binding(1), group(1)]]
var s_diffuse: sampler;

[[stage(fragment)]]
fn main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex);
}
