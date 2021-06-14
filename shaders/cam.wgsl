[[block]]
struct Uniforms {
  view: mat4x4<f32>;
  proj: mat4x4<f32>;
  model: mat4x4<f32>;
};

[[group(0), binding(0)]] var<uniform> uniforms: Uniforms;

struct Output {
  [[builtin(position)]] pos: vec4<f32>;
  [[location(0)]] col: vec3<f32>;
};

[[stage(vertex)]]
fn vs_main([[location(0)]] pos: vec3<f32>, [[location(1)]] col: vec3<f32>) -> Output {
  var out: Output;

  let model_pos: vec4<f32> = uniforms.model * vec4<f32>(pos, 1.0);
  let view_pos: vec4<f32> = uniforms.view * model_pos;

  out.pos = uniforms.proj * view_pos;
  return out;
}
