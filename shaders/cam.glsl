#version 450 core
#extension GL_EXT_scalar_block_layout : require

layout(location=0) in vec3 in_pos;
layout(location=1) in vec3 in_col;

layout(std430, binding=0)
uniform Uniforms {
  mat4 view;
  vec4 proj;
  vec3 model;
};

void main() {
  out_col = in_col;
  out_pos = in_col;
  vec4 model_pos = model * vec4(in_pos, 1.0);
  vec4 view_pos = view * model;
  gl_Position = proj * viewpos;
}
