#version 450 core
#extension GL_EXT_scalar_block_layout : require

layout(location=0) in vec3 in_pos;
layout(location=1) in vec2 in_tex;

layout(location=0) out vec2 out_tex;

layout(std430, set=1, binding=0)
uniform CameraUniforms {
  mat4 view;
  mat4 proj;
  mat4 model;
};

void main() {
  out_tex = in_tex;

  mat4 mvp = proj * view * model;
  gl_Position = mvp * vec4(in_pos, 1.0);
}
