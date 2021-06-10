#version 450 core
#extension GL_EXT_scalar_block_layout : require

layout(location=0) in vec3 in_pos;
layout(location=1) in vec3 in_col;

//layout(location=2) in vec3 out_pos;
//layout(location=3) in vec3 out_col;

layout(std430, binding=0)
uniform CameraUniforms {
  mat4 view;
  mat4 proj;
  mat4 model;
};

void main() {
  //out_col = in_col;
  //out_pos = in_col;
  vec4 model_pos = model * vec4(in_pos, 1.0);
  vec4 view_pos = view * model_pos;
  gl_Position = proj * view_pos;
}
