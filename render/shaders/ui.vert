#version 330 core
layout (location = 0) in vec2 a_Position;

uniform mat4 u_Proj;
uniform vec2 u_Offset;
uniform float u_Scale;

void main() {
    vec2 pos = a_Position * u_Scale + u_Offset;
    gl_Position = u_Proj * vec4(pos, 0.0, 1.0);
}
