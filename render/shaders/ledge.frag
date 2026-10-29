#version 330 core
uniform vec3 u_GlowColor;
uniform float u_Alpha;

out vec4 FragColor;

void main() {
    FragColor = vec4(u_GlowColor, u_Alpha);
}
