#version 330 core
in vec3 v_FragPos;
in vec3 v_Normal;
in vec2 v_UV;
in vec3 v_Color;

uniform vec3 u_LightDir;
uniform vec3 u_CamPos;

out vec4 FragColor;

void main() {
    vec3 norm = normalize(v_Normal);
    vec3 lightDir = normalize(u_LightDir);

    // Ambient
    float ambientStrength = 0.35;
    vec3 ambient = ambientStrength * vec3(0.85, 0.90, 0.95);

    // Diffuse
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 0.98, 0.92);

    // Procedural grid pattern on horizontal surfaces
    float grid = 1.0;
    if (abs(norm.y) > 0.7) {
        vec2 coord = v_FragPos.xz;
        vec2 gridVec = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
        float line = min(gridVec.x, gridVec.y);
        grid = 0.85 + 0.15 * min(line, 1.0);
    }

    vec3 result = (ambient + diffuse) * v_Color * grid;
    FragColor = vec4(result, 1.0);
}
