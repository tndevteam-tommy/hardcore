#version 330 core
in vec3 v_FragPos;
in vec3 v_Normal;
in vec2 v_UV;
in float v_BoneID;

uniform vec3 u_LightDir;
uniform vec3 u_CamPos;

out vec4 FragColor;

void main() {
    vec3 norm = normalize(v_Normal);
    vec3 lightDir = normalize(u_LightDir);
    vec3 viewDir = normalize(u_CamPos - v_FragPos);

    // Sleeve vs Skin tone
    vec3 baseColor;
    int b = int(v_BoneID);
    if (b == 0 || b == 1 || b == 19 || b == 20) {
        // Sleeves (Upper arm / Clavicle)
        baseColor = vec3(0.14, 0.15, 0.18);
    } else {
        // Skin & Athletic Gloves (Forearms, palms, fingers)
        baseColor = vec3(0.78, 0.58, 0.46);
    }

    // Ambient
    vec3 ambient = 0.45 * baseColor;

    // Diffuse
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * baseColor * 0.75;

    // Specular
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfDir), 0.0), 16.0);
    vec3 specular = vec3(0.25) * spec;

    FragColor = vec4(ambient + diffuse + specular, 1.0);
}
