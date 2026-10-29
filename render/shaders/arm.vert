#version 330 core
layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_UV;
layout (location = 3) in vec4 a_BoneIDs;
layout (location = 4) in vec4 a_Weights;

uniform mat4 u_Bones[38];
uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Proj;

out vec3 v_FragPos;
out vec3 v_Normal;
out vec2 v_UV;
out float v_BoneID;

void main() {
    int b0 = int(a_BoneIDs.x);
    int b1 = int(a_BoneIDs.y);
    int b2 = int(a_BoneIDs.z);
    int b3 = int(a_BoneIDs.w);

    mat4 boneTransform = u_Bones[b0] * a_Weights.x;
    if (a_Weights.y > 0.0) boneTransform += u_Bones[b1] * a_Weights.y;
    if (a_Weights.z > 0.0) boneTransform += u_Bones[b2] * a_Weights.z;
    if (a_Weights.w > 0.0) boneTransform += u_Bones[b3] * a_Weights.w;

    vec4 localPos = vec4(a_Position, 1.0);
    vec4 worldPos = boneTransform * localPos;
    v_FragPos = worldPos.xyz;

    mat3 rotMat = mat3(boneTransform);
    v_Normal = normalize(rotMat * a_Normal);
    v_UV = a_UV;
    v_BoneID = float(b0);

    gl_Position = u_Proj * u_View * worldPos;
}
