# version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoord;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat3 normalMatrix;

void main()
{
    vs_out.FragPos = vec3(model * vec4(aPosition, 1.0));
    vs_out.Normal = normalMatrix * aNormal;
    vs_out.TexCoord = aTexCoord;
    gl_Position = projection * view * vec4(vs_out.FragPos, 1.0);
}