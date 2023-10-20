# version 330 core

out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoord;
} fs_in;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform sampler2D texture1;

vec3 BlinnPhong(vec3 normal, vec3 fragPos, vec3 lightColor)
{
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    vec3 halfwayDir = normalize(lightDir + viewDir);  
    vec3 specular = vec3(0.0);
    if (diff > 0.0) {
        float spec = pow(max(dot(normal, halfwayDir), 0.0), 5.0);
        specular = spec * lightColor;
    }
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (distance * distance);
    diffuse *= attenuation;
    specular *= attenuation;
    return diffuse + specular;
}

void main() {
    vec3 color = texture(texture1, fs_in.TexCoord).rgb;
    vec3 lighting = BlinnPhong(normalize(fs_in.Normal), fs_in.FragPos, vec3(0.3));
    color *= lighting;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    FragColor = vec4(color, 1.0);
}