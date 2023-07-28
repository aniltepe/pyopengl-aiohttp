# version 330 core

out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
} fs_in;

uniform vec3 lightPos;
uniform vec3 viewPos;

vec3 BlinnPhong(vec3 normal, vec3 fragPos, vec3 lightColor)
{
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    vec3 halfwayDir = normalize(lightDir + viewDir);  
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = spec * lightColor;
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (distance * distance);
    diffuse *= attenuation;
    specular *= attenuation;
    return diffuse + specular;
}

void main() {
    vec3 color = vec3(0.8235, 0.6078, 0.5529);
    vec3 lighting = BlinnPhong(normalize(fs_in.Normal), fs_in.FragPos, vec3(0.3));
    color *= lighting;
    color = pow(color, vec3(1.0 / 2.2));
    FragColor = vec4(color, 1.0);
}