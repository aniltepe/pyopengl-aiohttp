#version 330 core
out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoord;
} fs_in;

struct Light {
    vec3 position;
    vec3 color;
    float constant;
    float linear;
    float quadratic;
};

uniform sampler2D albedoMap;
uniform samplerCube depthMap0;
uniform samplerCube depthMap1;
uniform Light light0;
uniform Light light1;
uniform vec3 viewPos;
uniform float farPlane;

vec3 gridSamplingDisk[20] = vec3[]
(
   vec3(1, 1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1, 1,  1), 
   vec3(1, 1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1),
   vec3(1, 1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1, 1,  0),
   vec3(1, 0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1, 0, -1),
   vec3(0, 1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0, 1, -1)
);

float ShadowCalculation(Light light, samplerCube depthMap) {
    vec3 fragToLight = fs_in.FragPos - light.position;
    float currentDepth = length(fragToLight);
    float shadow = 0.0;
    float bias = 0.15;
    float viewDistance = length(viewPos - fs_in.FragPos);
    float diskRadius = (1.0 + (viewDistance / farPlane)) / 25.0;
    for (int i = 0; i < 20; ++i) {
        float closestDepth = texture(depthMap, fragToLight + gridSamplingDisk[i] * diskRadius).r;
        closestDepth *= farPlane;
        if(currentDepth - bias > closestDepth)
            shadow += 1.0;
    }
    shadow /= float(20);
    return shadow;
}

vec3 LightCalculation(vec3 albedo, vec3 normal, Light light, samplerCube depthMap) {
    vec3 lightColor = light.color;
    vec3 ambient = 0.3 * albedo;
    vec3 lightDir = normalize(light.position - fs_in.FragPos);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    vec3 specular = vec3(0.0);
    if (diff > 0.0) {
        float spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
        specular = spec * lightColor;
    }
    float shadow = ShadowCalculation(light, depthMap);
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * albedo;
    return lighting;
}

void main()
{           
    vec3 albedo = texture(albedoMap, fs_in.TexCoord).rgb;
    vec3 normal = normalize(fs_in.Normal);
    vec3 lighting1 = LightCalculation(albedo, normal, light0, depthMap0);
    vec3 lighting2 = LightCalculation(albedo, normal, light1, depthMap1);
    
    FragColor = vec4(lighting1 + lighting2, 1.0);
}