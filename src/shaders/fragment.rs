use vulkano_shaders::shader;

pub mod fs {
    use vulkano_shaders::shader;

    shader! {
        ty: "fragment",
        src: "
            #version 450

            layout(location = 0) in vec3 fragNormal;
            layout(location = 1) in vec2 fragTexCoord;
            layout(location = 2) in vec3 fragPosition;

            layout(location = 0) out vec4 outColor;

            void main() {
                vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
                vec3 normal = normalize(fragNormal);
                
                // Ambient lighting
                float ambientStrength = 0.2;
                vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
                
                // Diffuse lighting
                float diff = max(dot(normal, lightDir), 0.0);
                vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
                
                // Base color (could be from texture)
                vec3 baseColor = vec3(0.7, 0.2, 0.2);
                
                // Final color
                vec3 result = (ambient + diffuse) * baseColor;
                outColor = vec4(result, 1.0);
            }
        "
    }
}

pub fn load(device: std::sync::Arc<vulkano::device::Device>) -> Result<std::sync::Arc<vulkano::shader::ShaderModule>, vulkano::shader::ShaderCreationError> {
    fs::load(device)
}
