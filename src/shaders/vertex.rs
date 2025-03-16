use vulkano_shaders::shader;

pub mod vs {
    use vulkano_shaders::shader;

    shader! {
        ty: "vertex",
        src: "
            #version 450

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec2 tex_coords;

            layout(binding = 0) uniform UniformBufferObject {
                mat4 model;
                mat4 view;
                mat4 proj;
            } ubo;

            layout(location = 0) out vec3 fragNormal;
            layout(location = 1) out vec2 fragTexCoord;
            layout(location = 2) out vec3 fragPosition;

            void main() {
                gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
                fragNormal = mat3(ubo.model) * normal;
                fragTexCoord = tex_coords;
                fragPosition = (ubo.model * vec4(position, 1.0)).xyz;
            }
        "
    }
}

pub fn load(device: std::sync::Arc<vulkano::device::Device>) -> Result<std::sync::Arc<vulkano::shader::ShaderModule>, vulkano::shader::ShaderCreationError> {
    vs::load(device)
}
