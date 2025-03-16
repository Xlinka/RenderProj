use anyhow::Result;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::shader::ShaderModule;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Loads a shader from a file
pub fn load_shader(
    device: Arc<Device>,
    shader_type: ShaderType,
    path: &str,
) -> Result<Arc<ShaderModule>> {
    // Read the shader file
    let mut file = File::open(Path::new(path))?;
    let mut shader_code = String::new();
    file.read_to_string(&mut shader_code)?;

    // Create the shader module
    // Convert GLSL to SPIR-V using shaderc
    let mut compiler = shaderc::Compiler::new().ok_or_else(|| anyhow::anyhow!("Failed to create shader compiler"))?;
    let binary = match shader_type {
        ShaderType::Vertex => {
            let binary = compiler.compile_into_spirv(
                &shader_code,
                shaderc::ShaderKind::Vertex,
                path,
                "main",
                None,
            )?;
            binary.as_binary_u8().to_vec()
        },
        ShaderType::Fragment => {
            let binary = compiler.compile_into_spirv(
                &shader_code,
                shaderc::ShaderKind::Fragment,
                path,
                "main",
                None,
            )?;
            binary.as_binary_u8().to_vec()
        },
        ShaderType::Compute => {
            let binary = compiler.compile_into_spirv(
                &shader_code,
                shaderc::ShaderKind::Compute,
                path,
                "main",
                None,
            )?;
            binary.as_binary_u8().to_vec()
        },
    };

    // Create the shader module from SPIR-V
    let shader_module = unsafe {
        ShaderModule::from_bytes(device, &binary)?
    };

    Ok(shader_module)
}

/// Shader types
pub enum ShaderType {
    Vertex,
    Fragment,
    Compute,
}

/// A struct to manage shader modules
pub struct ShaderManager {
    vertex_shader: Option<Arc<ShaderModule>>,
    fragment_shader: Option<Arc<ShaderModule>>,
    compute_shader: Option<Arc<ShaderModule>>,
}

impl ShaderManager {
    /// Creates a new shader manager
    pub fn new() -> Self {
        Self {
            vertex_shader: None,
            fragment_shader: None,
            compute_shader: None,
        }
    }

    /// Loads a vertex shader
    pub fn load_vertex_shader(&mut self, device: Arc<Device>, path: &str) -> Result<Arc<ShaderModule>> {
        let shader = load_shader(device, ShaderType::Vertex, path)?;
        self.vertex_shader = Some(shader.clone());
        Ok(shader)
    }

    /// Loads a fragment shader
    pub fn load_fragment_shader(&mut self, device: Arc<Device>, path: &str) -> Result<Arc<ShaderModule>> {
        let shader = load_shader(device, ShaderType::Fragment, path)?;
        self.fragment_shader = Some(shader.clone());
        Ok(shader)
    }

    /// Loads a compute shader
    pub fn load_compute_shader(&mut self, device: Arc<Device>, path: &str) -> Result<Arc<ShaderModule>> {
        let shader = load_shader(device, ShaderType::Compute, path)?;
        self.compute_shader = Some(shader.clone());
        Ok(shader)
    }

    /// Gets the vertex shader
    pub fn get_vertex_shader(&self) -> Option<Arc<ShaderModule>> {
        self.vertex_shader.clone()
    }

    /// Gets the fragment shader
    pub fn get_fragment_shader(&self) -> Option<Arc<ShaderModule>> {
        self.fragment_shader.clone()
    }

    /// Gets the compute shader
    pub fn get_compute_shader(&self) -> Option<Arc<ShaderModule>> {
        self.compute_shader.clone()
    }

    /// Unloads all shaders
    pub fn unload_all(&mut self) {
        self.vertex_shader = None;
        self.fragment_shader = None;
        self.compute_shader = None;
    }

    /// Unloads the vertex shader
    pub fn unload_vertex_shader(&mut self) {
        self.vertex_shader = None;
    }

    /// Unloads the fragment shader
    pub fn unload_fragment_shader(&mut self) {
        self.fragment_shader = None;
    }

    /// Unloads the compute shader
    pub fn unload_compute_shader(&mut self) {
        self.compute_shader = None;
    }
}

impl Drop for ShaderManager {
    fn drop(&mut self) {
        // Ensure all shaders are unloaded when the manager is dropped
        self.unload_all();
    }
}
