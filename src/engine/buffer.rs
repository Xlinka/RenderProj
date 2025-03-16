use anyhow::Result;
use log::info;
use nalgebra::Matrix4;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};

/// Vertex structure for our 3D models
#[derive(Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

// Implement Pod and Zeroable for Vertex
unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

/// Uniform buffer object for model-view-projection matrices
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct UniformBufferObject {
    pub model: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
}

// Implement Pod and Zeroable for UniformBufferObject
unsafe impl bytemuck::Pod for UniformBufferObject {}
unsafe impl bytemuck::Zeroable for UniformBufferObject {}

/// Creates a vertex buffer from a list of vertices
pub fn create_vertex_buffer(
    allocator: &StandardMemoryAllocator,
    vertices: &[Vertex],
) -> Result<Subbuffer<[Vertex]>> {
    let buffer = Buffer::from_iter(
        allocator,
        BufferCreateInfo::default(),
        AllocationCreateInfo::default(),
        vertices.iter().cloned(),
    )?;

    info!("Vertex buffer created with {} vertices", vertices.len());
    Ok(buffer)
}

/// Creates an index buffer from a list of indices
pub fn create_index_buffer(
    allocator: &StandardMemoryAllocator,
    indices: &[u32],
) -> Result<Subbuffer<[u32]>> {
    let buffer = Buffer::from_iter(
        allocator,
        BufferCreateInfo::default(),
        AllocationCreateInfo::default(),
        indices.iter().cloned(),
    )?;

    info!("Index buffer created with {} indices", indices.len());
    Ok(buffer)
}

/// Creates a uniform buffer for storing transformation matrices
pub fn create_uniform_buffer(
    allocator: &StandardMemoryAllocator,
) -> Result<Subbuffer<UniformBufferObject>> {
    let buffer = Buffer::from_data(
        allocator,
        BufferCreateInfo::default(),
        AllocationCreateInfo::default(),
        UniformBufferObject {
            model: Matrix4::identity(),
            view: Matrix4::identity(),
            proj: Matrix4::identity(),
        },
    )?;

    info!("Uniform buffer created successfully");
    Ok(buffer)
}

/// Creates a simple cube mesh
pub fn create_cube() -> (Vec<Vertex>, Vec<u32>) {
    // Vertices for a cube
    let vertices = vec![
        // Front face
        Vertex { position: [-0.5, -0.5, 0.5], normal: [0.0, 0.0, 1.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [0.5, -0.5, 0.5], normal: [0.0, 0.0, 1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [0.5, 0.5, 0.5], normal: [0.0, 0.0, 1.0], tex_coords: [1.0, 1.0] },
        Vertex { position: [-0.5, 0.5, 0.5], normal: [0.0, 0.0, 1.0], tex_coords: [0.0, 1.0] },
        
        // Back face
        Vertex { position: [-0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [-0.5, 0.5, -0.5], normal: [0.0, 0.0, -1.0], tex_coords: [1.0, 1.0] },
        Vertex { position: [0.5, 0.5, -0.5], normal: [0.0, 0.0, -1.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], tex_coords: [0.0, 0.0] },
        
        // Top face
        Vertex { position: [-0.5, 0.5, -0.5], normal: [0.0, 1.0, 0.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [-0.5, 0.5, 0.5], normal: [0.0, 1.0, 0.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [0.5, 0.5, 0.5], normal: [0.0, 1.0, 0.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [0.5, 0.5, -0.5], normal: [0.0, 1.0, 0.0], tex_coords: [1.0, 1.0] },
        
        // Bottom face
        Vertex { position: [-0.5, -0.5, -0.5], normal: [0.0, -1.0, 0.0], tex_coords: [1.0, 1.0] },
        Vertex { position: [0.5, -0.5, -0.5], normal: [0.0, -1.0, 0.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [0.5, -0.5, 0.5], normal: [0.0, -1.0, 0.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [-0.5, -0.5, 0.5], normal: [0.0, -1.0, 0.0], tex_coords: [1.0, 0.0] },
        
        // Right face
        Vertex { position: [0.5, -0.5, -0.5], normal: [1.0, 0.0, 0.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [0.5, 0.5, -0.5], normal: [1.0, 0.0, 0.0], tex_coords: [1.0, 1.0] },
        Vertex { position: [0.5, 0.5, 0.5], normal: [1.0, 0.0, 0.0], tex_coords: [0.0, 1.0] },
        Vertex { position: [0.5, -0.5, 0.5], normal: [1.0, 0.0, 0.0], tex_coords: [0.0, 0.0] },
        
        // Left face
        Vertex { position: [-0.5, -0.5, -0.5], normal: [-1.0, 0.0, 0.0], tex_coords: [0.0, 0.0] },
        Vertex { position: [-0.5, -0.5, 0.5], normal: [-1.0, 0.0, 0.0], tex_coords: [1.0, 0.0] },
        Vertex { position: [-0.5, 0.5, 0.5], normal: [-1.0, 0.0, 0.0], tex_coords: [1.0, 1.0] },
        Vertex { position: [-0.5, 0.5, -0.5], normal: [-1.0, 0.0, 0.0], tex_coords: [0.0, 1.0] },
    ];

    // Indices for the cube (6 faces, 2 triangles per face, 3 indices per triangle)
    let indices = vec![
        0, 1, 2, 2, 3, 0,       // Front face
        4, 5, 6, 6, 7, 4,       // Back face
        8, 9, 10, 10, 11, 8,    // Top face
        12, 13, 14, 14, 15, 12, // Bottom face
        16, 17, 18, 18, 19, 16, // Right face
        20, 21, 22, 22, 23, 20, // Left face
    ];

    (vertices, indices)
}
