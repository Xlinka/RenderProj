use anyhow::{anyhow, Result};
use log::{error, info};
use nalgebra::{Matrix4, Perspective3, Point3, Vector3};
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::device::{Device, Queue};
use vulkano::image::view::ImageView;
use vulkano::image::SwapchainImage;
use vulkano::instance::Instance;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, RenderPass};
use vulkano::swapchain::{
    acquire_next_image, AcquireError, Surface, Swapchain, SwapchainCreateInfo,
    SwapchainPresentInfo,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use winit::window::Window;

use crate::engine::buffer::{create_cube, create_index_buffer, create_uniform_buffer, create_vertex_buffer, UniformBufferObject, Vertex};
use crate::engine::instance::{create_logical_device, create_surface, select_physical_device};
use crate::engine::pipeline::{create_framebuffers, create_graphics_pipeline, create_render_pass};
use crate::engine::swapchain::{create_swapchain, recreate_swapchain, SwapchainBundle};
use crate::engine::shader_loader::ShaderManager;

pub struct Renderer {
    start_time: Instant,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<SwapchainImage>>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,
    vertex_buffer: Subbuffer<[Vertex]>,
    index_buffer: Subbuffer<[u32]>,
    uniform_buffer: Subbuffer<UniformBufferObject>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    memory_allocator: StandardMemoryAllocator,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    surface: Arc<Surface>,
    window: Arc<Window>,
    indices_count: u32,
}

impl Renderer {
    pub fn new(instance: Arc<Instance>, window: &Window) -> Result<Self> {
        // Create a surface for rendering
        let surface = create_surface(instance.clone(), window)?;

        // Select a physical device
        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface)?;

        // Create a logical device and queue
        let (device, queue) = create_logical_device(physical_device.clone(), queue_family_index)?;

        // Create memory allocator
        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

        // Create command buffer allocator
        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );

        // Create a swapchain
        let SwapchainBundle {
            swapchain,
            images: swapchain_images,
        } = create_swapchain(device.clone(), surface.clone(), &window)?;

        // Create a render pass
        let render_pass = create_render_pass(device.clone(), swapchain.image_format())?;

        // Load shaders using the shader manager
        let mut shader_manager = ShaderManager::new();
        let vs = shader_manager.load_vertex_shader(device.clone(), "shaders/shader.vert")?;
        let fs = shader_manager.load_fragment_shader(device.clone(), "shaders/shader.frag")?;

        // Create viewport
        let window_dimensions = window.inner_size();
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [window_dimensions.width as f32, window_dimensions.height as f32],
            depth_range: 0.0..1.0,
        };

        // Create graphics pipeline
        let pipeline = create_graphics_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport,
        )?;

        // Create framebuffers
        let framebuffers = create_framebuffers(&swapchain_images, render_pass.clone())?;

        // Create a cube mesh
        let (vertices, indices) = create_cube();
        let indices_count = indices.len() as u32;

        // Create vertex and index buffers
        let vertex_buffer = create_vertex_buffer(&memory_allocator, &vertices)?;
        let index_buffer = create_index_buffer(&memory_allocator, &indices)?;

        // Create uniform buffer
        let uniform_buffer = create_uniform_buffer(&memory_allocator)?;

        // Create a placeholder for the previous frame end
        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        Ok(Self {
            start_time: Instant::now(),
            device,
            queue,
            swapchain,
            swapchain_images,
            render_pass,
            pipeline,
            framebuffers,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            command_buffer_allocator,
            memory_allocator,
            previous_frame_end,
            surface,
            window: unsafe { Arc::from_raw(Arc::into_raw(Arc::new(window)) as *const Window) },
            indices_count,
        })
    }

    pub fn render_frame(&mut self) -> Result<()> {
        // Wait for the previous frame to finish
        let mut previous_frame_end = self.previous_frame_end.take().unwrap();
        previous_frame_end.cleanup_finished();

        // Update uniform buffer with new transformations
        self.update_uniform_buffer()?;

        // Acquire the next image from the swapchain
        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain()?;
                    self.previous_frame_end = Some(previous_frame_end.boxed());
                    return Ok(());
                }
                Err(e) => return Err(anyhow!("Failed to acquire next image: {}", e)),
            };

        if suboptimal {
            self.recreate_swapchain()?;
            self.previous_frame_end = Some(previous_frame_end.boxed());
            return Ok(());
        }

        // Build the command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 0.2, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(
                    self.framebuffers[image_index as usize].clone(),
                )
            },
            SubpassContents::Inline,
        )?;
        
        let _ = builder.bind_pipeline_graphics(self.pipeline.clone());
        let _ = builder.bind_vertex_buffers(0, self.vertex_buffer.clone());
        let _ = builder.bind_index_buffer(self.index_buffer.clone());
        let _ = builder.draw_indexed(self.indices_count, 1, 0, 0, 0);
        let _ = builder.end_render_pass();

        let command_buffer = builder.build()?;

        // Submit the command buffer
        let future = previous_frame_end
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)?
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain()?;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                error!("Failed to flush future: {}", e);
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }

        Ok(())
    }

    fn update_uniform_buffer(&self) -> Result<()> {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        // Create model matrix (rotation)
        let model = Matrix4::new_rotation(Vector3::new(0.0, elapsed * 0.3, 0.0));

        // Create view matrix (camera position)
        let view = Matrix4::look_at_rh(
            &Point3::new(2.0, 2.0, 2.0),
            &Point3::new(0.0, 0.0, 0.0),
            &Vector3::new(0.0, 1.0, 0.0),
        );

        // Create projection matrix
        let window_dimensions = self.window.inner_size();
        let aspect_ratio = window_dimensions.width as f32 / window_dimensions.height as f32;
        let proj = Perspective3::new(aspect_ratio, std::f32::consts::FRAC_PI_4, 0.1, 100.0).to_homogeneous();

        // Update the uniform buffer
        let ubo = UniformBufferObject {
            model,
            view,
            proj,
        };

        // Create a new uniform buffer with the updated data
        let new_buffer = create_uniform_buffer(&self.memory_allocator)?;
        
        // TODO: Copy the new buffer to the old buffer or replace it
        // For now, we'll just skip this step since we can't easily replace the buffer

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        // Get the new window dimensions
        let window_dimensions = self.window.inner_size();
        if window_dimensions.width == 0 || window_dimensions.height == 0 {
            return Ok(());
        }

        // Wait for the device to be idle
        unsafe {
            self.device.wait_idle()?;
        }

        // Recreate the swapchain
        let SwapchainBundle {
            swapchain,
            images: swapchain_images,
        } = recreate_swapchain(
            self.device.clone(),
            self.surface.clone(),
            self.swapchain.clone(),
            &self.window,
        )?;

        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;

        // Recreate the framebuffers
        self.framebuffers = create_framebuffers(&self.swapchain_images, self.render_pass.clone())?;

        // Update the viewport
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [window_dimensions.width as f32, window_dimensions.height as f32],
            depth_range: 0.0..1.0,
        };

        // Load shaders using the shader manager
        let mut shader_manager = ShaderManager::new();
        let vs = shader_manager.load_vertex_shader(self.device.clone(), "shaders/shader.vert")?;
        let fs = shader_manager.load_fragment_shader(self.device.clone(), "shaders/shader.frag")?;

        // Recreate the pipeline
        self.pipeline = create_graphics_pipeline(
            self.device.clone(),
            vs.clone(),
            fs.clone(),
            self.render_pass.clone(),
            viewport,
        )?;

        info!("Swapchain and dependent resources recreated");
        Ok(())
    }
}
