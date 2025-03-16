use anyhow::{anyhow, Result};
use log::info;
use std::sync::Arc;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::SwapchainImage;
use vulkano::instance::Instance;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout};
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    acquire_next_image, AcquireError, Surface, Swapchain, SwapchainPresentInfo,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::buffer::Subbuffer;
use nalgebra::Matrix4;

use crate::engine::buffer::{create_uniform_buffer, UniformBufferObject};
use crate::engine::instance::{create_logical_device, select_physical_device};
use crate::engine::swapchain::{create_swapchain, recreate_swapchain};

/// Renderer handles all drawing operations
pub struct Renderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    surface: Arc<Surface>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,
    swapchain_images: Vec<Arc<SwapchainImage>>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    viewport: Viewport,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    uniform_buffer: Subbuffer<UniformBufferObject>,
}

impl Renderer {
    /// Create a new renderer with the given instance and surface
    pub fn new(instance: Arc<Instance>, surface: Arc<Surface>) -> Result<Self> {
        // Select a suitable physical device and get a queue family index
        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface)?;

        // Create a logical device and queue
        let (device, queue) = create_logical_device(physical_device, queue_family_index)?;

        // Create a memory allocator
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        
        // Create a command buffer allocator
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Create a render pass
        let render_pass = create_render_pass(device.clone())?;

        // Create a swapchain, swapchain images, etc.
        let swapchain_bundle = create_swapchain(device.clone(), surface.clone())?;
        
        // Extract swapchain and images from the bundle
        let swapchain = swapchain_bundle.swapchain;
        let swapchain_images = swapchain_bundle.images;
        
        // Create framebuffers from swapchain images
        let framebuffers = create_framebuffers(&swapchain_images, &render_pass)?;
        
        // Create viewport
        let dimensions = swapchain.image_extent();
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        // Create a graphics pipeline
        let pipeline = create_pipeline(
            device.clone(),
            render_pass.clone(),
            viewport.clone(),
        )?;

        // Create a uniform buffer
        let uniform_buffer = create_uniform_buffer(&memory_allocator)?;

        // Create a fence for synchronization
        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        info!("Renderer initialized successfully");

        Ok(Self {
            device,
            queue,
            surface,
            swapchain,
            render_pass,
            pipeline,
            framebuffers,
            swapchain_images,
            memory_allocator,
            command_buffer_allocator,
            viewport,
            previous_frame_end,
            uniform_buffer,
        })
    }

    /// Render a frame
    pub fn render_frame(&mut self) -> Result<()> {
        // Wait for the previous frame to finish
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        // Get the next image from the swapchain
        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    // Recreate the swapchain if it's out of date
                    self.recreate_swapchain()?;
                    return Ok(());
                }
                Err(e) => return Err(anyhow!("Failed to acquire next image: {}", e)),
            };

        // Recreate the swapchain if it's suboptimal
        if suboptimal {
            self.recreate_swapchain()?;
            return Ok(());
        }

        let _ubo = UniformBufferObject {
            model: Matrix4::identity(),
            view: Matrix4::identity(),
            proj: Matrix4::identity(),
        };
        
        // Create a new uniform buffer with the updated values
        let _new_buffer = create_uniform_buffer(&self.memory_allocator)?;

        // Create a command buffer builder
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        // Begin the render pass
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassContents::Inline,
            )?
            .end_render_pass()?;

        // Build the command buffer
        let command_buffer = builder.build()?;

        // Submit the command buffer and advance to the next frame
        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)?
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        // Handle the result
        self.previous_frame_end = match future {
            Ok(future) => Some(future.boxed()),
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain()?;
                Some(sync::now(self.device.clone()).boxed())
            }
            Err(e) => return Err(anyhow!("Failed to flush future: {}", e)),
        };

        Ok(())
    }

    /// Recreate the swapchain
    fn recreate_swapchain(&mut self) -> Result<()> {
        // Recreate the swapchain and related resources
        let swapchain_bundle = recreate_swapchain(
            self.device.clone(),
            self.surface.clone(),
            self.swapchain.clone(),
        )?;

        // Update the renderer's fields
        self.swapchain = swapchain_bundle.swapchain;
        self.swapchain_images = swapchain_bundle.images;
        
        // Recreate framebuffers with new swapchain images
        self.framebuffers = create_framebuffers(&self.swapchain_images, &self.render_pass)?;
        
        // Update viewport with new dimensions
        let dimensions = self.swapchain.image_extent();
        self.viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        // Create a new pipeline with the updated viewport
        self.pipeline = create_pipeline(
            self.device.clone(),
            self.render_pass.clone(),
            self.viewport.clone(),
        )?;

        info!("Swapchain recreated successfully");
        Ok(())
    }
}

/// Creates a render pass compatible with our swapchain
fn create_render_pass(device: Arc<Device>) -> Result<Arc<RenderPass>> {
    let render_pass = vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: Format::B8G8R8A8_SRGB,
                samples: vulkano::image::SampleCount::Sample1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )?;

    Ok(render_pass)
}

/// Creates framebuffers from swapchain images
fn create_framebuffers(
    swapchain_images: &[Arc<SwapchainImage>],
    render_pass: &Arc<RenderPass>,
) -> Result<Vec<Arc<Framebuffer>>> {
    let framebuffers = swapchain_images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone())?;
            
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .map_err(|e| anyhow!("Failed to create framebuffer: {}", e))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(framebuffers)
}

/// Creates a graphics pipeline
fn create_pipeline(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Result<Arc<GraphicsPipeline>> {
    // Load shaders
    use crate::shaders::fragment::fs;
    use crate::shaders::vertex::vs;

    let vs = vs::load(device.clone())?;
    let fs = fs::load(device.clone())?;

    // Create pipeline layout
    let pipeline_layout = PipelineLayout::new(
        device.clone(), 
        PipelineLayoutCreateInfo {
            set_layouts: vec![],
            push_constant_ranges: vec![],
            ..Default::default()
        }
    )?;

    // For render pass, we need to convert to Subpass
    let subpass = vulkano::render_pass::Subpass::from(render_pass.clone(), 0).unwrap();

    // Create the pipeline - the builder completes with with_pipeline_layout
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(VertexInputState::new())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_dynamic(1))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(subpass) // Use subpass instead of render_pass
        .with_pipeline_layout(device.clone(), pipeline_layout)?;

    // The pipeline is already built by with_pipeline_layout
    Ok(pipeline)
}