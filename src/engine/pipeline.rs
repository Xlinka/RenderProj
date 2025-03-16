use anyhow::Result;
use log::info;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineLayout};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;

use crate::engine::buffer::Vertex;

/// Creates a render pass for our rendering pipeline
pub fn create_render_pass(device: Arc<Device>, format: Format) -> Result<Arc<RenderPass>> {
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: format,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )?;

    info!("Render pass created successfully");
    Ok(render_pass)
}

/// Creates framebuffers for each swapchain image
pub fn create_framebuffers(
    images: &[Arc<vulkano::image::SwapchainImage>],
    render_pass: Arc<RenderPass>,
) -> Result<Vec<Arc<Framebuffer>>> {
    let framebuffers = images
        .iter()
        .map(|image| {
            let view = vulkano::image::view::ImageView::new_default(image.clone())?;
            Ok(Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )?)
        })
        .collect::<Result<Vec<_>>>()?;

    info!("Created {} framebuffers", framebuffers.len());
    Ok(framebuffers)
}

/// Creates a graphics pipeline for rendering
pub fn create_graphics_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Result<Arc<GraphicsPipeline>> {
    // Create a vertex input state
    let vertex_input_state = VertexInputState::new();

    // Create a viewport state
    let viewport_state = ViewportState::viewport_fixed_scissor_irrelevant([viewport]);

    // Create a pipeline layout
    let pipeline_layout = PipelineLayout::new(
        device.clone(),
        vulkano::pipeline::layout::PipelineLayoutCreateInfo {
            push_constant_ranges: vec![],
            set_layouts: vec![],
            ..Default::default()
        },
    )?;

    // Create the graphics pipeline
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(vertex_input_state)
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::default())
        .viewport_state(viewport_state)
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .color_blend_state(ColorBlendState::new(1).blend_alpha())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .with_pipeline_layout(device.clone(), pipeline_layout)?;

    info!("Graphics pipeline created successfully");
    Ok(pipeline)
}
