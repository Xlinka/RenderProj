use anyhow::Result;
use log::info;
use std::sync::Arc;
use vulkano::instance::{Instance, InstanceCreateInfo};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

mod engine;
mod shaders;
use engine::renderer::Renderer;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    info!("Starting Vulkan rendering engine");

    // Create an event loop
    let event_loop = EventLoop::new();

    // Create a window
    let window = WindowBuilder::new()
        .with_title("Vulkan Rendering Engine")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
        .build(&event_loop)?;

    // Create a Vulkan instance
    let library = vulkano::VulkanLibrary::new().expect("No local Vulkan library");
    let enabled_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library.clone(),
        InstanceCreateInfo {
            enabled_extensions,
            ..Default::default()
        },
    )?;

    // Create a renderer
    let mut renderer = Renderer::new(instance, &window)?;

    // Run the event loop
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::MainEventsCleared => {
                // Render a frame
                if let Err(e) = renderer.render_frame() {
                    eprintln!("Error rendering frame: {}", e);
                    *control_flow = ControlFlow::Exit;
                }
            }
            _ => (),
        }
    });
}