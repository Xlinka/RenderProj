use anyhow::Result;
use log::info;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::swapchain::{
    AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
};
use winit::window::Window;

pub struct SwapchainBundle {
    pub swapchain: Arc<Swapchain>,
    pub images: Vec<Arc<SwapchainImage>>,
}

/// Creates a swapchain for rendering
pub fn create_swapchain(
    device: Arc<Device>,
    surface: Arc<Surface>,
    window: &Window,
) -> Result<SwapchainBundle> {
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(&surface, Default::default())?;

    // Choose the best available format
    let surface_formats = device
        .physical_device()
        .surface_formats(&surface, Default::default())?;
    
    let format = surface_formats
        .iter()
        .find(|(format, color_space)| {
            *format == Format::B8G8R8A8_SRGB && *color_space == vulkano::swapchain::ColorSpace::SrgbNonLinear
        })
        .map(|(format, color_space)| (*format, *color_space))
        .unwrap_or_else(|| (surface_formats[0].0, surface_formats[0].1));

    // Get the window dimensions
    let window_dimensions = window.inner_size();

    // Create the swapchain and its images
    let (swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count.max(2),
            image_format: Some(format.0),
            image_color_space: format.1,
            image_extent: [window_dimensions.width, window_dimensions.height],
            image_usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT,
            composite_alpha: vulkano::swapchain::CompositeAlpha::Opaque,
            ..Default::default()
        },
    )?;

    info!(
        "Swapchain created with format {:?} and {} images",
        format.0,
        images.len()
    );

    Ok(SwapchainBundle { swapchain, images })
}

/// Recreates the swapchain when needed (e.g., window resize)
pub fn recreate_swapchain(
    device: Arc<Device>,
    surface: Arc<Surface>,
    old_swapchain: Arc<Swapchain>,
    window: &Window,
) -> Result<SwapchainBundle> {
    let window_dimensions = window.inner_size();
    
    let (swapchain, images) = old_swapchain.recreate(SwapchainCreateInfo {
        image_extent: [window_dimensions.width, window_dimensions.height],
        ..old_swapchain.create_info()
    })?;

    info!("Swapchain recreated with {} images", images.len());
    
    Ok(SwapchainBundle { swapchain, images })
}
