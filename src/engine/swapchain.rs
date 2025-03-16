use anyhow::Result;
use log::info;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::{SwapchainImage};
use vulkano::swapchain::{
    Surface, Swapchain, SwapchainCreateInfo,
};

pub struct SwapchainBundle {
    pub swapchain: Arc<Swapchain>,
    pub images: Vec<Arc<SwapchainImage>>,
}

/// Creates a swapchain for rendering
pub fn create_swapchain(
    device: Arc<Device>,
    surface: Arc<Surface>,
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

    // Get dimensions from surface capabilities
    let dimensions = surface_capabilities.current_extent.unwrap_or([800, 600]);

    // Create the swapchain and its images
    let (swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count.max(2),
            image_format: Some(format.0),
            image_color_space: format.1,
            image_extent: dimensions,
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
) -> Result<SwapchainBundle> {
    // Get dimensions from surface capabilities 
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(&surface, Default::default())?;
    
    let dimensions = surface_capabilities.current_extent.unwrap_or_else(|| {
        // If current_extent is None, use the dimensions from the old swapchain
        old_swapchain.image_extent()
    });
        
    let (swapchain, images) = old_swapchain.recreate(SwapchainCreateInfo {
        image_extent: dimensions,
        ..old_swapchain.create_info()
    })?;

    info!("Swapchain recreated with {} images", images.len());
        
    Ok(SwapchainBundle { swapchain, images })
}