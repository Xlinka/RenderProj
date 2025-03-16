use anyhow::{anyhow, Result};
use log::info;
use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;

/// Selects the most suitable physical device (GPU) for our rendering engine
pub fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
) -> Result<(Arc<PhysicalDevice>, u32)> {
    // Get a list of all available physical devices
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()?
        .filter(|p| {
            // Check if device supports the required extensions
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {
            // Find a queue family that supports graphics and presentation
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(vulkano::device::QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            // Score physical devices to find the best one
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .ok_or_else(|| anyhow!("No suitable physical device found"))?;

    // Log the selected device
    info!(
        "Selected physical device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    Ok((physical_device, queue_family_index))
}

/// Creates a logical device and returns it along with the queue
pub fn create_logical_device(
    physical_device: Arc<PhysicalDevice>,
    queue_family_index: u32,
) -> Result<(Arc<Device>, Arc<vulkano::device::Queue>)> {
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    // Create the logical device and queues
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )?;

    // Get the first queue
    let queue = queues.next().ok_or_else(|| anyhow!("Failed to get device queue"))?;

    info!("Logical device created successfully");
    Ok((device, queue))
}