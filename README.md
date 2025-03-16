# Vulkan 3D Rendering Engine

A 3D rendering engine built with Rust and Vulkan.

## Features

- Vulkan-based rendering pipeline
- Shader support (GLSL)
- 3D model rendering
- Basic lighting
- Window management with winit

## Requirements

- Rust (latest stable version)
- Vulkan SDK
- Compatible GPU with Vulkan support

## Building

To build the project, run:

```bash
cargo build
```

For a release build:

```bash
cargo build --release
```

## Running

To run the application:

```bash
cargo run
```

Or with the release build:

```bash
cargo run --release
```

## Project Structure

- `src/engine/`: Core rendering engine components
  - `instance.rs`: Vulkan instance and device initialization
  - `swapchain.rs`: Swapchain management
  - `pipeline.rs`: Graphics pipeline setup
  - `buffer.rs`: Buffer management (vertex, index, uniform)
  - `renderer.rs`: Main renderer implementation
- `src/shaders/`: GLSL shaders
  - `vertex.rs`: Vertex shader
  - `fragment.rs`: Fragment shader

## License

This project is licensed under the MIT License - see the LICENSE file for details.
