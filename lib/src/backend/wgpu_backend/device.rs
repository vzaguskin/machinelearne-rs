//! GPU device management for WGPU backend.

use std::sync::Arc;
use wgpu::{Device, Features, Instance, Limits, Queue};

/// GPU device handle for WGPU backend.
///
/// Contains the wgpu device and queue for executing GPU operations.
/// Cloning is cheap as it uses Arc internally.
#[derive(Clone)]
pub struct WgpuDevice {
    pub(crate) device: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
}

impl WgpuDevice {
    /// Creates a new WgpuDevice by selecting the best available GPU.
    pub async fn new() -> Self {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found. Please ensure you have a GPU with Vulkan/Metal/D3D12 support.");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("machinelearne-rs WgpuDevice"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: Default::default(),
                },
                None, // trace path
            )
            .await
            .expect("Failed to create GPU device");

        WgpuDevice {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
    }

    /// Enumerates all available GPU adapters.
    pub async fn enumerate_adapters() -> Vec<AdapterInfo> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        adapters
            .into_iter()
            .map(|adapter| {
                let info = adapter.get_info();
                AdapterInfo {
                    name: info.name,
                    backend: format!("{:?}", info.backend),
                    device_type: format!("{:?}", info.device_type),
                }
            })
            .collect()
    }

    /// Creates a WgpuDevice from a specific adapter index.
    pub async fn from_index(index: usize) -> Self {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        let adapter = adapters.get(index).unwrap_or_else(|| {
            panic!(
                "GPU adapter index {} out of bounds ({} adapters available)",
                index,
                adapters.len()
            )
        });

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some(&format!("machinelearne-rs WgpuDevice [{}]", index)),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: Default::default(),
                },
                None, // trace path
            )
            .await
            .expect("Failed to create GPU device");

        WgpuDevice {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
    }
}

impl std::fmt::Debug for WgpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuDevice")
            .field("device", &"Arc<wgpu::Device>")
            .field("queue", &"Arc<wgpu::Queue>")
            .finish()
    }
}

/// Information about a GPU adapter.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    /// The name of the adapter.
    pub name: String,
    /// The backend API (Vulkan, Metal, D3D12, etc.).
    pub backend: String,
    /// The device type (DiscreteGpu, IntegratedGpu, Cpu, etc.).
    pub device_type: String,
}
