//! GPU device management for WGPU backend.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};
use wgpu::{Device, Features, Instance, Limits, Queue};

use super::accumulator::CommandAccumulator;
use super::bind_group_cache::BindGroupCache;
use super::buffer_pool::BufferPool;
use super::dynamic_uniform::DynamicUniformBuffer;

/// Global device singleton for WGPU backend.
/// Using a single device ensures all buffers are compatible.
static GLOBAL_DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

// Thread-local buffer pool for reusing GPU memory.
thread_local! {
    static BUFFER_POOL: RefCell<BufferPool> = RefCell::new(BufferPool::new());
}

// Thread-local command accumulator for batching operations.
thread_local! {
    static COMMAND_ACCUMULATOR: RefCell<CommandAccumulator> = RefCell::new(CommandAccumulator::new());
}

// Thread-local bind group cache for reducing bind group creation overhead.
thread_local! {
    static BIND_GROUP_CACHE: RefCell<BindGroupCache> = RefCell::new(BindGroupCache::new());
}

// Thread-local dynamic uniform buffer for params.
// This is lazily initialized when first accessed.
thread_local! {
    static DYNAMIC_UNIFORM: RefCell<Option<DynamicUniformBuffer>> = RefCell::new(None);
}

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
    async fn create() -> Self {
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

    /// Returns the global WgpuDevice, creating it if necessary.
    /// This ensures all tensors share the same device for buffer compatibility.
    pub fn global() -> Self {
        GLOBAL_DEVICE
            .get_or_init(|| pollster::block_on(Self::create()))
            .clone()
    }

    /// Creates a new WgpuDevice (for advanced use cases).
    /// Note: Buffers created on different devices are not compatible.
    pub async fn new() -> Self {
        Self::create().await
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

impl WgpuDevice {
    /// Executes a function with access to the thread-local buffer pool.
    pub fn with_pool<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut BufferPool) -> R,
    {
        BUFFER_POOL.with(|pool| f(&mut pool.borrow_mut()))
    }

    /// Executes a function with access to the thread-local command accumulator.
    pub fn with_accumulator<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut CommandAccumulator) -> R,
    {
        COMMAND_ACCUMULATOR.with(|acc| f(&mut acc.borrow_mut()))
    }

    /// Flushes any pending operations to the GPU.
    ///
    /// This submits all accumulated operations to the GPU for execution.
    /// Called automatically when tensor data is read back.
    pub fn flush(&self) {
        COMMAND_ACCUMULATOR.with(|acc| {
            acc.borrow_mut().flush(&self.device, &self.queue);
        });
    }

    /// Returns the number of pending operations waiting to be executed.
    pub fn pending_ops(&self) -> usize {
        COMMAND_ACCUMULATOR.with(|acc| acc.borrow().pending_count())
    }

    /// Checks if the accumulator should be flushed based on threshold.
    pub fn should_flush(&self) -> bool {
        COMMAND_ACCUMULATOR.with(|acc| acc.borrow().should_flush())
    }

    /// Flushes if the pending operation count exceeds the threshold.
    pub fn flush_if_needed(&self) {
        if self.should_flush() {
            self.flush();
        }
    }

    /// Executes a function with access to the thread-local bind group cache.
    pub fn with_bind_group_cache<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut BindGroupCache) -> R,
    {
        BIND_GROUP_CACHE.with(|cache| f(&mut cache.borrow_mut()))
    }

    /// Executes a function with access to the dynamic uniform buffer.
    ///
    /// The buffer is lazily initialized on first access.
    pub fn with_dynamic_uniform<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut DynamicUniformBuffer) -> R,
    {
        DYNAMIC_UNIFORM.with(|buf| {
            let mut buf = buf.borrow_mut();
            if buf.is_none() {
                *buf = Some(DynamicUniformBuffer::new(&self.device));
            }
            f(buf.as_mut().unwrap())
        })
    }

    /// Allocates space in the dynamic uniform buffer for params.
    ///
    /// Returns the offset to use as a dynamic offset, or None if buffer is full.
    pub fn allocate_params(&self, data: &[u8]) -> Option<u32> {
        self.with_dynamic_uniform(|buf| buf.allocate(data))
    }

    /// Resets the dynamic uniform buffer (call after flush).
    pub fn reset_dynamic_uniform(&self) {
        DYNAMIC_UNIFORM.with(|buf| {
            if let Some(ref mut b) = *buf.borrow_mut() {
                b.reset();
            }
        });
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
