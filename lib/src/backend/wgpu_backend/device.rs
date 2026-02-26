//! GPU device management for WGPU backend.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};
use wgpu::{Device, Features, Instance, Limits, Queue};

use super::accumulator::CommandAccumulator;
use super::bind_group_cache::BindGroupCache;
use super::buffer_pool::BufferPool;
use super::dynamic_uniform::DynamicUniformBuffer;
use super::staging_pool::StagingBufferPool;

/// Global device singleton for WGPU backend.
/// Using a single device ensures all buffers are compatible.
/// Stores Option to handle the case where no GPU is available.
static GLOBAL_DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();

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
    static DYNAMIC_UNIFORM: RefCell<Option<DynamicUniformBuffer>> = const { RefCell::new(None) };
}

// Thread-local staging buffer pool for efficient CPU readback.
// Buffers are reused across to_vec() calls to reduce allocation overhead.
thread_local! {
    static STAGING_POOL: RefCell<StagingBufferPool> = RefCell::new(StagingBufferPool::default());
}

// Thread-local debug mode flag for eager flushing.
// When enabled, operations flush immediately after each dispatch.
thread_local! {
    static DEBUG_MODE: RefCell<bool> = const { RefCell::new(false) };
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
    async fn create() -> Option<Self> {
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
            .await?;

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
            .ok()?;

        Some(WgpuDevice {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Checks if a GPU adapter is available.
    ///
    /// This is useful for tests that should be skipped when no GPU is present.
    pub fn is_available() -> bool {
        Self::try_global().is_some()
    }

    /// Returns the global WgpuDevice, creating it if necessary.
    /// This ensures all tensors share the same device for buffer compatibility.
    ///
    /// # Panics
    /// Panics if no suitable GPU adapter is found.
    pub fn global() -> Self {
        GLOBAL_DEVICE
            .get_or_init(|| pollster::block_on(Self::create()))
            .clone()
            .expect("No suitable GPU adapter found. Please ensure you have a GPU with Vulkan/Metal/D3D12 support.")
    }

    /// Tries to get the global WgpuDevice, returning None if no GPU is available.
    ///
    /// This is useful for tests that should be skipped when no GPU is present.
    pub fn try_global() -> Option<Self> {
        GLOBAL_DEVICE
            .get_or_init(|| pollster::block_on(Self::create()))
            .clone()
    }

    /// Creates a new WgpuDevice (for advanced use cases).
    /// Note: Buffers created on different devices are not compatible.
    ///
    /// # Panics
    /// Panics if no suitable GPU adapter is found.
    pub async fn new() -> Self {
        Self::create()
            .await
            .expect("No suitable GPU adapter found. Please ensure you have a GPU with Vulkan/Metal/D3D12 support.")
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

    /// Adds a command to the accumulator and flushes if in debug mode.
    /// This is the preferred way to add commands when debug mode is desired.
    pub fn add_command(&self, command: super::accumulator::ExecutableCommand) {
        self.with_accumulator(|acc| {
            acc.add_command(command);
            if acc.should_flush_after_add() {
                acc.flush(&self.device, &self.queue);
            }
        });
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

    /// Sets the flush threshold for the command accumulator.
    ///
    /// Operations will be auto-flushed when the pending count reaches this threshold.
    /// Default is 500 operations. There is also a memory threshold (256MB default)
    /// that triggers flush when estimated queued command memory is exceeded.
    ///
    /// # Performance Tuning
    ///
    /// - **Lower threshold** (e.g., 50-100): More frequent flushes, lower latency per operation,
    ///   but more GPU-CPU synchronization overhead. Good for debugging.
    /// - **Higher threshold** (e.g., 500-1000): Fewer flushes, better throughput for batch
    ///   operations, but higher memory usage. Good for training loops.
    /// - **Debug mode**: Use `set_debug_mode(true)` to flush after every operation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = WgpuDevice::global();
    /// device.set_flush_threshold(1000); // Batch up to 1000 operations
    /// ```
    pub fn set_flush_threshold(&self, threshold: usize) {
        COMMAND_ACCUMULATOR.with(|acc| acc.borrow_mut().set_flush_threshold(threshold));
    }

    /// Returns the current flush threshold (operation count).
    pub fn flush_threshold(&self) -> usize {
        COMMAND_ACCUMULATOR.with(|acc| acc.borrow().stats().flush_threshold)
    }

    /// Sets the memory threshold for the command accumulator.
    ///
    /// Commands will be auto-flushed when estimated memory usage exceeds this threshold.
    /// Default is 256MB. This prevents memory exhaustion from too many queued operations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = WgpuDevice::global();
    /// device.set_memory_threshold(128 * 1024 * 1024); // 128MB limit
    /// ```
    pub fn set_memory_threshold(&self, threshold: usize) {
        COMMAND_ACCUMULATOR.with(|acc| acc.borrow_mut().set_memory_threshold(threshold));
    }

    /// Returns the current memory threshold (bytes).
    pub fn memory_threshold(&self) -> usize {
        COMMAND_ACCUMULATOR.with(|acc| acc.borrow().stats().memory_threshold)
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

    /// Executes a function with access to the thread-local staging buffer pool.
    pub fn with_staging_pool<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&StagingBufferPool) -> R,
    {
        STAGING_POOL.with(|pool| f(&pool.borrow()))
    }

    /// Enables or disables debug mode for eager flushing.
    ///
    /// When debug mode is enabled, every operation is immediately flushed to GPU,
    /// making debugging easier at the cost of performance.
    pub fn set_debug_mode(&self, enabled: bool) {
        DEBUG_MODE.with(|mode| *mode.borrow_mut() = enabled);
        // Also update accumulator's debug mode
        COMMAND_ACCUMULATOR.with(|acc| acc.borrow_mut().set_debug_mode(enabled));
    }

    /// Returns whether debug mode is enabled.
    pub fn is_debug_mode(&self) -> bool {
        DEBUG_MODE.with(|mode| *mode.borrow())
    }

    /// Flushes immediately if debug mode is enabled.
    /// Called after each operation dispatch when debug mode is on.
    pub fn flush_if_debug(&self) {
        if self.is_debug_mode() {
            self.flush();
        }
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
