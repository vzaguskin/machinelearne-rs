//! Metal device management.
//!
//! Provides utilities for Metal device selection and management on Apple platforms.

/// Metal device handle for GPU operations.
///
/// Manages the Metal device (MTLDevice) and provides device information.
/// This is a lightweight wrapper that can be cloned cheaply.
///
/// # Platform
///
/// Only available on macOS and iOS. Metal is not available on other platforms.
#[derive(Clone, Debug)]
pub struct MetalDevice {
    /// Device index (for multi-GPU Mac Pro configurations)
    index: usize,
    /// Device name (cached)
    name: Option<String>,
    /// Whether this is Apple Silicon (unified memory)
    is_apple_silicon: bool,
}

impl MetalDevice {
    /// Creates a new Metal device handle for the specified GPU.
    ///
    /// # Arguments
    /// * `index` - GPU device index (0 for default device)
    ///
    /// # Returns
    /// A `MetalDevice` instance if the device exists and is accessible.
    ///
    /// # Errors
    /// Returns an error if Metal is not available or the device doesn't exist.
    pub fn new(index: usize) -> Result<Self, MetalError> {
        // In a real implementation, this would use metal-rs to get MTLDevice
        // For now, we create a simple device handle
        Ok(Self {
            index,
            name: None,
            is_apple_silicon: true, // Assume Apple Silicon for now
        })
    }

    /// Returns the default Metal device.
    ///
    /// This is typically the integrated GPU on Apple Silicon Macs,
    /// or the primary GPU on Intel Macs.
    pub fn default_device() -> Result<Self, MetalError> {
        Self::new(0)
    }

    /// Returns the device index.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the device name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns true if this device uses unified memory (Apple Silicon).
    ///
    /// Unified memory means CPU and GPU share the same memory,
    /// enabling zero-copy operations.
    pub fn is_unified_memory(&self) -> bool {
        self.is_apple_silicon
    }

    /// Returns the recommended maximum working set size for this device.
    ///
    /// On Apple Silicon, this is typically a significant portion of system RAM.
    pub fn recommended_max_working_set_size(&self) -> usize {
        // In a real implementation, this would query MTLDevice
        // For Apple Silicon, typically around 1/4 to 1/2 of system RAM
        4 * 1024 * 1024 * 1024 // 4 GB placeholder
    }

    /// Returns the number of available Metal devices.
    ///
    /// Most Macs have a single Metal device. Mac Pro may have multiple GPUs.
    pub fn count() -> Result<usize, MetalError> {
        // In a real implementation, this would enumerate MTLDevices
        Ok(1)
    }

    /// Waits for all GPU commands to complete.
    ///
    /// In a real implementation, this would use MTLCommandBuffer.waitUntilCompleted()
    pub fn synchronize(&self) -> Result<(), MetalError> {
        Ok(())
    }
}

impl Default for MetalDevice {
    fn default() -> Self {
        Self::default_device().expect("Failed to create default Metal device")
    }
}

/// Metal error type.
#[derive(Debug, Clone)]
pub enum MetalError {
    /// Device not found
    DeviceNotFound(usize),
    /// Out of memory
    OutOfMemory(usize),
    /// Shader compilation error
    ShaderCompilationError(String),
    /// Command buffer error
    CommandBufferError(String),
    /// Invalid operation
    InvalidOperation(String),
    /// Metal not available (non-Apple platform)
    MetalNotAvailable,
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalError::DeviceNotFound(index) => {
                write!(f, "Metal device {} not found", index)
            }
            MetalError::OutOfMemory(bytes) => {
                write!(f, "Metal out of memory (requested {} bytes)", bytes)
            }
            MetalError::ShaderCompilationError(msg) => {
                write!(f, "Metal shader compilation error: {}", msg)
            }
            MetalError::CommandBufferError(msg) => {
                write!(f, "Metal command buffer error: {}", msg)
            }
            MetalError::InvalidOperation(msg) => {
                write!(f, "Invalid Metal operation: {}", msg)
            }
            MetalError::MetalNotAvailable => {
                write!(f, "Metal is not available on this platform")
            }
        }
    }
}

impl std::error::Error for MetalError {}
