//! SPIR-V/Vulkan device management.
//!
//! Provides utilities for Vulkan device selection and management.

/// Vulkan device handle for GPU operations.
///
/// Manages the Vulkan physical/logical device and provides device information.
/// This is a lightweight wrapper that can be cloned cheaply.
///
/// # Platform Support
///
/// Works on any platform with Vulkan support:
/// - Linux with NVIDIA, AMD, Intel, or Mesa drivers
/// - Windows with NVIDIA, AMD, or Intel drivers
/// - macOS via MoltenVK (requires separate setup)
#[derive(Clone, Debug)]
pub struct SpirvDevice {
    /// Physical device index
    index: usize,
    /// Device name (cached)
    name: Option<String>,
    /// Vulkan API version
    api_version: Option<(u32, u32)>,
    /// Whether device supports compute shaders
    supports_compute: bool,
}

impl SpirvDevice {
    /// Creates a new Vulkan device handle for the specified GPU.
    ///
    /// # Arguments
    /// * `index` - GPU device index (0 for first GPU)
    ///
    /// # Returns
    /// A `SpirvDevice` instance if the device exists and is accessible.
    ///
    /// # Errors
    /// Returns an error if Vulkan is not available or the device doesn't exist.
    pub fn new(index: usize) -> Result<Self, SpirvError> {
        // In a real implementation, this would use vulkano to enumerate devices
        Ok(Self {
            index,
            name: None,
            api_version: None,
            supports_compute: true,
        })
    }

    /// Returns the default Vulkan device.
    ///
    /// This is typically the first discrete GPU found, or the integrated GPU
    /// if no discrete GPU is available.
    pub fn default_device() -> Result<Self, SpirvError> {
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

    /// Returns the Vulkan API version as (major, minor).
    pub fn api_version(&self) -> Option<(u32, u32)> {
        self.api_version
    }

    /// Returns true if this device supports compute shaders.
    pub fn supports_compute(&self) -> bool {
        self.supports_compute
    }

    /// Returns the number of available Vulkan devices.
    ///
    /// # Errors
    /// Returns an error if Vulkan is not available.
    pub fn count() -> Result<usize, SpirvError> {
        // In a real implementation, this would enumerate physical devices
        Ok(1)
    }

    /// Waits for all GPU commands to complete.
    ///
    /// In a real implementation, this would call vkDeviceWaitIdle
    pub fn synchronize(&self) -> Result<(), SpirvError> {
        Ok(())
    }

    /// Returns the maximum work group size for compute shaders.
    pub fn max_work_group_size(&self) -> [usize; 3] {
        // Typical value for most GPUs
        [1024, 1024, 64]
    }

    /// Returns the preferred work group size for compute shaders.
    pub fn preferred_work_group_size(&self) -> usize {
        // Good default for most operations
        256
    }
}

impl Default for SpirvDevice {
    fn default() -> Self {
        Self::default_device().expect("Failed to create default Vulkan device")
    }
}

/// SPIR-V/Vulkan error type.
#[derive(Debug, Clone)]
pub enum SpirvError {
    /// Device not found
    DeviceNotFound(usize),
    /// Out of memory
    OutOfMemory(usize),
    /// Shader compilation error
    ShaderCompilationError(String),
    /// Pipeline creation error
    PipelineError(String),
    /// Invalid operation
    InvalidOperation(String),
    /// Vulkan not available
    VulkanNotAvailable,
    /// Driver error
    DriverError(String),
}

impl std::fmt::Display for SpirvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpirvError::DeviceNotFound(index) => {
                write!(f, "Vulkan device {} not found", index)
            }
            SpirvError::OutOfMemory(bytes) => {
                write!(f, "Vulkan out of memory (requested {} bytes)", bytes)
            }
            SpirvError::ShaderCompilationError(msg) => {
                write!(f, "SPIR-V shader compilation error: {}", msg)
            }
            SpirvError::PipelineError(msg) => {
                write!(f, "Vulkan pipeline error: {}", msg)
            }
            SpirvError::InvalidOperation(msg) => {
                write!(f, "Invalid Vulkan operation: {}", msg)
            }
            SpirvError::VulkanNotAvailable => {
                write!(f, "Vulkan is not available on this system")
            }
            SpirvError::DriverError(msg) => {
                write!(f, "Vulkan driver error: {}", msg)
            }
        }
    }
}

impl std::error::Error for SpirvError {}
