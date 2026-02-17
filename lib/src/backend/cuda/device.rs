//! CUDA device management.
//!
//! Provides utilities for CUDA device selection and management.

/// CUDA device handle for GPU operations.
///
/// Manages the CUDA context and provides device information.
/// This is a lightweight wrapper that can be cloned cheaply.
#[derive(Clone, Debug)]
pub struct CudaDevice {
    /// Device ordinal (GPU index)
    ordinal: usize,
    /// Device name (cached)
    name: Option<String>,
    /// Compute capability (cached)
    compute_capability: Option<(i32, i32)>,
}

impl CudaDevice {
    /// Creates a new CUDA device handle for the specified GPU.
    ///
    /// # Arguments
    /// * `ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Returns
    /// A `CudaDevice` instance if the device exists and is accessible.
    ///
    /// # Errors
    /// Returns an error if CUDA is not available or the device doesn't exist.
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        // In a real implementation, this would initialize CUDA and query device info
        // For now, we create a simple device handle
        Ok(Self {
            ordinal,
            name: None,
            compute_capability: None,
        })
    }

    /// Returns the device ordinal (GPU index).
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Returns the device name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the compute capability as (major, minor).
    pub fn compute_capability(&self) -> Option<(i32, i32)> {
        self.compute_capability
    }

    /// Returns the number of available CUDA devices.
    ///
    /// # Errors
    /// Returns an error if CUDA driver is not available.
    pub fn count() -> Result<usize, CudaError> {
        // In a real implementation, this would call cuDeviceGetCount
        // For the stub implementation, return 1
        Ok(1)
    }

    /// Synchronizes the device, blocking until all operations complete.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        // In a real implementation, this would call cuCtxSynchronize
        Ok(())
    }
}

impl Default for CudaDevice {
    fn default() -> Self {
        Self::new(0).expect("Failed to create default CUDA device")
    }
}

/// CUDA error type.
#[derive(Debug, Clone)]
pub enum CudaError {
    /// Device not found
    DeviceNotFound(usize),
    /// Out of memory
    OutOfMemory(usize),
    /// Driver error
    DriverError(String),
    /// Invalid operation
    InvalidOperation(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::DeviceNotFound(ordinal) => {
                write!(f, "CUDA device {} not found", ordinal)
            }
            CudaError::OutOfMemory(bytes) => {
                write!(f, "CUDA out of memory (requested {} bytes)", bytes)
            }
            CudaError::DriverError(msg) => {
                write!(f, "CUDA driver error: {}", msg)
            }
            CudaError::InvalidOperation(msg) => {
                write!(f, "Invalid CUDA operation: {}", msg)
            }
        }
    }
}

impl std::error::Error for CudaError {}
