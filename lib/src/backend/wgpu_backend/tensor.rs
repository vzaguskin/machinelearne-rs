//! GPU tensor types for WGPU backend.

use super::device::WgpuDevice;
use super::shaders::{get_registry, BinaryOp, ScalarOp, UnaryOp};
use crate::preprocessing::PreprocessingError;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages};

// Helper structs for uniform buffer params with proper alignment

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BinaryParams {
    op: u32,
    len: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Binary2DParams {
    op: u32,
    rows: u32,
    cols: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Scalar1DParams {
    op: u32,
    len: u32,
    scalar: f32,
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Scalar2DParams {
    op: u32,
    rows: u32,
    cols: u32,
    scalar: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UnaryParams {
    op: u32,
    len_or_rows: u32,
    cols_or_zero: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatVecParams {
    rows: u32,
    cols: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatMulParams {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposeParams {
    rows: u32,
    cols: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ColRowParams {
    rows: u32,
    cols: u32,
    _padding: [u32; 2],
}

/// 1D tensor stored on GPU.
#[derive(Clone)]
pub struct WgpuTensor1D {
    buffer: Arc<Buffer>,
    len: usize,
}

impl WgpuTensor1D {
    /// Creates a 1D tensor filled with zeros.
    pub async fn zeros(device: &WgpuDevice, len: usize) -> Self {
        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WgpuTensor1D::zeros"),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Fill with zeros
        let zeros = vec![0.0f32; len];
        device
            .queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&zeros));

        WgpuTensor1D {
            buffer: Arc::new(buffer),
            len,
        }
    }

    /// Creates a 1D tensor from a vector of f32 values.
    pub async fn from_vec(device: &WgpuDevice, data: Vec<f32>) -> Self {
        let len = data.len();
        let buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("WgpuTensor1D::from_vec"),
            contents: bytemuck::cast_slice(&data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        WgpuTensor1D {
            buffer: Arc::new(buffer),
            len,
        }
    }

    /// Returns the length of the tensor.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies the tensor data to CPU as a Vec<f64>.
    pub async fn to_vec(&self) -> Vec<f64> {
        // This requires a staging buffer to read back from GPU
        // For simplicity, we'll use a blocking approach
        let device = pollster::block_on(WgpuDevice::new());

        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer_1d"),
            size: (self.len * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("to_vec_encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &staging_buffer,
            0,
            (self.len * std::mem::size_of::<f32>()) as u64,
        );
        device.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        device.device.poll(wgpu::Maintain::Wait);

        let result = rx.await.expect("Failed to receive map result");
        result.expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f64> = bytemuck::cast_slice(&data)
            .iter()
            .map(|&x: &f32| x as f64)
            .collect();
        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Performs a binary operation with another tensor.
    pub async fn binary_op(&self, device: &WgpuDevice, other: &Self, op: BinaryOp) -> Self {
        let registry = get_registry(&device.device);
        let pipeline = &registry.binary_1d;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("binary_op_output"),
            size: (self.len * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = BinaryParams {
            op: op as u32,
            len: self.len as u32,
            _padding: [0u32; 2],
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary_1d_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: other.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("binary_1d_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binary_1d_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = self.len.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor1D {
            buffer: Arc::new(output_buffer),
            len: self.len,
        }
    }

    /// Performs a scalar operation.
    pub async fn scalar_op(&self, device: &WgpuDevice, scalar: f64, op: ScalarOp) -> Self {
        let registry = get_registry(&device.device);
        let pipeline = &registry.scalar_1d;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scalar_op_output"),
            size: (self.len * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Scalar1DParams {
            op: op as u32,
            len: self.len as u32,
            scalar: scalar as f32,
            _padding: 0u32,
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scalar_1d_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scalar_1d_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scalar_1d_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = self.len.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor1D {
            buffer: Arc::new(output_buffer),
            len: self.len,
        }
    }

    /// Performs a unary operation.
    pub async fn unary_op(&self, device: &WgpuDevice, op: UnaryOp) -> Self {
        let registry = get_registry(&device.device);
        let pipeline = &registry.unary_1d;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unary_op_output"),
            size: (self.len * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = UnaryParams {
            op: op as u32,
            len_or_rows: self.len as u32,
            cols_or_zero: 0u32,
            _padding: 0u32,
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unary_1d_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("unary_1d_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("unary_1d_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = self.len.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor1D {
            buffer: Arc::new(output_buffer),
            len: self.len,
        }
    }

    /// Computes the sum of all elements.
    pub async fn sum(&self, device: &WgpuDevice) -> f64 {
        // For small tensors, use CPU
        if self.len <= 256 {
            let data = self.to_vec().await;
            return data.iter().sum();
        }

        // Multi-stage reduction for larger tensors
        let registry = get_registry(&device.device);
        let pipeline = &registry.sum_1d;

        let workgroup_count = self.len.div_ceil(256);
        let partial_sums = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("partial_sums"),
            size: (workgroup_count * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sum_1d_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partial_sums.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sum_1d_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sum_1d_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        // Read back partial sums and reduce on CPU
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_sum"),
            size: (workgroup_count * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sum_copy_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &partial_sums,
            0,
            &staging_buffer,
            0,
            (workgroup_count * std::mem::size_of::<f32>()) as u64,
        );
        device.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        device.device.poll(wgpu::Maintain::Wait);

        let result = rx.await.expect("Failed to receive map result");
        result.expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let partials: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        partials.iter().map(|&x| x as f64).sum()
    }
}

/// 2D tensor stored on GPU.
#[derive(Clone)]
pub struct WgpuTensor2D {
    buffer: Arc<Buffer>,
    rows: usize,
    cols: usize,
}

impl WgpuTensor2D {
    /// Creates a 2D tensor filled with zeros.
    pub async fn zeros(device: &WgpuDevice, rows: usize, cols: usize) -> Self {
        let total = rows * cols;
        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WgpuTensor2D::zeros"),
            size: (total * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let zeros = vec![0.0f32; total];
        device
            .queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&zeros));

        WgpuTensor2D {
            buffer: Arc::new(buffer),
            rows,
            cols,
        }
    }

    /// Creates a 2D tensor from row-major data.
    pub async fn from_vec(device: &WgpuDevice, data: Vec<f32>, rows: usize, cols: usize) -> Self {
        let buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("WgpuTensor2D::from_vec"),
            contents: bytemuck::cast_slice(&data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        WgpuTensor2D {
            buffer: Arc::new(buffer),
            rows,
            cols,
        }
    }

    /// Returns the shape of the tensor as (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Performs a binary operation with another tensor.
    pub async fn binary_op(&self, device: &WgpuDevice, other: &Self, op: BinaryOp) -> Self {
        let registry = get_registry(&device.device);
        let pipeline = &registry.binary_2d;

        let total = self.rows * self.cols;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("binary_op_output_2d"),
            size: (total * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Binary2DParams {
            op: op as u32,
            rows: self.rows as u32,
            cols: self.cols as u32,
            _padding: 0u32,
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary_2d_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: other.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("binary_2d_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binary_2d_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = self.cols.div_ceil(16);
            let workgroups_y = self.rows.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor2D {
            buffer: Arc::new(output_buffer),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Performs a scalar operation.
    pub async fn scalar_op(&self, device: &WgpuDevice, scalar: f64, op: ScalarOp) -> Self {
        let registry = get_registry(&device.device);
        let pipeline = &registry.scalar_2d;

        let total = self.rows * self.cols;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scalar_op_output_2d"),
            size: (total * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Scalar2DParams {
            op: op as u32,
            rows: self.rows as u32,
            cols: self.cols as u32,
            scalar: scalar as f32,
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scalar_2d_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scalar_2d_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scalar_2d_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = self.cols.div_ceil(16);
            let workgroups_y = self.rows.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor2D {
            buffer: Arc::new(output_buffer),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Performs a unary operation.
    pub async fn unary_op(&self, device: &WgpuDevice, op: UnaryOp) -> Self {
        let registry = get_registry(&device.device);
        let pipeline = &registry.unary_2d;

        let total = self.rows * self.cols;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unary_op_output_2d"),
            size: (total * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Binary2DParams {
            op: op as u32,
            rows: self.rows as u32,
            cols: self.cols as u32,
            _padding: 0u32,
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unary_2d_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("unary_2d_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("unary_2d_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = self.cols.div_ceil(16);
            let workgroups_y = self.rows.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor2D {
            buffer: Arc::new(output_buffer),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Computes the sum of all elements.
    pub async fn sum(&self, device: &WgpuDevice) -> f64 {
        let total = self.rows * self.cols;
        if total <= 256 {
            // Read back and compute on CPU for small tensors
            let data = self.read_to_vec(device).await;
            return data.iter().map(|&x| x as f64).sum();
        }

        let registry = get_registry(&device.device);
        let pipeline = &registry.sum_2d;

        let workgroup_count = total.div_ceil(256);
        let partial_sums = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("partial_sums_2d"),
            size: (workgroup_count * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sum_2d_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partial_sums.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sum_2d_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sum_2d_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        // Read back and reduce
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_sum_2d"),
            size: (workgroup_count * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sum_copy_encoder_2d"),
            });
        encoder.copy_buffer_to_buffer(
            &partial_sums,
            0,
            &staging_buffer,
            0,
            (workgroup_count * std::mem::size_of::<f32>()) as u64,
        );
        device.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        device.device.poll(wgpu::Maintain::Wait);

        let result = rx.await.expect("Failed to receive map result");
        result.expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let partials: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        partials.iter().map(|&x| x as f64).sum()
    }

    /// Matrix-vector multiplication.
    pub async fn matvec(&self, device: &WgpuDevice, x: &WgpuTensor1D) -> WgpuTensor1D {
        let registry = get_registry(&device.device);
        let pipeline = &registry.matvec;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matvec_output"),
            size: (self.rows * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = MatVecParams {
            rows: self.rows as u32,
            cols: self.cols as u32,
            _padding: [0u32; 2],
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("matvec_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matvec_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matvec_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = self.rows.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor1D {
            buffer: Arc::new(output_buffer),
            len: self.rows,
        }
    }

    /// Transposed matrix-vector multiplication.
    pub async fn matvec_transposed(&self, device: &WgpuDevice, x: &WgpuTensor1D) -> WgpuTensor1D {
        let registry = get_registry(&device.device);
        let pipeline = &registry.matvec_transposed;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matvec_t_output"),
            size: (self.cols * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = MatVecParams {
            rows: self.rows as u32,
            cols: self.cols as u32,
            _padding: [0u32; 2],
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("matvec_t_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec_t_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matvec_t_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matvec_t_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = self.cols.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor1D {
            buffer: Arc::new(output_buffer),
            len: self.cols,
        }
    }

    /// Matrix multiplication.
    pub async fn matmul(&self, device: &WgpuDevice, b: &Self) -> Self {
        let registry = get_registry(&device.device);
        let pipeline = &registry.matmul;

        let m = self.rows;
        let n = b.cols;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_output"),
            size: (m * n * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = MatMulParams {
            m: m as u32,
            k: self.cols as u32,
            n: n as u32,
            _padding: 0u32,
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("matmul_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = n.div_ceil(8);
            let workgroups_y = m.div_ceil(8);
            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor2D {
            buffer: Arc::new(output_buffer),
            rows: m,
            cols: n,
        }
    }

    /// Matrix transpose.
    pub async fn transpose(&self, device: &WgpuDevice) -> Self {
        let registry = get_registry(&device.device);
        let pipeline = &registry.transpose;

        let total = self.rows * self.cols;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transpose_output"),
            size: (total * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = MatVecParams {
            rows: self.rows as u32,
            cols: self.cols as u32,
            _padding: [0u32; 2],
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("transpose_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transpose_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("transpose_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transpose_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = self.cols.div_ceil(16);
            let workgroups_y = self.rows.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor2D {
            buffer: Arc::new(output_buffer),
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Flattens to 1D tensor.
    pub async fn ravel(&self, _device: &WgpuDevice) -> WgpuTensor1D {
        // Just wrap the buffer differently - no copy needed
        WgpuTensor1D {
            buffer: self.buffer.clone(),
            len: self.rows * self.cols,
        }
    }

    /// Column-wise sum.
    pub async fn col_sum(&self, device: &WgpuDevice) -> WgpuTensor1D {
        let registry = get_registry(&device.device);
        let pipeline = &registry.col_sum;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("col_sum_output"),
            size: (self.cols * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = MatVecParams {
            rows: self.rows as u32,
            cols: self.cols as u32,
            _padding: [0u32; 2],
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("col_sum_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("col_sum_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("col_sum_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("col_sum_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = self.cols.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor1D {
            buffer: Arc::new(output_buffer),
            len: self.cols,
        }
    }

    /// Row-wise sum.
    pub async fn row_sum(&self, device: &WgpuDevice) -> WgpuTensor1D {
        let registry = get_registry(&device.device);
        let pipeline = &registry.row_sum;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("row_sum_output"),
            size: (self.rows * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = MatVecParams {
            rows: self.rows as u32,
            cols: self.cols as u32,
            _padding: [0u32; 2],
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("row_sum_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("row_sum_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("row_sum_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("row_sum_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = self.rows.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor1D {
            buffer: Arc::new(output_buffer),
            len: self.rows,
        }
    }

    /// Column-wise mean.
    pub async fn col_mean(&self, device: &WgpuDevice) -> WgpuTensor1D {
        let sum = self.col_sum(device).await;
        let n = self.rows as f64;
        sum.scalar_op(device, 1.0 / n, ScalarOp::Mul).await
    }

    /// Column-wise standard deviation.
    pub async fn col_std(&self, device: &WgpuDevice, ddof: usize) -> WgpuTensor1D {
        let mean = self.col_mean(device).await;

        // Compute variance: E[(X - mean)^2]
        let centered = self.broadcast_op(device, &mean, BinaryOp::Sub).await;
        let squared = centered.binary_op(device, &centered, BinaryOp::Mul).await;
        let variance = squared.col_sum(device).await;

        // std = sqrt(variance / (n - ddof))
        let n = self.rows as f64 - ddof as f64;
        let scaled = variance.scalar_op(device, 1.0 / n, ScalarOp::Mul).await;
        scaled.unary_op(device, UnaryOp::Sqrt).await
    }

    /// Column-wise min.
    pub async fn col_min(&self, device: &WgpuDevice) -> WgpuTensor1D {
        // For now, use CPU fallback
        let data = self.read_to_vec(device).await;
        let mins: Vec<f64> = (0..self.cols)
            .map(|col| {
                (0..self.rows)
                    .map(|row| data[row * self.cols + col] as f64)
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        WgpuTensor1D::from_vec(device, mins.iter().map(|&x| x as f32).collect()).await
    }

    /// Column-wise max.
    pub async fn col_max(&self, device: &WgpuDevice) -> WgpuTensor1D {
        // For now, use CPU fallback
        let data = self.read_to_vec(device).await;
        let maxes: Vec<f64> = (0..self.cols)
            .map(|col| {
                (0..self.rows)
                    .map(|row| data[row * self.cols + col] as f64)
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();

        WgpuTensor1D::from_vec(device, maxes.iter().map(|&x| x as f32).collect()).await
    }

    /// Broadcast operation with 1D tensor.
    pub async fn broadcast_op(&self, device: &WgpuDevice, v: &WgpuTensor1D, op: BinaryOp) -> Self {
        let registry = get_registry(&device.device);
        let pipeline = &registry.broadcast_2d;

        let total = self.rows * self.cols;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("broadcast_output"),
            size: (total * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Binary2DParams {
            op: op as u32,
            rows: self.rows as u32,
            cols: self.cols as u32,
            _padding: 0u32,
        };
        let params_buffer = device.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("broadcast_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("broadcast_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: v.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("broadcast_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("broadcast_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = self.cols.div_ceil(16);
            let workgroups_y = self.rows.div_ceil(16);
            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        device.queue.submit(std::iter::once(encoder.finish()));

        WgpuTensor2D {
            buffer: Arc::new(output_buffer),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Read tensor data to CPU as Vec<f32>.
    async fn read_to_vec(&self, device: &WgpuDevice) -> Vec<f32> {
        let total = self.rows * self.cols;
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_2d"),
            size: (total * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &staging_buffer,
            0,
            (total * std::mem::size_of::<f32>()) as u64,
        );
        device.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        device.device.poll(wgpu::Maintain::Wait);

        let result = rx.await.expect("Failed to receive map result");
        result.expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Horizontal concatenation.
    pub async fn hcat(
        device: &WgpuDevice,
        tensors: &[Self],
        rows: usize,
        total_cols: usize,
    ) -> Result<Self, PreprocessingError> {
        // For simplicity, use CPU fallback
        let mut result = vec![0.0f32; rows * total_cols];
        let mut col_offset = 0;

        for tensor in tensors {
            let data = tensor.read_to_vec(device).await;
            for row in 0..rows {
                for col in 0..tensor.cols {
                    result[row * total_cols + col_offset + col] = data[row * tensor.cols + col];
                }
            }
            col_offset += tensor.cols;
        }

        Ok(Self::from_vec(device, result, rows, total_cols).await)
    }

    /// Select columns by indices.
    pub async fn select_columns(
        &self,
        device: &WgpuDevice,
        columns: &[usize],
        rows: usize,
    ) -> Self {
        // For simplicity, use CPU fallback
        let data = self.read_to_vec(device).await;
        let mut result = vec![0.0f32; rows * columns.len()];

        for row in 0..rows {
            for (out_col, &in_col) in columns.iter().enumerate() {
                result[row * columns.len() + out_col] = data[row * self.cols + in_col];
            }
        }

        Self::from_vec(device, result, rows, columns.len()).await
    }

    /// Create one-hot encoded matrix.
    pub async fn one_hot(
        device: &WgpuDevice,
        indices: &WgpuTensor1D,
        num_classes: usize,
        n: usize,
    ) -> Self {
        // For simplicity, use CPU fallback
        let indices_data = indices.to_vec().await;
        let mut result = vec![0.0f32; n * num_classes];

        for (i, &idx) in indices_data.iter().enumerate() {
            let class_idx = idx as usize;
            assert!(
                class_idx < num_classes,
                "Index {} >= num_classes {}",
                class_idx,
                num_classes
            );
            result[i * num_classes + class_idx] = 1.0;
        }

        Self::from_vec(device, result, n, num_classes).await
    }
}
