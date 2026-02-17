## 1. Infrastructure Setup

- [ ] 1.1 Add metal dependency to Cargo.toml under `metal` feature
- [ ] 1.2 Add macOS target guard for metal feature
- [ ] 1.3 Create `lib/src/backend/metal/` module structure
- [ ] 1.4 Add MetalDevice struct for device management
- [ ] 1.5 Implement Metal device enumeration

## 2. MPS Integration

- [ ] 2.1 Create MPS matrix wrapper types
- [ ] 2.2 Implement MPS-backed matmul
- [ ] 2.3 Implement MPS-backed matvec
- [ ] 2.4 Implement MPS-backed reductions (sum, mean)

## 3. Custom Shader Development

- [ ] 3.1 Write Metal Shading Language element-wise kernels
- [ ] 3.2 Write scalar operation kernels
- [ ] 3.3 Write math function kernels (exp, log, sigmoid)
- [ ] 3.4 Write column/row operation kernels
- [ ] 3.5 Write transpose kernel
- [ ] 3.6 Write broadcasting kernels

## 4. Memory Management

- [ ] 4.1 Define MetalTensor1D (MTLBuffer wrapper with shared storage)
- [ ] 4.2 Define MetalTensor2D (MTLBuffer wrapper with shared storage)
- [ ] 4.3 Implement zero-copy CPU access
- [ ] 4.4 Implement buffer allocation with shared storage mode

## 5. Backend Implementation

- [ ] 5.1 Implement constructor methods (zeros_1d, zeros_2d, from_vec_*)
- [ ] 5.2 Implement data access methods (to_vec_1d, len_1d, len_2d, shape)
- [ ] 5.3 Implement element-wise operations using shaders
- [ ] 5.4 Implement reduction operations using MPS/shaders
- [ ] 5.5 Implement mathematical functions using shaders
- [ ] 5.6 Implement linear algebra using MPS
- [ ] 5.7 Implement column/row operations using shaders
- [ ] 5.8 Implement broadcasting operations using shaders

## 6. Command Queue Management

- [ ] 6.1 Create command queue per device
- [ ] 6.2 Implement command buffer submission
- [ ] 6.3 Implement synchronization (completion handler)

## 7. Testing

- [ ] 7.1 Test basic tensor operations match CPU backend
- [ ] 7.2 Test all Backend trait methods
- [ ] 7.3 Benchmark vs CpuBackend for various tensor sizes
- [ ] 7.4 Test on Apple Silicon (M1/M2/M3)
- [ ] 7.5 Test on Intel Mac with discrete GPU
- [ ] 7.6 Test error handling

## 8. Documentation

- [ ] 8.1 Add doc comments to all public types
- [ ] 8.2 Document macOS version requirement
- [ ] 8.3 Document Apple Silicon vs Intel Mac differences
- [ ] 8.4 Update CHANGELOG.md
- [ ] 8.5 Add usage examples
