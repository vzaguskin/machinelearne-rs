## 1. Infrastructure Setup

- [x] 1.1 Add metal feature to Cargo.toml
- [ ] 1.2 Add metal-rs dependency (optional, for real Metal support)
- [x] 1.3 Create `lib/src/backend/metal/` module structure
- [x] 1.4 Add MetalDevice struct for device management
- [ ] 1.5 Implement Metal device enumeration (future)

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

- [x] 4.1 Define MetalTensor1D (MTLBuffer wrapper with shared storage)
- [x] 4.2 Define MetalTensor2D (MTLBuffer wrapper with shared storage)
- [x] 4.3 Implement zero-copy CPU access (via from_vec methods)
- [x] 4.4 Implement buffer allocation (host-side, future real Metal)
- [ ] 4.5 Implement real unified memory with metal-rs

## 5. Backend Implementation

- [x] 5.1 Implement constructor methods (zeros_1d, zeros_2d, from_vec_*)
- [x] 5.2 Implement data access methods (to_vec_1d, len_1d, len_2d, shape)
- [x] 5.3 Implement element-wise operations (host-side, future Metal shaders)
- [x] 5.4 Implement reduction operations (host-side, future MPS/shaders)
- [x] 5.5 Implement mathematical functions (host-side, future shaders)
- [x] 5.6 Implement linear algebra (host-side, future MPS integration)
- [x] 5.7 Implement column/row operations (host-side, future shaders)
- [x] 5.8 Implement broadcasting operations (host-side, future shaders)

## 6. Command Queue Management

- [ ] 6.1 Create command queue per device
- [ ] 6.2 Implement command buffer submission
- [ ] 6.3 Implement synchronization (completion handler)

## 7. Testing

- [x] 7.1 Test basic tensor operations match CPU backend
- [x] 7.2 Test all Backend trait methods (103 tests passing)
- [ ] 7.3 Benchmark vs CpuBackend for various tensor sizes
- [ ] 7.4 Test on Apple Silicon (M1/M2/M3)
- [ ] 7.5 Test on Intel Mac with discrete GPU
- [ ] 7.6 Test error handling

## 8. Documentation

- [x] 8.1 Add doc comments to all public types
- [x] 8.2 Document macOS version requirement
- [ ] 8.3 Document Apple Silicon vs Intel Mac differences (future)
- [ ] 8.4 Update CHANGELOG.md
- [ ] 8.5 Add usage examples

## Implementation Notes

The current implementation provides a complete Backend trait implementation using host-side
computation. This serves as:

1. **Foundation**: All 60+ Backend methods are implemented and tested
2. **Reference**: Correct behavior validated against CpuBackend
3. **Migration path**: Easy to swap host-side ops for Metal shaders later

Future work:
- Integrate `metal-rs` crate for real Metal operations
- Add MPS (Metal Performance Shaders) for optimized linear algebra
- Write custom Metal Shading Language (MSL) kernels
- Implement command queue management for async operations
- Leverage unified memory on Apple Silicon for zero-copy operations

## Platform Requirements

- macOS 12.0+ recommended (for best MPS support)
- iOS 14.0+ (future support)
- Apple Silicon (M1/M2/M3) or Intel Mac with Metal-supporting GPU
