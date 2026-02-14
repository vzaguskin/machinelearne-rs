# SGD Performance Analysis: Rust vs scikit-learn

## Executive Summary

Rust's SGD implementation is approximately **1875x slower** than scikit-learn's SGDRegressor (9546ms vs 5.09ms for 1000 iterations with 2 features).

## Benchmark Results

### Comparison Table (2 features, California Housing)

| Implementation | Config | Time (ms) | Speedup | MSE | MAE | R² |
|----------------|---------|-------------|---------|-----|-----|-----|
| **scikit-learn SGDRegressor** | lr_constant (1000 iter) | **5.09** | **1875x faster** | 0.6697 | 0.5978 | 0.4889 |
| **scikit-learn SGDRegressor** | lr_adaptive (1000 iter) | **25.38** | **376x faster** | 0.6630 | 0.6061 | 0.4941 |
| **Rust (machinelearne-rs)** | 50 epochs, bs=32 | 496.00 | - | 0.6963 | 0.6351 | 0.5221 |
| **Rust (machinelearne-rs)** | 1000 epochs, bs=32 | 9546.00 | - | 0.6963 | 0.6351 | 0.5221 |
| **Rust (machinelearne-rs)** | 1000 epochs, bs=128 | 8517.00 | - | 0.6687 | 0.6206 | 0.5411 |

## Performance Gap Analysis

### Why is Rust 1875x slower?

#### 1. **BLAS Library Usage** (Major Factor)

**scikit-learn:**
- Uses **Intel MKL** (Math Kernel Library) for all linear algebra operations
- Hardware-accelerated with SIMD (AVX2/AVX-512) instructions
- Multi-threaded automatically
- Cache-aware blocking algorithms
- Hand-optimized assembly for critical loops

```
numpy: 1.20.3
blas_mkl_info:
  libraries = ['mkl_rt', 'pthread']
  library_dirs = ['/home/victor/anaconda3/lib']
  define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
```

**Rust CPU Backend:**
- Pure Rust implementation
- No SIMD vectorization
- No multi-threading
- Naïve algorithms
- Hand-coded loops in Rust

#### 2. **Algorithm Implementation**

**scikit-learn (via numpy/scipy):**
- Uses optimized BLAS DGEMV (double-precision general matrix-vector multiply)
- Sophisticated blocking for cache efficiency
- Prefetching and pipelining
- Auto-tuned for specific CPU microarchitecture

**Rust (from lib/src/backend/cpu.rs:591-602):**
```rust
fn _matvec_unchecked(a: &CpuTensor2D, x: &Vec<f64>) -> Vec<f64> {
    let CpuTensor2D(data, rows, cols) = a;
    let mut result = Vec::with_capacity(*rows);
    for i in 0..*rows {
        let mut sum = 0.0;
        for j in 0..*cols {
            sum += data[i * *cols + j] * x[j];  // Naïve triple loop
        }
        result.push(sum);
    }
    result
}
```

**Characteristics:**
- O(n²) naive implementation
- Row-major access pattern (suboptimal for cache)
- No unrolling
- No SIMD
- Single-threaded

#### 3. **Precision Differences**

| Library | Precision | Impact |
|----------|-----------|---------|
| scikit-learn | Float (usually) | Faster on hardware with efficient float operations |
| Rust | Double (f64) | 2x memory bandwidth, less cache locality |
| Note: The CPU backend explicitly uses f64 per design for stability |

#### 4. **Early Stopping**

**scikit-learn:**
- `tol=1e-3` parameter for early stopping
- Stops when improvement falls below threshold
- Reduces actual iterations in many cases

**Rust:**
- No early stopping mechanism
- Always runs all epochs

## Detailed Performance Breakdown

### Per-Epoch Cost

**Rust:**
- 1000 epochs: 9546ms
- Per epoch: ~9.5ms
- Operations per epoch:
  - 16512 samples / 32 batch size = 516 batches
  - Each batch: matvec (forward), tdot (backward), updates
  - Total: ~1548 matvec operations per epoch
  - Per matvec: ~6.1μs (2×2 matrix × 32 samples)

**scikit-learn:**
- 1000 iterations: 5.09ms
- Per iteration: ~5.1μs
- **1875x faster per iteration!**

### What Makes Intel MKL So Fast?

1. **SIMD Vectorization**
   - AVX2: 256-bit registers (8 doubles)
   - AVX-512: 512-bit registers (8 doubles or 16 floats)
   - Processes 8 elements per instruction instead of 1

2. **Multi-threading**
   - Automatically uses all CPU cores
   - Parallelizes matrix operations
   - Linear speedup with core count

3. **Cache Optimization**
   - Cache-blocking algorithms
   - Prefetching data before needed
   - Exploiting locality of reference

4. **Assembly Optimization**
   - Hand-written assembly for critical loops
   - Optimized for specific CPU families
   - Uses CPU-specific instruction sets

5. **Just-In-Time Compilation**
   - Compiles for exact CPU at runtime
   - Selects optimal code paths dynamically

## Comparison with Other Backends

### ndarray Backend (if available)

The `ndarray` feature provides an alternative backend that uses the ndarray crate:

**Potential advantages:**
- Could leverage BLAS via `ndarray-linalg`
- Better code reuse with scientific Python ecosystem

**Current status:**
- May not have BLAS acceleration enabled
- Still using pure Rust algorithms

## Optimization Opportunities

### High Impact (100-1000x speedup)

1. **Integrate BLAS Library**
   ```rust
   // Use crates like:
   // - nalgebra-blas
   // - ndarray + ndarray-blas
   // - faer (pure Rust with SIMD)
   // - matrixmultiply (optimized SIMD implementation)
   ```
   - **Expected speedup: 50-500x**

2. **SIMD Vectorization**
   ```rust
   #[cfg(target_arch = "x86_64")]
   use std::arch::x86_64::*;
   // Manually vectorize matvec with AVX2/AVX-512
   ```
   - **Expected speedup: 4-8x** (depending on instruction set)

3. **Multi-threading**
   ```rust
   use rayon::prelude::*;
   // Parallelize batch processing
   batches.par_iter().for_each(|batch| {
       // Process batch
   });
   ```
   - **Expected speedup: 2-8x** (depending on core count)

4. **Use Optimized Matrix Library**
   - `matrixmultiply` crate: 4-8x faster than naive
   - `faer`: Pure Rust, SIMD-accelerated
   - `ndarray-linalg`: BLAS integration

### Medium Impact (2-10x speedup)

5. **Cache-Aware Algorithms**
   - Block matrix operations for L1/L2 cache
   - Tile loops for better locality

6. **Loop Unrolling**
   - Manually unroll inner loops
   - Compiler may do some of this automatically

7. **Optimize Memory Layout**
   - Column-major vs row-major considerations
   - Aligned memory allocations

### Low Impact (1.5-2x speedup)

8. **Switch to Float (f32)**
   - Reduce memory bandwidth by 2x
   - Better cache utilization
   - Tradeoff: numerical precision

9. **Compiler Optimizations**
   - Enable LTO (Link-Time Optimization)
   - Profile-guided optimization (PGO)
   - Use `cargo build --release` with aggressive flags

10. **Early Stopping**
    - Stop when loss improvement < tolerance
    - Reduces unnecessary epochs

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
1. Switch to `matrixmultiply` crate for matvec/mul operations
2. Enable Rayon for parallel batch processing
3. Add early stopping to trainer

**Expected speedup: 10-50x**

### Phase 2: BLAS Integration (2-4 weeks)
1. Add `ndarray` BLAS backend as an option
2. Integrate with OpenBLAS or MKL via FFI
3. Benchmark different BLAS implementations

**Expected speedup: 100-500x**

### Phase 3: SIMD & Micro-optimizations (4-8 weeks)
1. Hand-vectorized matvec using AVX2/AVX-512
2. Cache-tuned algorithms
3. Profile-guided optimization

**Expected speedup: 2-10x** (on top of BLAS)

## Recommendations

### For Current Use

1. **For Prototyping**: Current implementation is fine
   - Clean code
   - Easy to understand
   - No external dependencies

2. **For Production**: Use BLAS-accelerated backend
   - Add feature flag for optimized backend
   - Keep pure-Rust as default for portability

3. **For Benchmarks**: Use 50-100 epochs
   - Sufficient for convergence
   - Reduces benchmark time significantly
   - More fair comparison vs sklearn's early stopping

### For Future Development

1. **Architecture Decision**: Adopt BLAS as standard
   - Most ML frameworks use BLAS (PyTorch, TensorFlow, JAX)
   - Industry standard for performance
   - Acceptable tradeoff: external dependency for speed

2. **Backend Abstraction**: Make it easy to swap backends
   - Pure Rust (current)
   - BLAS-accelerated (new)
   - GPU (future)

3. **Benchmark Suite**: Comprehensive performance tracking
   - Track matvec, matmul, element-wise ops separately
   - Profile with perf/VTune
   - Compare against BLAS baseline

## Conclusion

The 1875x performance gap is due to scikit-learn using highly optimized Intel MKL library versus Rust's naive pure-Rust implementation. This is a known tradeoff in numerical computing libraries:

| Aspect | Pure Rust | BLAS-Accelerated |
|--------|-----------|-------------------|
| Dependencies | Zero | External (MKL, OpenBLAS, etc.) |
| Portability | Excellent | Good (with compiled libs) |
| Performance | Baseline (slow) | Hardware-optimized (fast) |
| Debuggability | Very easy | Harder (black box) |
| Maintainability | Clear code | FFI boundaries |

**Recommendation**: Implement BLAS-accelerated backend as an optional feature to close the performance gap while maintaining the pure-Rust implementation for debugging and portability.

## References

1. Intel MKL Documentation: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
2. matrixmultiply crate: https://github.com/bluss/matrixmultiply
3. faer crate: https://github.com/sarah-ek/faer-rs
4. ndarray BLAS integration: https://github.com/rust-ndarray/ndarray-linalg
5. Rust SIMD tutorial: https://rust-lang.github.io/std/arch/
