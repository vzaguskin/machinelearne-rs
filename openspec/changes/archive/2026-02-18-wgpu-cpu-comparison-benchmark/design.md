## Context

The WGPU backend was implemented with full GPU compute shader support. We now need to validate the GPU acceleration benefits and ensure numerical equivalence between CPU and GPU backends. This benchmark will run identical operations on both backends with precise timing and accuracy measurements.

## Goals / Non-Goals

**Goals:**
- Provide a reproducible benchmark comparing WGPU vs CPU performance
- Validate that both backends produce numerically equivalent results
- Show where GPU acceleration provides benefit vs overhead
- Document expected behavior for users

**Non-Goals:**
- Optimizing WGPU shader performance (separate concern)
- Adding new WGPU features or operations
- Automated CI benchmarks (requires GPU runners)

## Decisions

### Benchmark Structure
Create a single example file with shared code paths for both backends:
- Use generics to share training logic between backends
- Measure only training time (excluding data loading, standardization)
- Run multiple iterations and report averages

**Alternatives considered:**
- Separate benchmark files per backend - rejected to ensure exact same operations
- Using criterion for benchmarking - rejected to keep example simple and runnable without additional dependencies

### Timing Methodology
Use `std::time::Instant` for precise measurements:
- Measure only the `trainer.fit()` call
- Exclude data loading, tensor creation, and standardization
- Report in milliseconds with microsecond precision

### Dataset Sizes
Test with three sizes to show GPU overhead vs benefit:
- Small: 1,000 samples (GPU overhead likely dominates)
- Medium: 10,000 samples (transition point)
- Large: 20,640 samples (full California Housing, GPU should be faster)

### Accuracy Comparison
Compare final metrics (MSE, MAE, R²) after training:
- Both backends should produce equivalent results within floating-point tolerance
- R² difference should be < 0.01 for numerical equivalence
- Small differences are expected due to floating-point operation ordering

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| GPU may be slower for small datasets | Document this in output, test multiple sizes |
| Different GPU drivers may produce slightly different results | Use tolerance-based comparison, document expected behavior |
| WGPU may not initialize on headless systems | Provide clear error message and feature gate |
| Timing may vary between runs | Note that results are system-dependent |
