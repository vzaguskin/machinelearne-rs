# Benchmarks

Performance benchmarks for `machinelearne-rs` comparing against scikit-learn.

## Quick Start

```bash
# Run fair comparison benchmark (recommended)
python3 benchmarks/scripts/run_fair_comparison.py

# Or run individual components:
cargo run --release -p benchmarks --bin fair_comparison
python3 benchmarks/scripts/fair_comparison_sklearn.py
```

## Benchmark Suite

### Fair Comparison (`fair_comparison.rs`)

**Purpose:** Apples-to-apples comparison with identical configurations.

**What it tests:**
- Same number of epochs
- Same batch sizes
- Same learning rates
- Same data preprocessing

**Key configurations tested:**
| Name | Batch | LR | Epochs | Notes |
|------|-------|------|--------|-------|
| Full-batch LR=0.5 | full | 0.5 | 5, 50 | Optimal for Rust |
| Full-batch LR=0.1 | full | 0.1 | 5, 50 | sklearn-stable |
| Mini-batch | 32 | 0.01 | 5, 50 | Default sklearn LR |

### Other Benchmarks

| Binary | Purpose |
|--------|---------|
| `backend_comparison.rs` | Compare CPU vs ndarray backends |
| `full_batch_comparison.rs` | Full-batch training analysis |
| `learning_rate_search.rs` | Find optimal learning rates |
| `sgd_comparison.rs` | SGD hyperparameter comparison |
| `collect_metrics.rs` | Collect performance metrics |

---

### Backend Comparison (`backend_comparison.rs`)

**Purpose:** Compare performance between CPU and ndarray backends.

```bash
cargo run --release -p benchmarks --bin backend_comparison --features ndarray
```

**What it tests:**
- Same training workload on different backends
- Training time comparison
- Model accuracy (MSE, R²) for each backend

**Output:** JSON file with timing and metrics for each backend.

---

### Full Batch Comparison (`full_batch_comparison.rs`)

**Purpose:** Analyze full-batch gradient descent performance.

```bash
cargo run --release -p benchmarks --bin full_batch_comparison
```

**What it tests:**
- Full-batch vs mini-batch training
- Convergence characteristics
- Optimal epoch counts

**Output:** Timing and accuracy metrics for different configurations.

---

### Learning Rate Search (`learning_rate_search.rs`)

**Purpose:** Find optimal learning rates for different batch sizes.

```bash
cargo run --release -p benchmarks --bin learning_rate_search
```

**What it tests:**
- Multiple learning rates (0.001 to 1.0)
- Different batch sizes
- Convergence stability

**Output:** Learning rate vs convergence analysis.

---

### SGD Comparison (`sgd_comparison.rs`)

**Purpose:** Compare Rust SGD against sklearn's SGDRegressor.

```bash
cargo run --release -p benchmarks --bin sgd_comparison
```

**What it tests:**
- Matches sklearn's SGDRegressor configuration (max_iter=1000, tol=1e-3)
- Same preprocessing (StandardScaler)
- Training time and accuracy comparison

**Output:** JSON file with comparison metrics.

---

### Collect Metrics (`collect_metrics.rs`)

**Purpose:** Collect performance metrics for different feature counts.

```bash
cargo run --release -p benchmarks --bin collect_metrics
```

**What it tests:**
- Training time scaling with feature count (1, 2, 4, 8 features)
- MSE, MAE, R² metrics for each configuration
- Consistent evaluation methodology

**Output:** `benchmarks/results/rust_metrics.json` with structured metrics.

## Results

Results are saved to `benchmarks/results/`:

| File | Description |
|------|-------------|
| `FAIR_COMPARISON.md` | Human-readable comparison report |
| `rust_fair_comparison.json` | Raw Rust benchmark data |
| `sklearn_fair_comparison.json` | Raw sklearn benchmark data |

## Key Findings

### Performance

| Method | Batch | LR | Time (5 epochs) | R² |
|--------|-------|------|-----------------|-----|
| sklearn | full | 0.01 | ~2.8 ms | 0.52 |
| **Rust** | mini (32) | 0.01 | ~2.7 ms | 0.52 |
| **Rust** | full | 0.5 | ~1.5 ms | 0.52 |

**Rust is competitive** for small datasets (<100K samples)!

### Quality

Both implementations achieve **R² ≈ 0.52** with proper configuration.

### Learning Rate Trade-offs

- **Full-batch**: Needs higher LR (0.5) for fast convergence
- **Mini-batch**: Lower LR (0.01) works well due to gradient noise

## Dataset

Uses California Housing dataset (20,640 samples, 8 features):
- Preprocessing: z-score standardization
- Split: 80% train / 20% test
- Location: `benchmarks/datasets/california_housing.csv`

## Requirements

**Rust:**
```bash
cargo build --release -p benchmarks
```

**Python (for sklearn comparison):**
```bash
pip install numpy scikit-learn
```

## Interpreting Results

### When Rust is Faster
- Small datasets (<100K samples)
- Few features (<100)
- Simple matrix operations

### When sklearn is Faster
- Large datasets (>100K samples)
- Many features (>100)
- Complex matrix operations (BLAS-optimized)

## Future Improvements

1. **Add BLAS backend** (OpenBLAS, Intel MKL)
2. **Add SIMD** (AVX2/AVX-512)
3. **Add multi-threading** (rayon)
4. **Add f32 backend option**
