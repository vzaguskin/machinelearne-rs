# Benchmark Comparison: machinelearne-rs vs scikit-learn

**Generated:** 2026-02-15 01:46:54

## Environment

| Component | Version |
|-----------|---------|
| Rust | rustc 1.91.0 (f8297e351 2025-10-28) |
| Python | Python 3.7.9 |
| OS | Linux |

## Dataset

### California Housing Dataset

- **Samples:** 20,640
- **Features:** 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Target:** Median House Value (in $100,000s)
- **Split:** 80% train / 20% test

## Side-by-Side Comparison

### Linear Regression (scikit-learn vs machinelearne-rs)

| Implementation | Features | Train Time (ms) | Pred Time (ms) | MSE | MAE | R² |
|--------------|----------|-----------------|----------------|-----|-----|-----|
| scikit-learn | 1 | 0.85 | 0.08 | 0.7091 | 0.6299 | 0.4589 |
| machinelearne-rs | 1 | 341.00 (0.00x) | N/A | 0.6943 (-2.1%) | 0.6505 | 0.5235 |
| | | | | | |
| scikit-learn | 2 | 0.61 | 0.06 | 0.6630 | 0.6060 | 0.4941 |
| machinelearne-rs | 2 | 436.00 (0.00x) | N/A | 0.6963 (+5.0%) | 0.6351 | 0.5221 |
| | | | | | |
| scikit-learn | 4 | 0.72 | 0.07 | 0.6436 | 0.5804 | 0.5089 |
| machinelearne-rs | 4 | 698.00 (0.00x) | N/A | 0.6670 (+3.6%) | 0.6105 | 0.5423 |
| | | | | | |
| scikit-learn | 8 | 2.08 | 0.08 | 0.5559 | 0.5332 | 0.5758 |
| machinelearne-rs | 8 | 1264.00 (0.00x) | N/A | 0.4904 (-11.8%) | 0.5244 | 0.6634 |
| | | | | | |

## scikit-learn Full Results

| Model | Features | Train Time (ms) | Pred Time (ms) | MSE | MAE | R² |
|-------|----------|-----------------|----------------|-----|-----|-----|
| LinearRegression features 0 | 1.0 | 0.85 ± 1.10 | 0.08 ± 0.03 | 0.7091 | 0.6299 | 0.4589 |
| LinearRegression features 0 1 | 2.0 | 0.61 ± 0.14 | 0.06 ± 0.01 | 0.6630 | 0.6060 | 0.4941 |
| LinearRegression features 0 1 2 3 | 4.0 | 0.72 ± 0.15 | 0.07 ± 0.01 | 0.6436 | 0.5804 | 0.5089 |
| LinearRegression features 0 1 2 3 4 5 6 7 | 8.0 | 2.08 ± 0.10 | 0.08 ± 0.02 | 0.5559 | 0.5332 | 0.5758 |
| Ridge alpha 0.1 features 0 1 | 2.0 | 0.50 ± 0.09 | 0.06 ± 0.01 | 0.6630 | 0.6060 | 0.4941 |
| Ridge alpha 1.0 features 0 1 | 2.0 | 0.70 ± 0.12 | 0.09 ± 0.02 | 0.6630 | 0.6060 | 0.4941 |
| Ridge alpha 10.0 features 0 1 | 2.0 | 0.51 ± 0.04 | 0.06 ± 0.01 | 0.6630 | 0.6060 | 0.4941 |
| Ridge alpha 0.1 features 0 1 2 3 | 4.0 | 0.61 ± 0.04 | 0.07 ± 0.01 | 0.6436 | 0.5804 | 0.5089 |
| Ridge alpha 1.0 features 0 1 2 3 | 4.0 | 0.64 ± 0.07 | 0.09 ± 0.02 | 0.6435 | 0.5804 | 0.5090 |
| Ridge alpha 10.0 features 0 1 2 3 | 4.0 | 0.57 ± 0.04 | 0.08 ± 0.01 | 0.6425 | 0.5806 | 0.5097 |
| Ridge alpha 0.1 features 0 1 2 3 4 5 6 7 | 8.0 | 0.83 ± 0.26 | 0.09 ± 0.01 | 0.5559 | 0.5332 | 0.5758 |
| Ridge alpha 1.0 features 0 1 2 3 4 5 6 7 | 8.0 | 0.76 ± 0.12 | 0.10 ± 0.02 | 0.5558 | 0.5332 | 0.5759 |
| Ridge alpha 10.0 features 0 1 2 3 4 5 6 7 | 8.0 | 0.90 ± 0.03 | 0.09 ± 0.01 | 0.5550 | 0.5332 | 0.5764 |
| SGDRegressor lr constant features 0 1 | 2.0 | 5.09 ± 0.23 | 0.07 ± 0.01 | 0.6697 | 0.5978 | 0.4889 |
| SGDRegressor lr adaptive features 0 1 | 2.0 | 25.38 ± 4.46 | 0.06 ± 0.03 | 0.6630 | 0.6061 | 0.4941 |
| SGDRegressor lr constant features 0 1 2 3 | 4.0 | 6.21 ± 1.85 | 0.05 ± 0.01 | 0.7362 | 0.5935 | 0.4382 |
| SGDRegressor lr adaptive features 0 1 2 3 | 4.0 | 29.73 ± 2.96 | 0.06 ± 0.02 | 0.6433 | 0.5806 | 0.5091 |
| SGDRegressor lr constant features 0 1 2 3 4 5 6 7 | 8.0 | 5.07 ± 0.39 | 0.08 ± 0.04 | 6993777197015006969856.0000 | 28611322355.0348 | -5337094993353519398912.0000 |
| SGDRegressor lr adaptive features 0 1 2 3 4 5 6 7 | 8.0 | 34.39 ± 4.36 | 0.09 ± 0.05 | 6971646923003376.0000 | 55462164.8197 | -5320206926819254.0000 |
| RANSACRegressor features 0 1 | 2.0 | 11.92 ± 1.58 | 0.12 ± 0.02 | 0.7590 | 0.6057 | 0.4208 |

## Rust (machinelearne-rs) Metrics Summary

| Features | Train Time (ms) | MSE | MAE | R² |
|----------|-----------------|-----|-----|-----|
| 1 | 341.00 | 0.6943 | 0.6505 | 0.5235 |
| 2 | 436.00 | 0.6963 | 0.6351 | 0.5221 |
| 4 | 698.00 | 0.6670 | 0.6105 | 0.5423 |
| 8 | 1264.00 | 0.4904 | 0.5244 | 0.6634 |

## Rust (machinelearne-rs) Criterion Benchmarks

| Benchmark | Mean Time (ms) |
|-----------|----------------|
| mae_large_1M | 1.0166 |
| mse_large_1M | 0.9991 |
| predict_single | 0.0000 |
| predict_warmup | 0.0000 |
| r_squared_large_1M | 3.0745 |
| train_1_feature_with_metrics | 23.9972 |
| train_2_features_with_metrics | 28.0424 |
| train_4_features_with_metrics | 37.4616 |
| train_8_features_with_metrics | 62.4508 |

## Analysis

### Key Findings

### Performance Gaps

**1 Features:**
- Training: scikit-learn 0.85ms vs Rust 341.00ms (0.00x faster for sklearn)
- Accuracy (MSE): sklearn 0.7091 vs Rust 0.6943 (-2.1% difference)

**2 Features:**
- Training: scikit-learn 0.61ms vs Rust 436.00ms (0.00x faster for sklearn)
- Accuracy (MSE): sklearn 0.6630 vs Rust 0.6963 (+5.0% difference)

**4 Features:**
- Training: scikit-learn 0.72ms vs Rust 698.00ms (0.00x faster for sklearn)
- Accuracy (MSE): sklearn 0.6436 vs Rust 0.6670 (+3.6% difference)

**8 Features:**
- Training: scikit-learn 2.08ms vs Rust 1264.00ms (0.00x faster for sklearn)
- Accuracy (MSE): sklearn 0.5559 vs Rust 0.4904 (-11.8% difference)

### Key Findings

1. **Training Performance:** Comparison of training times across different feature counts.

2. **Prediction Performance:** Comparison of prediction latency for single and batch predictions.

3. **Accuracy Parity:** Comparison of MSE, MAE, and R² metrics between implementations.

4. **Scalability:** How performance scales with:
   - Number of features
   - Dataset size
   - Batch size


## Recommendations

### For Users

- **For production use:** Choose the library that best fits your use case and ecosystem.

- **For performance:** Consider the trade-offs between training speed, prediction speed, and accuracy.

- **For memory constraints:** Evaluate the memory footprint of each implementation.


### For Developers

1. **Optimization Opportunities:**
   - Vector operations optimization
   - Memory allocation reduction
   - Parallel processing for batch operations

2. **Feature Parity:**
   - Additional optimizers (Adam, RMSprop)
   - More loss functions
   - Advanced regularization techniques

3. **Benchmark Improvements:**
   - Memory profiling
   - CPU utilization metrics
   - GPU backend benchmarks


## Appendix

### Benchmark Methodology

**Training Benchmarks:**
- Models trained with SGD optimizer
- Learning rates: 0.001, 0.01, 0.1
- Batch sizes: 16, 32, 64, 128
- Epochs: 10, 50, 100
- 10 runs per configuration, mean ± std dev reported

**Prediction Benchmarks:**
- Single sample prediction latency
- Batch prediction for sizes: 10, 100, 1000, 10000
- 10,000 iterations for single prediction
- 5 warmup iterations before measurement

**Metrics:**
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

