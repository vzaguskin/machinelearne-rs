# Proposal: MLP (Multi-Layer Perceptron) Model

## Problem

The library currently only supports linear models. Users need more expressive models for non-linear problems like:
- XOR classification
- Complex regression tasks
- Feature learning from raw data

## Why MLP?

MLP is the ideal next model because:
1. **Perfect architecture fit**: Uses existing gradient-based training infrastructure
2. **No interface changes**: Works with existing Trainer, Optimizer, Loss, Regularizer
3. **Backend agnostic**: Naturally works with CPU, ndarray, and WGPU backends
4. **Foundation**: Enables future CNNs, RNNs, Transformers

### Why not Random Forest or Gradient Boosting?

| Algorithm | Interface Fit | Changes Needed |
|-----------|---------------|----------------|
| MLP | Excellent | None (new model only) |
| Random Forest | Poor | New trainer architecture |
| Gradient Boosting | Moderate | Tree infrastructure |

## Goals

1. Add MLP model with configurable layers and activations
2. Support all existing training features (early stopping, gradient clipping, regularization)
3. Enable ONNX export for deployment
4. Benchmark CPU vs WGPU performance

## Non-Goals

- CNN architecture (future work)
- Custom layer types beyond Dense
- Advanced optimizers (Adam, etc.) - future work

## Success Criteria

- MLP trains successfully on XOR problem (non-linear)
- MLP regression on California Housing with R² > 0.6
- ONNX export produces valid model files
- WGPU backend works (even if slower than CPU for small datasets)
