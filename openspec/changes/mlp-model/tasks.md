# Tasks: MLP Model Implementation

## Phase 1: Backend Extensions
- [x] Add `relu_1d`, `relu_2d` to Backend trait
- [x] Add `tanh_1d`, `tanh_2d` to Backend trait
- [x] Implement in CPU backend
- [ ] Implement in ndarray backend (if feature enabled)
- [x] Add unit tests for new operations (via existing test suite)

## Phase 2: Activation Module
- [x] Create `lib/src/model/activation.rs`
- [x] Define `Activation` enum (ReLU, Sigmoid, Tanh, Identity)
- [x] Implement `forward_1d` for each activation
- [x] Implement `backward_1d` for each activation
- [x] Add unit tests

## Phase 3: MLPParams
- [x] Create `lib/src/model/mlp.rs`
- [x] Define `LayerParams<B>` struct
- [x] Define `MLPParams<B>` struct
- [x] Implement `ParamOps<B>` for `MLPParams<B>`
- [x] Add `SerializableMLPParams` for serde
- [x] Add unit tests

## Phase 4: MLPModel Core
- [x] Define `MLPModel<B, S>` struct
- [x] Implement `new()` constructor with layer_sizes and activations
- [x] Implement weight initialization (Xavier - zeros pending random init)
- [x] Implement forward pass
- [x] Add type aliases (MLP, MLPRegressor)

## Phase 5: Training Support
- [x] Implement `ForwardCache` for storing activations
- [x] Implement `forward_with_cache()` method
- [x] Implement `backward()` with full backpropagation
- [x] Implement `TrainableModel<B>` trait
- [ ] Test with XOR dataset

## Phase 6: Inference Support
- [x] Implement `InferenceModel<B>` for `MLPModel<B, Fitted>`
- [x] Implement `predict()` for single sample
- [x] Implement `predict_batch()` for batch inference
- [x] Implement serialization (save/load)

## Phase 7: Integration
- [x] Implement `Optimizer<B, MLPParams<B>>` for `SGD<B>`
- [x] Implement `Regularizer<B, MLP<B>>` for L2
- [x] Add exports to `lib/src/model/mod.rs`
- [x] Test full training pipeline

## Phase 8: Examples & Docs
- [x] Create `train_mlp.rs` example (XOR)
- [ ] Create `train_mlp_california.rs` example
- [ ] Update CHANGELOG.md
- [ ] Update CLAUDE.md with MLP section

## Phase 9: ONNX Export
- [ ] Implement `OnnxNodeBuilder` for MLP layers
- [ ] Implement `OnnxExportable` for fitted MLP
- [ ] Add ONNX export tests
- [ ] Create `export_mlp_onnx.rs` example

## Phase 10: Benchmarks
- [ ] Create `mlp_cpu_wgpu_comparison.rs` benchmark
- [ ] Test on California Housing dataset
- [ ] Compare training time: CPU vs WGPU
- [ ] Compare inference time: CPU vs WGPU vs ONNX Runtime
- [ ] Document performance results in example output
