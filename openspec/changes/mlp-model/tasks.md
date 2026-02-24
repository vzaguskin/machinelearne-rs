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
- [ ] Create `lib/src/model/mlp.rs`
- [ ] Define `LayerParams<B>` struct
- [ ] Define `MLPParams<B>` struct
- [ ] Implement `ParamOps<B>` for `MLPParams<B>`
- [ ] Add `SerializableMLPParams` for serde
- [ ] Add unit tests

## Phase 4: MLPModel Core
- [ ] Define `MLPModel<B, S>` struct
- [ ] Implement `new()` constructor with layer_sizes and activations
- [ ] Implement weight initialization (Xavier)
- [ ] Implement forward pass
- [ ] Add type aliases (MLP, MLPRegressor)

## Phase 5: Training Support
- [ ] Implement `ForwardCache` for storing activations
- [ ] Implement `forward_with_cache()` method
- [ ] Implement `backward()` with full backpropagation
- [ ] Implement `TrainableModel<B>` trait
- [ ] Test with XOR dataset

## Phase 6: Inference Support
- [ ] Implement `InferenceModel<B>` for `MLPModel<B, Fitted>`
- [ ] Implement `predict()` for single sample
- [ ] Implement `predict_batch()` for batch inference
- [ ] Implement serialization (save/load)

## Phase 7: Integration
- [ ] Implement `Optimizer<B, MLPParams<B>>` for `SGD<B>`
- [ ] Implement `Regularizer<B, MLP<B>>` for L2
- [ ] Add exports to `lib/src/model/mod.rs`
- [ ] Test full training pipeline

## Phase 8: Examples & Docs
- [ ] Create `train_mlp.rs` example (XOR)
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
