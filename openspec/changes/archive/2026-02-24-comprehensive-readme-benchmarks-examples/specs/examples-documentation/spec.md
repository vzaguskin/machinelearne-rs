## ADDED Requirements

### Requirement: Examples are categorized by skill level and topic

The examples documentation SHALL organize all examples into logical categories to help users find relevant code for their learning path.

Categories SHALL include at minimum:
- **Basic**: Simple introductory examples (linear regression, basic training)
- **Intermediate**: Examples with multiple components (regularization, different losses)
- **Advanced**: Complex workflows (pipelines, grid search, real datasets)
- **GPU/WGPU**: Examples demonstrating GPU acceleration
- **ONNX/Deployment**: Export and deployment examples

#### Scenario: User finds examples by skill level

- **WHEN** user opens the examples README
- **THEN** examples are grouped into clear categories with skill level indicators

#### Scenario: Examples show learning progression

- **WHEN** user reviews the basic category
- **THEN** examples are listed in recommended learning order

---

### Requirement: Each example has complete documentation entry

Each example SHALL have a documentation entry containing:
1. Example name and file reference
2. One-sentence description of what it demonstrates
3. Required Cargo features (if any)
4. Run command
5. Expected output or behavior notes

#### Scenario: Example entry shows all required information

- **WHEN** user views an example entry
- **THEN** the entry includes description, features, run command, and expected output

#### Scenario: Feature requirements are clearly marked

- **WHEN** example requires specific Cargo features
- **THEN** the features are listed in the documentation entry with the run command

---

### Requirement: Examples include feature requirements matrix

The documentation SHALL provide a feature requirements matrix showing which examples require which features.

#### Scenario: User identifies feature dependencies at a glance

- **WHEN** user views the feature matrix
- **THEN** all examples and their required features are listed in tabular format

---

### Requirement: GPU examples document performance expectations

Examples involving GPU/WGPU backends SHALL document expected performance characteristics and hardware requirements.

#### Scenario: GPU example shows expected speedup

- **WHEN** user reviews a GPU example
- **THEN** the documentation notes expected performance improvements vs CPU

#### Scenario: Hardware requirements are documented

- **WHEN** GPU example has specific hardware requirements
- **THEN** requirements are noted in the documentation

---

### Requirement: ONNX examples document end-to-end workflows

ONNX-related examples SHALL document the complete workflow from training to deployment.

#### Scenario: ONNX export example shows complete workflow

- **WHEN** user follows an ONNX export example
- **THEN** the example shows training, export, and inference verification

#### Scenario: Server example documents API endpoints

- **WHEN** user reviews the server example
- **THEN** API endpoints and request formats are documented

---

### Requirement: Benchmark documentation is comprehensive

The benchmarks README SHALL document:
1. Purpose of each benchmark binary
2. How to run each benchmark
3. How to interpret results
4. System requirements (Python dependencies, etc.)

#### Scenario: Benchmark entry shows run commands

- **WHEN** user views a benchmark entry
- **THEN** run commands for both Rust and comparison scripts are provided

#### Scenario: Results interpretation is documented

- **WHEN** user reviews benchmark results
- **THEN** the documentation explains what the metrics mean and how to compare them
