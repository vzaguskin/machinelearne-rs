## Context

The library has 21+ examples in `lib/examples/` and 6+ benchmark binaries in `benchmarks/src/bin/`. Currently:
- Main README lists examples with single-line commands
- `benchmarks/README.md` documents the fair_comparison benchmark well but lacks other binaries
- No dedicated `lib/examples/README.md` exists
- Users must read source code to understand example purposes

## Goals / Non-Goals

**Goals:**
- Create comprehensive `lib/examples/README.md` with categorized examples
- Document feature requirements and run commands
- Provide expected outputs and interpretation guidance
- Enhance benchmark documentation

**Non-Goals:**
- Modifying example code (documentation only)
- Adding new examples
- Creating automated example testing
- Generating documentation from code comments

## Decisions

### Decision 1: Create dedicated `lib/examples/README.md`

**Rationale:** Examples are numerous and deserve their own documentation file, similar to how benchmarks have `benchmarks/README.md`.

**Alternative considered:** Expand main README - rejected because it would make main README too long.

### Decision 2: Organize by category with skill levels

**Categories:**
1. **Basic Training** - Simple linear regression examples
2. **Regularization** - L1/L2 regularization examples
3. **Classification** - Logistic regression examples
4. **MLP Neural Networks** - Neural network examples
5. **Preprocessing Pipelines** - Data transformation examples
6. **Model Selection** - Grid search and cross-validation
7. **GPU/WGPU** - GPU acceleration examples
8. **ONNX Export** - Model export examples
9. **Deployment** - Server and inference examples

**Rationale:** Categorical organization helps users find relevant examples faster than alphabetical listing.

### Decision 3: Include feature requirements matrix

**Format:** Markdown table with examples as rows, features as columns.

**Rationale:** Users often want to know which examples work with which feature combinations without reading each file.

### Decision 4: Enhance existing `benchmarks/README.md`

**Approach:** Add sections for each benchmark binary:
- `backend_comparison.rs`
- `full_batch_comparison.rs`
- `learning_rate_search.rs`
- `sgd_comparison.rs`
- `collect_metrics.rs`

**Rationale:** Keep benchmark documentation in one place rather than creating new files.

## Risks / Trade-offs

**Risk: Documentation becomes stale** → Each example file should have a brief header comment that matches the README description. CI does not currently validate this.

**Risk: Too much information** → Keep each example entry concise (3-5 lines) with links to source files for details.

**Trade-off: One README vs per-example docs** → Chose single README for discoverability. Per-example docs would be more detailed but harder to navigate.

## Open Questions

None - straightforward documentation task.
