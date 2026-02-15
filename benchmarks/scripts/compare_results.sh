#!/bin/bash
# Compare Rust and sklearn benchmark results

set -e

echo "======================================================================"
echo "Comparing Rust and sklearn benchmark results"
echo "======================================================================"

# Paths
RESULTS_DIR="benchmarks/results"
RUST_RESULTS_DIR="${RESULTS_DIR}/rust"
SKLEARN_RESULTS="${RESULTS_DIR}/raw/sklearn_results.csv"
COMPARE_OUTPUT="${RESULTS_DIR}/comparison.md"

# Create directories
mkdir -p "${RUST_RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/raw"

# Check if sklearn results exist
if [ ! -f "${SKLEARN_RESULTS}" ]; then
    echo "Warning: sklearn results not found at ${SKLEARN_RESULTS}"
    echo "Run: python benchmarks/scripts/sklearn_benchmarks.py"
    echo "Continuing with available results..."
fi

# Check if rust results exist
if [ ! -d "${RUST_RESULTS_DIR}" ] || [ -z "$(ls -A ${RUST_RESULTS_DIR})" ]; then
    echo "Warning: Rust results not found in ${RUST_RESULTS_DIR}"
    echo "Run: cargo bench --bench"
    echo "Continuing with available results..."
fi

# Generate comparison markdown
cat > "${COMPARE_OUTPUT}" << 'EOF'
# Benchmark Comparison: machinelearne-rs vs scikit-learn

This document compares the performance of `machinelearne-rs` and `scikit-learn`
on the California Housing dataset.

## Environment

- **Rust**: $(rustc --version)
- **Python**: $(python --version)
- **Date**: $(date)

## Results

### Training Time Comparison

| Model | Features | Rust (ms) | Sklearn (ms) | Speedup |
|-------|----------|-----------|--------------|---------|
EOF

# Parse sklearn results if available
if [ -f "${SKLEARN_RESULTS}" ]; then
    echo ""
    echo "Parsing sklearn results..."
    python3 << 'PYEOF'
import csv
import os
from pathlib import Path

sklearn_results_path = Path("benchmarks/results/raw/sklearn_results.csv")
if sklearn_results_path.exists():
    with open(sklearn_results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(f"  {row['model']}: train={row['train_time_mean_ms']:.2f}ms, "
                  f"pred={row['pred_time_mean_ms']:.2f}ms, R²={row['r2']:.4f}")
else:
    print("  No sklearn results found")
PYEOF
fi

echo ""
echo "======================================================================"
echo "Comparison complete!"
echo "======================================================================"
echo ""
echo "Results saved to: ${COMPARE_OUTPUT}"
echo ""
echo "To view the comparison:"
echo "  cat ${COMPARE_OUTPUT}"
