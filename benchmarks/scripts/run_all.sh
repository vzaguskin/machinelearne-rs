#!/bin/bash
# Comprehensive benchmark runner for machinelearne-rs and sklearn

set -e

echo "======================================================================"
echo "Comprehensive Benchmark Suite: machinelearne-rs vs scikit-learn"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section header
print_section() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
    echo ""
}

# Function to run command with error handling
run_cmd() {
    echo -e "${GREEN}▶${NC} $1"
    if eval "$1"; then
        echo -e "${GREEN}✓${NC} Success"
    else
        echo -e "${RED}✗${NC} Failed"
        return 1
    fi
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${YELLOW}Warning: python3 not found. Skipping sklearn benchmarks.${NC}"
        return 1
    fi
    return 0
}

# Check if Rust is available
check_rust() {
    if ! command -v cargo &> /dev/null; then
        echo -e "${YELLOW}Warning: cargo not found. Skipping Rust benchmarks.${NC}"
        return 1
    fi
    return 0
}

# Main script
print_section "Step 1: Check Prerequisites"

PYTHON_AVAILABLE=false
RUST_AVAILABLE=false

if check_python; then
    PYTHON_AVAILABLE=true
    echo -e "${GREEN}✓${NC} Python found: $(python --version)"
fi

if check_rust; then
    RUST_AVAILABLE=true
    echo -e "${GREEN}✓${NC} Rust found: $(rustc --version)"
fi

if ! $PYTHON_AVAILABLE && ! $RUST_AVAILABLE; then
    echo -e "${RED}Error: Neither Python nor Rust found. Exiting.${NC}"
    exit 1
fi

# Step 2: Setup directories
print_section "Step 2: Setup Benchmark Directories"
mkdir -p benchmarks/results/rust
mkdir -p benchmarks/results/raw
echo -e "${GREEN}✓${NC} Directories created"

# Step 3: Check dataset
print_section "Step 3: Check Dataset"
if [ -f "benchmarks/datasets/california_housing.csv" ]; then
    echo -e "${GREEN}✓${NC} Dataset found"
    TOTAL_SAMPLES=$(wc -l < benchmarks/datasets/california_housing.csv)
    echo "  Total samples: $((TOTAL_SAMPLES - 1))"  # Subtract header
else
    echo -e "${YELLOW}Warning: Dataset not found. Attempting to download...${NC}"
    if $PYTHON_AVAILABLE; then
        run_cmd "python benchmarks/scripts/download_dataset.py"
    else
        echo -e "${RED}Error: Cannot download dataset without Python${NC}"
        exit 1
    fi
fi

# Step 4: Run sklearn benchmarks
if $PYTHON_AVAILABLE; then
    print_section "Step 4: Run Sklearn Benchmarks"
    run_cmd "python benchmarks/scripts/sklearn_benchmarks.py"
else
    print_section "Step 4: Sklearn Benchmarks (Skipped)"
    echo -e "${YELLOW}Skipped: Python not available${NC}"
fi

# Step 5: Run Rust benchmarks
if $RUST_AVAILABLE; then
    print_section "Step 5: Run Rust Benchmarks"

    echo "Running linear regression training benchmarks..."
    run_cmd "cargo bench --bench train_1_feature -- --save-baseline main"

    echo "Running linear regression training benchmarks (2 features)..."
    run_cmd "cargo bench --bench train_2_features -- --save-baseline main"

    echo "Running linear regression training benchmarks (4 features)..."
    run_cmd "cargo bench --bench train_4_features -- --save-baseline main"

    echo "Running linear regression training benchmarks (8 features)..."
    run_cmd "cargo bench --bench train_8_features -- --save-baseline main"

    echo "Running prediction benchmarks..."
    run_cmd "cargo bench --bench predict -- --save-baseline main"

    echo "Running metrics benchmarks..."
    run_cmd "cargo bench --bench metrics -- --save-baseline main"
else
    print_section "Step 5: Rust Benchmarks (Skipped)"
    echo -e "${YELLOW}Skipped: Rust not available${NC}"
fi

# Step 6: Generate comparison
print_section "Step 6: Generate Comparison"
if $PYTHON_AVAILABLE; then
    run_cmd "python benchmarks/scripts/generate_report.py"
else
    echo -e "${YELLOW}Skipping report generation: Python not available${NC}"
fi

# Step 7: Summary
print_section "Benchmark Complete!"

echo "Results saved to:"
echo "  - benchmarks/results/raw/     (Raw CSV data)"
echo "  - benchmarks/results/rust/   (Rust criterion output)"
echo "  - benchmarks/results/BENCHMARK_REPORT.md  (Comparison report)"
echo ""

echo "To view the report:"
echo "  cat benchmarks/results/BENCHMARK_REPORT.md"
echo ""

if $PYTHON_AVAILABLE; then
    echo "To run individual benchmarks:"
    echo "  # Sklearn:"
    echo "  python benchmarks/scripts/sklearn_benchmarks.py"
    echo ""
fi

if $RUST_AVAILABLE; then
    echo "  # Rust:"
    echo "  cargo bench --bench train_1_feature"
    echo "  cargo bench --bench predict"
    echo "  cargo bench --bench metrics"
    echo ""
fi

echo "To compare results:"
echo "  bash benchmarks/scripts/compare_results.sh"
echo ""
