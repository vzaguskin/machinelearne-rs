#!/usr/bin/env python3
"""
Run full pipeline comparison: sklearn vs Rust

Compares three configurations:
1. Naive baseline
2. Default baseline (StandardScaler)
3. Optimized (GridSearchCV over scalers + L2 + polynomial)

All evaluated on holdout test set after save/load verification.
"""

import subprocess
import json
import sys
from pathlib import Path


def main():
    print("=== Pipeline Selection Comparison ===\n")
    print("This will run both sklearn and Rust implementations and compare results.\n")

    # Ensure output directories exist
    Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
    Path("saved_models").mkdir(exist_ok=True)

    # 1. Run sklearn comparison
    print("=" * 60)
    print("Running sklearn comparison...")
    print("=" * 60)

    result = subprocess.run(
        ["python3", "benchmarks/scripts/pipeline_comparison_sklearn.py"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"sklearn comparison failed with return code {result.returncode}")
        sys.exit(1)

    sklearn_results_path = Path("benchmarks/results/pipeline_comparison_sklearn.json")
    if not sklearn_results_path.exists():
        print("sklearn results file not found!")
        sys.exit(1)

    with open(sklearn_results_path) as f:
        sklearn_results = json.load(f)

    # 2. Run Rust comparison
    print("\n" + "=" * 60)
    print("Running Rust comparison...")
    print("=" * 60)

    result = subprocess.run(
        ["cargo", "run", "--release", "--example", "real_world_pipeline"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"Rust example failed with return code {result.returncode}")
        sys.exit(1)

    rust_results_path = Path("benchmarks/results/pipeline_comparison_rust.json")
    if not rust_results_path.exists():
        print("Rust results file not found!")
        sys.exit(1)

    with open(rust_results_path) as f:
        rust_results = json.load(f)

    # 3. Print comparison table
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)

    print("\n{:<25} {:>12} {:>12} {:>12}".format("Configuration", "sklearn R2", "Rust R2", "Diff"))
    print("-" * 70)

    for config in ["naive_baseline", "default_baseline", "optimized"]:
        sk_r2 = sklearn_results.get(config, {}).get("test_r2", 0)
        rs_r2 = rust_results.get(config, {}).get("test_r2", 0)
        diff = rs_r2 - sk_r2
        print("{:<25} {:>12.4f} {:>12.4f} {:>+12.4f}".format(config, sk_r2, rs_r2, diff))

    # 4. Show improvement from optimization
    print("\n" + "-" * 70)
    print("Improvement from pipeline optimization (naive -> optimized):")
    sk_naive = sklearn_results["naive_baseline"]["test_r2"]
    sk_opt = sklearn_results["optimized"]["test_r2"]
    rs_naive = rust_results["naive_baseline"]["test_r2"]
    rs_opt = rust_results["optimized"]["test_r2"]

    sk_improvement = (
        ((sk_opt - sk_naive) / abs(sk_naive) * 100) if abs(sk_naive) > 0.001 else 0
    )
    rs_improvement = (
        ((rs_opt - rs_naive) / abs(rs_naive) * 100) if abs(rs_naive) > 0.001 else 0
    )

    print(
        "  sklearn: {:.4f} -> {:.4f} (improvement: +{:.1f}%)".format(
            sk_naive, sk_opt, sk_improvement
        )
    )
    print(
        "  Rust:    {:.4f} -> {:.4f} (improvement: +{:.1f}%)".format(
            rs_naive, rs_opt, rs_improvement
        )
    )

    # 5. Show best parameters found
    print("\n" + "-" * 70)
    print("Best parameters found:")
    print(f"  sklearn: {sklearn_results['optimized']['best_params']}")
    print(f"  Rust:    {rust_results['optimized']['best_params']}")

    # 6. Save combined results
    combined = {
        "sklearn": sklearn_results,
        "rust": rust_results,
        "comparison": {
            "sklearn_improvement_pct": sk_improvement,
            "rust_improvement_pct": rs_improvement,
        },
    }

    combined_path = Path("benchmarks/results/pipeline_comparison_combined.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
