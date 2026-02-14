#!/usr/bin/env python3
"""
Generate comprehensive benchmark report comparing machinelearne-rs and sklearn.

This script parses benchmark results from both Rust and Python and generates
a detailed markdown report with comparisons and analysis.
"""

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class BenchmarkReport:
    """Generate benchmark comparison report."""

    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.raw_dir = self.results_dir / "raw"
        self.rust_dir = Path("target/criterion")  # Criterion stores results here
        self.report_path = self.results_dir / "BENCHMARK_REPORT.md"

    def parse_sklearn_results(self) -> List[Dict[str, Any]]:
        """Parse sklearn benchmark results from CSV."""
        results_file = self.raw_dir / "sklearn_results.csv"

        if not results_file.exists():
            print(f"Warning: sklearn results not found at {results_file}")
            return []

        with open(results_file) as f:
            reader = csv.DictReader(f)
            results = []
            for row in reader:
                # Convert numeric fields to floats
                converted = {}
                for key, value in row.items():
                    if key in ["train_time_mean_ms", "train_time_std_ms",
                               "pred_time_mean_ms", "pred_time_std_ms",
                               "mse", "mae", "r2", "n_train_samples",
                               "n_test_samples", "n_features", "n_runs"]:
                        converted[key] = float(value)
                    else:
                        converted[key] = value
                results.append(converted)
            return results

    def parse_rust_criterion_results(self) -> Dict[str, float]:
        """
        Parse Rust criterion benchmark results.

        Looks for criterion estimates.json files in the target/criterion directory.
        Returns a dictionary mapping benchmark names to mean times in ms.
        """
        results = {}

        # Look for criterion estimates.json files in the "new" subdirectories
        for estimates_file in self.rust_dir.glob("*/new/estimates.json"):
            try:
                with open(estimates_file) as f:
                    data = json.load(f)

                # Get benchmark name from directory path
                # Path is like "target/criterion/train_1_feature_with_metrics/new/estimates.json"
                bench_dir = estimates_file.parent.parent
                bench_name = str(bench_dir.name)

                # Extract mean time from estimates
                if "mean" in data and "point_estimate" in data["mean"]:
                    # Criterion stores time in nanoseconds
                    mean_ns = data["mean"]["point_estimate"]
                    mean_ms = mean_ns / 1_000_000
                    results[bench_name] = mean_ms
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse {estimates_file}: {e}")

        return results

    def get_rust_version(self) -> str:
        """Get Rust version."""
        try:
            result = subprocess.run(
                ["rustc", "--version"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def get_python_version(self) -> str:
        """Get Python version."""
        try:
            result = subprocess.run(
                ["python", "--version"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def extract_feature_count(self, name: str) -> Optional[int]:
        """Extract feature count from benchmark name."""
        # Match patterns like "train_1_feature", "train_2_features", etc.
        match = re.search(r"train_(\d+)_feature", name)
        if match:
            return int(match.group(1))
        match = re.search(r"features_(\d+)", name)
        if match:
            return int(match.group(1))
        match = re.search(r"features_\[(\d+)\]", name)
        if match:
            return int(match.group(1))
        return None

    def generate_report(self):
        """Generate the complete benchmark report."""
        sklearn_results = self.parse_sklearn_results()
        rust_results = self.parse_rust_criterion_results()

        with open(self.report_path, "w") as f:
            # Header
            f.write("# Benchmark Comparison: machinelearne-rs vs scikit-learn\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Environment
            f.write("## Environment\n\n")
            f.write("| Component | Version |\n")
            f.write("|-----------|---------|\n")
            f.write(f"| Rust | {self.get_rust_version()} |\n")
            f.write(f"| Python | {self.get_python_version()} |\n")
            f.write("| OS | Linux |\n")
            f.write("\n")

            # Dataset
            f.write("## Dataset\n\n")
            f.write("### California Housing Dataset\n\n")
            f.write("- **Samples:** 20,640\n")
            f.write("- **Features:** 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)\n")
            f.write("- **Target:** Median House Value (in $100,000s)\n")
            f.write("- **Split:** 80% train / 20% test\n")
            f.write("\n")

            # Sklearn Results
            if sklearn_results:
                f.write("## scikit-learn Results\n\n")
                self._write_sklearn_table(f, sklearn_results)
                f.write("\n")

            # Rust Results
            if rust_results:
                f.write("## Rust (machinelearne-rs) Results\n\n")
                self._write_rust_table(f, rust_results)
                f.write("\n")

            # Comparison
            if sklearn_results and rust_results:
                f.write("## Performance Comparison\n\n")
                self._write_comparison(f, sklearn_results, rust_results)
                f.write("\n")

            # Analysis
            f.write("## Analysis\n\n")
            f.write("### Key Findings\n\n")
            self._write_analysis(f, sklearn_results, rust_results)
            f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### For Users\n\n")
            self._write_recommendations(f, sklearn_results, rust_results)
            f.write("\n")

            f.write("### For Developers\n\n")
            self._write_dev_recommendations(f, sklearn_results, rust_results)
            f.write("\n")

            # Appendix
            f.write("## Appendix\n\n")
            f.write("### Benchmark Methodology\n\n")
            f.write("**Training Benchmarks:**\n")
            f.write("- Models trained with SGD optimizer\n")
            f.write("- Learning rates: 0.001, 0.01, 0.1\n")
            f.write("- Batch sizes: 16, 32, 64, 128\n")
            f.write("- Epochs: 10, 50, 100\n")
            f.write("- 10 runs per configuration, mean ± std dev reported\n")
            f.write("\n")
            f.write("**Prediction Benchmarks:**\n")
            f.write("- Single sample prediction latency\n")
            f.write("- Batch prediction for sizes: 10, 100, 1000, 10000\n")
            f.write("- 10,000 iterations for single prediction\n")
            f.write("- 5 warmup iterations before measurement\n")
            f.write("\n")
            f.write("**Metrics:**\n")
            f.write("- MSE (Mean Squared Error)\n")
            f.write("- MAE (Mean Absolute Error)\n")
            f.write("- R² (Coefficient of Determination)\n")
            f.write("\n")

        print(f"Report generated at: {self.report_path}")
        print(f"Total sklearn results: {len(sklearn_results)}")
        print(f"Total rust results: {len(rust_results)}")

    def _write_sklearn_table(self, f, results: List[Dict[str, Any]]):
        """Write sklearn results table."""
        f.write("| Model | Features | Train Time (ms) | Pred Time (ms) | MSE | MAE | R² |\n")
        f.write("|-------|----------|-----------------|----------------|-----|-----|-----|\n")

        for r in results:
            n_features = r.get("n_features", 0)
            model = r.get("model", "unknown").replace("_", " ")
            row = f"| {model} | {n_features} | {r['train_time_mean_ms']:.2f} ± {r['train_time_std_ms']:.2f} | {r['pred_time_mean_ms']:.2f} ± {r['pred_time_std_ms']:.2f} | {r['mse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} |\n"
            f.write(row)

    def _write_rust_table(self, f, results: Dict[str, float]):
        """Write Rust results table."""
        f.write("| Benchmark | Mean Time (ms) |\n")
        f.write("|-----------|----------------|\n")

        for name, time_ms in sorted(results.items()):
            f.write(f"| {name} | {time_ms:.4f} |\n")

    def _write_comparison(
        self,
        f,
        sklearn_results: List[Dict[str, Any]],
        rust_results: Dict[str, float],
    ):
        """Write comparison table."""
        f.write("### Training Time Comparison\n\n")
        f.write("| Features | Sklearn (ms) | Rust (ms) | Speedup |\n")
        f.write("|----------|-------------|-----------|---------|\n")

        # Group sklearn results by feature count
        sklearn_by_features = {}
        for r in sklearn_results:
            n_features = r.get("n_features", 0)
            if "LinearRegression" in r.get("model", ""):
                sklearn_by_features[n_features] = r.get("train_time_mean_ms", 0)

        # Find corresponding Rust results
        for n_features, sklearn_time in sorted(sklearn_by_features.items()):
            # Look for matching Rust benchmark (n_features is float, convert to int for pattern)
            rust_bench_name = None
            pattern = f"train_{int(n_features)}"
            for name in rust_results.keys():
                if pattern in name:
                    rust_bench_name = name
                    break

            if rust_bench_name:
                rust_time = rust_results[rust_bench_name]
                speedup = sklearn_time / rust_time if rust_time > 0 else 0
                f.write(
                    f"| {n_features} | {sklearn_time:.2f} | {rust_time:.2f} | "
                    f"{speedup:.2f}x |\n"
                )
            else:
                f.write(f"| {n_features} | {sklearn_time:.2f} | N/A | N/A |\n")

    def _write_analysis(
        self,
        f,
        sklearn_results: List[Dict[str, Any]],
        rust_results: Dict[str, float],
    ):
        """Write analysis section."""
        f.write("1. **Training Performance:** ")
        f.write("Comparison of training times across different feature counts.\n\n")

        f.write("2. **Prediction Performance:** ")
        f.write("Comparison of prediction latency for single and batch predictions.\n\n")

        f.write("3. **Accuracy Parity:** ")
        f.write("Comparison of MSE, MAE, and R² metrics between implementations.\n\n")

        f.write("4. **Scalability:** ")
        f.write("How performance scales with:\n")
        f.write("   - Number of features\n")
        f.write("   - Dataset size\n")
        f.write("   - Batch size\n\n")

    def _write_recommendations(
        self,
        f,
        sklearn_results: List[Dict[str, Any]],
        rust_results: Dict[str, float],
    ):
        """Write user recommendations."""
        f.write("- **For production use:** ")
        f.write("Choose the library that best fits your use case and ecosystem.\n\n")

        f.write("- **For performance:** ")
        f.write("Consider the trade-offs between training speed, prediction speed, and accuracy.\n\n")

        f.write("- **For memory constraints:** ")
        f.write("Evaluate the memory footprint of each implementation.\n\n")

    def _write_dev_recommendations(
        self,
        f,
        sklearn_results: List[Dict[str, Any]],
        rust_results: Dict[str, float],
    ):
        """Write developer recommendations."""
        f.write("1. **Optimization Opportunities:**\n")
        f.write("   - Vector operations optimization\n")
        f.write("   - Memory allocation reduction\n")
        f.write("   - Parallel processing for batch operations\n\n")

        f.write("2. **Feature Parity:**\n")
        f.write("   - Additional optimizers (Adam, RMSprop)\n")
        f.write("   - More loss functions\n")
        f.write("   - Advanced regularization techniques\n\n")

        f.write("3. **Benchmark Improvements:**\n")
        f.write("   - Memory profiling\n")
        f.write("   - CPU utilization metrics\n")
        f.write("   - GPU backend benchmarks\n\n")


def main():
    """Main entry point."""
    report = BenchmarkReport()
    report.generate_report()


if __name__ == "__main__":
    main()
