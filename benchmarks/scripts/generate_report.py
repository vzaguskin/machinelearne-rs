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

    def parse_rust_metrics(self) -> Dict[int, Dict[str, Any]]:
        """
        Parse Rust metrics from rust_metrics.json file.

        Returns a dictionary mapping feature count to metrics dict.
        """
        metrics_file = self.results_dir / "rust_metrics.json"

        if not metrics_file.exists():
            print(f"Warning: Rust metrics not found at {metrics_file}")
            return {}

        with open(metrics_file) as f:
            data = json.load(f)

        results = {}
        for item in data.get("results", []):
            n_features = item.get("n_features")
            if n_features is not None:
                # Only include results with valid metrics (not null/NaN)
                if item.get("mse") is not None and item.get("mse") != "":
                    results[n_features] = item

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
        rust_metrics = self.parse_rust_metrics()
        rust_criterion = self.parse_rust_criterion_results()

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

            # Side-by-Side Comparison
            if sklearn_results and rust_metrics:
                f.write("## Side-by-Side Comparison\n\n")
                f.write("### Linear Regression (scikit-learn vs machinelearne-rs)\n\n")
                self._write_side_by_side_comparison(f, sklearn_results, rust_metrics)
                f.write("\n")

            # Sklearn Results (full table)
            if sklearn_results:
                f.write("## scikit-learn Full Results\n\n")
                self._write_sklearn_table(f, sklearn_results)
                f.write("\n")

            # Rust Metrics Summary
            if rust_metrics:
                f.write("## Rust (machinelearne-rs) Metrics Summary\n\n")
                self._write_rust_metrics_table(f, rust_metrics)
                f.write("\n")

            # Rust Criterion Timings
            if rust_criterion:
                f.write("## Rust (machinelearne-rs) Criterion Benchmarks\n\n")
                self._write_rust_table(f, rust_criterion)
                f.write("\n")

            # Analysis
            f.write("## Analysis\n\n")
            f.write("### Key Findings\n\n")
            self._write_analysis(f, sklearn_results, rust_metrics, rust_criterion)
            f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### For Users\n\n")
            self._write_recommendations(f, sklearn_results, rust_metrics)
            f.write("\n")

            f.write("### For Developers\n\n")
            self._write_dev_recommendations(f, sklearn_results, rust_metrics)
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
        print(f"Total rust metrics: {len(rust_metrics)}")
        print(f"Total rust criterion: {len(rust_criterion)}")

    def _write_rust_metrics_table(self, f, results: Dict[int, Dict[str, Any]]):
        """Write Rust metrics summary table."""
        f.write("| Features | Train Time (ms) | MSE | MAE | R² |\n")
        f.write("|----------|-----------------|-----|-----|-----|\n")

        for n_features, metrics in sorted(results.items()):
            train_time = metrics.get("train_time_ms", "N/A")
            mse = metrics.get("mse", "N/A")
            mae = metrics.get("mae", "N/A")
            r2 = metrics.get("r2", "N/A")

            # Format numbers if available
            train_time_str = f"{train_time:.2f}" if train_time != "N/A" else "N/A"
            mse_str = f"{mse:.4f}" if mse != "N/A" else "N/A"
            mae_str = f"{mae:.4f}" if mae != "N/A" else "N/A"
            r2_str = f"{r2:.4f}" if r2 != "N/A" else "N/A"

            f.write(f"| {n_features} | {train_time_str} | {mse_str} | {mae_str} | {r2_str} |\n")

    def _write_side_by_side_comparison(
        self,
        f,
        sklearn_results: List[Dict[str, Any]],
        rust_metrics: Dict[int, Dict[str, Any]],
    ):
        """Write side-by-side comparison table."""
        f.write("| Implementation | Features | Train Time (ms) | Pred Time (ms) | MSE | MAE | R² |\n")
        f.write("|--------------|----------|-----------------|----------------|-----|-----|-----|\n")

        # Get sklearn LinearRegression results grouped by features
        sklearn_by_features = {}
        for r in sklearn_results:
            if "LinearRegression" in r.get("model", ""):
                n_features = int(r.get("n_features", 0))
                if n_features not in sklearn_by_features:
                    sklearn_by_features[n_features] = r

        # Write comparisons for each feature count
        for n_features in sorted(sklearn_by_features.keys()):
            sklearn = sklearn_by_features[n_features]
            rust = rust_metrics.get(n_features)

            # Sklearn row
            sklearn_train = sklearn.get("train_time_mean_ms", 0)
            sklearn_pred = sklearn.get("pred_time_mean_ms", 0)
            sklearn_mse = sklearn.get("mse", 0)
            sklearn_mae = sklearn.get("mae", 0)
            sklearn_r2 = sklearn.get("r2", 0)

            f.write(
                f"| scikit-learn | {n_features} | "
                f"{sklearn_train:.2f} | {sklearn_pred:.2f} | "
                f"{sklearn_mse:.4f} | {sklearn_mae:.4f} | {sklearn_r2:.4f} |\n"
            )

            # Rust row (if available)
            if rust:
                rust_train = rust.get("train_time_ms", 0)
                rust_mse = rust.get("mse", 0)
                rust_mae = rust.get("mae", 0)
                rust_r2 = rust.get("r2", 0)

                # Compute speedup
                train_speedup = sklearn_train / rust_train if rust_train > 0 else 0
                mse_diff = ((rust_mse - sklearn_mse) / sklearn_mse * 100) if sklearn_mse > 0 else 0

                f.write(
                    f"| machinelearne-rs | {n_features} | "
                    f"{rust_train:.2f} ({train_speedup:.2f}x) | N/A | "
                    f"{rust_mse:.4f} ({mse_diff:+.1f}%) | {rust_mae:.4f} | {rust_r2:.4f} |\n"
                )
            else:
                f.write(f"| machinelearne-rs | {n_features} | N/A | N/A | N/A | N/A | N/A |\n")

            f.write("| | | | | | |\n")  # Empty row for spacing

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
        rust_metrics: Dict[int, Dict[str, Any]],
        rust_criterion: Dict[str, float],
    ):
        """Write analysis section."""

        # Calculate performance gaps
        if rust_metrics and sklearn_results:
            f.write("### Performance Gaps\n\n")

            # Get sklearn LinearRegression results
            sklearn_lr = {}
            for r in sklearn_results:
                if "LinearRegression" in r.get("model", ""):
                    n_features = int(r.get("n_features", 0))
                    sklearn_lr[n_features] = r

            for n_features in sorted(rust_metrics.keys()):
                sklearn = sklearn_lr.get(n_features)
                rust = rust_metrics[n_features]

                if sklearn and rust:
                    sklearn_time = sklearn.get("train_time_mean_ms", 0)
                    rust_time = rust.get("train_time_ms", 0)
                    sklearn_mse = sklearn.get("mse", 0)
                    rust_mse = rust.get("mse", 0)

                    speedup = sklearn_time / rust_time if rust_time > 0 else 0
                    mse_diff_pct = ((rust_mse - sklearn_mse) / sklearn_mse * 100) if sklearn_mse > 0 else 0

                    f.write(f"**{n_features} Features:**\n")
                    f.write(f"- Training: scikit-learn {sklearn_time:.2f}ms vs Rust {rust_time:.2f}ms ")
                    f.write(f"({speedup:.2f}x faster for sklearn)\n")
                    f.write(f"- Accuracy (MSE): sklearn {sklearn_mse:.4f} vs Rust {rust_mse:.4f} ")
                    f.write(f"({mse_diff_pct:+.1f}% difference)\n\n")

        f.write("### Key Findings\n\n")

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
        rust_metrics: Dict[int, Dict[str, Any]],
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
        rust_metrics: Dict[int, Dict[str, Any]],
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
