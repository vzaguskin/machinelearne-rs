#!/usr/bin/env python3
"""
Fair comparison benchmark for sklearn vs machinelearne-rs.

This script runs sklearn with EXACT configurations to match the Rust benchmark.
Key principle: Compare identical workloads, not different defaults.

Test configurations:
1. Full-batch, 5 epochs (match sklearn's early stopping behavior)
2. Full-batch, 50 epochs (match Rust's default)
3. Mini-batch simulation (process same number of samples)
4. Convergence test (stop at same loss threshold)
"""

import csv
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FairBenchmark:
    """Fair benchmark suite comparing sklearn configurations."""

    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.results_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)

    def load_data(self, n_features: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load California Housing dataset with feature selection."""
        print(f"Loading California Housing dataset ({n_features} features)...")
        data = fetch_california_housing()
        X, y = data.data[:, :n_features], data.target

        # Same split as Rust (80/20, no shuffle for reproducibility)
        n_total = len(X)
        train_size = int(n_total * 0.8)

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Standardize features (same as Rust)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def benchmark_closed_form(self, n_features: int, n_runs: int = 10) -> Dict[str, Any]:
        """Benchmark sklearn's closed-form LinearRegression (OLS solution)."""
        X_train, X_test, y_train, y_test = self.load_data(n_features)

        print(f"\nBenchmarking sklearn LinearRegression (closed-form, {n_features} features)...")

        train_times = []
        for i in range(n_runs):
            model = LinearRegression(fit_intercept=True)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            train_times.append(elapsed)

        # Metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        result = {
            "test": "closed_form",
            "implementation": "sklearn",
            "n_features": n_features,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "train_time_mean_ms": np.mean(train_times) * 1000,
            "train_time_std_ms": np.std(train_times) * 1000,
            "train_time_min_ms": np.min(train_times) * 1000,
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2),
            "weights": model.coef_.tolist(),
            "bias": float(model.intercept_),
            "n_runs": n_runs,
        }

        print(f"  Time: {result['train_time_mean_ms']:.3f} +/- {result['train_time_std_ms']:.3f} ms")
        print(f"  MSE: {mse:.6f}, R2: {r2:.6f}")
        print(f"  Weights: {model.coef_}, Bias: {model.intercept_:.6f}")

        return result

    def benchmark_sgd_exact_epochs(
        self,
        n_features: int,
        n_epochs: int,
        batch_size: int,
        learning_rate: float = 0.01,
        n_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark sklearn SGDRegressor with EXACT epoch count.

        Key: Disable early stopping to run exact number of epochs.
        """
        X_train, X_test, y_train, y_test = self.load_data(n_features)

        config_name = f"sgd_epochs_{n_epochs}_batch_{batch_size}"
        print(f"\nBenchmarking sklearn {config_name} ({n_features} features)...")

        # Calculate iterations based on batch size
        # sklearn's max_iter = number of epochs, but it processes all samples per epoch
        # For fair comparison with mini-batch, we need to calculate equivalent iterations

        n_samples = len(X_train)

        train_times = []
        final_models = []

        for run in range(n_runs):
            # Disable early stopping by setting tol=None
            model = SGDRegressor(
                max_iter=n_epochs,
                tol=None,  # Disable early stopping - run exact epochs
                learning_rate='constant',
                eta0=learning_rate,
                fit_intercept=True,
                random_state=42 + run,
                verbose=0,
            )

            start = time.perf_counter()
            model.fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            train_times.append(elapsed)
            final_models.append(model)

        # Use last model for metrics
        model = final_models[-1]
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Calculate actual iterations ran
        n_iter = model.n_iter_

        # Calculate samples processed
        samples_processed = n_epochs * n_samples

        result = {
            "test": config_name,
            "implementation": "sklearn",
            "n_features": n_features,
            "n_epochs": n_epochs,
            "actual_iterations": n_iter,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "n_train_samples": n_samples,
            "n_test_samples": len(X_test),
            "samples_processed": samples_processed,
            "train_time_mean_ms": np.mean(train_times) * 1000,
            "train_time_std_ms": np.std(train_times) * 1000,
            "train_time_min_ms": np.min(train_times) * 1000,
            "time_per_epoch_ms": np.mean(train_times) * 1000 / n_epochs,
            "time_per_sample_us": np.mean(train_times) * 1e6 / samples_processed,
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2),
            "weights": model.coef_.tolist(),
            "bias": float(model.intercept_),
            "n_runs": n_runs,
        }

        print(f"  Time: {result['train_time_mean_ms']:.3f} +/- {result['train_time_std_ms']:.3f} ms")
        print(f"  Time per epoch: {result['time_per_epoch_ms']:.3f} ms")
        print(f"  Time per sample: {result['time_per_sample_us']:.4f} us")
        print(f"  MSE: {mse:.6f}, R2: {r2:.6f}")

        return result

    def benchmark_sgd_with_early_stopping(
        self,
        n_features: int,
        max_epochs: int = 1000,
        tolerance: float = 1e-3,
        learning_rate: float = 0.01,
        n_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark sklearn SGDRegressor with early stopping enabled.

        This matches sklearn's default behavior.
        """
        X_train, X_test, y_train, y_test = self.load_data(n_features)

        config_name = f"sgd_early_stop_tol_{tolerance}"
        print(f"\nBenchmarking sklearn {config_name} ({n_features} features)...")

        n_samples = len(X_train)

        train_times = []
        iterations_ran = []
        final_models = []

        for run in range(n_runs):
            model = SGDRegressor(
                max_iter=max_epochs,
                tol=tolerance,
                learning_rate='constant',
                eta0=learning_rate,
                fit_intercept=True,
                random_state=42 + run,
                verbose=0,
            )

            start = time.perf_counter()
            model.fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            train_times.append(elapsed)
            iterations_ran.append(model.n_iter_)
            final_models.append(model)

        # Use last model for metrics
        model = final_models[-1]
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mean_iterations = np.mean(iterations_ran)

        result = {
            "test": config_name,
            "implementation": "sklearn",
            "n_features": n_features,
            "max_epochs": max_epochs,
            "actual_iterations_mean": mean_iterations,
            "actual_iterations_std": np.std(iterations_ran),
            "tolerance": tolerance,
            "learning_rate": learning_rate,
            "n_train_samples": n_samples,
            "n_test_samples": len(X_test),
            "train_time_mean_ms": np.mean(train_times) * 1000,
            "train_time_std_ms": np.std(train_times) * 1000,
            "train_time_min_ms": np.min(train_times) * 1000,
            "time_per_iteration_ms": np.mean(train_times) * 1000 / mean_iterations,
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2),
            "weights": model.coef_.tolist(),
            "bias": float(model.intercept_),
            "n_runs": n_runs,
        }

        print(f"  Converged in: {mean_iterations:.1f} iterations")
        print(f"  Time: {result['train_time_mean_ms']:.3f} +/- {result['train_time_std_ms']:.3f} ms")
        print(f"  MSE: {mse:.6f}, R2: {r2:.6f}")

        return result

    def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """Run all fair comparison benchmarks."""
        print("=" * 70)
        print("SKLEARN FAIR COMPARISON BENCHMARKS")
        print("=" * 70)

        results = []

        # Test with 2 features (primary comparison)
        n_features = 2

        # 1. Closed-form solution (baseline)
        results.append(self.benchmark_closed_form(n_features, n_runs=10))

        # 2. SGD with exact 5 epochs, full batch, LR=0.5 (optimal for full-batch)
        results.append(self.benchmark_sgd_exact_epochs(
            n_features, n_epochs=5, batch_size=16512, learning_rate=0.5, n_runs=10
        ))

        # 3. SGD with exact 50 epochs, full batch, LR=0.5
        results.append(self.benchmark_sgd_exact_epochs(
            n_features, n_epochs=50, batch_size=16512, learning_rate=0.5, n_runs=5
        ))

        # 4. SGD with LR=0.1 for comparison (sklearn-stable)
        results.append(self.benchmark_sgd_exact_epochs(
            n_features, n_epochs=5, batch_size=16512, learning_rate=0.1, n_runs=10
        ))

        # 5. SGD with LR=0.01 for comparison (baseline)
        results.append(self.benchmark_sgd_exact_epochs(
            n_features, n_epochs=5, batch_size=16512, learning_rate=0.01, n_runs=10
        ))

        # 5. SGD with early stopping (sklearn default behavior)
        results.append(self.benchmark_sgd_with_early_stopping(
            n_features, max_epochs=1000, tolerance=1e-3, learning_rate=0.01, n_runs=10
        ))

        # 6. Test with 8 features
        n_features = 8
        results.append(self.benchmark_closed_form(n_features, n_runs=10))
        results.append(self.benchmark_sgd_exact_epochs(
            n_features, n_epochs=5, batch_size=16512, learning_rate=0.1, n_runs=10
        ))
        results.append(self.benchmark_sgd_exact_epochs(
            n_features, n_epochs=50, batch_size=16512, learning_rate=0.1, n_runs=5
        ))

        # Save results
        self.save_results(results)

        return results

    def save_results(self, results: List[Dict[str, Any]], filename: str = "sklearn_fair_comparison.json"):
        """Save benchmark results to JSON."""
        filepath = self.raw_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved {len(results)} results to {filepath}")


def main():
    benchmark = FairBenchmark()
    results = benchmark.run_all_benchmarks()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    # Print summary table
    print("\nSummary:")
    print(f"{'Test':<40} {'Time (ms)':<15} {'MSE':<12} {'R2':<10}")
    print("-" * 77)
    for r in results:
        test = r.get("test", "unknown")[:38]
        time_ms = r.get("train_time_mean_ms", 0)
        mse = r.get("mse", 0)
        r2 = r.get("r2", 0)
        print(f"{test:<40} {time_ms:<15.3f} {mse:<12.6f} {r2:<10.4f}")


if __name__ == "__main__":
    main()
