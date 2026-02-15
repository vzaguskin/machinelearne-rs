#!/usr/bin/env python3
"""
Comprehensive sklearn benchmarking for comparison with machinelearne-rs.

This script benchmarks various sklearn models on the California Housing dataset
and collects metrics for training time, prediction time, and accuracy.
"""

import csv
import time
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    SGDRegressor,
    LogisticRegression,
    RANSACRegressor,
    HuberRegressor,
    TheilSenRegressor,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SklearnBenchmark:
    """Benchmark suite for sklearn models."""

    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.results_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load California Housing dataset and split into train/test."""
        if self.data is None:
            print("Loading California Housing dataset...")
            data = fetch_california_housing()
            X, y = data.data, data.target
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.data = (X, y)
            print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def select_features(
        self, X: np.ndarray, feature_indices: List[int]
    ) -> np.ndarray:
        """Select a subset of features."""
        return X[:, feature_indices]

    def benchmark_model(
        self,
        model,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_runs: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Benchmark a single model.

        Returns:
            Dictionary with training_time, prediction_time, mse, mae, r2
        """
        print(f"Benchmarking {model_name}...")

        # Training time measurement
        train_times = []
        for _ in range(n_runs):
            model_copy = model.__class__(**kwargs)
            start = time.perf_counter()
            model_copy.fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            train_times.append(elapsed)

        # Use the last trained model for predictions
        fitted_model = model_copy

        # Prediction time measurement
        pred_times = []
        predictions_list = []
        for _ in range(n_runs):
            start = time.perf_counter()
            predictions = fitted_model.predict(X_test)
            elapsed = time.perf_counter() - start
            pred_times.append(elapsed)
            predictions_list.append(predictions)

        # Use the last predictions for metrics
        y_pred = predictions_list[-1]

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Calculate statistics
        train_mean = np.mean(train_times) * 1000  # Convert to ms
        train_std = np.std(train_times) * 1000
        pred_mean = np.mean(pred_times) * 1000
        pred_std = np.std(pred_times) * 1000

        result = {
            "model": model_name,
            "train_time_mean_ms": train_mean,
            "train_time_std_ms": train_std,
            "pred_time_mean_ms": pred_mean,
            "pred_time_std_ms": pred_std,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "n_runs": n_runs,
        }

        print(f"  Train time: {train_mean:.2f} ± {train_std:.2f} ms")
        print(f"  Pred time:  {pred_mean:.2f} ± {pred_std:.2f} ms")
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")

        return result

    def benchmark_linear_regression(
        self,
        feature_indices: List[int],
        n_runs: int = 10,
    ) -> Dict[str, Any]:
        """Benchmark sklearn LinearRegression."""
        X_train, X_test, y_train, y_test = self.load_data()
        X_train_subset = self.select_features(X_train, feature_indices)
        X_test_subset = self.select_features(X_test, feature_indices)

        model = LinearRegression()
        feature_str = "_".join(map(str, feature_indices))
        model_name = f"LinearRegression_features_{feature_str}"

        return self.benchmark_model(
            model,
            model_name,
            X_train_subset,
            y_train,
            X_test_subset,
            y_test,
            n_runs=n_runs,
            fit_intercept=True,
        )

    def benchmark_ridge(
        self,
        feature_indices: List[int],
        alphas: List[float] = [0.1, 1.0, 10.0],
        n_runs: int = 10,
    ) -> List[Dict[str, Any]]:
        """Benchmark sklearn Ridge with different alphas."""
        X_train, X_test, y_train, y_test = self.load_data()
        X_train_subset = self.select_features(X_train, feature_indices)
        X_test_subset = self.select_features(X_test, feature_indices)

        results = []
        for alpha in alphas:
            model = Ridge()
            feature_str = "_".join(map(str, feature_indices))
            model_name = f"Ridge_alpha_{alpha}_features_{feature_str}"

            result = self.benchmark_model(
                model,
                model_name,
                X_train_subset,
                y_train,
                X_test_subset,
                y_test,
                n_runs=n_runs,
                alpha=alpha,
                fit_intercept=True,
            )
            results.append(result)

        return results

    def benchmark_sgd_regressor(
        self,
        feature_indices: List[int],
        learning_rates: List[str] = ["constant", "invscaling", "adaptive"],
        n_runs: int = 10,
    ) -> List[Dict[str, Any]]:
        """Benchmark sklearn SGDRegressor with different learning rates."""
        X_train, X_test, y_train, y_test = self.load_data()
        X_train_subset = self.select_features(X_train, feature_indices)
        X_test_subset = self.select_features(X_test, feature_indices)

        # Scale data for SGD
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_test_scaled = scaler.transform(X_test_subset)

        results = []
        for lr in learning_rates:
            model = SGDRegressor(
                max_iter=1000,
                tol=1e-3,
                random_state=42,
            )
            feature_str = "_".join(map(str, feature_indices))
            model_name = f"SGDRegressor_lr_{lr}_features_{feature_str}"

            result = self.benchmark_model(
                model,
                model_name,
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test,
                n_runs=n_runs,
                learning_rate=lr,
                eta0=0.01,
            )
            results.append(result)

        return results

    def benchmark_ransac(
        self,
        feature_indices: List[int],
        n_runs: int = 5,  # RANSAC is slower
    ) -> Dict[str, Any]:
        """Benchmark sklearn RANSACRegressor."""
        X_train, X_test, y_train, y_test = self.load_data()
        X_train_subset = self.select_features(X_train, feature_indices)
        X_test_subset = self.select_features(X_test, feature_indices)

        model = RANSACRegressor(random_state=42)
        feature_str = "_".join(map(str, feature_indices))
        model_name = f"RANSACRegressor_features_{feature_str}"

        return self.benchmark_model(
            model,
            model_name,
            X_train_subset,
            y_train,
            X_test_subset,
            y_test,
            n_runs=n_runs,
        )

    def run_all_benchmarks(self):
        """Run all benchmarks and save results."""
        print("=" * 60)
        print("Running sklearn benchmarks")
        print("=" * 60)

        all_results = []

        # Linear Regression with different feature counts
        print("\n--- Linear Regression ---")
        for n_features in [1, 2, 4, 8]:
            feature_indices = list(range(n_features))
            result = self.benchmark_linear_regression(feature_indices, n_runs=10)
            all_results.append(result)

        # Ridge Regression with different alphas and features
        print("\n--- Ridge Regression ---")
        for n_features in [2, 4, 8]:
            feature_indices = list(range(n_features))
            results = self.benchmark_ridge(
                feature_indices, alphas=[0.1, 1.0, 10.0], n_runs=10
            )
            all_results.extend(results)

        # SGD Regressor
        print("\n--- SGD Regressor ---")
        for n_features in [2, 4, 8]:
            feature_indices = list(range(n_features))
            results = self.benchmark_sgd_regressor(
                feature_indices, learning_rates=["constant", "adaptive"], n_runs=10
            )
            all_results.extend(results)

        # RANSAC (only with 2 features for speed)
        print("\n--- RANSAC Regressor ---")
        feature_indices = [0, 1]
        result = self.benchmark_ransac(feature_indices, n_runs=5)
        all_results.append(result)

        # Save results to CSV
        self.save_results(all_results)

        return all_results

    def save_results(self, results: List[Dict[str, Any]], filename: str = "sklearn_results.csv"):
        """Save benchmark results to CSV."""
        filepath = self.raw_dir / filename

        if not results:
            print("No results to save!")
            return

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved {len(results)} results to {filepath}")


def main():
    """Main entry point."""
    benchmark = SklearnBenchmark()
    results = benchmark.run_all_benchmarks()

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    # Print summary
    print("\nSummary:")
    print(f"Total benchmarks run: {len(results)}")
    for result in results:
        print(
            f"  {result['model']}: "
            f"train={result['train_time_mean_ms']:.2f}ms, "
            f"pred={result['pred_time_mean_ms']:.2f}ms, "
            f"R²={result['r2']:.4f}"
        )


if __name__ == "__main__":
    main()
