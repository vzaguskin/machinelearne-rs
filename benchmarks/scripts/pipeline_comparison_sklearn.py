"""
sklearn Pipeline Comparison Script

Runs three configurations with holdout test set:
1. Naive baseline: LinearRegression on raw data
2. Default baseline: StandardScaler + LinearRegression
3. Optimized: GridSearchCV over scalers + L2 + polynomial features

After finding best params via CV, retrains on full train and evaluates on test.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time
import json
from pathlib import Path


def run_comparison():
    """Run sklearn pipeline comparison with holdout test set."""
    # Load data
    X, y = fetch_california_housing(return_X_y=True)

    # HOLDOUT TEST SET: 80/20 split (no shuffle for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    print(f"Dataset split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    results = {'train_size': X_train.shape[0], 'test_size': X_test.shape[0]}

    # 1. Naive baseline (raw data, no preprocessing) - evaluate on TEST
    print("\n1. Naive Baseline...")
    start = time.perf_counter()
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    naive_time = time.perf_counter() - start

    results['naive_baseline'] = {
        'test_r2': float(r2_score(y_test, y_pred)),
        'test_mse': float(mean_squared_error(y_test, y_pred)),
        'test_mae': float(mean_absolute_error(y_test, y_pred)),
        'test_rmse': float(mean_squared_error(y_test, y_pred, squared=False)),
        'train_time_ms': naive_time * 1000
    }

    # 2. Default baseline (StandardScaler + LinearRegression) - evaluate on TEST
    print("2. Default Baseline (StandardScaler)...")
    start = time.perf_counter()
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    default_time = time.perf_counter() - start

    results['default_baseline'] = {
        'test_r2': float(r2_score(y_test, y_pred)),
        'test_mse': float(mean_squared_error(y_test, y_pred)),
        'test_mae': float(mean_absolute_error(y_test, y_pred)),
        'test_rmse': float(mean_squared_error(y_test, y_pred, squared=False)),
        'train_time_ms': default_time * 1000,
        'config': {'scaler': 'StandardScaler', 'model': 'LinearRegression'}
    }

    # 3. Optimized pipeline (GridSearchCV over scalers + L2 + polynomial)
    print("3. Optimized Pipeline (GridSearchCV)...")
    print("   Search space: 4 scalers x 5 L2 values x 2 poly degrees = 40 combinations")

    pipeline = Pipeline([
        ('poly', 'passthrough'),  # Placeholder for PolynomialFeatures
        ('scaler', 'passthrough'),
        ('model', Ridge())
    ])

    param_grid = {
        'poly': ['passthrough', PolynomialFeatures(degree=2, include_bias=False)],
        'scaler': ['passthrough', StandardScaler(), MinMaxScaler(), RobustScaler()],
        'model__alpha': [0.0, 0.001, 0.01, 0.1, 1.0],  # L2 regularization
    }

    cv = KFold(n_splits=5, shuffle=False)

    start = time.perf_counter()
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    search.fit(X_train, y_train)  # CV on TRAIN only
    cv_time = time.perf_counter() - start

    # Get best params from CV
    best_params = search.best_params_
    print(f"   Best CV R2: {search.best_score_:.4f}")
    scaler_name = type(best_params['scaler']).__name__ if best_params['scaler'] != 'passthrough' else 'None'
    poly_str = 'degree 2' if best_params['poly'] != 'passthrough' else 'none'
    print(f"   Best params: scaler={scaler_name}, alpha={best_params['model__alpha']}, poly={poly_str}")

    # RETRAIN on full training data with best params
    start = time.perf_counter()
    best_pipeline = search.best_estimator_
    # Already fitted on full train by GridSearchCV's refit=True (default)
    retrain_time = time.perf_counter() - start

    # Save pipeline
    Path("saved_models").mkdir(exist_ok=True)
    model_path = Path("saved_models/sklearn_best_pipeline.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"   Saved pipeline to {model_path}")

    # Load and verify
    loaded_pipeline = joblib.load(model_path)
    y_pred_loaded = loaded_pipeline.predict(X_test)

    # Evaluate on TEST set
    y_pred = best_pipeline.predict(X_test)

    results['optimized'] = {
        'test_r2': float(r2_score(y_test, y_pred)),
        'test_mse': float(mean_squared_error(y_test, y_pred)),
        'test_mae': float(mean_absolute_error(y_test, y_pred)),
        'test_rmse': float(mean_squared_error(y_test, y_pred, squared=False)),
        'cv_time_ms': cv_time * 1000,
        'retrain_time_ms': retrain_time * 1000,
        'best_cv_score': float(search.best_score_),
        'best_params': {
            'scaler': scaler_name,
            'l2_alpha': best_params['model__alpha'],
            'polynomial': poly_str
        },
        'n_combinations': len(search.cv_results_['mean_test_score']),
        'load_verify_match': bool((y_pred == y_pred_loaded).all())
    }

    # Print summary
    print("\n" + "="*60)
    print("sklearn RESULTS (evaluated on holdout test set)")
    print("="*60)
    print(f"{'Method':<25} {'R2':>8} {'RMSE':>10} {'MAE':>8}")
    print("-"*60)
    for name in ['naive_baseline', 'default_baseline', 'optimized']:
        r = results[name]
        print(f"{name:<25} {r['test_r2']:>8.4f} {r['test_rmse']:>10.4f} {r['test_mae']:>8.4f}")

    # Save results
    Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
    with open('benchmarks/results/pipeline_comparison_sklearn.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_comparison()
