// Benchmark utilities and common modules
// This is the main entry point for running benchmarks

fn main() {
    println!("Machinelearne-rs Benchmark Suite");
    println!();
    println!("Usage:");
    println!("  cargo bench --package benchmarks --all-benchmarks");
    println!("  cargo bench --package benchmarks --bench <benchmark_name>");
    println!();
    println!("Available benchmarks:");
    println!("  - train_1_feature: Training benchmarks with 1 feature");
    println!("  - train_2_features: Training benchmarks with 2 features");
    println!("  - train_4_features: Training benchmarks with 4 features");
    println!("  - train_8_features: Training benchmarks with 8 features");
    println!("  - predict: Prediction latency and throughput benchmarks");
    println!("  - metrics: Metrics computation benchmarks");
    println!();
    println!("To run all benchmarks:");
    println!("  bash benchmarks/scripts/run_all.sh");
}
