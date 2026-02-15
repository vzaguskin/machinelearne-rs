use std::time::{Duration, Instant};

/// Timer for measuring elapsed time.
#[derive(Debug)]
pub struct Timer {
    start: Option<Instant>,
    total: Duration,
}

impl Timer {
    /// Create a new timer.
    pub fn new() -> Self {
        Self {
            start: None,
            total: Duration::ZERO,
        }
    }

    /// Start the timer.
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Stop the timer and add the elapsed time to the total.
    pub fn stop(&mut self) -> Duration {
        if let Some(start) = self.start.take() {
            let elapsed = start.elapsed();
            self.total += elapsed;
            elapsed
        } else {
            Duration::ZERO
        }
    }

    /// Get the total elapsed time.
    pub fn total(&self) -> Duration {
        self.total
    }

    /// Get the total elapsed time in milliseconds.
    pub fn total_ms(&self) -> f64 {
        self.total.as_secs_f64() * 1000.0
    }

    /// Get the total elapsed time in seconds.
    pub fn total_secs(&self) -> f64 {
        self.total.as_secs_f64()
    }

    /// Reset the timer.
    pub fn reset(&mut self) {
        self.start = None;
        self.total = Duration::ZERO;
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Run a function and measure its execution time.
///
/// # Arguments
///
/// * `f` - Function to execute
///
/// # Returns
///
/// A tuple of (result, elapsed_time)
pub fn time_fn<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    (result, elapsed)
}

/// Run a function multiple times and return the mean and std dev of execution times.
///
/// # Arguments
///
/// * `iterations` - Number of iterations to run
/// * `f` - Function to execute
///
/// # Returns
///
/// A tuple of (results, mean_time_ms, std_dev_ms)
pub fn benchmark_fn<F, R>(iterations: usize, mut f: F) -> (Vec<R>, f64, f64)
where
    F: FnMut() -> R,
{
    let mut results = Vec::with_capacity(iterations);
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let (result, elapsed) = time_fn(&mut f);
        results.push(result);
        times.push(elapsed.as_secs_f64() * 1000.0);
    }

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std_dev = variance.sqrt();

    (results, mean, std_dev)
}

/// Run a function multiple times with warmup and return statistics.
///
/// # Arguments
///
/// * `warmup` - Number of warmup iterations (not counted in results)
/// * `iterations` - Number of measurement iterations
/// * `f` - Function to execute
///
/// # Returns
///
/// A tuple of (results, mean_time_ms, std_dev_ms, min_ms, max_ms)
pub fn benchmark_with_warmup<F, R>(
    warmup: usize,
    iterations: usize,
    mut f: F,
) -> (Vec<R>, f64, f64, f64, f64)
where
    F: FnMut() -> R,
{
    // Warmup
    for _ in 0..warmup {
        let _ = f();
    }

    // Actual benchmarking
    let mut results = Vec::with_capacity(iterations);
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        results.push(result);
        times.push(elapsed);
    }

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std_dev = variance.sqrt();
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    (results, mean, std_dev, min, max)
}

/// Statistics for benchmarking results.
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    pub mean_ms: f64,
    pub std_dev_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

impl BenchmarkStats {
    /// Calculate statistics from a list of times in milliseconds.
    pub fn from_times(mut times: Vec<f64>) -> Self {
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
        let std_dev = variance.sqrt();
        let min = times.first().copied().unwrap_or(0.0);
        let max = times.last().copied().unwrap_or(0.0);

        let n = times.len();
        let median = if n.is_multiple_of(2) {
            (times[n / 2 - 1] + times[n / 2]) / 2.0
        } else {
            times[n / 2]
        };

        let p95_idx = ((n as f64 * 0.95) as usize).min(n - 1);
        let p95 = times[p95_idx];

        let p99_idx = ((n as f64 * 0.99) as usize).min(n - 1);
        let p99 = times[p99_idx];

        Self {
            mean_ms: mean,
            std_dev_ms: std_dev,
            min_ms: min,
            max_ms: max,
            median_ms: median,
            p95_ms: p95,
            p99_ms: p99,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer() {
        let mut timer = Timer::new();
        timer.start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed.as_millis() >= 10);
        assert!(timer.total_ms() >= 10.0);
    }

    #[test]
    fn test_benchmark_fn() {
        let fn_to_test = || {
            let mut sum = 0i32;
            for i in 0..1000 {
                sum += i;
            }
            sum
        };

        let (results, mean, std_dev) = benchmark_fn(10, fn_to_test);
        assert_eq!(results.len(), 10);
        assert!(mean > 0.0);
        assert!(std_dev >= 0.0);
    }

    #[test]
    fn test_benchmark_with_warmup() {
        let fn_to_test = || 42;

        let (results, mean, std_dev, min, max) = benchmark_with_warmup(5, 10, fn_to_test);
        assert_eq!(results.len(), 10);
        assert_eq!(results[0], 42);
        assert!(mean >= 0.0);
        assert!(std_dev >= 0.0);
        assert!(max >= min);
    }

    #[test]
    fn test_benchmark_stats() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = BenchmarkStats::from_times(times);

        assert!((stats.mean_ms - 3.0).abs() < 1e-6);
        assert!((stats.median_ms - 3.0).abs() < 1e-6);
        assert!((stats.min_ms - 1.0).abs() < 1e-6);
        assert!((stats.max_ms - 5.0).abs() < 1e-6);
    }
}
