//! Dataset loading utilities
//!
//! Loads California Housing dataset from CSV file.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Load California Housing dataset from CSV.
///
/// The dataset is expected to be at `benchmarks/datasets/california_housing.csv`.
///
/// # Format
/// - 8 features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
/// - 1 target: MedHouseVal (median house value in $100k)
/// - No header row expected (if there is, it will fail to parse as numbers)
///
/// # Returns
/// Tuple of (features, targets) where:
/// - features: Vec<Vec<f32>> with shape (n_samples, 8)
/// - targets: Vec<f32> with shape (n_samples,)
pub fn load_california_housing() -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn std::error::Error>> {
    let path = Path::new("benchmarks/datasets/california_housing.csv");

    load_california_housing_from_path(path)
}

/// Load California Housing dataset from a specific path.
pub fn load_california_housing_from_path(
    path: &Path,
) -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut features = Vec::new();
    let mut targets = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Skip header if present (starts with non-numeric character)
        let first_char = line.chars().next();
        if let Some(c) = first_char {
            if !c.is_ascii_digit() && c != '-' && c != '+' && c != '.' {
                continue;
            }
        }

        // Parse values
        let values: Vec<f32> = line
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if values.len() >= 9 {
            // Columns 0-7 are features, column 8 is target
            targets.push(values[8]);
            features.push(values[0..8].to_vec());
        } else if line_num > 0 {
            // Only warn for non-header lines
            eprintln!(
                "Warning: Line {} has {} values, expected 9",
                line_num + 1,
                values.len()
            );
        }
    }

    if features.is_empty() {
        return Err(format!("No valid data found in {:?}", path).into());
    }

    Ok((features, targets))
}

/// Get feature names for California Housing dataset.
pub fn california_housing_feature_names() -> &'static [&'static str] {
    &[
        "MedInc",     // Median income in block group
        "HouseAge",   // Median house age in block group
        "AveRooms",   // Average number of rooms per household
        "AveBedrms",  // Average number of bedrooms per household
        "Population", // Block group population
        "AveOccup",   // Average number of household members
        "Latitude",   // Block group latitude
        "Longitude",  // Block group longitude
    ]
}

/// Main function for running as a standalone example.
fn main() {
    match load_california_housing() {
        Ok((features, targets)) => {
            println!("Loaded California Housing dataset:");
            println!("  Samples: {}", features.len());
            println!("  Features: {}", features[0].len());
            println!(
                "  Target range: [{:.2}, {:.2}]",
                targets.iter().cloned().fold(f32::INFINITY, f32::min),
                targets.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
        Err(e) => {
            eprintln!("Failed to load dataset: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_california_housing() {
        // This test will only pass if the dataset exists
        if let Ok((features, targets)) = load_california_housing() {
            assert_eq!(features.len(), targets.len());
            assert!(!features.is_empty());

            // Each feature row should have 8 features
            for (i, row) in features.iter().enumerate() {
                assert_eq!(
                    row.len(),
                    8,
                    "Row {} has {} features, expected 8",
                    i,
                    row.len()
                );
            }

            // Should be around 20640 samples
            assert!(
                features.len() > 20000,
                "Expected ~20640 samples, got {}",
                features.len()
            );
        }
    }

    #[test]
    fn test_feature_names() {
        let names = california_housing_feature_names();
        assert_eq!(names.len(), 8);
    }
}
