use csv::ReaderBuilder;
use machinelearne_rs::dataset::InMemoryDataset;
use std::fs::File;
use std::io::{self, BufReader};

/// California Housing dataset loader.
///
/// Loads the California Housing dataset from CSV and provides methods
/// to select subsets of features and split into train/validation/test sets.
///
/// The dataset contains 20640 samples with 8 features:
/// - MedInc: Median income in block group
/// - HouseAge: Median house age in block group
/// - AveRooms: Average number of rooms per household
/// - AveBedrms: Average number of bedrooms per household
/// - Population: Block group population
/// - AveOccup: Average number of household members
/// - Latitude: Block group latitude
/// - Longitude: Block group longitude
///
/// Target variable: MedHouseVal (Median house value for California districts)
#[derive(Debug, Clone)]
pub struct CaliforniaHousingDataset {
    features: Vec<Vec<f32>>,
    target: Vec<f32>,
    feature_names: Vec<&'static str>,
}

impl CaliforniaHousingDataset {
    /// Load the California Housing dataset from CSV.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the CSV file (default: "benchmarks/datasets/california_housing.csv")
    ///
    /// # Example
    ///
    /// ```no_run
    /// use benchmarks::data::CaliforniaHousingDataset;
    ///
    /// let dataset = CaliforniaHousingDataset::load(
    ///     "benchmarks/datasets/california_housing.csv"
    /// ).unwrap();
    /// ```
    pub fn load(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut rdr = ReaderBuilder::new().from_reader(reader);

        let mut features = Vec::new();
        let mut target = Vec::new();

        for result in rdr.records() {
            let record = result?;
            // Columns: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, MedHouseVal
            let med_inc: f32 = record[0].parse().unwrap_or(0.0);
            let house_age: f32 = record[1].parse().unwrap_or(0.0);
            let ave_rooms: f32 = record[2].parse().unwrap_or(0.0);
            let ave_bedrms: f32 = record[3].parse().unwrap_or(0.0);
            let population: f32 = record[4].parse().unwrap_or(0.0);
            let ave_occup: f32 = record[5].parse().unwrap_or(0.0);
            let latitude: f32 = record[6].parse().unwrap_or(0.0);
            let longitude: f32 = record[7].parse().unwrap_or(0.0);
            let med_house_val: f32 = record[8].parse().unwrap_or(0.0);

            features.push(vec![
                med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude,
                longitude,
            ]);
            target.push(med_house_val);
        }

        Ok(Self {
            features,
            target,
            feature_names: vec![
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
            ],
        })
    }

    /// Get the number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Get the number of features.
    pub fn n_features(&self) -> usize {
        self.features.first().map(|f| f.len()).unwrap_or(0)
    }

    /// Get feature names.
    pub fn feature_names(&self) -> &[&'static str] {
        &self.feature_names
    }

    /// Select a subset of features.
    ///
    /// # Arguments
    ///
    /// * `feature_indices` - Indices of features to select (0-indexed)
    ///
    /// # Example
    ///
    /// ```no_run
    /// let dataset = CaliforniaHousingDataset::load(...).unwrap();
    /// // Select only MedInc and HouseAge (first 2 features)
    /// let subset = dataset.select_features(&[0, 1]);
    /// ```
    pub fn select_features(&self, feature_indices: &[usize]) -> Self {
        let features: Vec<Vec<f32>> = self
            .features
            .iter()
            .map(|row| {
                feature_indices
                    .iter()
                    .map(|&idx| row.get(idx).copied().unwrap_or(0.0))
                    .collect()
            })
            .collect();
        let feature_names: Vec<&'static str> = feature_indices
            .iter()
            .map(|&idx| self.feature_names.get(idx).copied().unwrap_or("Unknown"))
            .collect();

        Self {
            features,
            target: self.target.clone(),
            feature_names,
        }
    }

    /// Split the dataset into train and test sets.
    ///
    /// # Arguments
    ///
    /// * `train_ratio` - Fraction of data to use for training (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// (train_dataset, test_dataset)
    ///
    /// # Example
    ///
    /// ```no_run
    /// let dataset = CaliforniaHousingDataset::load(...).unwrap();
    /// let (train, test) = dataset.split(0.8); // 80% train, 20% test
    /// ```
    pub fn split(&self, train_ratio: f32) -> (Self, Self) {
        let n_samples = self.len();
        let n_train = (n_samples as f32 * train_ratio) as usize;

        let train_features = self.features[..n_train].to_vec();
        let train_target = self.target[..n_train].to_vec();

        let test_features = self.features[n_train..].to_vec();
        let test_target = self.target[n_train..].to_vec();

        (
            Self {
                features: train_features,
                target: train_target,
                feature_names: self.feature_names.clone(),
            },
            Self {
                features: test_features,
                target: test_target,
                feature_names: self.feature_names.clone(),
            },
        )
    }

    /// Split the dataset into train, validation, and test sets.
    ///
    /// # Arguments
    ///
    /// * `train_ratio` - Fraction of data to use for training
    /// * `val_ratio` - Fraction of data to use for validation
    ///
    /// # Returns
    ///
    /// (train_dataset, val_dataset, test_dataset)
    pub fn split_train_val_test(&self, train_ratio: f32, val_ratio: f32) -> (Self, Self, Self) {
        let n_samples = self.len();
        let n_train = (n_samples as f32 * train_ratio) as usize;
        let n_val = (n_samples as f32 * val_ratio) as usize;

        let train_features = self.features[..n_train].to_vec();
        let train_target = self.target[..n_train].to_vec();

        let val_features = self.features[n_train..n_train + n_val].to_vec();
        let val_target = self.target[n_train..n_train + n_val].to_vec();

        let test_features = self.features[n_train + n_val..].to_vec();
        let test_target = self.target[n_train + n_val..].to_vec();

        (
            Self {
                features: train_features,
                target: train_target,
                feature_names: self.feature_names.clone(),
            },
            Self {
                features: val_features,
                target: val_target,
                feature_names: self.feature_names.clone(),
            },
            Self {
                features: test_features,
                target: test_target,
                feature_names: self.feature_names.clone(),
            },
        )
    }

    /// Convert to an InMemoryDataset for use with machinelearne-rs.
    pub fn to_in_memory_dataset(&self) -> Result<InMemoryDataset, String> {
        InMemoryDataset::new(self.features.clone(), self.target.clone())
    }

    /// Get a reference to the features.
    pub fn features(&self) -> &[Vec<f32>] {
        &self.features
    }

    /// Get a reference to the target values.
    pub fn target(&self) -> &[f32] {
        &self.target
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_dataset() {
        let dataset = CaliforniaHousingDataset::load("benchmarks/datasets/california_housing.csv")
            .expect("Failed to load dataset");
        assert_eq!(dataset.len(), 20640);
        assert_eq!(dataset.n_features(), 8);
        assert_eq!(dataset.feature_names().len(), 8);
    }

    #[test]
    fn test_select_features() {
        let dataset = CaliforniaHousingDataset::load("benchmarks/datasets/california_housing.csv")
            .expect("Failed to load dataset");
        let subset = dataset.select_features(&[0, 1]); // MedInc and HouseAge
        assert_eq!(subset.n_features(), 2);
        assert_eq!(subset.feature_names(), &["MedInc", "HouseAge"]);
    }

    #[test]
    fn test_split() {
        let dataset = CaliforniaHousingDataset::load("benchmarks/datasets/california_housing.csv")
            .expect("Failed to load dataset");
        let (train, test) = dataset.split(0.8);
        assert_eq!(train.len(), 16512);
        assert_eq!(test.len(), 4128);
    }

    #[test]
    fn test_to_in_memory_dataset() {
        let dataset = CaliforniaHousingDataset::load("benchmarks/datasets/california_housing.csv")
            .expect("Failed to load dataset");
        let im_dataset = dataset.to_in_memory_dataset().expect("Failed to convert");
        assert_eq!(im_dataset.len(), Some(20640));
    }
}
