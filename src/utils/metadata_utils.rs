//! Port of `smartloop/utils/metadata_utils.py`.
//!
//! Utility functions for handling model training metadata and dataset
//! operations.  The metadata file is `<models_dir>/<safe_model_id>_training_metadata.json`,
//! a JSON object keyed by dataset hash (indent=2, same as Python).

use serde_json::{Map, Value};

/// Calculate an MD5 hex digest of the dataset file to uniquely identify it.
/// Returns `None` (after printing a warning) when the file can't be read.
pub fn calculate_dataset_hash(dataset_path: &str) -> Option<String> {
    let _ = dataset_path;
    todo!()
}

/// Path to the training metadata file for `model_id` (slashes and
/// backslashes replaced with `_`).
pub fn get_training_metadata_path(models_dir: &str, model_id: &str) -> String {
    let _ = (models_dir, model_id);
    todo!()
}

/// Load training metadata if it exists; an empty map otherwise.
pub fn load_training_metadata(models_dir: &str, model_id: &str) -> Map<String, Value> {
    let _ = (models_dir, model_id);
    todo!()
}

/// Save training metadata (dataset hash, merged model dir, training date,
/// model id) for future reference.  Errors are printed, never returned.
pub fn save_training_metadata(
    models_dir: &str,
    model_id: &str,
    dataset_hash: &str,
    merged_model_dir: &str,
) {
    let _ = (models_dir, model_id, dataset_hash, merged_model_dir);
    todo!()
}

/// Check whether the model has already been trained for this dataset;
/// returns the merged model directory when it still exists.  A stale entry
/// whose directory is missing is removed from the metadata file.
pub fn check_existing_training(
    models_dir: &str,
    model_id: &str,
    dataset_hash: Option<&str>,
) -> Option<String> {
    let _ = (models_dir, model_id, dataset_hash);
    todo!()
}
