//! Port of `smartloop/utils/metadata_utils.py`.
//!
//! Utility functions for handling model training metadata and dataset
//! operations.  The metadata file is `<models_dir>/<safe_model_id>_training_metadata.json`,
//! a JSON object keyed by dataset hash (indent=2, same as Python).

use std::io::Read;
use std::path::Path;

use console::style;
use md5::{Digest, Md5};
use serde_json::{Map, Value};

use crate::utils::py_compat::{py_json_dumps_indent, py_str};

/// `[bold yellow][13][/bold yellow] <msg>`
fn print_warning(msg: &str) {
    println!("{} {msg}", style("[13]").yellow().bold());
}

/// `[bold green][1][/bold green] <msg>`
fn print_success(msg: &str) {
    println!("{} {msg}", style("[1]").green().bold());
}

/// `[bold cyan][14][/bold cyan] <msg>`
fn print_detail(msg: &str) {
    println!("{} {msg}", style("[14]").cyan().bold());
}

fn hash_file(dataset_path: &str) -> std::io::Result<String> {
    let mut hasher = Md5::new();
    let mut f = std::fs::File::open(dataset_path)?;
    // Read file in chunks to handle large datasets
    let mut buf = [0u8; 4096];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Calculate an MD5 hex digest of the dataset file to uniquely identify it.
/// Returns `None` (after printing a warning) when the file can't be read.
pub fn calculate_dataset_hash(dataset_path: &str) -> Option<String> {
    match hash_file(dataset_path) {
        Ok(digest) => Some(digest),
        Err(e) => {
            print_warning(&format!("Could not calculate dataset hash: {e}"));
            None
        }
    }
}

/// Path to the training metadata file for `model_id` (slashes and
/// backslashes replaced with `_`).
pub fn get_training_metadata_path(models_dir: &str, model_id: &str) -> String {
    let safe_model_id = model_id.replace('/', "_").replace('\\', "_");
    let metadata_file = format!("{safe_model_id}_training_metadata.json");
    Path::new(models_dir).join(metadata_file).to_string_lossy().into_owned()
}

/// Load training metadata if it exists; an empty map otherwise.
pub fn load_training_metadata(models_dir: &str, model_id: &str) -> Map<String, Value> {
    let metadata_path = get_training_metadata_path(models_dir, model_id);
    if Path::new(&metadata_path).exists() {
        let loaded = std::fs::read_to_string(&metadata_path)
            .map_err(|e| e.to_string())
            .and_then(|text| serde_json::from_str::<Value>(&text).map_err(|e| e.to_string()));
        match loaded {
            Ok(Value::Object(map)) => return map,
            Ok(_) => {}
            // Python prints "[info] ..."; Rich swallows the unknown "[info]" tag.
            Err(e) => println!("Could not load training metadata: {e}"),
        }
    }
    Map::new()
}

fn write_metadata(metadata_path: &str, metadata: &Map<String, Value>) -> std::io::Result<()> {
    std::fs::write(metadata_path, py_json_dumps_indent(&Value::Object(metadata.clone()), 2))
}

/// Save training metadata (dataset hash, merged model dir, training date,
/// model id) for future reference.  Errors are printed, never returned.
pub fn save_training_metadata(
    models_dir: &str,
    model_id: &str,
    dataset_hash: &str,
    merged_model_dir: &str,
) {
    let metadata_path = get_training_metadata_path(models_dir, model_id);
    let mut metadata = load_training_metadata(models_dir, model_id);

    let mut entry = Map::new();
    entry.insert("dataset_hash".into(), Value::String(dataset_hash.to_string()));
    entry.insert("merged_model_dir".into(), Value::String(merged_model_dir.to_string()));
    // str(pd.Timestamp.now())
    entry.insert(
        "training_date".into(),
        Value::String(chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.6f").to_string()),
    );
    entry.insert("model_id".into(), Value::String(model_id.to_string()));
    metadata.insert(dataset_hash.to_string(), Value::Object(entry));

    match write_metadata(&metadata_path, &metadata) {
        Ok(()) => print_success("Training metadata saved"),
        Err(e) => print_warning(&format!("Could not save training metadata: {e}")),
    }
}

/// Check whether the model has already been trained for this dataset;
/// returns the merged model directory when it still exists.  A stale entry
/// whose directory is missing is removed from the metadata file.
pub fn check_existing_training(
    models_dir: &str,
    model_id: &str,
    dataset_hash: Option<&str>,
) -> Option<String> {
    let dataset_hash = match dataset_hash {
        Some(h) if !h.is_empty() => h,
        _ => return None,
    };

    let mut metadata = load_training_metadata(models_dir, model_id);

    let entry = metadata.get(dataset_hash)?;
    let merged_model_dir = entry.get("merged_model_dir").map(py_str).unwrap_or_default();
    if !merged_model_dir.is_empty() && Path::new(&merged_model_dir).exists() {
        let training_date = entry.get("training_date").map(py_str).unwrap_or_else(|| "unknown".to_string());
        print_success("Found existing training for this dataset");
        print_detail(&format!("Training date: {training_date}"));
        print_detail(&format!("Model location: {merged_model_dir}"));
        return Some(merged_model_dir);
    }

    print_warning(&format!("Training metadata found but model directory missing: {merged_model_dir}"));
    // Clean up invalid metadata entry
    metadata.shift_remove(dataset_hash);
    let metadata_path = get_training_metadata_path(models_dir, model_id);
    let _ = write_metadata(&metadata_path, &metadata);

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().to_str().unwrap();
        let data = dir.path().join("data.jsonl");
        std::fs::write(&data, b"hello").unwrap();
        let hash = calculate_dataset_hash(data.to_str().unwrap()).unwrap();
        assert_eq!(hash, "5d41402abc4b2a76b9719d911017c592");
        assert!(calculate_dataset_hash(dir.path().join("missing").to_str().unwrap()).is_none());

        assert_eq!(
            get_training_metadata_path(models, "org/model"),
            dir.path().join("org_model_training_metadata.json").to_string_lossy()
        );

        let merged = dir.path().join("merged");
        std::fs::create_dir(&merged).unwrap();
        save_training_metadata(models, "org/model", &hash, merged.to_str().unwrap());
        assert_eq!(
            check_existing_training(models, "org/model", Some(&hash)).as_deref(),
            merged.to_str()
        );
        assert!(check_existing_training(models, "org/model", None).is_none());

        std::fs::remove_dir(&merged).unwrap();
        assert!(check_existing_training(models, "org/model", Some(&hash)).is_none());
        assert!(load_training_metadata(models, "org/model").is_empty());
    }
}
