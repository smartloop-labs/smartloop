//! Port of `smartloop/utils/__init__.py`.

pub mod chunk_converter;
pub mod device_utils;
pub mod doc_splitter;
pub mod log_utils;
pub mod logging_setup;
pub mod mcp_result_formatter;
pub mod metadata_utils;
pub mod path_utils;

pub use chunk_converter::convert_chunks_to_document_chunks;
pub use device_utils::{get_device_config, is_apple_silicon, DeviceConfig};
pub use doc_splitter::{is_markdown_table, json_to_table, DocumentSplitter};
pub use log_utils::{ConsoleLogger, LoggerLike};
pub use logging_setup::{get_logger, Logger};
pub use mcp_result_formatter::ResultFormatter;
pub use metadata_utils::{
    calculate_dataset_hash, check_existing_training, load_training_metadata,
    save_training_metadata,
};
pub use path_utils::normalize_path;
