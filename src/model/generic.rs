//! Port of `smartloop/model/generic.py`.
//!
//! [`GenericModel`] is the shared implementation behind every concrete model:
//! weight download, document ingestion (files, directories, URLs, Google
//! Drive links), skills, status, train/build and the cached LLM instance.  The
//! per-model facts (id, family version, capabilities, ...) come from the
//! concrete model's [`ModelClass`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::llms::base_llm::BaseLLM;
#[allow(unused_imports)]
use crate::loader::ModelLoader;
use crate::model::downloader::slp_downloader::{SLPDownloader, SharedLogger, SharedWorkspace};
use crate::model::slp_model::{DocumentProgressCallback, DocumentSource, ModelClass, SLPModel, TrainProgressCallback};
#[allow(unused_imports)]
use crate::paths::expand_home;
use crate::types::{ModelCapability, ModelType};
#[allow(unused_imports)]
use crate::utils::normalize_path;
#[allow(unused_imports)]
use crate::workspace::types::{Document, DocumentStatus, Skill};

/// Shared implementation of [`SLPModel`] for all concrete models.
pub struct GenericModel {
    pub workspace: SharedWorkspace,
    pub logger: SharedLogger,
    pub downloader: Arc<dyn SLPDownloader>,
    /// The concrete model's static facts (the Python subclass's static methods).
    pub class: ModelClass,
    /// The concrete model's modality (the Python `model_type` property).
    pub model_type: ModelType,
    /// Cache for the loaded model.
    instance_cache: Mutex<Option<Arc<dyn BaseLLM>>>,
}

impl GenericModel {
    /// Build a generic model for `class` with modality `model_type`.
    pub fn new(
        workspace: SharedWorkspace,
        downloader: Arc<dyn SLPDownloader>,
        logger: SharedLogger,
        class: ModelClass,
        model_type: ModelType,
    ) -> Self {
        Self { workspace, logger, downloader, class, model_type, instance_cache: Mutex::new(None) }
    }

    /// MD5 hex digest of a file's contents.
    fn calculate_file_hash(&self, file_path: &Path) -> anyhow::Result<String> {
        let _ = file_path;
        todo!()
    }

    /// Whether `source` parses as a URL with both scheme and netloc.
    pub fn is_url(&self, source: &str) -> bool {
        let _ = source;
        todo!()
    }

    /// Whether `url` is a Google Drive link.
    pub fn is_google_drive_url(&self, url: &str) -> bool {
        let _ = url;
        todo!()
    }

    /// Extract the file ID from a Google Drive URL (`/file/d/<id>` or
    /// `id=<id>`); errors with "Could not extract file ID from Google Drive
    /// URL: <url>".
    pub fn extract_google_drive_id(&self, url: &str) -> anyhow::Result<String> {
        let _ = url;
        todo!()
    }

    /// Copy the original file into `dest_dir`, appending `_<n>` to the stem on
    /// name collisions; returns the destination path.
    pub fn copy_to_project_dir(&self, file_path: &Path, dest_dir: &Path) -> anyhow::Result<PathBuf> {
        let _ = (file_path, dest_dir);
        todo!()
    }

    /// Use the LLM to produce a short descriptive snake_case filename from the
    /// first 512 characters of the file; `None` on any failure.
    pub fn generate_filename(&self, file_path: &Path, llm: &dyn BaseLLM) -> Option<String> {
        let _ = (file_path, llm);
        todo!()
    }

    /// Download a Google Drive file (handling the large-file confirm cookie)
    /// into a fresh temp directory.
    fn download_from_google_drive(&self, url: &str) -> anyhow::Result<PathBuf> {
        let _ = url;
        todo!()
    }

    /// Download a URL into a fresh temp directory.
    fn download_url(&self, url: &str) -> anyhow::Result<PathBuf> {
        let _ = url;
        todo!()
    }

    /// Emit `{"stage", "filename", "message"}` if a callback is provided.
    fn emit_progress(&self, callback: Option<&DocumentProgressCallback>, stage: &str, filename: &str, message: &str) {
        let _ = (callback, stage, filename, message);
        todo!()
    }

    /// Error unless `model_dir` contains `*.safetensors` files.
    pub fn check_safetensors(&self, model_dir: &Path) -> anyhow::Result<()> {
        let _ = model_dir;
        todo!()
    }
}

impl SLPModel for GenericModel {
    fn workspace(&self) -> &SharedWorkspace {
        &self.workspace
    }

    fn logger(&self) -> &SharedLogger {
        &self.logger
    }

    fn init(&self) -> anyhow::Result<()> {
        todo!()
    }

    /// Copies the original file(s) into the project's documents directory and
    /// registers them with PENDING status (duplicates by hash are skipped).
    /// `cwd` never redirects where documents are stored.
    fn add_document(
        &self,
        source: DocumentSource,
        progress_callback: Option<DocumentProgressCallback>,
        cwd: Option<&str>,
        generate_filename: bool,
    ) -> anyhow::Result<Vec<Document>> {
        let _ = (source, progress_callback, cwd, generate_filename);
        todo!()
    }

    fn add_skill(&self, name: &str, content: &str) -> anyhow::Result<()> {
        let _ = (name, content);
        todo!()
    }

    fn status(&self) -> anyhow::Result<()> {
        todo!()
    }

    fn build(&self, progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()> {
        let _ = progress_callback;
        todo!()
    }

    fn train(&self, progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()> {
        let _ = progress_callback;
        todo!()
    }

    fn model_type(&self) -> ModelType {
        self.model_type
    }

    fn model_id(&self) -> String {
        (self.class.model_id)()
    }

    fn model_version(&self) -> i64 {
        (self.class.model_version)()
    }

    fn param_size_in_b(&self) -> f64 {
        (self.class.param_size_in_b)()
    }

    fn default_capabilities(&self) -> Vec<ModelCapability> {
        (self.class.default_capabilities)()
    }

    fn thinking_default(&self) -> bool {
        (self.class.thinking_default)()
    }

    fn vision_chat_handler(&self) -> Option<String> {
        (self.class.vision_chat_handler)()
    }

    fn get_instance(&self) -> anyhow::Result<Arc<dyn BaseLLM>> {
        let _ = &self.instance_cache;
        todo!()
    }
}

/// Define a concrete model type wrapping [`GenericModel`]: a newtype with
/// inherent static accessors mirroring the Python static methods (these are
/// what the config registry's [`ModelClass`] points at), `Deref` to the
/// generic model, and an [`SLPModel`] impl.
#[macro_export]
macro_rules! generic_model_type {
    (
        $(#[$meta:meta])*
        $name:ident {
            model_id: $model_id:expr,
            model_type: $model_type:expr,
            model_version: $version:expr,
            param_size_in_b: $size:expr,
            default_capabilities: $caps:expr,
            thinking_default: $thinking:expr,
            vision_chat_handler: $handler:expr $(,)?
        }
    ) => {
        $(#[$meta])*
        pub struct $name(pub $crate::model::generic::GenericModel);

        #[allow(non_camel_case_types)]
        impl $name {
            /// Construct the model (`Model(workspace, downloader, logger)`).
            pub fn new(
                workspace: $crate::model::downloader::slp_downloader::SharedWorkspace,
                downloader: ::std::sync::Arc<dyn $crate::model::downloader::slp_downloader::SLPDownloader>,
                logger: $crate::model::downloader::slp_downloader::SharedLogger,
            ) -> Self {
                Self($crate::model::generic::GenericModel::new(workspace, downloader, logger, Self::class(), $model_type))
            }

            /// This model's class descriptor.
            pub fn class() -> $crate::model::slp_model::ModelClass {
                $crate::model::slp_model::ModelClass {
                    class_name: stringify!($name),
                    model_id: Self::model_id,
                    model_version: Self::model_version,
                    param_size_in_b: Self::param_size_in_b,
                    default_capabilities: Self::default_capabilities,
                    thinking_default: Self::thinking_default,
                    vision_chat_handler: Self::vision_chat_handler,
                }
            }

            /// The HuggingFace-style model id.
            pub fn model_id() -> String {
                ($model_id).to_string()
            }
            /// Model family version.
            pub fn model_version() -> i64 {
                $version
            }
            /// Parameter count in billions.
            pub fn param_size_in_b() -> f64 {
                $size
            }
            /// Declared capabilities.
            pub fn default_capabilities() -> Vec<$crate::types::ModelCapability> {
                $caps
            }
            /// Whether reasoning defaults on.
            pub fn thinking_default() -> bool {
                $thinking
            }
            /// Vision chat-handler name.
            pub fn vision_chat_handler() -> Option<String> {
                let handler: Option<&str> = $handler;
                handler.map(str::to_string)
            }
        }

        impl ::std::ops::Deref for $name {
            type Target = $crate::model::generic::GenericModel;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl $crate::model::slp_model::SLPModel for $name {
            fn workspace(&self) -> &$crate::model::downloader::slp_downloader::SharedWorkspace {
                $crate::model::slp_model::SLPModel::workspace(&self.0)
            }
            fn logger(&self) -> &$crate::model::downloader::slp_downloader::SharedLogger {
                $crate::model::slp_model::SLPModel::logger(&self.0)
            }
            fn init(&self) -> ::anyhow::Result<()> {
                $crate::model::slp_model::SLPModel::init(&self.0)
            }
            fn add_document(
                &self,
                source: $crate::model::slp_model::DocumentSource,
                progress_callback: Option<$crate::model::slp_model::DocumentProgressCallback>,
                cwd: Option<&str>,
                generate_filename: bool,
            ) -> ::anyhow::Result<Vec<$crate::workspace::types::Document>> {
                $crate::model::slp_model::SLPModel::add_document(&self.0, source, progress_callback, cwd, generate_filename)
            }
            fn add_skill(&self, name: &str, content: &str) -> ::anyhow::Result<()> {
                $crate::model::slp_model::SLPModel::add_skill(&self.0, name, content)
            }
            fn status(&self) -> ::anyhow::Result<()> {
                $crate::model::slp_model::SLPModel::status(&self.0)
            }
            fn build(&self, progress_callback: Option<$crate::model::slp_model::TrainProgressCallback>) -> ::anyhow::Result<()> {
                $crate::model::slp_model::SLPModel::build(&self.0, progress_callback)
            }
            fn train(&self, progress_callback: Option<$crate::model::slp_model::TrainProgressCallback>) -> ::anyhow::Result<()> {
                $crate::model::slp_model::SLPModel::train(&self.0, progress_callback)
            }
            fn model_type(&self) -> $crate::types::ModelType {
                $crate::model::slp_model::SLPModel::model_type(&self.0)
            }
            fn model_id(&self) -> String {
                $crate::model::slp_model::SLPModel::model_id(&self.0)
            }
            fn model_version(&self) -> i64 {
                $crate::model::slp_model::SLPModel::model_version(&self.0)
            }
            fn param_size_in_b(&self) -> f64 {
                $crate::model::slp_model::SLPModel::param_size_in_b(&self.0)
            }
            fn default_capabilities(&self) -> Vec<$crate::types::ModelCapability> {
                $crate::model::slp_model::SLPModel::default_capabilities(&self.0)
            }
            fn thinking_default(&self) -> bool {
                $crate::model::slp_model::SLPModel::thinking_default(&self.0)
            }
            fn vision_chat_handler(&self) -> Option<String> {
                $crate::model::slp_model::SLPModel::vision_chat_handler(&self.0)
            }
            fn get_instance(&self) -> ::anyhow::Result<::std::sync::Arc<dyn $crate::llms::base_llm::BaseLLM>> {
                $crate::model::slp_model::SLPModel::get_instance(&self.0)
            }
        }
    };
}
