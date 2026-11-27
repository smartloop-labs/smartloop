//! Progress frames `download_and_stream` sends while a model downloads.

use std::sync::Arc;

use futures::StreamExt;
use serial_test::serial;
use smartloop::api::routers::init::{self as init_router, InitDownloader};
use smartloop::api::services::ModelService;
use smartloop::model::downloader::artifact_source::ArtifactMap;
use smartloop::model::downloader::cancellation::{self, CancelToken};
use smartloop::model::downloader::slp_downloader::{DownloadProgress, ProgressCallback};

/// Reports a little progress, then a long verification, then completion.
struct VerifyingDownloader;

impl InitDownloader for VerifyingDownloader {
    fn resolve_artifacts(&self, _model_id: &str) -> anyhow::Result<ArtifactMap> {
        Ok(ArtifactMap::new())
    }

    fn set_cancel_token(&self, _token: CancelToken) {}

    fn download(
        &self,
        _model_id: &str,
        progress: Option<ProgressCallback>,
        _artifacts: Option<ArtifactMap>,
    ) -> anyhow::Result<()> {
        let progress = progress.unwrap();
        progress(DownloadProgress::new("Big", 100, 100, "downloading"));
        progress(DownloadProgress::new("Big", 100, 100, "verifying"));
        progress(DownloadProgress::new("Big", 100, 100, "completed"));
        Ok(())
    }

    fn bootstrap_download(&self, _progress: Option<ProgressCallback>) -> anyhow::Result<()> {
        Ok(())
    }

    fn list_bootstrap_models(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        Ok(Vec::new())
    }
}

#[tokio::test(flavor = "multi_thread")]
#[serial]
async fn test_a_verifying_file_gets_its_own_frame() {
    cancellation::reset_for_tests();
    let stream = init_router::download_and_stream(
        Arc::new(VerifyingDownloader),
        "org/big".to_string(),
        "big".to_string(),
        false,
        None,
        None::<Arc<ModelService>>,
    );
    let frames: Vec<String> = stream.filter_map(|f| async move { f.ok() }).collect().await;
    let verifying: Vec<&String> = frames.iter().filter(|f| f.contains("\"status\": \"verifying\"")).collect();
    assert_eq!(verifying.len(), 1, "{frames:#?}");
    assert!(verifying[0].contains("Verifying Big..."), "{}", verifying[0]);
}
