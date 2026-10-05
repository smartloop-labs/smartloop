//! Port of `tests/api/conftest.py`.

// ======================= BEGIN unit: model =======================
/// API test fixtures.
///
/// Nothing here may reach the network, and one guarantee needs stating
/// outright: **no test downloads a model.**  The routers under test build
/// their own `SLPArtifactDownloader` rather than receiving one, and `/v1/init`
/// hands the transfer to a blocking worker whose thread outlives the response.
/// A test that closed the stream to avoid "running a real download" therefore
/// left one running: the weights kept arriving, into a temp dir that would
/// later be deleted, long after the assertion had passed.  A full run wrote
/// tens of gigabytes of real GGUFs this way and filled the disk.
///
/// So the download entry points are stubbed for every test in this package,
/// process-wide rather than at any one call site — three call sites build a
/// downloader (`/v1/init`, `/v1/bootstrap`, and `GenericModel::init` by way of
/// `ModelFactory`), and a stub attached to one of them silently misses the
/// others.  Every API test holds a [`NoRealDownloads`] guard (Python's autouse
/// `no_real_downloads` fixture) and, because the stubs are process-global, is
/// `#[serial]`.
#[allow(dead_code)]
pub mod model_support {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use parking_lot::Mutex;
    use smartloop::model::downloader::artifact_source::ArtifactMap;
    use smartloop::model::downloader::slp_downloader::{
        DownloadStubCall, DownloadStubFn, DownloadStubs, ResolveStubFn, set_download_stubs,
    };

    /// Files a real download leaves behind, complete or interrupted.
    pub const WEIGHT_SUFFIXES: [&str; 2] = ["gguf", "part"];

    /// One intercepted download: `("download", model_id)` or
    /// `("bootstrap_download", model_name)`.
    pub type DownloadCall = (String, Option<String>);

    /// Stub every path that would fetch weights, and prove none was taken.
    ///
    /// Un-opt-out-able by design: a download is never what an API test is
    /// asserting, and the failure mode when one slips through is silent — the test
    /// passes, and the machine fills up minutes later from a thread nobody is
    /// watching.
    ///
    /// The stubs stand in for a *successful* download, so a route that goes on to
    /// register the model still behaves as it would in life.  What they do not do
    /// is touch the network or the disk.
    pub struct NoRealDownloads {
        tmp: tempfile::TempDir,
        calls: Arc<Mutex<Vec<DownloadCall>>>,
        stubs: Mutex<DownloadStubs>,
    }

    impl NoRealDownloads {
        /// Install the stubs (Python's `no_real_downloads` fixture).
        pub fn new() -> Self {
            let tmp = tempfile::tempdir().expect("temp dir");
            let calls: Arc<Mutex<Vec<DownloadCall>>> = Arc::new(Mutex::new(Vec::new()));
            let tmp_path = tmp.path().to_path_buf();

            let download_calls = Arc::clone(&calls);
            let download_root = tmp_path.clone();
            let fake_download: DownloadStubFn = Arc::new(move |call: DownloadStubCall| {
                match call {
                    DownloadStubCall::Download { model_id, models_dir } => {
                        download_calls.lock().push(("download".to_string(), Some(model_id)));
                        Ok(models_dir.unwrap_or_else(|| download_root.clone()))
                    }
                    DownloadStubCall::BootstrapDownload { model_name, models_dir } => {
                        download_calls.lock().push(("bootstrap_download".to_string(), model_name));
                        Ok(models_dir.unwrap_or_else(|| download_root.clone()))
                    }
                }
            });

            let stubs = DownloadStubs {
                download: Some(Arc::clone(&fake_download)),
                bootstrap_download: Some(fake_download),
                resolve_artifacts: None,
            };
            set_download_stubs(Some(stubs.clone()));
            NoRealDownloads { tmp, calls, stubs: Mutex::new(stubs) }
        }

        /// The temp directory the stubs report as the download location
        /// (`tmp_path`), also the tree checked for leaked weights on drop.
        pub fn tmp_path(&self) -> &Path {
            self.tmp.path()
        }

        /// Every download the stubs intercepted, in order.
        pub fn calls(&self) -> Vec<DownloadCall> {
            self.calls.lock().clone()
        }

        /// Replace the `download` stub (e.g. with one that raises
        /// `DownloadCancelled`), as a test's own `monkeypatch.setattr` would.
        pub fn set_download(&self, stub: DownloadStubFn) {
            let mut stubs = self.stubs.lock();
            stubs.download = Some(stub);
            set_download_stubs(Some(stubs.clone()));
        }

        /// Replace `resolve_artifacts` with `stub`.
        pub fn set_resolve_artifacts(&self, stub: ResolveStubFn) {
            let mut stubs = self.stubs.lock();
            stubs.resolve_artifacts = Some(stub);
            set_download_stubs(Some(stubs.clone()));
        }

        /// `resolve_artifacts` answering a fixed map.
        pub fn resolve_to(&self, artifacts: ArtifactMap) {
            self.set_resolve_artifacts(Arc::new(move |_model_id: &str| Ok(artifacts.clone())));
        }

        /// Weight files under the temp dir — there must be none.
        pub fn leaked(&self) -> Vec<PathBuf> {
            walkdir::WalkDir::new(self.tmp.path())
                .into_iter()
                .flatten()
                .filter(|entry| entry.file_type().is_file())
                .map(|entry| entry.into_path())
                .filter(|path| {
                    path.extension().and_then(|e| e.to_str()).is_some_and(|ext| WEIGHT_SUFFIXES.contains(&ext))
                })
                .collect()
        }
    }

    impl Default for NoRealDownloads {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for NoRealDownloads {
        fn drop(&mut self) {
            set_download_stubs(None);
            // The stub can stop matching — a route that starts building its
            // downloader somewhere new, or a rename — and the only symptom would
            // be the disk filling again.  Assert the outcome rather than trusting
            // the patch.
            let leaked = self.leaked();
            if !std::thread::panicking() {
                assert!(
                    leaked.is_empty(),
                    "a real model download escaped the stub and wrote {leaked:?} — check whether a new code path \
                     builds its own downloader"
                );
            }
        }
    }

    /// Install the download stubs for the duration of a test.
    pub fn no_real_downloads() -> NoRealDownloads {
        NoRealDownloads::new()
    }
}
// ======================== END unit: model ========================
