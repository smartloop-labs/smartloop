//! Port of `tests/unit/test_indexer_waits_for_quiet.py`.
//!
//! Background indexing must not embed while the model is answering.
//!
//! Embedding and generation are the same GPU.  A bge-m3 pass over a few
//! hundred chunks took a streaming reply from ~57 tok/s to ~3 — the user
//! watching their answer arrive a word at a time so the workspace index could
//! be a minute fresher.  `ModelService::inference_slot` already counted "is the
//! model answering someone right now"; this is the reader it was built for.

use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};
use smartloop::db::workspace_indexer::IndexStats;
use smartloop::tasks::base_task::{TaskLogger, TaskOptions};
use smartloop::tasks::indexer_task::{IndexerTask, WorkspaceIndex};
use smartloop::utils::log_utils::LoggerLike;
use smartloop::workspace::workspace::{Project, Workspace};

/// Stand-in for the `MagicMock()` logger: swallows everything.
struct NullLogger;

impl LoggerLike for NullLogger {
    fn info(&self, _msg: &str) {}
    fn warning(&self, _msg: &str) {}
    fn debug(&self, _msg: &str) {}
    fn error(&self, _msg: &str) {}
}

/// Stand-in for the `MagicMock()` workspace indexer: records every `index`
/// call and reports nothing changed.
#[derive(Default)]
struct RecordingIndexer {
    calls: Mutex<Vec<String>>,
}

impl WorkspaceIndex for RecordingIndexer {
    fn index(&self, work_dir: &str) -> IndexStats {
        self.calls.lock().push(work_dir.to_string());
        IndexStats { indexed: 0, skipped: 0, removed: 0 }
    }
    fn file_count(&self) -> usize {
        0
    }
}

impl RecordingIndexer {
    fn assert_not_called(&self) {
        let calls = self.calls.lock();
        assert!(calls.is_empty(), "index was called: {calls:?}");
    }

    fn assert_called_once_with(&self, work_dir: &str) {
        assert_eq!(*self.calls.lock(), vec![work_dir.to_string()]);
    }
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

/// An IndexerTask wired to just enough state to run one scan decision.
///
/// `idle` stands in for the patched `device_idle_seconds` reading.
fn task(project_dir: &Path, idle: f64) -> (Arc<IndexerTask>, Arc<RecordingIndexer>) {
    let dir = project_dir.to_str().unwrap().to_string();

    let mut ws = Workspace::new("test-workspace");
    ws.workspace_dir = dir.clone();
    let workspace = Arc::new(RwLock::new(ws));

    let logger: TaskLogger = Arc::new(NullLogger);
    let task = IndexerTask::new(workspace, logger, None, None, TaskOptions::default());

    let indexer = Arc::new(RecordingIndexer::default());
    task.set_workspace_indexer(Some(indexer.clone()), Some(dir.clone()));
    task.set_last_workspace_scan(0.0);

    let project = Project { project_dir: dir, ..Default::default() };
    task.set_project_provider(Some(Arc::new(move || Some(project.clone()))));
    task.set_idle_seconds_provider(Some(Arc::new(move || idle)));

    // The cadence the Python helper pinned on the instance.
    assert_eq!(IndexerTask::WORKSPACE_SCAN_INTERVAL, 30.0);
    assert_eq!(IndexerTask::WORKSPACE_QUIET_SECONDS, 5.0);
    (task, indexer)
}

mod test_a_pass_waits_for_a_quiet_model {
    use super::*;

    #[test]
    fn test_a_pass_is_deferred_while_the_model_generates() {
        let td = tempfile::tempdir().unwrap();
        let (task, indexer) = task(td.path(), 0.0);
        task.scan_workspace_files().unwrap();
        indexer.assert_not_called();
    }

    #[test]
    fn test_a_pass_is_deferred_inside_the_gaps_between_calls_in_one_turn() {
        // One turn is many sequential calls — classify, skill match, tool
        // resolution, judge, then the answer.  A bare "is it generating right
        // now" check would slip a pass into a gap and run through the reply.
        let td = tempfile::tempdir().unwrap();
        let (task, indexer) = task(td.path(), 1.5);
        task.scan_workspace_files().unwrap();
        indexer.assert_not_called();
    }

    #[test]
    fn test_a_pass_runs_once_the_model_has_been_quiet() {
        let td = tempfile::tempdir().unwrap();
        let (task, indexer) = task(td.path(), 42.0);
        task.scan_workspace_files().unwrap();
        indexer.assert_called_once_with(td.path().to_str().unwrap());
    }

    #[test]
    fn test_a_model_that_never_ran_does_not_block_the_first_pass() {
        // idle_seconds() is inf before anything has generated; a fresh install
        // has to index without waiting for a turn that may never come.
        let td = tempfile::tempdir().unwrap();
        let (task, indexer) = task(td.path(), f64::INFINITY);
        task.scan_workspace_files().unwrap();
        indexer.assert_called_once_with(td.path().to_str().unwrap());
    }

    #[test]
    fn test_a_deferral_retries_within_a_quiet_period_not_a_full_interval() {
        // The dispatcher stamps the clock when it hands the pass over, so
        // leaving it there would turn every deferral into a 30s wait and a busy
        // stretch would starve the index instead of postponing it.
        let td = tempfile::tempdir().unwrap();
        let (task, _indexer) = task(td.path(), 0.0);
        task.scan_workspace_files().unwrap();

        let due_in = task.last_workspace_scan() + IndexerTask::WORKSPACE_SCAN_INTERVAL - now();
        assert!(
            0.0 < due_in && due_in <= IndexerTask::WORKSPACE_QUIET_SECONDS + 1.0,
            "due_in = {due_in}"
        );
    }
}

mod test_the_idle_reading_comes_from_the_right_process {
    //! The Python tests patched `embeddings._remote_embedder` with a mock.
    //! Here the real resolution runs: `SLP_MODEL_HOST` names a stand-in model
    //! host (a gRPC server answering `IdleSeconds`), or a dead address.

    use std::net::TcpListener as StdTcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serial_test::serial;
    use smartloop::api::services::model_service::ModelService;
    use smartloop::db::embeddings;
    use smartloop::model_host::client::set_device_owner;
    use smartloop::model_host::pb;
    use smartloop::model_host::pb::model_host_server::{ModelHost, ModelHostServer};
    use tonic::{Request, Response, Status};

    use super::*;

    /// Restores `SLP_MODEL_HOST` when dropped.
    struct HostEnv(Option<String>);

    impl HostEnv {
        fn set(value: Option<&str>) -> Self {
            let previous = std::env::var("SLP_MODEL_HOST").ok();
            // SAFETY: tests touching the environment are serialised.
            unsafe {
                match value {
                    Some(v) => std::env::set_var("SLP_MODEL_HOST", v),
                    None => std::env::remove_var("SLP_MODEL_HOST"),
                }
            }
            HostEnv(previous)
        }
    }

    impl Drop for HostEnv {
        fn drop(&mut self) {
            // SAFETY: tests touching the environment are serialised.
            unsafe {
                match self.0.take() {
                    Some(v) => std::env::set_var("SLP_MODEL_HOST", v),
                    None => std::env::remove_var("SLP_MODEL_HOST"),
                }
            }
        }
    }

    /// A model host that only answers `IdleSeconds`, counting the calls.
    struct IdleHost {
        idle: f64,
        calls: Arc<AtomicUsize>,
    }

    fn unimplemented<T>() -> Result<Response<T>, Status> {
        Err(Status::unimplemented("not part of this test"))
    }

    #[tonic::async_trait]
    impl ModelHost for IdleHost {
        async fn generate(
            &self,
            _: Request<pb::GenerateRequest>,
        ) -> Result<Response<pb::GenerateResponse>, Status> {
            unimplemented()
        }
        async fn generate_batch(
            &self,
            _: Request<pb::GenerateBatchRequest>,
        ) -> Result<Response<pb::GenerateBatchResponse>, Status> {
            unimplemented()
        }
        type StreamStream = futures::stream::Empty<Result<pb::StreamChunk, Status>>;
        async fn stream(&self, _: Request<pb::StreamRequest>) -> Result<Response<Self::StreamStream>, Status> {
            unimplemented()
        }
        async fn encode_texts(
            &self,
            _: Request<pb::EncodeTextsRequest>,
        ) -> Result<Response<pb::EncodeTextsResponse>, Status> {
            unimplemented()
        }
        async fn is_loaded(&self, _: Request<pb::ProjectRequest>) -> Result<Response<pb::BoolReply>, Status> {
            unimplemented()
        }
        async fn supports_thinking(
            &self,
            _: Request<pb::SupportsThinkingRequest>,
        ) -> Result<Response<pb::BoolReply>, Status> {
            unimplemented()
        }
        async fn supports_native_tools(
            &self,
            _: Request<pb::SupportsThinkingRequest>,
        ) -> Result<Response<pb::BoolReply>, Status> {
            unimplemented()
        }
        async fn model_name(&self, _: Request<pb::ProjectRequest>) -> Result<Response<pb::StringReply>, Status> {
            unimplemented()
        }
        async fn model_label(
            &self,
            _: Request<pb::ModelLabelRequest>,
        ) -> Result<Response<pb::StringReply>, Status> {
            unimplemented()
        }
        async fn count_tokens(
            &self,
            _: Request<pb::CountTokensRequest>,
        ) -> Result<Response<pb::IntReply>, Status> {
            unimplemented()
        }
        async fn max_context_tokens(
            &self,
            _: Request<pb::ProjectRequest>,
        ) -> Result<Response<pb::IntReply>, Status> {
            unimplemented()
        }
        async fn response_max_context_tokens(
            &self,
            _: Request<pb::ResponseMaxContextRequest>,
        ) -> Result<Response<pb::IntReply>, Status> {
            unimplemented()
        }
        async fn has_capability_model(
            &self,
            _: Request<pb::HasCapabilityRequest>,
        ) -> Result<Response<pb::BoolReply>, Status> {
            unimplemented()
        }
        async fn idle_seconds(&self, _: Request<pb::ProjectRequest>) -> Result<Response<pb::FloatReply>, Status> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Response::new(pb::FloatReply { value: self.idle }))
        }
        async fn load_model(&self, _: Request<pb::ProjectRequest>) -> Result<Response<pb::StringReply>, Status> {
            unimplemented()
        }
        async fn unload_auxiliary_models(
            &self,
            _: Request<pb::UnloadAuxiliaryRequest>,
        ) -> Result<Response<pb::Empty>, Status> {
            unimplemented()
        }
        async fn unload_embedding_model(
            &self,
            _: Request<pb::ProjectRequest>,
        ) -> Result<Response<pb::Empty>, Status> {
            unimplemented()
        }

        async fn unload_model(
            &self,
            _: Request<pb::UnloadModelRequest>,
        ) -> Result<Response<pb::BoolReply>, Status> {
            unimplemented()
        }
    }

    /// Serve an [`IdleHost`] on a loopback port; returns `host:port` and the
    /// call counter.  The server lives for the rest of the test process.
    fn spawn_idle_host(idle: f64) -> (String, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        listener.set_nonblocking(true).unwrap();
        let service = ModelHostServer::new(IdleHost { idle, calls: calls.clone() });
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                let incoming = async_stream::stream! {
                    loop {
                        yield listener.accept().await.map(|(stream, _)| stream);
                    }
                };
                let _ = tonic::transport::Server::builder()
                    .add_service(service)
                    .serve_with_incoming(incoming)
                    .await;
            });
        });
        (address, calls)
    }

    /// A loopback address nothing listens on.
    fn dead_address() -> String {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        drop(listener);
        address
    }

    #[test]
    #[serial(embeddings)]
    fn test_a_child_asks_the_host_that_owns_the_embedder() {
        // A child polling its own idle time would always read "quiet" and
        // answer for the wrong device — both the embedding and the generation
        // happen in the host.
        set_device_owner(false);
        let (address, calls) = spawn_idle_host(0.0);
        let _env = HostEnv::set(Some(&address));

        assert_eq!(embeddings::device_idle_seconds(), 0.0);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[serial(embeddings)]
    fn test_an_unreachable_host_reads_as_idle() {
        // Unknowable means idle: an indexer that never runs is worse than one
        // that overlaps.
        set_device_owner(false);
        let _env = HostEnv::set(Some(&dead_address()));

        assert_eq!(embeddings::device_idle_seconds(), f64::INFINITY);
    }

    #[test]
    #[serial(embeddings)]
    fn test_the_host_itself_reads_its_own_service() {
        // No host address: this process owns the embedder, so it reads its own
        // service — which last finished generating three seconds ago.
        let _env = HostEnv::set(None);
        let service = ModelService::instance();
        let saved = *service.inference_state.lock();
        {
            let mut state = service.inference_state.lock();
            state.active = 0;
            state.last_inference_at = now() - 3.0;
        }

        let idle = embeddings::device_idle_seconds();
        *service.inference_state.lock() = saved;

        assert!((3.0..3.5).contains(&idle), "idle = {idle}");
    }
}
