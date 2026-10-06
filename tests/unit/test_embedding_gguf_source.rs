//! Port of `tests/unit/test_embedding_gguf_source.py`.
//!
//! The embedding GGUF is fetched from our own bucket, not HuggingFace.
//!
//! The Hub gates GGUF downloads behind a user token and is slow from most
//! regions, so first run resolves in order: bundled copy → an
//! already-downloaded HF-cache copy on disk → our public bucket.  The Hub
//! itself is never called (there is no Hub client in the port at all).
//!
//! Where the Python tests patch `httpx.stream`, these serve the bucket from a
//! local HTTP server (or a closed port for an outage).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use smartloop::db::llama_embeddings::{
    cdn_gguf_path, hf_cached_gguf_path, load_embedder_with, EmbedBackend, LlamaLoadParams,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const FILENAME: &str = "bge-m3-Q4_K_M.gguf";
const REPO: &str = "gpustack/bge-m3-GGUF";
/// The production bucket URL; only used where the network must not be hit.
const BASE_URL: &str = "https://dl.smartloop.ai/embeddings";

/// A bucket at `<server>/embeddings` serving `body` for the GGUF.
async fn bucket(body: &'static [u8]) -> (MockServer, String) {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(format!("/embeddings/{FILENAME}")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
        .mount(&server)
        .await;
    let base = format!("{}/embeddings", server.uri());
    (server, base)
}

/// A base URL on a port nothing listens on — every request fails to connect.
fn unreachable_base_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{addr}/embeddings")
}

/// A one-shot server that promises `content_length` bytes but sends `body`
/// and hangs up.
fn short_body_server(body: &'static [u8], content_length: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body);
            let _ = stream.flush();
        }
    });
    format!("http://{addr}/embeddings")
}

fn listdir(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

/// `_cdn_gguf_path` off the async runtime (it blocks on its own download).
async fn cdn(cache_dir: Option<String>, base_url: String) -> Option<String> {
    tokio::task::spawn_blocking(move || cdn_gguf_path(cache_dir.as_deref(), &base_url, FILENAME))
        .await
        .unwrap()
}

mod test_cdn_download {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_downloads_and_caches_the_gguf() {
        let tmp = tempfile::tempdir().unwrap();
        let (_server, base) = bucket(b"ggufbody").await;
        let path = cdn(Some(tmp.path().to_string_lossy().into_owned()), base).await;

        let expected = tmp.path().join(FILENAME).to_string_lossy().into_owned();
        assert_eq!(path.as_deref(), Some(expected.as_str()));
        assert_eq!(std::fs::read(path.unwrap()).unwrap(), b"ggufbody");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_cached_file_short_circuits_the_network() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(FILENAME), b"already here").unwrap();

        // Any network hit despite a cached GGUF would fail the request.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;

        let path = cdn(
            Some(tmp.path().to_string_lossy().into_owned()),
            format!("{}/embeddings", server.uri()),
        )
        .await;
        let expected = tmp.path().join(FILENAME).to_string_lossy().into_owned();
        assert_eq!(path.as_deref(), Some(expected.as_str()));
        server.verify().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_truncated_download_is_rejected_and_leaves_nothing_behind() {
        // Server promised 100 bytes, delivered 8 — a resumed-but-incomplete
        // GGUF would fail deep inside llama.cpp instead of here.
        let tmp = tempfile::tempdir().unwrap();
        let base = short_body_server(b"ggufbody", 100);
        let path = cdn(Some(tmp.path().to_string_lossy().into_owned()), base).await;
        assert!(path.is_none());

        assert!(listdir(tmp.path()).is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_failed_request_falls_through_without_a_partial_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = cdn(Some(tmp.path().to_string_lossy().into_owned()), unreachable_base_url()).await;
        assert!(path.is_none());

        assert!(listdir(tmp.path()).is_empty());
    }

    /// No workspace cache to write into.
    #[test]
    fn test_skipped_without_a_cache_dir_or_base_url_no_cache_dir() {
        assert!(cdn_gguf_path(None, BASE_URL, FILENAME).is_none());
    }

    /// Explicitly opted out of the bucket.
    #[test]
    fn test_skipped_without_a_cache_dir_or_base_url_no_base_url() {
        assert!(cdn_gguf_path(Some("/tmp"), "", FILENAME).is_none());
    }
}

mod test_hf_cache_reuse {
    use super::*;

    #[test]
    fn test_finds_a_gguf_already_in_the_hf_cache_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join("models--gpustack--bge-m3-GGUF").join("snapshots").join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join(FILENAME), b"x").unwrap();

        let expected = snapshot.join(FILENAME).to_string_lossy().into_owned();
        assert_eq!(
            hf_cached_gguf_path(Some(tmp.path().to_str().unwrap()), REPO, FILENAME).as_deref(),
            Some(expected.as_str())
        );
    }

    #[test]
    fn test_absent_cache_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(hf_cached_gguf_path(Some(tmp.path().to_str().unwrap()), REPO, FILENAME).is_none());
        assert!(hf_cached_gguf_path(None, REPO, FILENAME).is_none());
    }
}

/// Stands in for `llama_cpp.Llama`: records the load parameters and hands
/// back an inert backend.
#[derive(Clone, Default)]
struct FakeLlama {
    calls: Arc<Mutex<Vec<LlamaLoadParams>>>,
}

struct InertBackend;

impl EmbedBackend for InertBackend {
    fn embed(&mut self, batch: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(batch.iter().map(|_| vec![1.0]).collect())
    }
}

impl FakeLlama {
    fn load(
        &self,
        cache_dir: &Path,
        base_url: &str,
    ) -> Option<smartloop::db::llama_embeddings::LlamaEmbedder> {
        let calls = self.calls.clone();
        load_embedder_with(
            Some(cache_dir.to_str().unwrap()),
            REPO,
            FILENAME,
            "cpu",
            base_url,
            &move |params: &LlamaLoadParams| {
                calls.lock().push(params.clone());
                Ok(Box::new(InertBackend) as Box<dyn EmbedBackend>)
            },
        )
    }
}

mod test_resolution_order {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_bucket_supplies_the_gguf_without_touching_the_hub() {
        let tmp = tempfile::tempdir().unwrap();
        let (_server, base) = bucket(b"gguf").await;
        let fake = FakeLlama::default();
        let cache = tmp.path().to_path_buf();
        let loader = fake.clone();
        let embedder = tokio::task::spawn_blocking(move || loader.load(&cache, &base).is_some())
            .await
            .unwrap();

        assert!(embedder);
        let calls = fake.calls.lock();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].model_path, tmp.path().join(FILENAME).to_string_lossy());
    }

    /// The Hub is never called — it is what the token requirement came from.
    ///
    /// With no GGUF reachable the loader returns None and retrieval degrades
    /// to TF-IDF, rather than reintroducing a token-gated download.
    #[test]
    fn test_bucket_outage_falls_back_to_tfidf_not_the_hub() {
        let tmp = tempfile::tempdir().unwrap();
        let fake = FakeLlama::default();
        let embedder = fake.load(tmp.path(), &unreachable_base_url());

        assert!(embedder.is_none());
        assert!(fake.calls.lock().is_empty());
    }
}
