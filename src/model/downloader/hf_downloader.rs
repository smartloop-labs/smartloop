//! Port of `smartloop/model/downloader/hf_downloader.py`.
//!
//! Downloads a model repository from the HuggingFace Hub with byte-level
//! progress.  `huggingface_hub` has no Rust equivalent, so the two calls it
//! made (`HfApi.list_repo_tree(recursive=True)` and `hf_hub_download`) are
//! hand-ported over reqwest against the Hub's public HTTP API.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use parking_lot::Mutex;

use crate::model::downloader::cancellation::Event;
use crate::model::downloader::slp_downloader::{
    AggregateProgress, DownloadProgress, ProgressCallback, SLPDownloader, SharedLogger, SharedWorkspace,
};

/// Parallel file downloads (`min(4, cpu_count)`).
pub fn max_workers() -> usize {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    4.min(cpus)
}

/// Seconds between progress polls of the on-disk file sizes.
pub const POLL_INTERVAL: f64 = 0.5;

/// Base URL of the HuggingFace Hub.
pub const HF_ENDPOINT: &str = "https://huggingface.co";

/// Revision every call resolves against (`huggingface_hub`'s default).
const DEFAULT_REVISION: &str = "main";

/// `HF_HUB_DOWNLOAD_TIMEOUT` / `HF_HUB_ETAG_TIMEOUT` defaults (seconds).
const HUB_TIMEOUT: Duration = Duration::from_secs(10);

/// Read buffer for file downloads.
const DOWNLOAD_CHUNK: usize = 10 * 1024 * 1024;

/// One file entry of a repository tree (`RepoFile` with a known size).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub struct RepoFile {
    pub path: String,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(rename = "type", default)]
    pub kind: String,
}

/// The Hub endpoint: `HF_ENDPOINT` when set (as `huggingface_hub` honours
/// it), else [`HF_ENDPOINT`].
fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|e| e.trim().trim_end_matches('/').to_string())
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| HF_ENDPOINT.to_string())
}

/// The token to send: the explicit one, else the locally saved one
/// (`huggingface_hub.get_token`: `HF_TOKEN`, then `$HF_HOME/token`).
fn resolve_token(token: Option<&str>) -> Option<String> {
    if let Some(token) = token.filter(|t| !t.is_empty()) {
        return Some(token.to_string());
    }
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            return Some(token);
        }
    }
    let hf_home = std::env::var("HF_HOME").map(PathBuf::from).ok().or_else(|| {
        std::env::var("XDG_CACHE_HOME")
            .map(|c| PathBuf::from(c).join("huggingface"))
            .ok()
            .or_else(|| dirs::home_dir().map(|h| h.join(".cache").join("huggingface")))
    })?;
    let token = fs::read_to_string(hf_home.join("token")).ok()?;
    let token = token.trim().to_string();
    (!token.is_empty()).then_some(token)
}

/// `urllib.parse.quote(s)` (safe `/`): percent-encode every byte that is not
/// an ASCII letter, digit, `_.-~` or `/`.
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-' | b'~' | b'/') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn client() -> anyhow::Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .connect_timeout(HUB_TIMEOUT)
        .timeout(HUB_TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(20))
        .build()?)
}

fn with_auth(request: reqwest::blocking::RequestBuilder, token: Option<&str>) -> reqwest::blocking::RequestBuilder {
    match token {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

/// `huggingface_hub.utils.hf_raise_for_status`: an error naming the status and
/// the URL for any 4xx/5xx response.
fn raise_for_status(response: &reqwest::blocking::Response) -> anyhow::Result<()> {
    let status = response.status();
    if status.is_client_error() || status.is_server_error() {
        let kind = if status.is_client_error() { "Client Error" } else { "Server Error" };
        anyhow::bail!(
            "{} {}: {} for url: {}",
            status.as_u16(),
            kind,
            status.canonical_reason().unwrap_or(""),
            response.url()
        );
    }
    Ok(())
}

/// The `rel="next"` target of a `Link` header, if any.
fn next_link(response: &reqwest::blocking::Response) -> Option<String> {
    let header = response.headers().get(reqwest::header::LINK)?.to_str().ok()?;
    header.split(',').find_map(|part| {
        let mut pieces = part.split(';');
        let target = pieces.next()?.trim();
        let is_next = pieces.any(|p| {
            let p = p.trim().replace(' ', "");
            p == "rel=\"next\"" || p == "rel=next"
        });
        if !is_next {
            return None;
        }
        Some(target.trim_start_matches('<').trim_end_matches('>').to_string())
    })
}

/// List every entry of `repo_id`'s tree recursively (`HfApi.list_repo_tree`).
pub fn list_repo_tree(repo_id: &str, token: Option<&str>) -> anyhow::Result<Vec<RepoFile>> {
    let token = resolve_token(token);
    let client = client()?;
    let mut url = format!(
        "{}/api/models/{}/tree/{}?recursive=True&expand=False",
        endpoint(),
        repo_id,
        quote(DEFAULT_REVISION)
    );
    let mut entries = Vec::new();
    loop {
        let response = with_auth(client.get(&url), token.as_deref()).send()?;
        raise_for_status(&response)?;
        let next = next_link(&response);
        let page: Vec<RepoFile> = response.json()?;
        entries.extend(page);
        match next {
            Some(next) => url = next,
            None => break,
        }
    }
    Ok(entries)
}

/// Download one file of `repo_id` into `local_dir/<filename>`, writing to a
/// `.incomplete` sidecar first (`hf_hub_download(local_dir=...)`).
pub fn hf_hub_download(repo_id: &str, filename: &str, local_dir: &Path, token: Option<&str>) -> anyhow::Result<PathBuf> {
    let token = resolve_token(token);
    let destination = local_dir.join(filename);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let incomplete = PathBuf::from(format!("{}.incomplete", destination.display()));

    let url = format!("{}/{}/resolve/{}/{}", endpoint(), repo_id, quote(DEFAULT_REVISION), quote(filename));
    let response = with_auth(client()?.get(&url), token.as_deref()).send()?;
    raise_for_status(&response)?;

    let result = (|| -> anyhow::Result<()> {
        let mut response = response;
        let mut file = File::create(&incomplete)?;
        let mut buf = vec![0u8; DOWNLOAD_CHUNK];
        loop {
            let n = match response.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            };
            file.write_all(&buf[..n])?;
        }
        file.flush()?;
        Ok(())
    })();
    if let Err(err) = result {
        let _ = fs::remove_file(&incomplete);
        return Err(err);
    }
    fs::rename(&incomplete, &destination)?;
    Ok(destination)
}

/// HuggingFace Hub downloader.
pub struct HFDownloader {
    pub workspace: SharedWorkspace,
    pub logger: SharedLogger,
}

impl HFDownloader {
    /// New downloader for `workspace`.
    pub fn new(workspace: SharedWorkspace, logger: SharedLogger) -> Self {
        Self { workspace, logger }
    }
}

/// Run `hf_hub_download` for every pending file on up to `workers` threads,
/// returning `(path, error)` for each failure in completion order.
fn download_all(
    workers: usize,
    model_id: &str,
    pending: &[RepoFile],
    local_dir: &Path,
    token: Option<&str>,
    logger: &SharedLogger,
) -> Vec<String> {
    let queue: Mutex<VecDeque<&RepoFile>> = Mutex::new(pending.iter().collect());
    let (tx, rx) = mpsc::channel::<(String, anyhow::Result<PathBuf>)>();
    let threads = workers.max(1).min(pending.len().max(1));
    let mut failed = Vec::new();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let tx = tx.clone();
            let queue = &queue;
            scope.spawn(move || loop {
                let Some(file) = queue.lock().pop_front() else { break };
                let result = hf_hub_download(model_id, &file.path, local_dir, token);
                let _ = tx.send((file.path.clone(), result));
            });
        }
        drop(tx);
        for (filename, result) in rx.iter() {
            if let Err(e) = result {
                logger.error(&format!("[error] {filename}: {e}"));
                failed.push(filename);
            }
        }
    });
    failed
}

impl SLPDownloader for HFDownloader {
    /// Download model from HuggingFace Hub with byte-level progress.
    fn download(&self, model_id: &str, progress_callback: Option<ProgressCallback>) -> anyhow::Result<PathBuf> {
        // Progress is reported via our own callback; nothing here draws bars.
        self.logger.info(&format!("Downloading model: {model_id} from HuggingFace Hub"));
        let models_dir = self.workspace.read().models_dir.clone();
        let local_dir = Path::new(&models_dir).join(model_id);
        fs::create_dir_all(&local_dir)?;

        let token = {
            let ws = self.workspace.read();
            ws.get_model(None).and_then(|m| m.config).and_then(|c| c.hf_token)
        };

        let repo_files = list_repo_tree(model_id, token.as_deref())?;
        let all_files: Vec<RepoFile> = repo_files.into_iter().filter(|f| f.size.is_some()).collect();

        let mut pending: Vec<RepoFile> = Vec::new();
        let mut skipped = 0usize;
        for f in all_files {
            let local_path = local_dir.join(&f.path);
            let up_to_date = local_path.is_file()
                && fs::metadata(&local_path).map(|m| Some(m.len()) == f.size).unwrap_or(false);
            if up_to_date {
                skipped += 1;
            } else {
                pending.push(f);
            }
        }

        if skipped > 0 {
            self.logger.info(&format!("[ok] {skipped} file(s) already up-to-date, skipped"));
        }

        if pending.is_empty() {
            self.logger.info(&format!("[ok] All files already downloaded at: {}", local_dir.display()));
            return Ok(local_dir);
        }

        let total_bytes: u64 = pending.iter().map(|f| f.size.unwrap_or(0)).sum();
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        self.logger.info(&format!("Downloading {} file(s), {total_mb:.1} MB total", pending.len()));

        // Roll the parallel per-file progress up into a single 0->100% bar so
        // the client sees one download, not the files fighting over the meter.
        let aggregate = AggregateProgress::new(total_bytes, progress_callback.clone(), model_id);

        let done = Event::new();

        let result = std::thread::scope(|scope| {
            let poll_progress = || {
                let mut last_reported: HashMap<String, u64> = HashMap::new();
                while !done.is_set() {
                    for f in &pending {
                        let size = f.size.unwrap_or(0);
                        let local_path = local_dir.join(&f.path);
                        let incomplete = PathBuf::from(format!("{}.incomplete", local_path.display()));
                        let mut file_downloaded = 0u64;
                        for p in [&local_path, &incomplete] {
                            if let Ok(meta) = fs::metadata(p) {
                                file_downloaded = meta.len().min(size);
                                break;
                            }
                        }
                        let prev = last_reported.get(&f.path).copied().unwrap_or(0);
                        if progress_callback.is_some() && file_downloaded > 0 && file_downloaded != prev {
                            last_reported.insert(f.path.clone(), file_downloaded);
                            let name = Path::new(&f.path)
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_else(|| f.path.clone());
                            aggregate.call(DownloadProgress::new(name, file_downloaded, size, "downloading"));
                        }
                    }
                    done.wait(POLL_INTERVAL);
                }
            };
            let monitor = scope.spawn(poll_progress);

            let outcome = (|| -> anyhow::Result<PathBuf> {
                let failed = download_all(max_workers(), model_id, &pending, &local_dir, token.as_deref(), &self.logger);

                if !failed.is_empty() {
                    anyhow::bail!("Failed to download {} file(s): {}", failed.len(), failed.join(", "));
                }

                aggregate.finish();

                self.logger.info(&format!("[ok] Model downloaded to: {}", local_dir.display()));
                Ok(local_dir.clone())
            })();

            done.set();
            let _ = monitor.join();
            outcome
        });

        if let Err(e) = &result {
            // `except (OSError, ValueError)`: only local I/O and bad-value
            // failures are logged here; a failed batch was already logged per
            // file above.
            if e.downcast_ref::<std::io::Error>().is_some() {
                self.logger.error(&format!("[error] Error downloading model: {e}"));
            }
        }
        result
    }
}
