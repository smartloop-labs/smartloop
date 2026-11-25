//! Test doubles for the model downloader (unit: model).
//!
//! The Python tests monkeypatch `httpx.stream` / `httpx.head` with scripted
//! fakes and build downloaders with `SLPArtifactDownloader.__new__`; these are
//! the Rust stand-ins, injected through `SLPArtifactDownloader::with_http`.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use smartloop::model::downloader::artifact_downloader::{
    DownloadHttp, HeadResponse, Headers, HttpError, SLPArtifactDownloader, StreamResponse, StreamTimeout,
    TransportErrorKind,
};
use smartloop::model::downloader::cancellation::CancelToken;
use smartloop::model::downloader::slp_downloader::SharedLogger;
use smartloop::utils::log_utils::LoggerLike;

/// `logging.getLogger("test")`: messages go to stderr (captured by the test
/// harness), nothing else.
#[derive(Debug, Default, Clone)]
pub struct TestLogger;

impl LoggerLike for TestLogger {
    fn info(&self, msg: &str) {
        eprintln!("INFO {msg}");
    }
    fn warning(&self, msg: &str) {
        eprintln!("WARNING {msg}");
    }
    fn debug(&self, msg: &str) {
        eprintln!("DEBUG {msg}");
    }
    fn error(&self, msg: &str) {
        eprintln!("ERROR {msg}");
    }
}

/// A shared [`TestLogger`].
pub fn test_logger() -> SharedLogger {
    Arc::new(TestLogger)
}

/// `httpx.ReadError(msg)`.
pub fn read_error(msg: &str) -> HttpError {
    HttpError::transport(TransportErrorKind::Read, msg)
}

/// `httpx.ConnectError(msg)`.
pub fn connect_error(msg: &str) -> HttpError {
    HttpError::transport(TransportErrorKind::Connect, msg)
}

/// `httpx.ReadTimeout(msg)`.
pub fn read_timeout(msg: &str) -> HttpError {
    HttpError::transport(TransportErrorKind::ReadTimeout, msg)
}

/// `httpx.RemoteProtocolError(msg)`.
pub fn remote_protocol_error(msg: &str) -> HttpError {
    HttpError::transport(TransportErrorKind::RemoteProtocol, msg)
}

/// `httpx.HTTPStatusError(..., response=httpx.Response(code))`.
pub fn status_error(code: u16) -> HttpError {
    HttpError::status(code, "error")
}

/// Runs before a body chunk is handed out, with the chunk's index.
pub type ChunkHook = Box<dyn Fn(usize) + Send>;

/// A stand-in for an httpx streaming response.  Dropping it marks it closed
/// (the `__exit__` of `with httpx.stream(...)`).
pub struct FakeResp {
    status_code: u16,
    chunks: VecDeque<Vec<u8>>,
    /// Index at which to fail with `exc` instead of yielding.
    raise_at: Option<usize>,
    exc: Option<HttpError>,
    index: usize,
    on_chunk: Option<ChunkHook>,
    closed: Arc<AtomicBool>,
    headers: Headers,
}

impl FakeResp {
    /// A response with `status_code` streaming `chunks`.
    pub fn new(status_code: u16, chunks: &[&[u8]]) -> Self {
        FakeResp {
            status_code,
            chunks: chunks.iter().map(|c| c.to_vec()).collect(),
            raise_at: None,
            exc: None,
            index: 0,
            on_chunk: None,
            closed: Arc::new(AtomicBool::new(false)),
            headers: Headers::new(),
        }
    }

    /// Replace the status code.
    pub fn with_status(mut self, status_code: u16) -> Self {
        self.status_code = status_code;
        self
    }

    /// Run `hook` with each chunk's index before handing it out.
    pub fn with_on_chunk(mut self, hook: impl Fn(usize) + Send + 'static) -> Self {
        self.on_chunk = Some(Box::new(hook));
        self
    }

    /// Add a response header.
    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.insert(name.to_string(), value.to_string());
        self
    }

    /// Fail with `exc` when chunk `raise_at` would be yielded.
    pub fn failing_at(mut self, raise_at: usize, exc: HttpError) -> Self {
        self.raise_at = Some(raise_at);
        self.exc = Some(exc);
        self
    }

    /// A response that trips `token` partway through the body (the cancel
    /// tests' `_CancellingResp`): the chunk at `cancel_at` is still yielded.
    pub fn cancelling(chunks: &[&[u8]], token: &CancelToken, cancel_at: usize) -> Self {
        let token = token.clone();
        let mut resp = FakeResp::new(200, chunks);
        resp.on_chunk = Some(Box::new(move |i| {
            if i == cancel_at {
                token.cancel();
            }
        }));
        resp
    }

    /// A flag that turns true once the response has been closed.
    pub fn closed_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.closed)
    }
}

impl Drop for FakeResp {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}

impl StreamResponse for FakeResp {
    fn status_code(&self) -> u16 {
        self.status_code
    }

    fn raise_for_status(&self) -> Result<(), HttpError> {
        if self.status_code >= 400 {
            return Err(status_error(self.status_code));
        }
        Ok(())
    }

    fn next_chunk(&mut self, _chunk_size: usize) -> Result<Option<Vec<u8>>, HttpError> {
        let index = self.index;
        if self.raise_at == Some(index) {
            if let Some(exc) = self.exc.clone() {
                return Err(exc);
            }
        }
        let Some(chunk) = self.chunks.pop_front() else {
            return Ok(None);
        };
        self.index += 1;
        if let Some(hook) = &self.on_chunk {
            hook(index);
        }
        Ok(Some(chunk))
    }

    fn header(&self, name: &str) -> Option<String> {
        smartloop::model::downloader::artifact_downloader::header_get(&self.headers, name).map(str::to_string)
    }
}

/// A HEAD replacement (`monkeypatch.setattr(ad.httpx, "head", ...)`).
pub type HeadFn = Box<dyn Fn(&str) -> Result<HeadResponse, HttpError> + Send + Sync>;

/// Replaces `httpx.stream` / `httpx.head`: hands out scripted responses and
/// records the headers and timeout of every stream call.
pub struct ScriptedHttp {
    responses: Mutex<VecDeque<FakeResp>>,
    pub header_calls: Mutex<Vec<Headers>>,
    pub timeout_calls: Mutex<Vec<StreamTimeout>>,
    head: Option<HeadFn>,
}

impl ScriptedHttp {
    /// Stream calls answer with `responses` in order.
    pub fn new(responses: Vec<FakeResp>) -> Arc<Self> {
        Arc::new(ScriptedHttp {
            responses: Mutex::new(responses.into()),
            header_calls: Mutex::new(Vec::new()),
            timeout_calls: Mutex::new(Vec::new()),
            head: None,
        })
    }

    /// No stream responses; HEAD answered by `head`.
    pub fn with_head(head: HeadFn) -> Arc<Self> {
        Arc::new(ScriptedHttp {
            responses: Mutex::new(VecDeque::new()),
            header_calls: Mutex::new(Vec::new()),
            timeout_calls: Mutex::new(Vec::new()),
            head: Some(head),
        })
    }

    /// Stream calls answer with `responses` in order; HEAD answered by `head`.
    pub fn with_head_and_responses(head: HeadFn, responses: Vec<FakeResp>) -> Arc<Self> {
        Arc::new(ScriptedHttp {
            responses: Mutex::new(responses.into()),
            header_calls: Mutex::new(Vec::new()),
            timeout_calls: Mutex::new(Vec::new()),
            head: Some(head),
        })
    }

    /// How many stream calls were made.
    pub fn calls(&self) -> usize {
        self.header_calls.lock().len()
    }
}

impl DownloadHttp for ScriptedHttp {
    fn head(&self, url: &str) -> Result<HeadResponse, HttpError> {
        match &self.head {
            Some(head) => head(url),
            None => panic!("unexpected HEAD request to {url}"),
        }
    }

    fn stream(&self, url: &str, headers: &Headers, timeout: StreamTimeout) -> Result<Box<dyn StreamResponse>, HttpError> {
        self.header_calls.lock().push(headers.clone());
        self.timeout_calls.lock().push(timeout);
        match self.responses.lock().pop_front() {
            Some(resp) => Ok(Box::new(resp)),
            None => panic!("no scripted response left for {url}"),
        }
    }
}

/// `{"Range": value}` as a recorded header map.
pub fn range_header(value: &str) -> Headers {
    let mut headers = Headers::new();
    headers.insert("Range".to_string(), value.to_string());
    headers
}

/// A downloader built without `__init__` (no network, no workspace), whose
/// backoff is instant but still honours a cancelled token (the `_no_sleep`
/// fixture), talking to `http`.
pub fn downloader(cancel: Option<CancelToken>, http: Option<Arc<ScriptedHttp>>) -> SLPArtifactDownloader {
    let token = cancel.unwrap_or_default();
    token.set_instant_wait(true);
    let dl = SLPArtifactDownloader::for_tests(None, test_logger(), Some(token));
    match http {
        Some(http) => dl.with_http(http),
        None => dl,
    }
}

/// A server for one object that answers `Range` requests from its bytes, so
/// tests can run concurrent ranges without scripting their order.
pub struct RangeServer {
    pub data: Vec<u8>,
    pub etag: Mutex<String>,
    /// Bytes at or past this offset are unreachable: a body stops just
    /// before it and a request starting there fails to connect.
    pub cut_at: Mutex<Option<u64>>,
    /// Body chunk size.
    pub piece: usize,
    /// Every `Range` (and `If-Range`) asked for, in order.
    pub requests: Mutex<Vec<(String, Option<String>)>>,
}

impl RangeServer {
    /// Serve `data` with a strong ETag.
    pub fn new(data: Vec<u8>) -> Arc<Self> {
        Arc::new(RangeServer {
            data,
            etag: Mutex::new("\"v1\"".to_string()),
            cut_at: Mutex::new(None),
            piece: 4,
            requests: Mutex::new(Vec::new()),
        })
    }

    /// The `Range` headers requested, in order.
    pub fn ranges(&self) -> Vec<String> {
        self.requests.lock().iter().map(|(r, _)| r.clone()).collect()
    }

    /// What a HEAD would report.
    pub fn info(&self) -> smartloop::model::downloader::artifact_downloader::RemoteFileInfo {
        smartloop::model::downloader::artifact_downloader::RemoteFileInfo {
            size: self.data.len() as u64,
            accept_ranges: true,
            etag: Some(self.etag.lock().clone()),
            ..Default::default()
        }
    }
}

impl DownloadHttp for RangeServer {
    fn head(&self, _url: &str) -> Result<HeadResponse, HttpError> {
        let mut headers = Headers::new();
        headers.insert("content-length".to_string(), self.data.len().to_string());
        headers.insert("accept-ranges".to_string(), "bytes".to_string());
        headers.insert("etag".to_string(), self.etag.lock().clone());
        Ok(HeadResponse { status_code: 200, headers })
    }

    fn stream(&self, _url: &str, headers: &Headers, _timeout: StreamTimeout) -> Result<Box<dyn StreamResponse>, HttpError> {
        let range = headers.get("Range").cloned().unwrap_or_default();
        let if_range = headers.get("If-Range").cloned();
        self.requests.lock().push((range.clone(), if_range.clone()));
        let total = self.data.len() as u64;
        let etag = self.etag.lock().clone();
        let (first, last) = range
            .strip_prefix("bytes=")
            .and_then(|r| r.split_once('-'))
            .map(|(a, b)| (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap_or(total - 1)))
            .unwrap_or((0, total - 1));
        if if_range.as_ref().is_some_and(|v| *v != etag) || range.is_empty() {
            let pieces: Vec<&[u8]> = self.data.chunks(self.piece).collect();
            return Ok(Box::new(FakeResp::new(200, &pieces).with_header("ETag", &etag)));
        }
        let cut = *self.cut_at.lock();
        if cut.is_some_and(|c| first >= c) {
            return Err(connect_error("unreachable"));
        }
        let stop = cut.map_or(last, |c| last.min(c - 1));
        let body = &self.data[first as usize..=stop as usize];
        let pieces: Vec<&[u8]> = body.chunks(self.piece).collect();
        let mut resp = FakeResp::new(206, &pieces)
            .with_header("Content-Range", &format!("bytes {first}-{last}/{total}"))
            .with_header("ETag", &etag);
        if stop < last {
            resp = resp.failing_at(pieces.len(), read_error("cut"));
        }
        Ok(Box::new(resp))
    }
}
