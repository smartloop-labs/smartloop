//! Port of `tests/unit/test_web_search_gate.py`.
//!
//! Web search must not be cancelled by an unrelated local retrieval hit.
//!
//! Vector/feedback retrieval always returns its nearest neighbours, related or
//! not, so gating the search on "did anything match locally?" let one indexed
//! document or one stored feedback Q&A silence it for every query.  Which of
//! the collected sources actually answers the question is decided downstream.
//!
//! The Python tests monkeypatched the explorer's search helpers; here the same
//! results come from the explorer's injectable collaborators — a fake web
//! searcher (`web_search_agent`), a fake model service (the duck-typed
//! `model_service` the searches call), a fake LLM for the follow-up gate, and
//! for MCP a fake resolver plus the tool caller's transport seam.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use serde_json::{Map, Value, json};

use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
use smartloop::agents::mcp::explorer::{
    ConversationTurn, DocumentGroups, Explorer, ExplorerSearchService, ProgressCallback, SearchContext, ToolResolver,
    WebSearcher,
};
use smartloop::agents::mcp::mcp_agent::ResolvedCall;
use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
use smartloop::agents::mcp::result_mapper::MCPResultMapper;
use smartloop::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationStream};
use smartloop::mcp::client::MCPToolCallResult;
use smartloop::parsers::DocumentChunk;
use smartloop::workspace::workspace::{MCPServer, MCPToolDefinition};

// ---------------------------------------------------------------- fakes

/// The explorer's model (`MagicMock()` / the follow-up gate's mock LLM).
///
/// `verdict` is what `invoke` answers; `None` makes the call fail
/// (`side_effect = RuntimeError("model unavailable")`).  Every prompt the
/// model is asked is recorded.
struct GateLlm {
    verdict: Option<String>,
    prompts: Mutex<Vec<String>>,
}

impl GateLlm {
    fn new(verdict: Option<&str>) -> Arc<Self> {
        Arc::new(Self { verdict: verdict.map(str::to_string), prompts: Mutex::new(Vec::new()) })
    }
}

impl BaseLLM for GateLlm {
    fn project_id(&self) -> &str {
        "p"
    }
    fn get_num_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }
    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }
    fn get_max_tokens(&self) -> usize {
        4096
    }
    fn call(&self, prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
        self.prompts.lock().push(prompt.to_string());
        match &self.verdict {
            Some(verdict) => Ok(verdict.clone()),
            None => Err(anyhow::anyhow!("model unavailable")),
        }
    }
    fn invoke(&self, prompt: &str, max_tokens: Option<usize>, thinking: Option<bool>) -> String {
        // A mock's `invoke` returns its value verbatim; a failure surfaces the
        // way the real wrapper reports one.
        match self.call(prompt, None, CallOptions { max_tokens, thinking }) {
            Ok(verdict) => verdict,
            Err(e) => format!("Error generating response: {e}"),
        }
    }
    fn stream<'a>(&'a self, _prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
        Box::new(std::iter::empty())
    }
    fn cancelled(&self) -> Option<CancelEvent> {
        None
    }
    fn set_cancelled(&self, _event: Option<CancelEvent>) {}
    fn llm_type(&self) -> &str {
        "mock"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `web_search_agent` with `is_enabled`, `last_sources` and `_search_web`
/// patched: every query searched is recorded in `searched`.
struct FakeWebSearch {
    enabled: bool,
    searched: Arc<Mutex<Vec<String>>>,
}

impl WebSearcher for FakeWebSearch {
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn search(&self, query: &str, _progress: Option<ProgressCallback>) -> anyhow::Result<String> {
        self.searched.lock().push(query.to_string());
        Ok("X2D II: 100MP medium format.".to_string())
    }
    fn last_sources(&self) -> Vec<String> {
        vec!["http://x".to_string()]
    }
    fn last_source_extracts(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// The model service the local searches query; empty unless a test sets a hit.
#[derive(Default)]
struct FakeSearchService {
    vector: Option<DocumentGroups>,
    feedback: Vec<DocumentChunk>,
    workspace: Vec<DocumentChunk>,
}

impl ExplorerSearchService for FakeSearchService {
    fn get_document_groups(
        &self,
        _query: Option<&str>,
        _skill_instruction: Option<&str>,
        _conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Option<DocumentGroups>> {
        Ok(self.vector.clone())
    }
    fn search_feedback(&self, _query: &str, _n_results: usize) -> anyhow::Result<Vec<DocumentChunk>> {
        Ok(self.feedback.clone())
    }
    fn search_workspace(
        &self,
        _query: &str,
        _work_dir: &str,
        _n_results: usize,
        _conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Vec<DocumentChunk>> {
        Ok(self.workspace.clone())
    }
}

fn chunk(document_id: &str, text: &str) -> DocumentChunk {
    DocumentChunk { document_id: document_id.to_string(), text: text.to_string(), ..DocumentChunk::default() }
}

/// `_explorer(monkeypatch, searched, enabled)`.
fn explorer(searched: &Arc<Mutex<Vec<String>>>, enabled: bool) -> Explorer {
    let llm: Arc<dyn BaseLLM> = GateLlm::new(Some(""));
    let mut ex = Explorer::new(llm, None, None, false);
    ex.web_search_agent = Arc::new(FakeWebSearch { enabled, searched: searched.clone() });
    ex
}

/// `collect_parallel(query, model_service=..., cwd=..., mcp_servers=...,
/// conversation_history=..., project_dir=...)` with every other argument at
/// its default.
fn collect(
    ex: &Explorer,
    query: &str,
    service: FakeSearchService,
    cwd: Option<&str>,
    mcp_servers: Option<&[MCPServer]>,
    history: Option<&[ConversationTurn]>,
    project_dir: Option<&str>,
) -> SearchContext {
    let service: Arc<dyn ExplorerSearchService> = Arc::new(service);
    ex.collect_parallel(query, Some(service), cwd, None, None, mcp_servers, None, history, None, project_dir, None, None)
}

fn searched_list(searched: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    searched.lock().clone()
}

fn turn(role: &str, content: &str) -> ConversationTurn {
    HashMap::from([("role".to_string(), role.to_string()), ("content".to_string(), content.to_string())])
}

// ---------------------------------------------------------------- tests

/// Collect never searches the web: the model does, when it asks to, through
/// the client's `web_search` tool.  Not even with search on for the project
/// and nothing local to answer from.
mod collect_never_searches {
    use super::*;

    #[test]
    fn test_no_local_hit_does_not_search() {
        let searched = Arc::new(Mutex::new(Vec::new()));
        let ex = explorer(&searched, true);

        let ctx = collect(&ex, "what is the Hasselblad X2D II", FakeSearchService::default(), None, None, None, None);

        assert!(searched_list(&searched).is_empty());
        assert!(ctx.web_context.is_empty());
        assert!(ctx.web_sources.is_empty());
    }

    #[test]
    fn test_local_hits_are_still_collected() {
        let searched = Arc::new(Mutex::new(Vec::new()));
        let ex = explorer(&searched, true);
        let service = FakeSearchService {
            feedback: vec![chunk("f1", "Q: cat breeds?\nA: Siberian cats shed little.")],
            workspace: vec![chunk("notes.md", "old notes")],
            ..Default::default()
        };

        let ctx = collect(&ex, "latest camera releases", service, Some("/tmp"), None, None, None);

        assert!(searched_list(&searched).is_empty());
        assert!(ctx.web_context.is_empty());
        assert!(!ctx.feedback_context.is_empty());
    }

    #[test]
    fn test_a_follow_up_does_not_search() {
        let searched = Arc::new(Mutex::new(Vec::new()));
        let mut ex = explorer(&searched, true);
        let llm: Arc<dyn BaseLLM> = GateLlm::new(Some("SEARCH"));
        ex.llm = Some(llm);
        let history = vec![
            turn("user", "Show me latest local events"),
            turn("assistant", "Jazz in the Park on Friday."),
        ];

        let ctx =
            collect(&ex, "what about next weekend?", FakeSearchService::default(), None, None, Some(&history), None);

        assert!(searched_list(&searched).is_empty());
        assert!(ctx.web_context.is_empty());
    }
}

mod mcp_alongside_no_search {
    use super::*;

    /// Resolves every directive to one read-only tool on the server asked.
    struct FakeResolver;

    impl ToolResolver for FakeResolver {
        fn resolve(
            &self,
            _directive: &str,
            mcp_servers: &[MCPServer],
            _context: Option<&str>,
            _llm: Option<&dyn BaseLLM>,
        ) -> Option<ResolvedCall> {
            let server = mcp_servers.first()?;
            Some(ResolvedCall { reasoned: true, ..ResolvedCall::new(server.id.clone(), "list_issues") })
        }
    }

    /// A connected server is still asked; the web is not.
    #[test]
    fn test_mcp_tool_result_is_collected_without_a_web_search() {
        let searched = Arc::new(Mutex::new(Vec::new()));
        let mut ex = explorer(&searched, true);

        // The MCP leg answers "MCP TOOL (...list_issues):\n#1 open".
        let tool_llm: Arc<dyn BaseLLM> = GateLlm::new(Some("{}"));
        let arg_extractor = Arc::new(MCPArgumentExtractor::new(tool_llm));
        let tool_caller = Arc::new(MCPToolCaller::new(
            arg_extractor.clone(),
            Arc::new(Mutex::new(MCPResultMapper::new())),
            None,
        ));
        tool_caller.set_call_tool_hook(Some(Arc::new(
            |_server: &MCPServer,
             _tool: &str,
             _args: Option<&Map<String, Value>>,
             _token: Option<&str>,
             _cwd: Option<&str>| MCPToolCallResult {
                success: true,
                content: Some(vec![json!({"type": "text", "text": "#1 open"})]),
                is_error: false,
                error: None,
                auth_expired: false,
            },
        )));
        ex.tool_caller = Some(tool_caller);
        ex.arg_extractor = Some(arg_extractor);
        ex.mcp_agent = Arc::new(FakeResolver);
        ex.follow_up_calls = Arc::new(|_server: &MCPServer, _tool: &str, _payload: &str| Vec::new());

        let server = MCPServer {
            id: "issues".to_string(),
            name: "issues".to_string(),
            tools: vec![MCPToolDefinition { name: "list_issues".to_string(), description: None, input_schema: None, ..Default::default() }],
            ..MCPServer::default()
        };
        let servers = [server];

        let ctx = collect(
            &ex,
            "list my issues",
            FakeSearchService::default(),
            None,
            Some(&servers),
            None,
            Some("/tmp/p"),
        );

        assert!(searched_list(&searched).is_empty());
        assert!(ctx.web_context.is_empty());
        assert!(!ctx.mcp_context.is_empty());
    }

    #[test]
    fn test_disabled_project_never_searches() {
        let searched = Arc::new(Mutex::new(Vec::new()));
        let ex = explorer(&searched, false);

        let ctx = collect(&ex, "what is the Hasselblad X2D II", FakeSearchService::default(), None, None, None, None);

        assert!(searched_list(&searched).is_empty());
        assert_eq!(ctx.web_context, "");
    }
}

/// Feedback retrieval applies the same quality floor as documents.
///
/// It was the one path with no floor — recall to a fixed 0.65 distance and
/// return everything — so unrelated stored Q&A landed in the prompt and the
/// model answered from it.
mod feedback_relevance_floor {
    use smartloop::api::services::model_service::ModelService;
    use smartloop::db::feedback_indexer::FeedbackIndexer;
    use smartloop::db::store::{Chunk, ChunkResult, VectorStore};
    use smartloop::model::model_manager::Model;
    use smartloop::workspace::workspace::{Project, Workspace};

    use super::*;

    /// A store whose every search answers `chunks` (`indexer.search.return_value`).
    struct FakeStore {
        chunks: Vec<ChunkResult>,
    }

    impl VectorStore for FakeStore {
        fn project_dir(&self) -> &str {
            "/tmp/p"
        }
        fn collection_name(&self) -> &str {
            "feedback"
        }
        fn base_dir(&self) -> Option<&str> {
            None
        }
        fn add_chunks(&self, _chunks: &[Chunk]) -> anyhow::Result<()> {
            Ok(())
        }
        fn query(&self, _query_text: &str, _n_results: Option<usize>) -> anyhow::Result<Vec<String>> {
            Ok(self.chunks.iter().map(|c| c.text.clone()).collect())
        }
        fn query_with_metadata(
            &self,
            _query_text: &str,
            _n_results: Option<usize>,
            _max_distance: f64,
            _where: Option<&Map<String, Value>>,
        ) -> anyhow::Result<Vec<ChunkResult>> {
            Ok(self.chunks.clone())
        }
        fn delete_document(&self, _document_id: &str) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn service(chunks: Vec<ChunkResult>) -> Arc<ModelService> {
        let svc = ModelService::new_bare();
        let project = Project {
            id: "p1".to_string(),
            project_dir: "/tmp/p".to_string(),
            current: true,
            ..Project::default()
        };
        let mut workspace = Workspace::new("ws");
        workspace.projects = vec![project];
        *svc.model.write() = Some(Arc::new(Model::new("m", Arc::new(RwLock::new(workspace)), None)));

        let indexer = FeedbackIndexer::with_store(Arc::new(FakeStore { chunks }));
        svc.caches("p1").lock().feedback_indexer = Some(Arc::new(indexer));
        svc
    }

    fn chunk(distance: f64) -> ChunkResult {
        ChunkResult {
            text: format!("Q: q?\nA: a. ({distance})"),
            document_id: "f1".to_string(),
            document_name: Some("feedback/s/m".to_string()),
            distance,
            metadata: Map::new(),
            source_content: Some(String::new()),
            ..ChunkResult::default()
        }
    }

    #[test]
    fn test_weak_best_match_returns_nothing() {
        let svc = service(vec![chunk(0.62)]);

        assert!(svc.search_feedback("what is the X2D II", 5).is_empty());
    }

    #[test]
    fn test_strong_match_kept_and_far_neighbours_trimmed() {
        let svc = service(vec![chunk(0.20), chunk(0.28), chunk(0.50)]);

        let kept = svc.search_feedback("refund window", 5);

        assert_eq!(kept.iter().map(|c| c.distance).collect::<Vec<_>>(), vec![0.20, 0.28]);
    }

    #[test]
    fn test_caps_at_requested_n_results() {
        let svc = service((0..10).map(|_| chunk(0.10)).collect());

        assert_eq!(svc.search_feedback("refund window", 3).len(), 3);
    }
}

/// The status check sits in front of every query, so it must be cheap and
/// must never suppress a search it cannot verify.
mod hosted_status_cache {
    use serial_test::serial;
    use smartloop::search::providers::{self, WebSearchCallError, WebSearchClient};

    use super::*;

    /// `setup_method` / `teardown_method` plus monkeypatch undo: forget the
    /// cached status and restore the real credential check and client.
    pub(super) struct Reset;

    impl Reset {
        pub(super) fn new() -> Self {
            providers::reset_status_cache();
            Reset
        }
    }

    impl Drop for Reset {
        fn drop(&mut self) {
            providers::reset_status_cache();
            providers::set_credentials_check(None);
            providers::set_platform_client_factory(None);
        }
    }

    pub(super) fn signed_in(has_credentials: bool) {
        providers::set_credentials_check(Some(Arc::new(move || has_credentials)));
    }

    type StatusFn = Arc<dyn Fn() -> Result<bool, WebSearchCallError> + Send + Sync>;

    /// A stub platform client whose status call is `status`.
    struct Stub {
        status: StatusFn,
        calls: Option<Arc<Mutex<usize>>>,
    }

    impl WebSearchClient for Stub {
        fn web_search(&self, _query: &str, _count: usize) -> Result<Vec<String>, WebSearchCallError> {
            unreachable!("only the status endpoint is stubbed")
        }
        fn web_search_status(&self) -> Result<bool, WebSearchCallError> {
            if let Some(calls) = &self.calls {
                *calls.lock() += 1;
            }
            (self.status)()
        }
    }

    /// Point providers at a stub platform client whose status call is `status`.
    fn client(status: StatusFn, calls: Option<Arc<Mutex<usize>>>) {
        providers::set_platform_client_factory(Some(Arc::new(move || {
            Box::new(Stub { status: status.clone(), calls: calls.clone() }) as Box<dyn WebSearchClient>
        })));
    }

    #[test]
    #[serial]
    fn test_result_is_cached_across_calls() {
        let _reset = Reset::new();
        let calls = Arc::new(Mutex::new(0));
        signed_in(true);
        client(Arc::new(|| Ok(true)), Some(calls.clone()));

        assert!(providers::hosted_search_available(None));
        assert!(providers::hosted_search_available(None));
        assert_eq!(*calls.lock(), 1, "second call should come from the cache");
    }

    #[test]
    #[serial]
    fn test_signed_out_short_circuits_without_a_request() {
        let _reset = Reset::new();
        signed_in(false);

        providers::set_platform_client_factory(Some(Arc::new(|| -> Box<dyn WebSearchClient> {
            panic!("must not call the endpoint without a token")
        })));

        assert!(!providers::hosted_search_available(None));
    }

    /// A health check that cannot complete says nothing about whether search
    /// works, and the search itself already skips on a 403.
    #[test]
    #[serial]
    fn test_a_failed_check_fails_open() {
        let _reset = Reset::new();
        signed_in(true);
        client(Arc::new(|| Err(WebSearchCallError::Other(anyhow::anyhow!("network down")))), None);

        assert!(providers::hosted_search_available(None));
    }

    /// Only an explicit 401/403 — surfaced by the client as a permission
    /// error — counts as a real "search is off".
    #[test]
    #[serial]
    fn test_forbidden_reports_unavailable() {
        let _reset = Reset::new();
        signed_in(true);
        client(Arc::new(|| Err(WebSearchCallError::Permission("HTTP 403".to_string()))), None);

        assert!(!providers::hosted_search_available(None));
    }
}

/// The search call itself goes through `SLPPlatformClient`; what stays here is
/// the translation from the client's errors to the agent's skip signals.
mod hosted_search_provider {
    use serial_test::serial;
    use smartloop::search::providers::{self, HostedSearchProvider, SearchProviderError, WebSearchCallError, WebSearchClient};

    use super::hosted_status_cache::{Reset, signed_in};
    use super::*;

    type SearchFn = Arc<dyn Fn(&str, usize) -> Result<Vec<String>, WebSearchCallError> + Send + Sync>;

    struct Stub {
        search: SearchFn,
    }

    impl WebSearchClient for Stub {
        fn web_search(&self, query: &str, count: usize) -> Result<Vec<String>, WebSearchCallError> {
            (self.search)(query, count)
        }
        fn web_search_status(&self) -> Result<bool, WebSearchCallError> {
            unreachable!("only the search endpoint is stubbed")
        }
    }

    fn client(search: SearchFn) {
        providers::set_platform_client_factory(Some(Arc::new(move || {
            Box::new(Stub { search: search.clone() }) as Box<dyn WebSearchClient>
        })));
    }

    #[test]
    #[serial]
    fn test_returns_links() {
        let _reset = Reset::new();
        signed_in(true);
        let seen: Arc<Mutex<Option<(String, usize)>>> = Arc::new(Mutex::new(None));

        let seen_in = seen.clone();
        client(Arc::new(move |query: &str, count: usize| {
            *seen_in.lock() = Some((query.to_string(), count));
            Ok(vec!["http://a".to_string(), "http://b".to_string()])
        }));

        assert_eq!(HostedSearchProvider::new().search("bikes", 5).unwrap(), vec!["http://a", "http://b"]);
        assert_eq!(*seen.lock(), Some(("bikes".to_string(), 5)));
    }

    #[test]
    #[serial]
    fn test_count_is_clamped() {
        let _reset = Reset::new();
        signed_in(true);
        let counts = Arc::new(Mutex::new(Vec::new()));
        let counts_in = counts.clone();
        client(Arc::new(move |_q: &str, c: usize| {
            counts_in.lock().push(c);
            Ok(Vec::new())
        }));

        HostedSearchProvider::new().search("bikes", 99).unwrap();
        HostedSearchProvider::new().search("bikes", 0).unwrap();
        assert_eq!(*counts.lock(), vec![20, 1]);
    }

    #[test]
    #[serial]
    fn test_no_credentials_raises() {
        let _reset = Reset::new();
        signed_in(false);

        providers::set_platform_client_factory(Some(Arc::new(|| -> Box<dyn WebSearchClient> {
            panic!("must not call the endpoint without credentials")
        })));

        let err = HostedSearchProvider::new().search("bikes", 5).unwrap_err();
        assert!(matches!(err, SearchProviderError::NoSearchCredentials(_)), "{err:?}");
    }

    #[test]
    #[serial]
    fn test_permission_error_becomes_search_auth_error() {
        let _reset = Reset::new();
        signed_in(true);
        client(Arc::new(|_q: &str, _c: usize| Err(WebSearchCallError::Permission("HTTP 403".to_string()))));

        let err = HostedSearchProvider::new().search("bikes", 5).unwrap_err();
        assert!(matches!(err, SearchProviderError::SearchAuthError(_)), "{err:?}");
    }

    /// A 500 or a dropped connection must stay distinguishable from "off" —
    /// the agent skips either way, but only 401/403 means search is disabled.
    #[test]
    #[serial]
    fn test_transient_error_is_not_an_auth_error() {
        let _reset = Reset::new();
        signed_in(true);
        client(Arc::new(|_q: &str, _c: usize| Err(WebSearchCallError::Other(anyhow::anyhow!("boom")))));

        let err = HostedSearchProvider::new().search("bikes", 5).unwrap_err();
        match err {
            SearchProviderError::Other(e) => assert_eq!(e.to_string(), "boom"),
            other => panic!("expected the connection error to propagate, got {other:?}"),
        }
    }
}
