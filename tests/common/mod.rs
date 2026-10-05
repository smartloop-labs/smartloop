//! Port of `tests/conftest.py` (shared fixtures).

// ============================================================ agents_mcp_explorer
// Fixtures added by the `agents_mcp_explorer` unit.  Other units: add yours in
// your own delimited block rather than editing this one.

/// Fakes for driving `smartloop::agents::mcp::explorer::Explorer` without a
/// model, a network or an MCP transport (Python: `Explorer.__new__(Explorer)`
/// with the collaborators set by hand, plus `MagicMock`s).
pub mod explorer_fixtures {
    use std::any::Any;
    use std::collections::HashMap;
    use std::sync::Arc;

    use parking_lot::Mutex;
    use serde_json::{Map, Value};

    use smartloop::agents::fetch_agent::FetchAgent;
    use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
    use smartloop::agents::mcp::explorer::{
        ConversationTurn, DocumentGroups, Explorer, ExplorerSearchService, FollowUpPlanner,
        ProgressCallback, ToolResolver, WebSearcher,
    };
    use smartloop::agents::mcp::mcp_agent::ResolvedCall;
    use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
    use smartloop::agents::mcp::result_mapper::MCPResultMapper;
    use smartloop::llms::base_llm::{CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
    use smartloop::llms::BaseLLM;
    use smartloop::mcp::client::MCPToolCallResult;
    use smartloop::parsers::DocumentChunk;
    use smartloop::workspace::workspace::MCPServer;

    /// One `progress(step, status, message)` call.
    pub type ProgressEvent = (String, String, String);

    /// A model that answers every `invoke` with a fixed reply and records the
    /// prompts it was given.
    pub struct ScriptedLLM {
        pub reply: String,
        pub prompts: Mutex<Vec<String>>,
    }

    impl ScriptedLLM {
        pub fn replying(reply: &str) -> Arc<Self> {
            Arc::new(Self { reply: reply.to_string(), prompts: Mutex::new(Vec::new()) })
        }
    }

    impl BaseLLM for ScriptedLLM {
        fn project_id(&self) -> &str {
            ""
        }
        fn get_num_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
        fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
            messages
        }
        fn get_max_tokens(&self) -> usize {
            4096
        }
        fn call(&self, prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
            self.prompts.lock().push(prompt.to_string());
            Ok(self.reply.clone())
        }
        fn stream<'a>(&'a self, prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
            self.prompts.lock().push(prompt.to_string());
            Box::new(std::iter::once(GenerationChunk { text: self.reply.clone() }))
        }
        fn invoke(&self, prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
            self.prompts.lock().push(prompt.to_string());
            self.reply.clone()
        }
        fn cancelled(&self) -> Option<CancelEvent> {
            None
        }
        fn set_cancelled(&self, _event: Option<CancelEvent>) {}
        fn llm_type(&self) -> &str {
            "scripted"
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// The calls a stubbed transport saw: `(server name, tool, arguments)`.
    pub type TransportLog = Arc<Mutex<Vec<(String, String, Map<String, Value>)>>>;

    /// `MCPToolCaller(None, None)` with its transport replaced by `answer`
    /// (Python: `caller.call_tool = MagicMock(...)`).  Every call is logged.
    pub fn stubbed_caller(
        answer: impl Fn(&MCPServer, &str, &Map<String, Value>) -> MCPToolCallResult + Send + Sync + 'static,
    ) -> (Arc<MCPToolCaller>, TransportLog) {
        let llm: Arc<dyn BaseLLM> = ScriptedLLM::replying("");
        let caller = MCPToolCaller::new(
            Arc::new(MCPArgumentExtractor::new(llm)),
            Arc::new(Mutex::new(MCPResultMapper::new())),
            None,
        );
        let log: TransportLog = Arc::new(Mutex::new(Vec::new()));
        let seen = log.clone();
        caller.set_call_tool_hook(Some(Arc::new(move |server, tool, arguments, _token, _cwd| {
            let arguments = arguments.cloned().unwrap_or_default();
            seen.lock().push((server.name.clone(), tool.to_string(), arguments.clone()));
            answer(server, tool, &arguments)
        })));
        (Arc::new(caller), log)
    }

    /// A successful result carrying one text block.
    pub fn text_result(text: &str) -> MCPToolCallResult {
        MCPToolCallResult {
            success: true,
            content: Some(vec![serde_json::json!({"type": "text", "text": text})]),
            ..Default::default()
        }
    }

    /// `mcp_agent.resolve.return_value = ...`; counts its calls.
    pub struct ScriptedResolver {
        pub answer: Option<ResolvedCall>,
        pub calls: Mutex<usize>,
    }

    impl ScriptedResolver {
        pub fn returning(answer: Option<ResolvedCall>) -> Arc<Self> {
            Arc::new(Self { answer, calls: Mutex::new(0) })
        }
    }

    impl ToolResolver for ScriptedResolver {
        fn resolve(
            &self,
            _directive: &str,
            _mcp_servers: &[MCPServer],
            _context: Option<&str>,
            _llm: Option<&dyn BaseLLM>,
        ) -> Option<ResolvedCall> {
            *self.calls.lock() += 1;
            self.answer.clone()
        }
    }

    /// A web search agent double (`MagicMock()` with `is_enabled`,
    /// `last_sources` and `search` set); records each search and the
    /// progress sink it was handed.
    pub struct FakeWebSearch {
        pub enabled: bool,
        pub result: String,
        pub sources: Vec<String>,
        pub searched_with: Mutex<Vec<(String, Option<ProgressCallback>)>>,
    }

    impl FakeWebSearch {
        pub fn new(enabled: bool, result: &str, sources: &[&str]) -> Arc<Self> {
            Arc::new(Self {
                enabled,
                result: result.to_string(),
                sources: sources.iter().map(|s| s.to_string()).collect(),
                searched_with: Mutex::new(Vec::new()),
            })
        }
    }

    impl WebSearcher for FakeWebSearch {
        fn is_enabled(&self) -> bool {
            self.enabled
        }
        fn search(&self, query: &str, progress: Option<ProgressCallback>) -> anyhow::Result<String> {
            self.searched_with.lock().push((query.to_string(), progress));
            Ok(self.result.clone())
        }
        fn last_sources(&self) -> Vec<String> {
            self.sources.clone()
        }
        fn last_source_extracts(&self) -> HashMap<String, String> {
            HashMap::new()
        }
    }

    /// A model service whose searches all come back empty (Python patched
    /// `_search_vector` → `None`, `_search_feedback` / `_search_workspace` →
    /// `""`).
    #[derive(Default)]
    pub struct EmptySearch;

    impl ExplorerSearchService for EmptySearch {
        fn get_document_groups(
            &self,
            _query: Option<&str>,
            _skill_instruction: Option<&str>,
            _conversation_history: Option<&[ConversationTurn]>,
        ) -> anyhow::Result<Option<DocumentGroups>> {
            Ok(None)
        }
        fn search_feedback(&self, _query: &str, _n_results: usize) -> anyhow::Result<Vec<DocumentChunk>> {
            Ok(Vec::new())
        }
        fn search_workspace(
            &self,
            _query: &str,
            _work_dir: &str,
            _n_results: usize,
            _conversation_history: Option<&[ConversationTurn]>,
        ) -> anyhow::Result<Vec<DocumentChunk>> {
            Ok(Vec::new())
        }
    }

    /// Follow-up planner returning `calls` whatever it is asked
    /// (`monkeypatch.setattr("smartloop.mcp.call_plans.follow_up_calls", ...)`).
    pub fn planned(calls: Vec<(&'static str, Value)>) -> FollowUpPlanner {
        Arc::new(move |_, _, _| {
            calls
                .iter()
                .map(|(tool, args)| (tool.to_string(), args.as_object().cloned().unwrap_or_default()))
                .collect()
        })
    }

    /// An explorer with every collaborator injected and no model unless one
    /// is given.
    pub fn bare_explorer(
        llm: Option<Arc<dyn BaseLLM>>,
        tool_caller: Option<Arc<MCPToolCaller>>,
        resolver: Arc<dyn ToolResolver>,
        web_search: Arc<dyn WebSearcher>,
        follow_up_calls: FollowUpPlanner,
    ) -> Explorer {
        Explorer {
            llm,
            tool_caller,
            arg_extractor: None,
            use_llm_history: false,
            fetch_agent: Arc::new(FetchAgent::new()),
            web_search_agent: web_search,
            mcp_agent: resolver,
            follow_up_calls,
        }
    }

    /// A progress sink recording every event.
    pub fn recording_sink() -> (ProgressCallback, Arc<Mutex<Vec<ProgressEvent>>>) {
        let events: Arc<Mutex<Vec<ProgressEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let seen = events.clone();
        let sink: ProgressCallback = Arc::new(move |step: &str, status: &str, message: &str| {
            seen.lock().push((step.to_string(), status.to_string(), message.to_string()));
        });
        (sink, events)
    }

    /// The messages of the recorded events (`c.args[2]` per call).
    pub fn messages(events: &Arc<Mutex<Vec<ProgressEvent>>>) -> Vec<String> {
        events.lock().iter().map(|(_, _, m)| m.clone()).collect()
    }
}
// =========================================================== /agents_mcp_explorer
