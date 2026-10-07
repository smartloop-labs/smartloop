//! Skills: matching them to a request, and shaping their output.
//!
//! Port of `smartloop/agents/orchestration/skill_processor.py`.
//!
//! A skill is a markdown document the user wrote describing a procedure.  This
//! module covers both ends of using one: picking which skill (if any) a request
//! maps to, and turning the steps' raw output back into the answer the skill's
//! body describes.
//!
//! Everything in between — planning and executing the steps — is the pipeline's
//! job.  A matched skill contributes guiding context to the planner; it does not
//! get its own control flow.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::agents::mcp::explorer::{Explorer, SkillDict as ExplorerSkillDict};
use crate::agents::mcp::server_selector::{
    BASH_TOOL, EXPLORE_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL,
};
use crate::agents::nodes::{AgentNode, NodeResult};
use crate::agents::orchestration::context::OrchestratorContext;
use crate::agents::orchestration::model_router::{
    ModelRouter, DECISION_MAX_TOKENS, SHORT_TEXT_MAX_TOKENS,
};
use crate::agents::orchestration::response_builder::TOOL_CODE_FENCE_RE;
use crate::agents::planner::ModelTier;
use crate::llms::base_llm::BaseLLM;
use crate::types::ModelCapability;

/// Python logger name.
const LOG_TARGET: &str = "smartloop.agents.orchestration.skill_processor";

/// A parsed skill (`name`, `description`, `content`, `skill_map`).
pub type SkillDict = Map<String, Value>;

/// `key=value` pairs in a skill's prose, read as default tool arguments.
pub static KV_PAIR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\w+)\s*=\s*(\w+)").unwrap());

/// A URL anywhere in the user's message.
static URL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"https?://[^\s<>'")\]]+"#).unwrap());

/// Separators ignored when comparing a requested skill name to a skill.
static NORM_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\s_\-]+").unwrap());

/// Start of the `## skill_map` section (the rest of the document follows).
static SKILL_MAP_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?ms)^##\s+skill_map\s*\n(.*)").unwrap());

/// Any following markdown heading (ends the `## skill_map` section).
static NEXT_HEADING_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^##\s").unwrap());

/// `- key: value` — a new skill_map item (or a key of the current one).
static ITEM_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^-\s+(\w+):\s*(.*)").unwrap());

/// `key: value` — a continuation key of the current skill_map item.
static KV_LINE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\w+):\s*(.*)").unwrap());

/// Test seam replacing [`Explorer::match_skills`] (Python tests patch the
/// explorer with a mock): `(user_message, candidates, llm) -> matched`.
pub type SkillMatcher = Arc<
    dyn Fn(&str, &[ExplorerSkillDict], Option<&dyn BaseLLM>) -> Vec<ExplorerSkillDict>
        + Send
        + Sync,
>;

/// Test seam replacing [`SkillProcessor::extract_tool_condition`]:
/// `(skill_text, tool_name) -> condition`.
pub type ConditionExtractor = Arc<dyn Fn(&str, &str) -> Option<String> + Send + Sync>;

// ---------------------------------------------------------------------------
// Small helpers shared by the orchestration modules of this unit.
// ---------------------------------------------------------------------------

/// The first `n` characters of `s` (Python `s[:n]`).
pub(crate) fn char_prefix(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// Characters `[start, end)` of `s` (Python `s[start:end]` with in-range,
/// non-negative bounds).
pub(crate) fn char_slice(s: &str, start: usize, end: usize) -> String {
    s.chars().skip(start).take(end.saturating_sub(start)).collect()
}

/// Per-character lowercase that keeps character positions aligned with the
/// original text (so an index found in the lowercased text slices the
/// original).
fn lower_chars(s: &str) -> Vec<char> {
    s.chars().map(|c| c.to_lowercase().next().unwrap_or(c)).collect()
}

/// Character index of the first occurrence of `needle` in `hay` at or after
/// `from`.
fn find_chars(hay: &[char], needle: &[char], from: usize) -> Option<usize> {
    if needle.is_empty() {
        return if from <= hay.len() { Some(from) } else { None };
    }
    if needle.len() > hay.len() {
        return None;
    }
    (from..=hay.len() - needle.len()).find(|&i| hay[i..i + needle.len()] == *needle)
}

/// Strip any of `chars` from both ends (Python `str.strip(chars)`).
pub(crate) fn strip_chars<'a>(s: &'a str, chars: &str) -> &'a str {
    s.trim_matches(|c: char| chars.contains(c))
}

/// Invoke an LLM, turning a panic into an `Err` (Python wrapped these calls in
/// `try/except Exception`; a Rust `invoke` reports failure in-band, but a
/// misbehaving backend or test double may still panic).
pub(crate) fn try_invoke(
    llm: &dyn BaseLLM,
    prompt: &str,
    max_tokens: Option<usize>,
) -> Result<String, String> {
    catch_unwind(AssertUnwindSafe(|| llm.invoke(prompt, max_tokens, None))).map_err(|e| {
        if let Some(s) = e.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "LLM call panicked".to_string()
        }
    })
}

/// Run `f`, turning a panic into an `Err` carrying its message.
pub(crate) fn try_call<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|e| {
        if let Some(s) = e.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "call panicked".to_string()
        }
    })
}

/// Python `repr()` of a string.
pub(crate) fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `repr()` of a JSON value (dicts/lists as Python prints them).
pub(crate) fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => py_str_repr(s),
        Value::Array(items) => {
            format!("[{}]", items.iter().map(py_repr).collect::<Vec<_>>().join(", "))
        }
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python `str()` of a JSON value: strings verbatim, everything else as repr.
pub(crate) fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python truthiness of a JSON value.
pub(crate) fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `dict.get(key, "")` for a string field of a skill record.
fn skill_str<'a>(skill: &'a SkillDict, key: &str) -> &'a str {
    skill.get(key).and_then(Value::as_str).unwrap_or("")
}

/// The explorer's string-only view of a parsed skill.
fn to_explorer_skill(skill: &SkillDict) -> ExplorerSkillDict {
    skill
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
        .collect::<HashMap<String, String>>()
}

/// Matches skills to a request and synthesizes their results.
///
/// NOTE: `find_server_hint`, `extract_default_args` and `compact_instruction`
/// have no callers — argument resolution goes through MCPArgumentExtractor and
/// presentation guidance uses the skill body verbatim. Moved here as-is rather
/// than deleted during the split; safe to drop.
pub struct SkillProcessor {
    pub(crate) models: Arc<ModelRouter>,
    pub explorer: Arc<Explorer>,
    /// Skills matched for the current request, kept so a later step can ask
    /// what matched without re-running the match.
    pub matched_skills: Mutex<Option<Vec<SkillDict>>>,
    /// Replaces the explorer's matcher when set (test seam).
    skill_matcher: RwLock<Option<SkillMatcher>>,
    /// Replaces [`Self::extract_tool_condition`] when set (test seam).
    condition_extractor: RwLock<Option<ConditionExtractor>>,
}

impl SkillProcessor {
    /// Build a processor over the model router and explorer.
    pub fn new(models: Arc<ModelRouter>, explorer: Arc<Explorer>) -> Self {
        Self {
            models,
            explorer,
            matched_skills: Mutex::new(None),
            skill_matcher: RwLock::new(None),
            condition_extractor: RwLock::new(None),
        }
    }

    /// Replace the skill matcher (`explorer.match_skills`) — a test seam.
    pub fn set_skill_matcher(&self, matcher: Option<SkillMatcher>) {
        *self.skill_matcher.write() = matcher;
    }

    /// Replace [`Self::extract_tool_condition`] — a test seam.
    pub fn set_condition_extractor(&self, extractor: Option<ConditionExtractor>) {
        *self.condition_extractor.write() = extractor;
    }

    /// The decision model, owned by the router.
    fn agent_llm(&self) -> Arc<dyn BaseLLM> {
        self.models.agent_llm.clone()
    }

    /// Select the user-picked skill, or LLM-match when none was picked.
    ///
    /// Returns a result with `data = {"matched_skills": [...]}` and sets
    /// `ctx.parsed_skills`.  (Python method name: `match`.)
    pub fn match_skills(&self, ctx: &mut OrchestratorContext, node: &AgentNode) -> NodeResult {
        let mut parsed_skills: Vec<SkillDict> = Vec::new();
        for sc in &ctx.skills {
            let mut name = String::new();
            let mut desc = String::new();
            for line in sc.split('\n') {
                let s = line.trim();
                if let Some(rest) = s.strip_prefix("name:") {
                    name = strip_chars(rest.trim(), "'\"").to_string();
                } else if let Some(rest) = s.strip_prefix("description:") {
                    desc = strip_chars(rest.trim(), "'\"").to_string();
                }
            }

            if !name.is_empty() {
                let skill_map = Self::parse_skill_map(sc);
                let mut skill = SkillDict::new();
                skill.insert("name".into(), Value::String(name));
                skill.insert("description".into(), Value::String(desc));
                skill.insert("content".into(), Value::String(sc.clone()));
                skill.insert(
                    "skill_map".into(),
                    serde_json::to_value(&skill_map).unwrap_or(Value::Array(Vec::new())),
                );
                parsed_skills.push(skill);
            }
        }
        ctx.parsed_skills = parsed_skills.iter().cloned().map(Value::Object).collect();

        let system_tool_names: HashSet<&str> =
            [FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL, BASH_TOOL, EXPLORE_TOOL]
                .into_iter()
                .collect();

        // The user explicitly picked a skill for this turn (the API resolved a
        // leading ``/fund-summary`` to a skill name). That IS the intent signal,
        // so use it directly and skip the LLM detection — a second LLM guess must
        // not be able to veto an explicit choice. Only a request naming a built-in
        // tool skill or a ``__``-prefixed system skill (never user-selectable), or
        // a name that matches nothing, falls through to normal detection.
        if let Some(requested_skill) = ctx.requested_skill.as_deref().filter(|s| !s.is_empty()) {
            let norm = |s: &str| NORM_RE.replace_all(&s.to_lowercase(), "").into_owned();
            let wanted = norm(requested_skill);
            let requested: Vec<SkillDict> = parsed_skills
                .iter()
                .filter(|s| {
                    let name = skill_str(s, "name");
                    norm(name) == wanted
                        && !system_tool_names.contains(name)
                        && !name.starts_with("__")
                })
                .cloned()
                .collect();
            if !requested.is_empty() {
                self.models.emit(
                    "analyze",
                    "processing",
                    &format!(
                        "Fulfilling your request with the '{}' skill",
                        skill_str(&requested[0], "name")
                    ),
                );
                *self.matched_skills.lock() = Some(requested.clone());
                return NodeResult::with_data(
                    node.node_id.clone(),
                    json!({ "matched_skills": requested }),
                );
            }
            tracing::info!(
                target: LOG_TARGET,
                "[skill_match] selected skill {} not selectable/not found among [{}] — using normal detection",
                py_str_repr(requested_skill),
                parsed_skills
                    .iter()
                    .map(|s| py_str_repr(skill_str(s, "name")))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }

        let has_url = URL_RE.is_match(&ctx.user_message);
        let mut skills_for_matching: Vec<SkillDict> = parsed_skills
            .iter()
            .filter(|s| !system_tool_names.contains(skill_str(s, "name")))
            .cloned()
            .collect();
        if !has_url {
            skills_for_matching.retain(|s| {
                let lname = skill_str(s, "name").to_lowercase();
                !["fetch", "browse", "scrape"].iter().any(|kw| lname.contains(kw))
            });
        }

        // Nothing to match — skip the LLM entirely so a project with no skills
        // never pays to load the heavier skills model.  Silently: announcing a
        // match against skills that were all filtered out describes work that
        // is not about to happen.
        if skills_for_matching.is_empty() {
            *self.matched_skills.lock() = Some(Vec::new());
            return NodeResult::with_data(node.node_id.clone(), json!({ "matched_skills": [] }));
        }

        // Announced here rather than on entry, and counting the candidates
        // rather than everything on disk.  ``ctx.skills`` also holds the
        // built-in tool skills (bash, file-read, file-write, …) and the system
        // skills, none of which are matchable — counting those told the user
        // their request was being weighed against skills they had never
        // created and that could not be picked.
        let n = skills_for_matching.len();
        self.models.emit(
            "analyze",
            "processing",
            &format!(
                "Analyzing your request against {} skill{}",
                n,
                if n != 1 { "s" } else { "" }
            ),
        );

        // Skill matching is an intent judgement the small orchestrator model is
        // poor at (it defaults to picking the only listed skill). Prefer the
        // 'reasoning'-capability model, but only when it is resident or fits — a
        // memory-tight device falls back to the mini instead of risking an OOM.
        let match_llm = self.models.decision_llm(Some(ModelCapability::Reasoning.value()), None);
        tracing::info!(
            target: LOG_TARGET,
            "[models] SKILL_MATCH → {}",
            ModelRouter::model_label(&*match_llm)
        );

        let candidates: Vec<ExplorerSkillDict> =
            skills_for_matching.iter().map(to_explorer_skill).collect();
        let matcher = self.skill_matcher.read().clone();
        let matched = match matcher {
            Some(m) => m(&ctx.user_message, &candidates, Some(&*match_llm)),
            None => self.explorer.match_skills(&ctx.user_message, &candidates, Some(&*match_llm)),
        };

        // The explorer works on a string-only view; map its picks back to the
        // full records (which also carry the parsed ``skill_map``).
        let matched_skills: Vec<SkillDict> = matched
            .iter()
            .map(|m| {
                let name = m.get("name").map(String::as_str).unwrap_or("");
                let content = m.get("content").map(String::as_str);
                skills_for_matching
                    .iter()
                    .find(|s| {
                        skill_str(s, "name") == name
                            && content.map_or(true, |c| skill_str(s, "content") == c)
                    })
                    .cloned()
                    .unwrap_or_else(|| {
                        m.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect()
                    })
            })
            .collect();
        *self.matched_skills.lock() = Some(matched_skills.clone());
        NodeResult::with_data(node.node_id.clone(), json!({ "matched_skills": matched_skills }))
    }

    /// Extract a conditional clause for a tool from skill text.
    ///
    /// Searches for the tool name in `skill_text` and uses the LLM to
    /// determine whether execution is conditional.  Returns an `llm:`
    /// condition string or `None`.
    pub fn extract_tool_condition(&self, skill_text: &str, tool_name: &str) -> Option<String> {
        let hook = self.condition_extractor.read().clone();
        if let Some(hook) = hook {
            return hook(skill_text, tool_name);
        }

        let text_lower = lower_chars(skill_text);
        let mut tool_pos = find_chars(&text_lower, &lower_chars(tool_name), 0);
        if tool_pos.is_none() {
            tool_pos = find_chars(&text_lower, &lower_chars(&tool_name.replace('_', " ")), 0);
        }
        let tool_pos = tool_pos?;

        let total = text_lower.len();
        let start = tool_pos.saturating_sub(300);
        let end = total.min(tool_pos + 200);
        let relevant_content = char_slice(skill_text, start, end);

        let prompt = format!(
            "Read this skill instruction and determine if executing \
             the tool '{tool_name}' is conditional on some prior result.\n\n\
             Skill text:\n{relevant_content}\n\n\
             If the tool should ONLY run when a condition is met, \
             reply with a single YES/NO question that captures the \
             condition (e.g. 'Were discrepancies found in the results?').\n\
             If the tool should ALWAYS run (no condition), reply: NONE"
        );

        match try_invoke(&*self.agent_llm(), &prompt, Some(DECISION_MAX_TOKENS)) {
            Ok(reply) => {
                let response = reply.trim().to_string();
                tracing::info!(
                    target: LOG_TARGET,
                    "[condition-extract] LLM response: {}",
                    char_prefix(&response, 120)
                );

                if response.is_empty() || char_prefix(&response.to_lowercase(), 20).contains("none")
                {
                    return None;
                }

                let mut question = strip_chars(response.trim(), "\"'").to_string();
                // A valid condition is a YES/NO question. A bare "YES"/"NO" (or any
                // answer with no question mark) is a malformed reply, not a real
                // condition — treat it as unconditional rather than fabricating an
                // ``llm:YES`` gate that can never be satisfied.
                if !question.contains('?') {
                    return None;
                }

                let parts: Vec<&str> = question.split('?').collect();
                for part in parts.iter().rev() {
                    let candidate = part.trim();
                    if !candidate.is_empty() && candidate.chars().count() > 10 {
                        question = format!("{candidate}?");
                        break;
                    }
                }

                Some(format!("llm:{question}"))
            }
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[condition-extract] LLM failed: {}", e);
                None
            }
        }
    }

    /// Find a server name hint near a tool reference in skill content.
    ///
    /// Looks for patterns like "save_issue on Linear MCP" or "tool in Linear MCP".
    pub fn find_server_hint(matched_skills: &[SkillDict], tool_name: &str) -> Option<String> {
        // Pattern: <tool_name> on <Server>
        let pattern = Regex::new(&format!(r"(?i){}\s+on\s+(\w+)", regex::escape(tool_name)))
            .expect("escaped tool name is a valid pattern");
        // Pattern: tool in <Server> MCP
        static PATTERN2: Lazy<Regex> =
            Lazy::new(|| Regex::new(r"(?i)tool\s+in\s+(\w+)\s+MCP").unwrap());
        for sk in matched_skills {
            let content = skill_str(sk, "content");
            if let Some(m) = pattern.captures(content) {
                return Some(m[1].to_string());
            }
            if let Some(m2) = PATTERN2.captures(content) {
                return Some(m2[1].to_string());
            }
        }
        None
    }

    /// Extract `key=value` default args from skill text near a tool reference.
    ///
    /// Scans the skill body for all occurrences of the tool name and picks up
    /// `key=value` pairs in the surrounding context.  For example, given
    /// "call save_issue on Linear MCP, use team=DEV, assignee=me" returns
    /// `{"team": "DEV", "assignee": "me"}`.
    pub fn extract_default_args(
        matched_skills: &[SkillDict],
        tool_name: &str,
    ) -> BTreeMap<String, String> {
        let mut defaults: BTreeMap<String, String> = BTreeMap::new();
        let mut order: Vec<String> = Vec::new();
        let skip_keys: HashSet<&str> =
            ["name", "description", "tools", "type", "step", "step_type"].into_iter().collect();
        let tool_lower = lower_chars(tool_name);
        let tool_len = tool_name.chars().count();
        for sk in matched_skills {
            let content = skill_str(sk, "content");
            let content_lower = lower_chars(content);
            let content_len = content_lower.len();
            // Scan all occurrences of the tool name
            let mut search_start = 0usize;
            while let Some(tool_pos) = find_chars(&content_lower, &tool_lower, search_start) {
                search_start = tool_pos + tool_lower.len();
                // Look at the surrounding window (200 chars before/after)
                let start = tool_pos.saturating_sub(200);
                let end = content_len.min(tool_pos + tool_len + 200);
                let window = char_slice(content, start, end);
                for m in KV_PAIR_RE.captures_iter(&window) {
                    let key = m[1].to_string();
                    let value = m[2].to_string();
                    if skip_keys.contains(key.to_lowercase().as_str()) {
                        continue;
                    }
                    if !defaults.contains_key(&key) {
                        order.push(key.clone());
                    }
                    defaults.insert(key, value);
                }
                if tool_lower.is_empty() {
                    // An empty needle matches everywhere; one window is enough.
                    break;
                }
            }
        }
        if !defaults.is_empty() {
            let shown = order
                .iter()
                .map(|k| format!("{}: {}", py_str_repr(k), py_str_repr(&defaults[k])))
                .collect::<Vec<_>>()
                .join(", ");
            tracing::info!(
                target: LOG_TARGET,
                "[fallback] extracted default_args for '{}': {{{}}}",
                tool_name,
                shown
            );
        }
        defaults
    }

    /// Strip YAML frontmatter and ```` ```mcp ```` blocks from skill content.
    ///
    /// Returns the body of the skill — the instructions that should be
    /// injected into context for the streaming LLM.  Fenced `mcp` blocks are
    /// execution directives, not prose — they must be removed so the LLM
    /// doesn't echo them verbatim.
    pub fn strip_frontmatter(content: &str) -> String {
        // Strip until nothing changes: a frontmatter block can be followed by
        // another.  The orchestrator prepends its own ``name:``/``description:``
        // header to a file that usually has one already, and a skill edited
        // from the template keeps the placeholder block above the real one —
        // leaving the second in the body put ``name: job-match`` first in the
        // message a bare ``/job-match`` sends, and its tools were called with
        // the skill's name as their query.
        let mut body = content.to_string();
        loop {
            let stripped = Self::strip_one_frontmatter(&body);
            if stripped == body {
                break;
            }
            body = stripped;
        }
        if body.is_empty() {
            body = content.to_string();
        }

        // Defensively strip any legacy ```mcp / ```tool_code fenced blocks so
        // they never reach the streaming LLM as instructions.
        TOOL_CODE_FENCE_RE.replace_all(&body, "").trim().to_string()
    }

    /// Remove the leading frontmatter block, fenced or not, from `content`.
    pub fn strip_one_frontmatter(content: &str) -> String {
        let lines: Vec<&str> = content.split('\n').collect();
        let mut body_start = 0usize;
        let mut in_frontmatter = false;
        let mut in_unfenced = false;
        const FM_KEYS: [&str; 3] = ["name:", "description:", "tools:"];

        for (i, line) in lines.iter().enumerate() {
            let stripped = line.trim();
            if stripped == "---" {
                if in_unfenced {
                    // Closing fence of a block that never opened one.
                    body_start = i + 1;
                    break;
                }
                if !in_frontmatter {
                    in_frontmatter = true;
                    continue;
                } else {
                    // End of fenced frontmatter block
                    body_start = i + 1;
                    break;
                }
            }
            if in_frontmatter {
                continue;
            }
            // Unfenced frontmatter: lines starting with known keys
            if FM_KEYS.iter().any(|k| stripped.starts_with(k)) {
                in_unfenced = true;
                body_start = i + 1;
                continue;
            }
            // First non-frontmatter, non-empty line
            if !stripped.is_empty() {
                body_start = i;
                break;
            }
        }

        let start = body_start.min(lines.len());
        lines[start..].join("\n").trim().to_string()
    }

    /// Synthesize all accumulated step outputs into a single document.
    ///
    /// Falls back to the raw accumulated context if synthesis fails.
    pub fn synthesize(
        &self,
        user_message: &str,
        accumulated_context: &str,
        skill_context: &str,
    ) -> String {
        let prompt = format!(
            "<system>\n\
             You are a data synthesis assistant. You have been given raw outputs \
             from several execution steps (tool calls, document retrieval, analysis). \
             Your task is to merge them into a single, coherent, factual document \
             that fully answers the user's request.\n\n\
             Rules:\n\
             - Include ALL factual values exactly as provided — do not round, \
             estimate, or omit numbers.\n\
             - Do NOT answer the user directly or add conversational framing.\n\
             - Do NOT include JSON, code blocks, or raw data structures.\n\
             - Write in clear prose with labelled sections if appropriate.\n\
             - If there is only one data source, summarise it concisely.\n\
             </system>\n\
             <user>\n\
             User request: {user_message}\n\n\
             Skill context:\n{}\n\n\
             Accumulated step outputs:\n{accumulated_context}\n\
             </user>",
            char_prefix(skill_context, 1000)
        );

        // Synthesis writes the final skill answer, so use a model declaring
        // the reasoning capability (the largest that fits — see
        // ModelService._get_llm_for_capability), not the small orchestrator.
        // Falls back to the large/agent model when no capable model fits.
        //
        // ``purpose`` is what the user is told.  The capability is how the
        // model is chosen, and saying "Loading model for reasoning…" directly
        // under "Synthesizing the final answer…" reports an internal routing
        // tag as though it were a second, unrelated step.
        let outcome = try_call(|| {
            let synth_llm = self.models.llm_for_tier(
                ModelTier::Large,
                Some(ModelCapability::Reasoning.value()),
                Some("the final answer"),
            );
            tracing::info!(
                target: LOG_TARGET,
                "[models] SYNTHESIZE → {}",
                ModelRouter::model_label(&*synth_llm)
            );
            synth_llm.invoke(&prompt, None, None).trim().to_string()
        });
        match outcome {
            Ok(result) => {
                if !result.is_empty() {
                    tracing::info!(
                        target: LOG_TARGET,
                        "[synthesize] produced {} chars from {} chars of accumulated data",
                        result.chars().count(),
                        accumulated_context.chars().count()
                    );
                    return result;
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[synthesize] synthesis failed, using raw context: {}",
                    e
                );
            }
        }

        accumulated_context.to_string()
    }

    /// Distil a compact instruction from skill context for the streaming LLM.
    ///
    /// Extracts only presentation/formatting guidance.
    pub fn compact_instruction(
        &self,
        skill_context: Option<&str>,
        user_message: &str,
    ) -> Option<String> {
        let skill_context = skill_context.filter(|s| !s.trim().is_empty())?;

        let prompt = format!(
            "<system>\n\
             You are reading a skill instruction file. All tool calls and \
             data-gathering steps have ALREADY been executed successfully.\n\n\
             Extract ONLY guidance about how to PRESENT or FORMAT the \
             results for the user — but ONLY if the skill text explicitly \
             specifies a format. Do not invent formatting guidance.\n\n\
             Do NOT include any tool call instructions, API actions, \
             function calls, or internal execution directives — those are done.\n\n\
             If the skill text does not explicitly specify a presentation \
             format, reply with: NONE\n\
             Otherwise reply with a concise 1-3 sentence instruction \
             that quotes the presentation directive from the skill text.\n\
             </system>\n\
             <user>\n\
             User request: {user_message}\n\n\
             Skill text:\n{}\n\
             </user>",
            char_prefix(skill_context, 2000)
        );

        match try_invoke(&*self.agent_llm(), &prompt, Some(SHORT_TEXT_MAX_TOKENS)) {
            Ok(reply) => {
                let result = reply.trim().to_string();
                if result.is_empty() || char_prefix(&result.to_lowercase(), 10).contains("none") {
                    return None;
                }
                tracing::info!(
                    target: LOG_TARGET,
                    "[skill] compact instruction: {}",
                    char_prefix(&result, 200)
                );
                Some(result)
            }
            Err(e) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[skill] compact instruction extraction failed: {}",
                    e
                );
                // No fallback to raw skill body — it contains execution
                // directives that the streaming LLM would follow verbatim,
                // fabricating responses based on skill instructions rather
                // than actual data.
                None
            }
        }
    }

    /// Parse a `## skill_map` section from skill markdown into step dicts.
    ///
    /// Each list item is a YAML-like block with keys: step, tool, server,
    /// instruction, step_type, condition.  Returns an empty list when no
    /// `## skill_map` section is found.  At most 5 steps are returned.
    pub fn parse_skill_map(content: &str) -> Vec<BTreeMap<String, String>> {
        // Find the ## skill_map section
        let Some(caps) = SKILL_MAP_RE.captures(content) else {
            return Vec::new();
        };

        let mut block = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        // Stop at the next markdown heading (if any)
        if let Some(next_heading) = NEXT_HEADING_RE.find(block) {
            block = &block[..next_heading.start()];
        }

        let mut steps: Vec<BTreeMap<String, String>> = Vec::new();
        let mut current: BTreeMap<String, String> = BTreeMap::new();

        for line in block.split('\n') {
            let stripped = line.trim();
            if stripped.is_empty() {
                continue;
            }

            // New item starts with "- key: value"
            if let Some(item) = ITEM_RE.captures(stripped) {
                // If this is the "step" key, save previous item and start new
                let key = item[1].to_string();
                let value = strip_chars(item[2].trim(), "\"'").to_string();
                if key == "step" && !current.is_empty() {
                    steps.push(std::mem::take(&mut current));
                }
                current.insert(key, value);
            } else if let Some(kv) = KV_LINE_RE.captures(stripped) {
                // Continuation key: value (no leading dash)
                let key = kv[1].to_string();
                let value = strip_chars(kv[2].trim(), "\"'").to_string();
                current.insert(key, value);
            }
        }

        if !current.is_empty() {
            steps.push(current);
        }

        steps.truncate(5);
        steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_frontmatter_removes_fenced_and_stacked_blocks() {
        let content = "---\nname: a\ndescription: b\n---\nname: job-match\ndescription: x\n---\n\nBody text.";
        assert_eq!(SkillProcessor::strip_frontmatter(content), "Body text.");
    }

    #[test]
    fn strip_frontmatter_removes_mcp_fences() {
        let content = "---\nname: a\n---\nDo it.\n```mcp\ncall x\n```\nDone.";
        assert_eq!(SkillProcessor::strip_frontmatter(content), "Do it.\n\nDone.");
    }

    #[test]
    fn strip_frontmatter_keeps_plain_body() {
        assert_eq!(SkillProcessor::strip_frontmatter("Just a body."), "Just a body.");
    }

    #[test]
    fn parse_skill_map_reads_items() {
        let content = "# Skill\n\n## skill_map\n- step: 1\n  tool: get_x\n  server: \"srv\"\n- step: 2\n  tool: get_y\n## Other\n- step: 3\n";
        let steps = SkillProcessor::parse_skill_map(content);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0]["tool"], "get_x");
        assert_eq!(steps[0]["server"], "srv");
        assert_eq!(steps[1]["step"], "2");
        assert!(SkillProcessor::parse_skill_map("no map here").is_empty());
    }

    #[test]
    fn extract_default_args_near_tool() {
        let mut sk = SkillDict::new();
        sk.insert(
            "content".into(),
            Value::String("call save_issue on Linear MCP, use team=DEV, assignee=me, name=x".into()),
        );
        let args = SkillProcessor::extract_default_args(&[sk.clone()], "save_issue");
        assert_eq!(args.get("team").map(String::as_str), Some("DEV"));
        assert_eq!(args.get("assignee").map(String::as_str), Some("me"));
        assert!(!args.contains_key("name"));
        assert_eq!(
            SkillProcessor::find_server_hint(&[sk], "save_issue").as_deref(),
            Some("Linear")
        );
    }

    #[test]
    fn py_repr_matches_python() {
        let v = json!({"symbol": "AAPL", "n": 1, "ok": true, "x": null});
        assert_eq!(py_repr(&v), "{'symbol': 'AAPL', 'n': 1, 'ok': True, 'x': None}");
        assert_eq!(py_str(&json!("AAPL")), "AAPL");
    }
}
