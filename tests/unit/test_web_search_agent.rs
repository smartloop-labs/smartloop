//! Port of `tests/unit/test_web_search_agent.py`.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;

use smartloop::agents::fetch_agent::UrlFetcher;
use smartloop::agents::web_search_agent::{normalize_search_query, WebSearchAgent};

use crate::shared::agents_core::FakeLlm;

mod normalize_search_query_tests {
    use super::*;

    #[test]
    fn test_strips_give_me_filler() {
        assert_eq!(normalize_search_query("give me a list latest canyon bikes"), "canyon bikes");
    }

    #[test]
    fn test_strips_please_and_question_mark() {
        assert_eq!(normalize_search_query("please tell me the weather"), "the weather");
    }

    #[test]
    fn test_strips_leading_filler_and_trailing_politeness() {
        assert_eq!(normalize_search_query("can you show me the new macbooks?"), "new macbooks");
    }

    #[test]
    fn test_keeps_meaningful_subject() {
        assert_eq!(normalize_search_query("what is the latest iphone release date"), "iphone release date");
    }

    #[test]
    fn test_no_filler_leaves_message_intact() {
        assert_eq!(normalize_search_query("latest canyon bikes"), "canyon bikes");
    }

    #[test]
    fn test_empty_falls_back_to_raw() {
        assert_eq!(normalize_search_query(""), "");
    }

    #[test]
    fn test_only_filler_falls_back_to_raw() {
        assert_eq!(normalize_search_query("give me"), "give me");
    }

    #[test]
    fn test_collapses_whitespace() {
        assert_eq!(normalize_search_query("  tell me   about  canyon   bikes "), "canyon bikes");
    }

    #[test]
    fn test_drops_weak_stopwords_so_endpoint_doesnt_latch_onto_them() {
        assert_eq!(normalize_search_query("the current stock prices for apple"), "stock prices apple");
    }

    #[test]
    fn test_keeps_single_subject_with_stopword() {
        assert_eq!(normalize_search_query("electric current"), "electric current");
    }
}

/// A fetcher answering from a fixed `url -> text` map (`""` when absent).
struct MapFetcher(HashMap<String, String>);

impl UrlFetcher for MapFetcher {
    fn fetch(&self, url: &str, _max_length: usize, _start_index: usize, _raw: bool) -> String {
        self.0.get(url).cloned().unwrap_or_default()
    }
}

/// A fetcher echoing the URL back (`lambda self, url, max_length=0: url`).
struct EchoFetcher;

impl UrlFetcher for EchoFetcher {
    fn fetch(&self, url: &str, _max_length: usize, _start_index: usize, _raw: bool) -> String {
        url.to_string()
    }
}

fn candidates() -> Vec<(String, String)> {
    vec![
        ("a.com".to_string(), "aaa".to_string()),
        ("b.com".to_string(), "bbb".to_string()),
        ("c.com".to_string(), "ccc".to_string()),
    ]
}

fn urls(ranked: &[(String, String)]) -> Vec<&str> {
    ranked.iter().map(|(u, _)| u.as_str()).collect()
}

mod rerank {
    use super::*;

    fn agent(llm: Option<&Arc<FakeLlm>>) -> WebSearchAgent {
        WebSearchAgent::new(Some(Arc::new(EchoFetcher)), 3, 10, llm.map(|l| l.shared()))
    }

    #[test]
    fn test_no_llm_keeps_endpoint_order() {
        let agent = agent(None);
        assert_eq!(agent.rerank("query", candidates()), candidates());
    }

    #[test]
    fn test_single_candidate_untouched() {
        let agent = agent(None);
        let one = vec![("a.com".to_string(), "aaa".to_string())];
        assert_eq!(agent.rerank("query", one.clone()), one);
    }

    #[test]
    fn test_reranks_by_model_ranking() {
        let llm = FakeLlm::new("3,1,2");
        let agent = agent(Some(&llm));
        let ranked = agent.rerank("canyon bikes", candidates());
        assert!(llm.calls()[0].contains("Query: canyon bikes"));
        assert_eq!(urls(&ranked), vec!["c.com", "a.com", "b.com"]);
    }

    #[test]
    fn test_out_of_range_and_duplicate_indices_are_guarded() {
        let llm = FakeLlm::new("5,1,1,2");
        let agent = agent(Some(&llm));
        let ranked = agent.rerank("query", candidates());
        assert_eq!(urls(&ranked), vec!["a.com", "b.com", "c.com"]);
    }

    #[test]
    fn test_rerank_failure_falls_back_to_endpoint_order() {
        let llm = FakeLlm::new("");
        llm.set_failure("model down");
        let agent = agent(Some(&llm));
        assert_eq!(agent.rerank("query", candidates()), candidates());
    }

    #[test]
    fn test_garbled_response_appends_unranked_at_end() {
        let llm = FakeLlm::new("3 blah");
        let agent = agent(Some(&llm));
        let ranked = agent.rerank("query", candidates());
        assert_eq!(urls(&ranked), vec!["c.com", "a.com", "b.com"]);
    }
}

mod enablement {
    //! The project flag and the hosted status must both allow a search.

    use super::*;

    use serial_test::serial;
    use smartloop::config::ModelConfig;

    use crate::shared::workspace_support::EnvGuard;
    use smartloop::search::providers::{self, WebSearchCallError, WebSearchClient};
    use smartloop::workspace::workspace::{Project, Workspace};

    /// The platform as the tests stub it: the status answer, how often it was
    /// asked, and every query searched.
    #[derive(Clone)]
    struct Platform {
        available: Arc<Mutex<bool>>,
        asked: Arc<Mutex<Vec<bool>>>,
        searched: Arc<Mutex<Vec<String>>>,
    }

    struct StubClient(Platform);

    impl WebSearchClient for StubClient {
        fn web_search(&self, query: &str, _count: usize) -> Result<Vec<String>, WebSearchCallError> {
            self.0.searched.lock().push(query.to_string());
            Ok(vec!["http://x".to_string()])
        }
        fn web_search_status(&self) -> Result<bool, WebSearchCallError> {
            self.0.asked.lock().push(true);
            Ok(*self.0.available.lock())
        }
    }

    /// Say the account can search unless a test says otherwise (the autouse
    /// `_hosted_available` fixture): signed in, and the status endpoint says
    /// yes.  Without this the result would depend on the network and on
    /// whoever runs the suite being signed in.  Restores everything on drop.
    ///
    /// `SLP_HOME` points at the temp dir too (pytest's `tmp_path` isolation):
    /// loading the workspace overlays each project's own state file from
    /// `<home>/<workspace id>/<project id>/`, so a stray `ws/p1` under the
    /// real home would otherwise override the flag these tests write.
    struct Fixture {
        platform: Platform,
        _home: EnvGuard,
        _tmp: tempfile::TempDir,
        dir: String,
    }

    impl Fixture {
        fn new() -> Self {
            let platform = Platform {
                available: Arc::new(Mutex::new(true)),
                asked: Arc::new(Mutex::new(Vec::new())),
                searched: Arc::new(Mutex::new(Vec::new())),
            };
            let for_factory = platform.clone();
            providers::reset_status_cache();
            providers::set_credentials_check(Some(Arc::new(|| true)));
            providers::set_platform_client_factory(Some(Arc::new(move || {
                Box::new(StubClient(for_factory.clone())) as Box<dyn WebSearchClient>
            })));
            Workspace::reset();
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().to_string_lossy().to_string();
            let home = EnvGuard::set("SLP_HOME", tmp.path());
            Self { platform, _home: home, _tmp: tmp, dir }
        }

        fn write_config(&self, enabled: bool, workspace_enabled: bool) {
            let cfg = format!("{}/config.json", self.dir);
            std::fs::write(
                &cfg,
                serde_json::to_string(&json!({
                    "id": "m",
                    "workspace": {
                        "id": "ws",
                        "workspace_dir": self.dir,
                        "models_dir": self.dir,
                        "models": [],
                        "web_search_enabled": workspace_enabled,
                        "projects": [
                            {"id": "other", "project_dir": self.dir, "current": false,
                             "name": "other", "web_search_enabled": true},
                            {"id": "p1", "project_dir": self.dir, "current": true,
                             "name": "p1", "web_search_enabled": enabled},
                        ],
                    },
                }))
                .unwrap(),
            )
            .unwrap();
            Workspace::set_config_file(Some(cfg));
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            providers::set_credentials_check(None);
            providers::set_platform_client_factory(None);
            providers::reset_status_cache();
            Workspace::set_config_file(None);
            Workspace::reset();
        }
    }

    #[test]
    #[serial]
    fn test_disabled_project_skips_search() {
        let fx = Fixture::new();
        fx.write_config(false, true);

        assert!(!WebSearchAgent::is_enabled());
        assert_eq!(WebSearchAgent::default().search("latest canyon bikes", None, None, None), "");
        assert!(fx.platform.searched.lock().is_empty());
    }

    #[test]
    #[serial]
    fn test_enabled_project_reports_enabled() {
        let fx = Fixture::new();
        fx.write_config(true, true);
        assert!(WebSearchAgent::is_enabled());
    }

    /// A project written before the flag existed (or freshly created) is on.
    #[test]
    #[serial]
    fn test_flag_absent_from_disk_defaults_to_enabled() {
        let fx = Fixture::new();
        let cfg = format!("{}/config.json", fx.dir);
        std::fs::write(
            &cfg,
            serde_json::to_string(&json!({
                "id": "m",
                "workspace": {
                    "id": "ws", "workspace_dir": fx.dir, "models_dir": fx.dir,
                    "models": [],
                    // No web_search_enabled key at all.
                    "projects": [{"id": "p1", "project_dir": fx.dir,
                                  "current": true, "name": "p1"}],
                },
            }))
            .unwrap(),
        )
        .unwrap();
        Workspace::set_config_file(Some(cfg));
        assert!(WebSearchAgent::is_enabled());
    }

    #[test]
    fn test_new_project_is_disabled_by_default() {
        let project = Project {
            id: "p1".to_string(),
            project_dir: "/tmp/p1".to_string(),
            current: true,
            name: Some("p1".to_string()),
            ..Default::default()
        };
        assert!(!project.web_search_enabled);
    }

    #[test]
    #[serial]
    fn test_disk_state_wins_over_stale_cached_workspace() {
        let fx = Fixture::new();
        // A worker process can hold a singleton loaded before the user toggled
        // web search off; the on-disk flag is the source of truth.
        let stale_dir = format!("{}/stale", fx.dir);
        std::fs::create_dir_all(&stale_dir).unwrap();
        let stale_cfg = format!("{}/config.json", stale_dir);
        std::fs::write(
            &stale_cfg,
            serde_json::to_string(&json!({
                "id": "m",
                "workspace": {
                    "id": "ws", "workspace_dir": fx.dir, "models_dir": fx.dir,
                    "models": [],
                    "projects": [{"id": "p1", "project_dir": fx.dir, "current": true,
                                  "name": "p1", "web_search_enabled": true}],
                },
            }))
            .unwrap(),
        )
        .unwrap();
        Workspace::set_config_file(Some(stale_cfg));
        let cached = Workspace::init(&ModelConfig::new("m"));
        assert!(cached.read().projects.iter().any(|p| p.id == "p1" && p.web_search_enabled));

        fx.write_config(false, true);
        assert!(!WebSearchAgent::is_enabled());
    }

    #[test]
    #[serial]
    fn test_enabled_project_with_an_available_account() {
        let fx = Fixture::new();
        fx.write_config(true, true);
        assert!(WebSearchAgent::is_enabled());
    }

    /// Signed out, or no provider registered — the endpoint decides, and the
    /// orchestrator skips instead of calling and discarding a 403.
    #[test]
    #[serial]
    fn test_hosted_status_can_veto_an_enabled_project() {
        let fx = Fixture::new();
        fx.write_config(true, true);
        *fx.platform.available.lock() = false;

        assert!(!WebSearchAgent::is_enabled());
    }

    /// The user's own switch is free to read, so it is checked first — a
    /// disabled project must not cost a network round-trip.
    #[test]
    #[serial]
    fn test_project_switch_wins_without_asking_the_endpoint() {
        let fx = Fixture::new();
        fx.write_config(false, true);

        assert!(!WebSearchAgent::is_enabled());
        assert!(fx.platform.asked.lock().is_empty());
    }
}

mod source_extracts {
    //! The kept pages' text is recorded so a link can be verified later.

    use super::*;

    fn agent(pages: &HashMap<String, String>) -> WebSearchAgent {
        WebSearchAgent::new(Some(Arc::new(MapFetcher(pages.clone()))), 2, 2, None)
    }

    #[test]
    fn test_fetch_top_records_text_per_kept_url() {
        // Pages must clear the usability floor (>= 200 chars of real text).
        let pages: HashMap<String, String> = HashMap::from([
            (
                "a.com".to_string(),
                format!("Canyon Aeroad CFR weighs 6.8kg. {}", "Aero road details. ".repeat(12)),
            ),
            (
                "b.com".to_string(),
                format!("Canyon Ultimate CF SLX starts at 4299 euros. {}", "Spec notes. ".repeat(16)),
            ),
        ]);
        let agent = agent(&pages);
        agent.fetch_top("canyon bikes", &["a.com".to_string(), "b.com".to_string()], 2, 2, None);

        assert_eq!(*agent.last_sources.read(), vec!["a.com".to_string(), "b.com".to_string()]);
        let expected: HashMap<String, String> =
            pages.iter().map(|(url, text)| (url.clone(), text.trim().to_string())).collect();
        assert_eq!(*agent.last_source_extracts.read(), expected);
    }

    #[test]
    fn test_unfetchable_links_leave_no_extracts() {
        let agent = agent(&HashMap::new());
        let out = agent.fetch_top("canyon bikes", &["a.com".to_string()], 1, 1, None);

        assert!(out.contains("a.com"));
        assert!(agent.last_source_extracts.read().is_empty());
    }
}
