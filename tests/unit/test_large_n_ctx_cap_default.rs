//! The large and capability models load with a capped context by default.
//!
//! Uncapped, each auto-sized its window from the memory free at load time as
//! if it were the only model, so `qwen3.5-2b` and `ministral3-3b` loaded back
//! to back each took n_ctx=65536.  A query over 402 documents then fanned out
//! 210 extractions and the agent was killed for memory.

use smartloop::config::app_settings::AppSettings;

#[test]
fn test_large_n_ctx_cap_defaults_to_16k() {
    assert_eq!(AppSettings::defaults().llm_large_n_ctx_cap, 16384);
}
