//! Port of `tests/unit/test_generation_rate.py`.
//!
//! How fast generation actually was, and what decided it.
//!
//! "Slow" is not actionable and neither is a wall-clock total: a long prompt on
//! a fast model and a short prompt on a model that fell back to CPU take the
//! same ten seconds and want opposite fixes.  Prefill (time to the first token)
//! and decode rate separate them, and the streaming path — the one a user
//! watches — measured neither.
//!
//! The load-time knobs behind both were fixed at 512 and 4 with no way to see
//! or change them; they are chosen from the device now, and logged.

use serial_test::serial;

use smartloop::constants::OPENING_TOKENS;
use smartloop::llms::llama_cpp::LlamaCpp;
use smartloop::llms::llm_factory::LlmFactory;

/// `time.time()`.
fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

mod the_rate_line {
    use super::*;

    #[test]
    fn test_it_separates_prefill_from_decode() {
        let start = now() - 5.0; // generation began 5s ago
        let first = now() - 4.0; // first token landed 1s in

        let line = LlamaCpp::rate(3200, 48, start, Some(first), None);

        assert!(line.contains("3200 prompt"), "{line}");
        assert!(line.contains("ttft 1.0"), "{line}");
        // 48 tokens over the 4s since the first one.
        assert!(line.contains("12.0 tok/s"), "{line}");
    }

    #[test]
    fn test_a_generation_that_produced_nothing_says_nothing() {
        assert_eq!(LlamaCpp::rate(3200, 0, now(), None, None), "");
        assert_eq!(LlamaCpp::rate(3200, 5, now(), None, None), "");
    }

    #[test]
    fn test_a_single_token_does_not_divide_by_zero() {
        let t = now();
        assert!(LlamaCpp::rate(10, 1, t, Some(t), None).contains("tok/s"));
    }

    /// An average hides the ramp, and the ramp is what a reader notices.
    #[test]
    fn test_the_opening_is_reported_apart_from_the_rest() {
        let t = now();
        let start = t - 11.0;
        let first = t - 10.0; // 10s of decode
        let opening = t - 6.0; // the first 32 tokens took 4s of it

        let line = LlamaCpp::rate(900, 132, start, Some(first), Some(opening));

        // 132/10s overall, 32 in 4s at the start, 100 in the 6s after.
        assert!(line.contains("13.2 tok/s"), "{line}");
        assert!(line.contains(&format!("first {OPENING_TOKENS}: 8.0")), "{line}");
        assert!(line.contains("then 16.7"), "{line}");
    }

    #[test]
    fn test_a_generation_too_short_to_have_two_halves_reports_one_rate() {
        let t = now();
        let line = LlamaCpp::rate(900, OPENING_TOKENS, t - 3.0, Some(t - 2.0), Some(t - 1.0));
        assert!(line.contains("tok/s"), "{line}");
        assert!(!line.contains("first"), "{line}");
    }
}

mod batch_and_threads {
    use super::*;

    /// Restores the two override variables when dropped (`monkeypatch`).
    struct EnvGuard(Vec<(&'static str, Option<String>)>);

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.0.drain(..) {
                // SAFETY: the tests touching these variables run serially.
                unsafe {
                    match value {
                        Some(v) => std::env::set_var(key, v),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
    }

    fn pick(n_gpu_layers: i32, env: &[(&str, &str)]) -> (usize, usize) {
        let keys = ["SLP_LLM_N_BATCH", "SLP_LLM_N_THREADS"];
        let _guard = EnvGuard(keys.iter().map(|k| (*k, std::env::var(k).ok())).collect());
        // SAFETY: the tests touching these variables run serially.
        unsafe {
            for key in keys {
                std::env::remove_var(key);
            }
            for (key, value) in env {
                std::env::set_var(key, value);
            }
        }
        LlmFactory::batch_and_threads(n_gpu_layers)
    }

    #[test]
    #[serial]
    fn test_a_gpu_load_gets_the_bigger_prefill_batch() {
        let (n_batch, _threads) = pick(-1, &[]);
        assert_eq!(n_batch, 1024);
    }

    /// Only the layers that are not offloaded run on them, so a CPU-only load
    /// is exactly where a flat 4 left the machine idle.
    #[test]
    #[serial]
    fn test_a_cpu_load_gets_the_machine_s_threads() {
        let (_batch, n_threads) = pick(0, &[]);
        let cpu_count = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        assert_eq!(n_threads, cpu_count);
    }

    #[test]
    #[serial]
    fn test_the_environment_wins() {
        let (n_batch, n_threads) = pick(-1, &[("SLP_LLM_N_BATCH", "256"), ("SLP_LLM_N_THREADS", "2")]);
        assert_eq!((n_batch, n_threads), (256, 2));
    }
}
