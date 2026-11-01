//! Port of `smartloop/api/services/__init__.py`.
//!
//! API services: the model service, the platform telemetry client and the
//! storage/project-io helpers, plus the process-wide singletons.

use std::sync::Arc;

use once_cell::sync::OnceCell;

pub mod api_client;
pub mod model_service;
pub mod output_writer;
pub mod project_io;
pub mod storage_service;

pub use api_client::SLPAPIClient;
pub use model_service::ModelService;
pub use storage_service::{compute_usage, discard_partial_download, run_cleanup};

/// Global `ModelService` singleton (Python `_model_service`).
static MODEL_SERVICE: OnceCell<Arc<ModelService>> = OnceCell::new();

/// Global `SLPAPIClient` singleton (Python `_api_client`).
static API_CLIENT: OnceCell<Arc<SLPAPIClient>> = OnceCell::new();

/// Stand-in for FastAPI's `app.dependency_overrides[get_model_service]`: when
/// set, [`get_model_service`] returns this instead of the singleton.
static MODEL_SERVICE_OVERRIDE: once_cell::sync::Lazy<parking_lot::RwLock<Option<Arc<ModelService>>>> =
    once_cell::sync::Lazy::new(|| parking_lot::RwLock::new(None));

/// Make [`get_model_service`] return `service` (or, with `None`, restore the
/// singleton).  Tests only — the analogue of overriding the dependency.
pub fn set_model_service_override(service: Option<Arc<ModelService>>) {
    *MODEL_SERVICE_OVERRIDE.write() = service;
}

/// Get the global `ModelService` singleton, creating it on first use.
pub fn get_model_service() -> Arc<ModelService> {
    if let Some(service) = MODEL_SERVICE_OVERRIDE.read().clone() {
        return service;
    }
    MODEL_SERVICE.get_or_init(ModelService::instance).clone()
}

/// Get the global `SLPAPIClient` singleton, creating it on first use.
pub fn get_api_client() -> Arc<SLPAPIClient> {
    API_CLIENT.get_or_init(|| Arc::new(SLPAPIClient::new(None))).clone()
}

/// `datetime.now(timezone.utc).isoformat()`: microsecond precision (omitted
/// when zero, as Python does) and a `+00:00` offset.
pub(crate) fn utc_now_isoformat() -> String {
    use chrono::Timelike;
    let now = chrono::Utc::now();
    let base = now.to_rfc3339_opts(chrono::SecondsFormat::Secs, false); // ...+00:00
    let micros = now.nanosecond() % 1_000_000_000 / 1_000;
    if micros == 0 {
        base
    } else {
        // Splice the fraction in before the offset.
        let (stamp, offset) = base.split_at(base.len() - "+00:00".len());
        format!("{stamp}.{micros:06}{offset}")
    }
}

/// Python-compatible JSON text (`json.dumps`), shared by the services that
/// write files other tools (and older builds) read back.
pub(crate) mod pyjson {
    use serde_json::Value;

    /// `json.dumps(value)` (no indent): `", "` / `": "` separators and
    /// `ensure_ascii=True`.
    pub fn dumps(value: &Value) -> String {
        let mut out = String::new();
        write_value(&mut out, value, None, 0);
        out
    }

    /// `json.dumps(value, indent=indent)`: newline-separated items (`","`
    /// before each newline), `": "` key separator, `ensure_ascii=True`.
    pub fn dumps_indent(value: &Value, indent: usize) -> String {
        let mut out = String::new();
        write_value(&mut out, value, Some(indent), 0);
        out
    }

    fn newline(out: &mut String, indent: Option<usize>, level: usize) {
        if let Some(n) = indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', n * level));
        }
    }

    fn write_value(out: &mut String, value: &Value, indent: Option<usize>, level: usize) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => out.push_str(&n.to_string()),
            Value::String(s) => write_str(out, s),
            Value::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                        if indent.is_none() {
                            out.push(' ');
                        }
                    }
                    newline(out, indent, level + 1);
                    write_value(out, item, indent, level + 1);
                }
                newline(out, indent, level);
                out.push(']');
            }
            Value::Object(map) => {
                if map.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (key, item)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                        if indent.is_none() {
                            out.push(' ');
                        }
                    }
                    newline(out, indent, level + 1);
                    write_str(out, key);
                    out.push_str(": ");
                    write_value(out, item, indent, level + 1);
                }
                newline(out, indent, level);
                out.push('}');
            }
        }
    }

    fn write_str(out: &mut String, s: &str) {
        out.push('"');
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{08}' => out.push_str("\\b"),
                '\u{0c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                    let mut buf = [0u16; 2];
                    for unit in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{:04x}", unit));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;

        #[test]
        fn matches_python_compact_and_indented_output() {
            let v = json!({"a": [1, "é"], "b": {}, "c": null});
            assert_eq!(dumps(&v), r#"{"a": [1, "\u00e9"], "b": {}, "c": null}"#);
            assert_eq!(
                dumps_indent(&v, 2),
                "{\n  \"a\": [\n    1,\n    \"\\u00e9\"\n  ],\n  \"b\": {},\n  \"c\": null\n}"
            );
        }
    }
}
